"""Decoder-only Transformer for next-token prediction.

A newly constructed instance has random weights. Random weights do not encode
language knowledge. Load a trained checkpoint before expecting coherent text.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from karllo.model.config import KarlloConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: KarlloConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.n_positions, config.n_positions)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _c = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_embd)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: KarlloConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.fc2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, config: KarlloConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class KarlloTransformer(nn.Module):
    def __init__(self, config: KarlloConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # Residual projections scaled like GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _b, t = idx.size()
        if t > self.config.n_positions:
            raise ValueError(
                f"Sequence length {t} exceeds context window {self.config.n_positions}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 40,
        eos_id: Optional[int] = None,
    ):
        """Autoregressive next-token generation. Yields token ids one at a time."""
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.n_positions :]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            token = int(next_id[0, 0].item())
            yield token
            if eos_id is not None and token == eos_id:
                break

    def parameter_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def weight_fingerprint(self) -> str:
        """Cheap hash of a few tensors so we can verify training actually moved weights."""
        import hashlib

        h = hashlib.sha256()
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.is_floating_point():
                    # Sample a few stats rather than hashing every byte (faster, still honest).
                    stats = torch.stack(
                        [
                            p.detach().float().mean().cpu(),
                            p.detach().float().std().cpu(),
                            p.detach().float().flatten()[0].cpu(),
                            p.detach().float().flatten()[-1].cpu(),
                        ]
                    )
                    h.update(name.encode())
                    h.update(stats.numpy().tobytes())
        return h.hexdigest()[:16]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
