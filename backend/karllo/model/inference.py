"""Load the active checkpoint and generate tokens. Does not train the model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import torch

from karllo.model.checkpoints import load_checkpoint, read_active
from karllo.model.tokenizer import KarlloTokenizer
from karllo.model.transformer import KarlloTransformer


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GenerationSettings:
    max_new_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 40


class InferenceEngine:
    def __init__(self) -> None:
        self.device = detect_device()
        self.model: Optional[KarlloTransformer] = None
        self.tokenizer: Optional[KarlloTokenizer] = None
        self.manifest: dict = {}
        self.checkpoint_path: Optional[str] = None
        self._stop: set[str] = set()

    def status(self) -> dict:
        trained = bool(self.manifest.get("trained"))
        return {
            "loaded": self.model is not None,
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "trained": trained,
            "notes": self.manifest.get("notes", ""),
            "architecture": self.manifest.get("architecture"),
            "parameters": self.manifest.get("parameters"),
            "config": self.manifest.get("config"),
            "kind": self.manifest.get("kind"),
            "honest_warning": None
            if trained
            else (
                "The active checkpoint is randomly initialized or untrained. "
                "It cannot answer questions reliably until you complete a real "
                "training run and activate that checkpoint."
            ),
        }

    def load_active(self) -> dict:
        pointer = read_active()
        if pointer is None:
            raise RuntimeError("No active checkpoint. Start the server once to create the original init.")
        return self.load_path(Path(pointer["path"]))

    def load_path(self, path: Path) -> dict:
        model, tokenizer, extra = load_checkpoint(path, map_location="cpu")
        model.to(self.device)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.manifest = extra.get("manifest") or {}
        self.checkpoint_path = str(path)
        return self.status()

    def request_stop(self, conversation_id: str) -> None:
        self._stop.add(conversation_id)

    def _build_prompt_ids(
        self,
        messages: list[dict[str, str]],
        prefix: str = "",
    ) -> torch.Tensor:
        assert self.tokenizer is not None
        toks = self.tokenizer
        ids: list[int] = [toks.bos_id]
        if prefix:
            ids.append(toks.system_id)
            ids.extend(toks.encode(prefix))
            ids.append(toks.eos_id)
        # Keep only the tail that fits in context minus generation budget.
        encoded_msgs: list[list[int]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            piece = [toks.assistant_id if role == "assistant" else toks.user_id]
            if role == "system":
                piece = [toks.system_id]
            piece.extend(toks.encode(content))
            piece.append(toks.eos_id)
            encoded_msgs.append(piece)
        encoded_msgs.append([toks.assistant_id])
        ctx = self.model.config.n_positions if self.model else 256
        flat: list[int] = []
        for piece in encoded_msgs:
            flat.extend(piece)
        if len(ids) + len(flat) > ctx:
            # Drop oldest messages first, keep system prefix.
            keep = ctx - len(ids) - 1
            flat = flat[-keep:]
            if flat and flat[0] not in (toks.user_id, toks.assistant_id, toks.system_id):
                flat = [toks.user_id] + flat
        ids.extend(flat)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def stream(
        self,
        messages: list[dict[str, str]],
        settings: Optional[GenerationSettings] = None,
        prefix: str = "",
        conversation_id: str = "",
    ) -> Generator[str, None, None]:
        if self.model is None or self.tokenizer is None:
            self.load_active()
        assert self.model is not None and self.tokenizer is not None
        settings = settings or GenerationSettings()
        idx = self._build_prompt_ids(messages, prefix=prefix)
        if conversation_id in self._stop:
            self._stop.discard(conversation_id)
        buf = ""
        for token in self.model.generate(
            idx,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            eos_id=self.tokenizer.eos_id,
        ):
            if conversation_id and conversation_id in self._stop:
                self._stop.discard(conversation_id)
                break
            if token == self.tokenizer.eos_id:
                break
            # Decode incrementally from a growing buffer of ids is expensive;
            # decode each token and skip incomplete utf-8 replacement noise.
            piece = self.tokenizer.decode([token], skip_special=True)
            if piece:
                buf += piece
                yield piece

    def generate_text(
        self,
        messages: list[dict[str, str]],
        settings: Optional[GenerationSettings] = None,
        prefix: str = "",
    ) -> str:
        return "".join(self.stream(messages, settings=settings, prefix=prefix))
