"""Architecture hyperparameters for the Karllo decoder-only Transformer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class KarlloConfig:
    vocab_size: int = 4096
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    n_positions: int = 256
    dropout: float = 0.1
    bias: bool = False
    tie_embeddings: bool = True
    # Architecture name is stored so checkpoints can be rejected if incompatible.
    architecture: str = "karllo-decoder-v1"

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.vocab_size < 256 + 8:
            raise ValueError("vocab_size must cover byte tokens plus specials")
        if self.n_positions < 8:
            raise ValueError("n_positions too small")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KarlloConfig":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in allowed}
        return cls(**filtered)

    @classmethod
    def tiny(cls) -> "KarlloConfig":
        """CPU-friendly config for tests and smoke training."""
        return cls(
            vocab_size=512,
            n_embd=64,
            n_head=4,
            n_layer=2,
            n_positions=64,
            dropout=0.0,
        )

    @classmethod
    def small(cls) -> "KarlloConfig":
        """Default experimental config. Trainable on CPU for tiny corpora."""
        return cls()

    @classmethod
    def medium(cls) -> "KarlloConfig":
        """Larger experimental config. Prefer a GPU / Colab notebook."""
        return cls(
            vocab_size=8192,
            n_embd=256,
            n_head=8,
            n_layer=8,
            n_positions=512,
            dropout=0.1,
        )

    def estimated_parameters(self) -> int:
        """Rough parameter count used for hardware estimates before building the net."""
        v, d, L, t = self.vocab_size, self.n_embd, self.n_layer, self.n_positions
        embed = v * d + t * d
        # qkv + proj + 2 layer norms + FFN (4d) per layer
        per_layer = 4 * d * d + 2 * d + 2 * d * (4 * d) + 2 * d
        head = d  # final ln
        if self.tie_embeddings:
            lm = 0
        else:
            lm = v * d
        return embed + L * per_layer + head + lm
