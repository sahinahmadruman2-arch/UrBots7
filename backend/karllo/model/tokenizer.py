"""Byte-level BPE tokenizer compatible with Karllo checkpoints.

Special tokens occupy the first ids. Base vocabulary is UTF-8 bytes (256).
Learned BPE merges grow the vocabulary up to config.vocab_size.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|mask|>",
]

PAD, UNK, BOS, EOS, USER, ASSISTANT, SYSTEM, MASK = range(8)
N_SPECIAL = len(SPECIAL_TOKENS)
BYTE_OFFSET = N_SPECIAL  # byte 0 lives at id 8


class KarlloTokenizer:
    def __init__(
        self,
        merges: Optional[list[tuple[int, int]]] = None,
        vocab_size: int = 4096,
    ) -> None:
        self.vocab_size = vocab_size
        self.merges = merges or []
        self.special = {name: i for i, name in enumerate(SPECIAL_TOKENS)}
        self.id_to_special = {i: name for name, i in self.special.items()}
        self._merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        # Reverse map for decode of merged tokens: id -> byte sequence
        self._id_to_bytes = {BYTE_OFFSET + b: bytes([b]) for b in range(256)}
        next_id = BYTE_OFFSET + 256
        for a, b in self.merges:
            seq = self._id_to_bytes[a] + self._id_to_bytes[b]
            self._id_to_bytes[next_id] = seq
            next_id += 1

    @property
    def pad_id(self) -> int:
        return PAD

    @property
    def unk_id(self) -> int:
        return UNK

    @property
    def bos_id(self) -> int:
        return BOS

    @property
    def eos_id(self) -> int:
        return EOS

    @property
    def user_id(self) -> int:
        return USER

    @property
    def assistant_id(self) -> int:
        return ASSISTANT

    @property
    def system_id(self) -> int:
        return SYSTEM

    def _merge_once(self, ids: list[int]) -> list[int]:
        if len(ids) < 2 or not self._merge_ranks:
            return ids
        while True:
            best_rank = None
            best_i = None
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            a, b = ids[best_i], ids[best_i + 1]
            new_id = BYTE_OFFSET + 256 + self._merge_ranks[(a, b)]
            ids = ids[:best_i] + [new_id] + ids[best_i + 2 :]
        return ids

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        raw = text.encode("utf-8")
        ids = [BYTE_OFFSET + b for b in raw]
        ids = self._merge_once(ids)
        ids = [i if i < self.vocab_size else UNK for i in ids]
        if add_special:
            return [BOS] + ids + [EOS]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        chunks: list[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.id_to_special:
                if skip_special:
                    continue
                chunks.append(self.id_to_special[i].encode("utf-8"))
                continue
            seq = self._id_to_bytes.get(i)
            if seq is None:
                continue
            chunks.append(seq)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def encode_chat(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        ids: list[int] = [BOS]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                ids.append(ASSISTANT)
            elif role == "system":
                ids.append(SYSTEM)
            else:
                ids.append(USER)
            ids.extend(self.encode(content))
            ids.append(EOS)
        if add_generation_prompt:
            ids.append(ASSISTANT)
        return ids

    def encode_qa(self, question: str, answer: Optional[str] = None) -> list[int]:
        text = f"Question: {question.strip()}\nAnswer:"
        ids = [BOS] + self.encode(text)
        if answer is not None:
            ids.extend(self.encode(" " + answer.strip()))
            ids.append(EOS)
        return ids

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "merges": [list(p) for p in self.merges],
            "special_tokens": SPECIAL_TOKENS,
            "architecture": "karllo-bpe-v1",
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KarlloTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(p) for p in data.get("merges", [])]
        return cls(merges=merges, vocab_size=int(data["vocab_size"]))

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "n_merges": len(self.merges),
            "special_tokens": SPECIAL_TOKENS,
        }


def _byte_ids(text: str) -> list[int]:
    return [BYTE_OFFSET + b for b in text.encode("utf-8")]


def train_bpe(
    texts: Iterable[str],
    vocab_size: int,
    max_chars: int = 2_000_000,
) -> KarlloTokenizer:
    """Learn BPE merges from raw text. Caps corpus size so we do not fill RAM."""
    if vocab_size < BYTE_OFFSET + 256:
        raise ValueError("vocab_size too small for byte-level BPE")
    collected: list[str] = []
    total = 0
    for t in texts:
        if not t:
            continue
        collected.append(t)
        total += len(t)
        if total >= max_chars:
            break
    corpus = collected
    sequences = [_byte_ids(t) for t in corpus if t]
    n_merges = vocab_size - (BYTE_OFFSET + 256)
    merges: list[tuple[int, int]] = []
    next_id = BYTE_OFFSET + 256

    def pair_counts(seqs: list[list[int]]) -> Counter:
        c: Counter = Counter()
        for seq in seqs:
            for a, b in zip(seq, seq[1:]):
                c[(a, b)] += 1
        return c

    def apply_merge(seqs: list[list[int]], pair: tuple[int, int], new_id: int) -> list[list[int]]:
        a, b = pair
        out: list[list[int]] = []
        for seq in seqs:
            i = 0
            new_seq: list[int] = []
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            out.append(new_seq)
        return out

    for _ in range(n_merges):
        counts = pair_counts(sequences)
        if not counts:
            break
        pair, freq = counts.most_common(1)[0]
        if freq < 2:
            break
        merges.append(pair)
        sequences = apply_merge(sequences, pair, next_id)
        next_id += 1

    return KarlloTokenizer(merges=merges, vocab_size=vocab_size)


_word_re = re.compile(r"\S+|\s+")
