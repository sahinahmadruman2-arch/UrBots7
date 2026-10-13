"""Streaming dataset loading, validation, and packing.

Supported:
  - .txt raw corpus (pretraining / continued pretraining)
  - .jsonl with {text} or {prompt,response} or {messages}
  - .json list of the same
  - Question:/Answer: books

Does not load enormous files entirely into RAM. Records metadata while streaming.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

from karllo.model.tokenizer import KarlloTokenizer
from karllo.paths import DATASETS_DIR, ensure_dirs

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".txt", ".jsonl", ".json"}
QA_Q = re.compile(r"^Question:\s*(.*)$", re.IGNORECASE)
QA_A = re.compile(r"^Answer:\s*(.*)$", re.IGNORECASE)

PRETRAIN_MODES = {"pretrain", "continue_pretrain"}
SFT_MODES = {"sft"}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_dataset_id(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    slug = slug or "dataset"
    return f"{slug}-{uuid.uuid4().hex[:8]}"


def iter_text_file(path: Path, chunk_chars: int = 4000) -> Iterator[str]:
    buf = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            piece = f.read(64 * 1024)
            if not piece:
                if buf.strip():
                    yield buf
                break
            buf += piece
            while len(buf) >= chunk_chars:
                cut = buf.rfind("\n", 0, chunk_chars)
                if cut < chunk_chars // 2:
                    cut = chunk_chars
                yield buf[:cut]
                buf = buf[cut:]


def parse_qa_book(text: str) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    question: Optional[str] = None
    answer_lines: list[str] = []
    for line in text.splitlines():
        qm = QA_Q.match(line.strip())
        am = QA_A.match(line.strip())
        if qm:
            if question is not None:
                examples.append(
                    {"prompt": question, "response": "\n".join(answer_lines).strip()}
                )
            question = qm.group(1).strip()
            answer_lines = []
        elif am and question is not None:
            answer_lines = [am.group(1)]
        elif question is not None:
            answer_lines.append(line)
    if question is not None:
        examples.append({"prompt": question, "response": "\n".join(answer_lines).strip()})
    return examples


def _normalize_obj(obj: Any) -> Optional[dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, str):
        if not obj.strip():
            return None
        return {"kind": "text", "text": obj}
    if not isinstance(obj, dict):
        return None
    if "messages" in obj and isinstance(obj["messages"], list):
        msgs = []
        for m in obj["messages"]:
            if isinstance(m, dict) and "content" in m:
                msgs.append(
                    {
                        "role": str(m.get("role", "user")),
                        "content": str(m.get("content", "")),
                    }
                )
        if not msgs:
            return None
        return {"kind": "chat", "messages": msgs}
    if "prompt" in obj and "response" in obj:
        return {
            "kind": "sft",
            "prompt": str(obj["prompt"]),
            "response": str(obj["response"]),
        }
    if "question" in obj and "answer" in obj:
        return {
            "kind": "sft",
            "prompt": str(obj["question"]),
            "response": str(obj["answer"]),
        }
    if "text" in obj:
        t = str(obj["text"])
        if not t.strip():
            return None
        return {"kind": "text", "text": t}
    return None


def iter_examples(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        raw = path.read_text(encoding="utf-8", errors="replace")
        if QA_Q.search(raw) and QA_A.search(raw):
            for ex in parse_qa_book(raw):
                yield {"kind": "sft", **ex}
            return
        for chunk in iter_text_file(path):
            if chunk.strip():
                yield {"kind": "text", "text": chunk}
        return
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    yield {"kind": "malformed", "line": line_no, "raw": line[:200]}
                    continue
                norm = _normalize_obj(obj)
                if norm is None:
                    yield {"kind": "empty", "line": line_no}
                else:
                    yield norm
        return
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            yield {"kind": "malformed", "line": 0, "raw": "invalid json"}
            return
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            yield {"kind": "malformed", "line": 0, "raw": "json root must be list"}
            return
        for i, obj in enumerate(data):
            norm = _normalize_obj(obj)
            if norm is None:
                yield {"kind": "empty", "line": i}
            else:
                yield norm
        return
    raise ValueError(f"Unsupported dataset type: {suffix}")


def example_key(ex: dict[str, Any]) -> str:
    if ex.get("kind") == "text":
        return _hash_text(ex.get("text", ""))
    if ex.get("kind") == "sft":
        return _hash_text(ex.get("prompt", "") + "\0" + ex.get("response", ""))
    if ex.get("kind") == "chat":
        return _hash_text(json.dumps(ex.get("messages"), ensure_ascii=False))
    return _hash_text(json.dumps(ex, ensure_ascii=False))


def analyze_dataset(
    path: Path, tokenizer: Optional[KarlloTokenizer] = None, preview_n: int = 6
) -> dict[str, Any]:
    counts = {
        "text": 0,
        "sft": 0,
        "chat": 0,
        "empty": 0,
        "malformed": 0,
        "duplicates": 0,
    }
    seen: set[str] = set()
    chars = 0
    tokens = 0
    preview: list[dict[str, Any]] = []
    for ex in iter_examples(path):
        kind = ex.get("kind")
        if kind in ("empty", "malformed"):
            counts[kind] += 1
            continue
        key = example_key(ex)
        if key in seen:
            counts["duplicates"] += 1
            continue
        seen.add(key)
        if kind in counts:
            counts[kind] += 1
        if kind == "text":
            body = ex["text"]
        elif kind == "sft":
            body = f"Question: {ex['prompt']}\nAnswer: {ex['response']}"
        else:
            body = "\n".join(m.get("content", "") for m in ex.get("messages", []))
        chars += len(body)
        if tokenizer is not None:
            tokens += len(tokenizer.encode(body))
        if len(preview) < preview_n:
            preview.append({"kind": kind, "excerpt": body[:400]})
    example_count = counts["text"] + counts["sft"] + counts["chat"]
    if tokenizer is None:
        tokens = chars // 4  # rough estimate only
        token_note = "estimated (chars/4) — tokenizer not available"
    else:
        token_note = "counted with Karllo tokenizer"
    return {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size,
        "example_count": example_count,
        "counts": counts,
        "chars": chars,
        "tokens": tokens,
        "token_note": token_note,
        "preview": preview,
        "suggested_mode": "sft"
        if (counts["sft"] + counts["chat"]) >= counts["text"]
        else "pretrain",
    }


def save_dataset_record(record: dict[str, Any]) -> Path:
    ensure_dirs()
    meta_path = DATASETS_DIR / record["id"] / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return meta_path


def list_datasets() -> list[dict[str, Any]]:
    ensure_dirs()
    out = []
    for child in sorted(DATASETS_DIR.iterdir()):
        meta = child / "meta.json"
        if meta.exists():
            out.append(json.loads(meta.read_text(encoding="utf-8")))
    return out


def get_dataset(ds_id: str) -> dict[str, Any]:
    # Prevent path traversal
    if "/" in ds_id or "\\" in ds_id or ".." in ds_id:
        raise ValueError("invalid dataset id")
    meta = DATASETS_DIR / ds_id / "meta.json"
    if not meta.exists():
        raise FileNotFoundError(ds_id)
    return json.loads(meta.read_text(encoding="utf-8"))


def dataset_file(record: dict[str, Any]) -> Path:
    return Path(record["file"])


def split_indices(n: int, val_ratio: float = 0.1) -> tuple[list[int], list[int]]:
    n_val = max(1, int(n * val_ratio)) if n > 5 else max(0, n // 5)
    n_val = min(n_val, n - 1) if n > 1 else 0
    train = list(range(0, n - n_val))
    val = list(range(n - n_val, n))
    return train, val


def load_corpus_texts(path: Path, max_examples: Optional[int] = None) -> list[str]:
    texts: list[str] = []
    for ex in iter_examples(path):
        if ex.get("kind") == "text":
            texts.append(ex["text"])
        elif ex.get("kind") == "sft":
            texts.append(f"Question: {ex['prompt']}\nAnswer: {ex['response']}")
        elif ex.get("kind") == "chat":
            texts.append("\n".join(m.get("content", "") for m in ex.get("messages", [])))
        if max_examples is not None and len(texts) >= max_examples:
            break
    return texts


def encode_pretrain_document(tokenizer: KarlloTokenizer, text: str) -> list[int]:
    return tokenizer.encode(text) + [tokenizer.eos_id]


def encode_sft_example(
    tokenizer: KarlloTokenizer, ex: dict[str, Any]
) -> tuple[list[int], list[int]]:
    """Return (token_ids, labels) with prompt tokens masked to -100."""
    ignore = -100
    if ex.get("kind") == "chat":
        ids = [tokenizer.bos_id]
        labels = [ignore]
        for msg in ex["messages"]:
            role = msg.get("role", "user")
            content = tokenizer.encode(msg.get("content", ""))
            if role == "assistant":
                ids.append(tokenizer.assistant_id)
                labels.append(ignore)
                ids.extend(content)
                labels.extend(content)
                ids.append(tokenizer.eos_id)
                labels.append(tokenizer.eos_id)
            else:
                role_id = tokenizer.system_id if role == "system" else tokenizer.user_id
                ids.append(role_id)
                labels.append(ignore)
                ids.extend(content)
                labels.extend([ignore] * len(content))
                ids.append(tokenizer.eos_id)
                labels.append(ignore)
        return ids, labels
    # prompt/response
    prompt = ex.get("prompt") or ex.get("question") or ""
    response = ex.get("response") or ex.get("answer") or ""
    prompt_ids = tokenizer.encode_qa(prompt, answer=None)
    answer_ids = tokenizer.encode(" " + response.strip()) + [tokenizer.eos_id]
    ids = prompt_ids + answer_ids
    labels = [ignore] * len(prompt_ids) + answer_ids
    return ids, labels


def packed_blocks(
    token_streams: Iterator[list[int]],
    block_size: int,
    pad_id: int,
    max_blocks: Optional[int] = None,
) -> list[list[int]]:
    blocks: list[list[int]] = []
    buf: list[int] = []
    for seq in token_streams:
        buf.extend(seq)
        while len(buf) >= block_size:
            blocks.append(buf[:block_size])
            buf = buf[block_size:]
            if max_blocks is not None and len(blocks) >= max_blocks:
                return blocks
    if buf:
        buf = buf + [pad_id] * (block_size - len(buf))
        blocks.append(buf[:block_size])
    return blocks


def new_dataset_record(
    *,
    name: str,
    file_path: Path,
    source: str,
    analysis: dict[str, Any],
    license_note: str = "",
) -> dict[str, Any]:
    ds_id = safe_dataset_id(name)
    dest_dir = DATASETS_DIR / ds_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / file_path.name
    if Path(file_path).resolve() != dest_file.resolve():
        dest_file.write_bytes(Path(file_path).read_bytes())
    record = {
        "id": ds_id,
        "name": name,
        "file": str(dest_file),
        "source": source,
        "created_at": time.time(),
        "license_note": license_note,
        "analysis": analysis,
        "split": {"val_ratio": 0.1},
    }
    save_dataset_record(record)
    return record
