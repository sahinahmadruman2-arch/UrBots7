"""User-approved conversation memory. This is NOT model training."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from karllo.paths import MEMORY_DIR, ensure_dirs

MEM_FILE = MEMORY_DIR / "memories.json"


def _load() -> list[dict[str, Any]]:
    ensure_dirs()
    if not MEM_FILE.exists():
        return []
    return json.loads(MEM_FILE.read_text(encoding="utf-8"))


def _save(items: list[dict[str, Any]]) -> None:
    ensure_dirs()
    MEM_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def list_memories() -> list[dict[str, Any]]:
    return _load()


def add_memory(content: str, source: str = "user") -> dict[str, Any]:
    content = content.strip()
    if not content:
        raise ValueError("Memory cannot be empty")
    if len(content) > 2000:
        raise ValueError("Memory too long")
    items = _load()
    item = {
        "id": uuid.uuid4().hex[:12],
        "content": content,
        "source": source,
        "enabled": True,
        "created_at": time.time(),
    }
    items.append(item)
    _save(items)
    return item


def update_memory(mem_id: str, content: str | None = None, enabled: bool | None = None) -> dict[str, Any]:
    items = _load()
    for item in items:
        if item["id"] == mem_id:
            if content is not None:
                item["content"] = content.strip()
            if enabled is not None:
                item["enabled"] = bool(enabled)
            _save(items)
            return item
    raise FileNotFoundError(mem_id)


def delete_memory(mem_id: str) -> None:
    items = _load()
    nxt = [i for i in items if i["id"] != mem_id]
    if len(nxt) == len(items):
        raise FileNotFoundError(mem_id)
    _save(nxt)


def enabled_prefix(max_chars: int = 800) -> str:
    parts = []
    n = 0
    for item in _load():
        if not item.get("enabled"):
            continue
        line = f"- {item['content']}"
        if n + len(line) > max_chars:
            break
        parts.append(line)
        n += len(line)
    if not parts:
        return ""
    return (
        "User-approved memories (preferences and facts; these are not model weights):\n"
        + "\n".join(parts)
    )
