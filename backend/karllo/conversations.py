"""Conversation persistence. Private to the owner. Not used as training data unless exported."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from karllo.paths import CONVERSATIONS_DIR, ensure_dirs
from karllo.security import safe_join


def _path(cid: str) -> Path:
    if not cid or any(c in cid for c in "/\\"):
        raise ValueError("invalid conversation id")
    return safe_join(CONVERSATIONS_DIR, f"{cid}.json")


def list_conversations() -> list[dict[str, Any]]:
    ensure_dirs()
    items = []
    for p in CONVERSATIONS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items.append(
                {
                    "id": data["id"],
                    "title": data.get("title") or "New conversation",
                    "updated_at": data.get("updated_at"),
                    "created_at": data.get("created_at"),
                    "n_messages": len(data.get("messages", [])),
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
    return items


def get_conversation(cid: str) -> dict[str, Any]:
    path = _path(cid)
    if not path.exists():
        raise FileNotFoundError(cid)
    return json.loads(path.read_text(encoding="utf-8"))


def create_conversation(title: str = "New conversation") -> dict[str, Any]:
    ensure_dirs()
    cid = uuid.uuid4().hex[:12]
    rec = {
        "id": cid,
        "title": title,
        "created_at": time.time(),
        "updated_at": time.time(),
        "messages": [],
        "allow_training_export": False,
    }
    _path(cid).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def save_conversation(rec: dict[str, Any]) -> dict[str, Any]:
    rec["updated_at"] = time.time()
    _path(rec["id"]).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def rename_conversation(cid: str, title: str) -> dict[str, Any]:
    rec = get_conversation(cid)
    rec["title"] = title.strip()[:80] or rec["title"]
    return save_conversation(rec)


def delete_conversation(cid: str) -> None:
    path = _path(cid)
    if path.exists():
        path.unlink()


def append_messages(cid: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    rec = get_conversation(cid)
    rec["messages"].extend(messages)
    if rec.get("title") in ("", "New conversation") and messages:
        first = next((m["content"] for m in messages if m.get("role") == "user"), "")
        if first:
            rec["title"] = first.strip().split("\n")[0][:48]
    return save_conversation(rec)
