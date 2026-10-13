"""Owner authentication, path safety, and upload limits."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from karllo.paths import DATA_DIR, ensure_dirs

OWNER_FILE = DATA_DIR / "owner.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"
PBKDF2_ITERS = 120_000
SESSION_TTL_S = 60 * 60 * 12
DEFAULT_DEV_PASSWORD = "karllo-owner"


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERS)
    return dk.hex()


def ensure_owner() -> dict:
    ensure_dirs()
    if OWNER_FILE.exists():
        return json.loads(OWNER_FILE.read_text(encoding="utf-8"))
    password = os.environ.get("KARLLO_OWNER_PASSWORD", DEFAULT_DEV_PASSWORD)
    salt = secrets.token_bytes(16)
    rec = {
        "salt": salt.hex(),
        "password_hash": _hash_password(password, salt),
        "created_at": time.time(),
        "must_change": os.environ.get("KARLLO_OWNER_PASSWORD") is None,
        "hint": (
            "Default password is 'karllo-owner'. Set KARLLO_OWNER_PASSWORD before deploying."
            if os.environ.get("KARLLO_OWNER_PASSWORD") is None
            else "Owner password loaded from environment."
        ),
    }
    OWNER_FILE.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def verify_password(password: str) -> bool:
    rec = ensure_owner()
    salt = bytes.fromhex(rec["salt"])
    expected = rec["password_hash"]
    got = _hash_password(password, salt)
    return hmac.compare_digest(expected, got)


def _sessions() -> dict:
    if not SESSIONS_FILE.exists():
        return {}
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_sessions(data: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(data), encoding="utf-8")


def create_session() -> str:
    token = secrets.token_urlsafe(32)
    data = _sessions()
    data[token] = time.time() + SESSION_TTL_S
    # drop expired
    now = time.time()
    data = {k: v for k, v in data.items() if v > now}
    _save_sessions(data)
    return token


def valid_session(token: Optional[str]) -> bool:
    if not token:
        return False
    data = _sessions()
    exp = data.get(token)
    if not exp:
        return False
    if exp < time.time():
        data.pop(token, None)
        _save_sessions(data)
        return False
    return True


def revoke_session(token: str) -> None:
    data = _sessions()
    data.pop(token, None)
    _save_sessions(data)


def safe_join(base: Path, *parts: str) -> Path:
    base = base.resolve()
    dest = base.joinpath(*parts).resolve()
    if not str(dest).startswith(str(base)):
        raise ValueError("path traversal blocked")
    return dest


def allowed_upload_name(filename: str) -> bool:
    name = Path(filename).name
    if name != filename or "/" in filename or "\\" in filename or ".." in filename:
        return False
    return Path(name).suffix.lower() in {".txt", ".jsonl", ".json"}
