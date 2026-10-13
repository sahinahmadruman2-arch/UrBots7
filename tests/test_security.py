from pathlib import Path

import pytest

from karllo.security import allowed_upload_name, safe_join, verify_password, ensure_owner


def test_path_traversal_blocked(tmp_path: Path, monkeypatch):
    with pytest.raises(ValueError):
        safe_join(tmp_path, "..", "etc", "passwd")
    dest = safe_join(tmp_path, "ok.txt")
    assert dest.parent == tmp_path.resolve()


def test_upload_names():
    assert allowed_upload_name("data.jsonl")
    assert allowed_upload_name("book.txt")
    assert not allowed_upload_name("../x.txt")
    assert not allowed_upload_name("x.exe")
    assert not allowed_upload_name("a/b.txt")


def test_password_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KARLLO_OWNER_PASSWORD", "secret-test")
    from karllo import paths, security

    monkeypatch.setattr(security, "OWNER_FILE", tmp_path / "owner.json")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    rec = ensure_owner()
    assert rec["must_change"] is False
    assert verify_password("secret-test")
    assert not verify_password("nope")
