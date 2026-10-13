"""Versioned checkpoint save/load with integrity and compatibility checks."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import torch

from karllo.model.config import KarlloConfig
from karllo.model.tokenizer import KarlloTokenizer
from karllo.model.transformer import KarlloTransformer
from karllo.paths import (
    ACTIVE_POINTER,
    BEST_CKPT_DIR,
    CHECKPOINTS_DIR,
    LATEST_CKPT_DIR,
    ORIGINAL_CKPT_DIR,
    RUNS_CKPT_DIR,
    TOKENIZER_DIR,
    ensure_dirs,
)

MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "model.pt"
OPTIM_NAME = "optimizer.pt"
TOKENIZER_NAME = "tokenizer.json"
CONFIG_NAME = "config.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_dir(kind: str, run_id: Optional[str] = None) -> Path:
    if kind == "original":
        return ORIGINAL_CKPT_DIR
    if kind == "best":
        return BEST_CKPT_DIR
    if kind == "latest":
        return LATEST_CKPT_DIR
    if kind == "run":
        if not run_id:
            raise ValueError("run checkpoints require run_id")
        return RUNS_CKPT_DIR / run_id
    raise ValueError(f"unknown checkpoint kind {kind}")


def save_checkpoint(
    model: KarlloTransformer,
    tokenizer: KarlloTokenizer,
    dest: Path,
    *,
    kind: str,
    training_mode: Optional[str] = None,
    step: int = 0,
    metrics: Optional[dict[str, Any]] = None,
    optimizer: Any = None,
    scheduler: Any = None,
    trained: bool = False,
    notes: str = "",
    run_id: Optional[str] = None,
    parent: Optional[str] = None,
) -> dict[str, Any]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    weights_path = dest / WEIGHTS_NAME
    config_path = dest / CONFIG_NAME
    tok_path = dest / TOKENIZER_NAME
    payload = {
        "model_state": model.state_dict(),
        "config": model.config.to_dict(),
        "architecture": model.config.architecture,
    }
    torch.save(payload, weights_path)
    config_path.write_text(json.dumps(model.config.to_dict(), indent=2), encoding="utf-8")
    tokenizer.save(tok_path)
    if optimizer is not None:
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "step": step,
            },
            dest / OPTIM_NAME,
        )
    ckpt_id = dest.name if dest.name not in {"original", "best", "latest"} else kind
    if kind == "run":
        ckpt_id = dest.name
    manifest = {
        "id": ckpt_id,
        "kind": kind,
        "run_id": run_id,
        "parent": parent,
        "created_at": time.time(),
        "architecture": model.config.architecture,
        "config": model.config.to_dict(),
        "parameters": model.parameter_count(),
        "training_mode": training_mode,
        "step": step,
        "metrics": metrics or {},
        "trained": bool(trained),
        "notes": notes,
        "weights_sha256": _sha256_file(weights_path),
        "tokenizer_sha256": _sha256_file(tok_path),
        "weight_fingerprint": model.weight_fingerprint(),
        "has_optimizer": (dest / OPTIM_NAME).exists(),
    }
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(dest: Path) -> dict[str, Any]:
    path = Path(dest) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"No manifest in {dest}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checkpoint(dest: Path, *, run_inference: bool = True) -> dict[str, Any]:
    dest = Path(dest)
    manifest = load_manifest(dest)
    weights = dest / WEIGHTS_NAME
    tok = dest / TOKENIZER_NAME
    issues: list[str] = []
    if not weights.exists():
        issues.append("missing model.pt")
    if not tok.exists():
        issues.append("missing tokenizer.json")
    if weights.exists():
        digest = _sha256_file(weights)
        if digest != manifest.get("weights_sha256"):
            issues.append("model.pt hash mismatch")
    if tok.exists():
        digest = _sha256_file(tok)
        if digest != manifest.get("tokenizer_sha256"):
            issues.append("tokenizer.json hash mismatch")
    model = None
    tokenizer = None
    inference_ok = False
    if not issues:
        try:
            model, tokenizer, _m = load_checkpoint(dest, map_location="cpu")
            if model.config.architecture != "karllo-decoder-v1":
                issues.append("incompatible architecture")
            if tokenizer.vocab_size != model.config.vocab_size:
                issues.append("tokenizer vocab_size does not match model")
            if run_inference:
                model.eval()
                dummy = torch.tensor([[tokenizer.bos_id]], dtype=torch.long)
                logits, _ = model(dummy)
                if not torch.isfinite(logits).all():
                    issues.append("inference produced non-finite logits")
                else:
                    inference_ok = True
        except Exception as exc:  # noqa: BLE001
            issues.append(f"load/inference failed: {exc}")
    return {
        "ok": not issues,
        "issues": issues,
        "inference_ok": inference_ok,
        "manifest": manifest,
        "path": str(dest),
    }


def load_checkpoint(
    dest: Path,
    map_location: str = "cpu",
    load_optimizer: bool = False,
) -> tuple[KarlloTransformer, KarlloTokenizer, dict[str, Any]]:
    dest = Path(dest)
    weights = torch.load(dest / WEIGHTS_NAME, map_location=map_location, weights_only=False)
    cfg = KarlloConfig.from_dict(weights["config"])
    model = KarlloTransformer(cfg)
    model.load_state_dict(weights["model_state"])
    tokenizer = KarlloTokenizer.load(dest / TOKENIZER_NAME)
    manifest = load_manifest(dest) if (dest / MANIFEST_NAME).exists() else {}
    extra: dict[str, Any] = {"manifest": manifest, "optimizer": None, "scheduler": None, "step": 0}
    if load_optimizer and (dest / OPTIM_NAME).exists():
        blob = torch.load(dest / OPTIM_NAME, map_location=map_location, weights_only=False)
        extra["optimizer"] = blob.get("optimizer")
        extra["scheduler"] = blob.get("scheduler")
        extra["step"] = blob.get("step", 0)
    return model, tokenizer, extra


def copy_checkpoint(src: Path, dest: Path) -> None:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def list_checkpoints() -> list[dict[str, Any]]:
    ensure_dirs()
    found: list[dict[str, Any]] = []
    for kind, base in (
        ("original", ORIGINAL_CKPT_DIR),
        ("best", BEST_CKPT_DIR),
        ("latest", LATEST_CKPT_DIR),
    ):
        if (base / MANIFEST_NAME).exists():
            m = load_manifest(base)
            m["path"] = str(base)
            m["kind"] = kind
            found.append(m)
    if RUNS_CKPT_DIR.exists():
        for child in sorted(RUNS_CKPT_DIR.iterdir()):
            if (child / MANIFEST_NAME).exists():
                m = load_manifest(child)
                m["path"] = str(child)
                found.append(m)
            # nested step checkpoints
            for step_dir in sorted(child.glob("step-*")):
                if (step_dir / MANIFEST_NAME).exists():
                    m = load_manifest(step_dir)
                    m["path"] = str(step_dir)
                    found.append(m)
    return found


def read_active() -> Optional[dict[str, Any]]:
    if not ACTIVE_POINTER.exists():
        return None
    return json.loads(ACTIVE_POINTER.read_text(encoding="utf-8"))


def set_active(path: Path, *, verified: bool = True) -> dict[str, Any]:
    path = Path(path)
    report = verify_checkpoint(path)
    if not report["ok"]:
        raise ValueError(f"Cannot activate checkpoint: {report['issues']}")
    pointer = {
        "path": str(path.resolve()),
        "activated_at": time.time(),
        "verified": verified,
        "manifest": report["manifest"],
    }
    ACTIVE_POINTER.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    return pointer


def default_tokenizer_path() -> Path:
    return TOKENIZER_DIR / "tokenizer.json"


def ensure_original_checkpoint(config: Optional[KarlloConfig] = None) -> dict[str, Any]:
    """Create a randomly initialized original checkpoint once. Never overwrite it."""
    ensure_dirs()
    dest = ORIGINAL_CKPT_DIR
    if (dest / MANIFEST_NAME).exists():
        return load_manifest(dest)
    config = config or KarlloConfig.small()
    tok_path = default_tokenizer_path()
    if tok_path.exists():
        tokenizer = KarlloTokenizer.load(tok_path)
        # Keep model vocab aligned with tokenizer.
        config.vocab_size = tokenizer.vocab_size
    else:
        tokenizer = KarlloTokenizer(vocab_size=config.vocab_size)
        tokenizer.save(tok_path)
    model = KarlloTransformer(config)
    manifest = save_checkpoint(
        model,
        tokenizer,
        dest,
        kind="original",
        trained=False,
        notes=(
            "Randomly initialized weights. This checkpoint has not been trained "
            "and does not contain language knowledge."
        ),
    )
    if read_active() is None:
        set_active(dest)
    return manifest
