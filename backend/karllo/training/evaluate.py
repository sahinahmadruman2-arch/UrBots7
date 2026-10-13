"""Compare checkpoints on a fixed evaluation set. Never auto-promotes production."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import torch

from karllo.model.checkpoints import load_checkpoint, verify_checkpoint
from karllo.model.inference import GenerationSettings, detect_device
from karllo.paths import EVAL_DIR, ensure_dirs
from karllo.training.data import encode_sft_example, iter_examples


def load_eval_set(path: Optional[Path] = None) -> list[dict[str, Any]]:
    ensure_dirs()
    path = path or (EVAL_DIR / "test.jsonl")
    if not path.exists():
        return []
    return [e for e in iter_examples(path) if e.get("kind") in ("sft", "chat", "text")]


def _loss_on_set(model, tokenizer, examples: list[dict[str, Any]], device: str) -> Optional[float]:
    if not examples:
        return None
    model.eval()
    losses = []
    ctx = model.config.n_positions
    with torch.no_grad():
        for ex in examples[:200]:
            if ex.get("kind") in ("sft", "chat"):
                ids, labels = encode_sft_example(tokenizer, ex)
            else:
                ids = tokenizer.encode(ex.get("text", ""), add_special=True)
                labels = ids[1:] + [-100]
                ids = ids[:ctx]
                labels = labels[:ctx]
                if len(ids) < 2:
                    continue
            ids = ids[:ctx]
            labels = labels[:ctx]
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = torch.tensor([labels[: len(ids)]], dtype=torch.long, device=device)
            if y.shape[1] < x.shape[1]:
                pad = torch.full((1, x.shape[1] - y.shape[1]), -100, dtype=torch.long, device=device)
                y = torch.cat([y, pad], dim=1)
            _, loss = model(x, targets=y, ignore_index=-100)
            if loss is not None and torch.isfinite(loss):
                losses.append(float(loss.item()))
    if not losses:
        return None
    return sum(losses) / len(losses)


def qualitative_generate(model, tokenizer, questions: list[str], device: str) -> list[dict[str, str]]:
    model.eval()
    out = []
    settings = GenerationSettings(max_new_tokens=40, temperature=0.7, top_k=20)
    for q in questions:
        ids = tokenizer.encode_qa(q, answer=None)
        ctx = model.config.n_positions
        x = torch.tensor([ids[-ctx:]], dtype=torch.long, device=device)
        tokens = []
        for tok in model.generate(
            x,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=settings.top_k,
            eos_id=tokenizer.eos_id,
        ):
            if tok == tokenizer.eos_id:
                break
            tokens.append(tok)
        text = tokenizer.decode(tokens)
        out.append({"question": q, "response": text})
    return out


def evaluate_checkpoint(
    ckpt_path: Path,
    eval_path: Optional[Path] = None,
    qualitative_questions: Optional[list[str]] = None,
) -> dict[str, Any]:
    report = verify_checkpoint(ckpt_path)
    if not report["ok"]:
        return {"ok": False, "error": report["issues"], "path": str(ckpt_path)}
    device = detect_device()
    model, tokenizer, extra = load_checkpoint(ckpt_path, map_location="cpu")
    model.to(device)
    examples = load_eval_set(eval_path)
    loss = _loss_on_set(model, tokenizer, examples, device)
    questions = qualitative_questions or [
        "What is Karllo?",
        "What is 2 + 2?",
        "Name a planet in the solar system.",
    ]
    samples = qualitative_generate(model, tokenizer, questions, device)
    trained = bool(extra.get("manifest", {}).get("trained"))
    return {
        "ok": True,
        "path": str(ckpt_path),
        "device": device,
        "n_eval_examples": len(examples),
        "eval_loss": loss,
        "trained": trained,
        "samples": samples,
        "manifest": extra.get("manifest"),
        "evaluated_at": time.time(),
        "notes": [
            "eval_loss is measured with the Karllo checkpoint on the fixed eval set.",
            "Generated samples are not human quality scores; inspect them before promoting.",
            "This function never activates a checkpoint automatically.",
        ],
    }


def compare_checkpoints(path_a: Path, path_b: Path, eval_path: Optional[Path] = None) -> dict[str, Any]:
    a = evaluate_checkpoint(path_a, eval_path)
    b = evaluate_checkpoint(path_b, eval_path)
    regression = None
    if a.get("eval_loss") is not None and b.get("eval_loss") is not None:
        regression = b["eval_loss"] > a["eval_loss"] + 0.05
    return {
        "previous": a,
        "candidate": b,
        "val_loss_regression": regression,
        "recommendation": (
            "Do not promote the candidate: evaluation failed."
            if not b.get("ok")
            else (
                "Candidate has higher eval loss. Review samples before replacing production."
                if regression
                else "Candidate eval loss is not worse. Still requires manual approval to activate."
            )
        ),
    }


def save_eval_report(report: dict[str, Any], name: str) -> Path:
    ensure_dirs()
    dest = EVAL_DIR / f"{name}.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dest
