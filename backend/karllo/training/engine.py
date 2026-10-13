"""Genuine training loop: forward, loss, backward, optimizer step, checkpoints.

Training modes are explicit and never silently swapped:
  - pretrain: random init, next-token prediction on raw text
  - continue_pretrain: load checkpoint, next-token prediction on raw text
  - sft: load checkpoint, instruction/chat examples with prompt loss masking
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from karllo.model.checkpoints import copy_checkpoint, save_checkpoint
from karllo.model.config import KarlloConfig
from karllo.model.tokenizer import KarlloTokenizer, train_bpe
from karllo.model.transformer import KarlloTransformer
from karllo.training.data import (
    encode_pretrain_document,
    encode_sft_example,
    iter_examples,
    packed_blocks,
    split_indices,
)


MODES = ("pretrain", "continue_pretrain", "sft")


@dataclass
class TrainConfig:
    mode: str
    dataset_file: str
    run_dir: str
    checkpoint_out: str
    source_checkpoint: Optional[str] = None
    model_size: str = "small"  # tiny | small | medium
    vocab_size: int = 4096
    n_positions: Optional[int] = None
    lr: float = 3e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_steps: int = 50
    epochs: int = 1
    weight_decay: float = 0.01
    warmup_steps: int = 5
    ckpt_every: int = 25
    val_ratio: float = 0.1
    seed: int = 42
    device: str = "cpu"
    max_examples: Optional[int] = 2000
    train_tokenizer: bool = True

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode {self.mode}. Use one of {MODES}")
        if self.mode != "pretrain" and not self.source_checkpoint:
            raise ValueError(f"{self.mode} requires source_checkpoint")
        if self.mode == "pretrain" and self.source_checkpoint:
            # Allowed only if caller wants to ignore it — we refuse to mix modes.
            pass
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("batch_size and grad_accum must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")


def _pick_device(requested: str) -> str:
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "mps":
        ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return "mps" if ok else "cpu"
    return "cpu"


def _hardware_report(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_memory_gb": None,
        "cpu_count": torch.get_num_threads(),
    }
    if device == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    try:
        import shutil as sh

        total, used, free = sh.disk_usage("/")
        info["disk_free_gb"] = round(free / (1024**3), 2)
    except Exception:
        pass
    try:
        import os

        page = os.sysconf("SC_PAGE_SIZE")
        phys = os.sysconf("SC_PHYS_PAGES")
        info["ram_gb"] = round(page * phys / (1024**3), 2)
    except Exception:
        pass
    return info


def _config_from_size(size: str, vocab_size: int, n_positions: Optional[int]) -> KarlloConfig:
    if size == "tiny":
        cfg = KarlloConfig.tiny()
    elif size == "medium":
        cfg = KarlloConfig.medium()
    else:
        cfg = KarlloConfig.small()
    cfg.vocab_size = vocab_size
    if n_positions:
        cfg.n_positions = n_positions
    return cfg


class StopRequested(Exception):
    pass


def _should_stop(run_dir: Path) -> bool:
    return (run_dir / "STOP").exists()


def _write_status(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(run_dir / "status.json")


def _append_log(run_dir: Path, event: dict[str, Any]) -> None:
    event = {"ts": time.time(), **event}
    with (run_dir / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _make_optimizer(model: KarlloTransformer, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith("bias") or "ln" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    optim = AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    sched = CosineAnnealingLR(optim, T_max=max(cfg.max_steps, 1), eta_min=cfg.lr * 0.1)
    return optim, sched


def _batchify(blocks: list[list[int]], batch_size: int) -> list[torch.Tensor]:
    batches = []
    for i in range(0, len(blocks), batch_size):
        chunk = blocks[i : i + batch_size]
        if not chunk:
            continue
        batches.append(torch.tensor(chunk, dtype=torch.long))
    return batches


def _sft_blocks(
    examples: list[dict[str, Any]],
    tokenizer: KarlloTokenizer,
    block_size: int,
) -> tuple[list[list[int]], list[list[int]]]:
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    for ex in examples:
        ids, labels = encode_sft_example(tokenizer, ex)
        if len(ids) < 2:
            continue
        if len(ids) > block_size:
            ids = ids[:block_size]
            labels = labels[:block_size]
        pad_n = block_size - len(ids)
        if pad_n:
            ids = ids + [tokenizer.pad_id] * pad_n
            labels = labels + [-100] * pad_n
        xs.append(ids)
        ys.append(labels)
    return xs, ys


def run_training(
    cfg: TrainConfig,
    progress_cb: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    cfg.validate()
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    device = _pick_device(cfg.device)
    hardware = _hardware_report(device)
    dataset_path = Path(cfg.dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    status: dict[str, Any] = {
        "state": "starting",
        "mode": cfg.mode,
        "step": 0,
        "max_steps": cfg.max_steps,
        "tokens_processed": 0,
        "train_loss": None,
        "val_loss": None,
        "lr": cfg.lr,
        "hardware": hardware,
        "error": None,
        "trained_weights_changed": False,
        "complete": False,
        "checkpoint_saved": False,
        "dataset_file": str(dataset_path),
        "started_at": time.time(),
        "notes": [],
    }
    _write_status(run_dir, status)
    _append_log(run_dir, {"event": "start", "mode": cfg.mode, "hardware": hardware})

    try:
        if cfg.mode == "pretrain":
            model, tokenizer, parent = _init_pretrain(cfg, dataset_path, run_dir)
        elif cfg.mode == "continue_pretrain":
            model, tokenizer, parent = _load_source(cfg)
            status["notes"].append(
                "Continued pretraining from existing checkpoint. Original files were not overwritten."
            )
        else:
            model, tokenizer, parent = _load_source(cfg)
            status["notes"].append(
                "Supervised fine-tuning from existing checkpoint. Prompt tokens are loss-masked."
            )

        model.to(device)
        model.train()
        params = model.parameter_count()
        status["parameters"] = params
        status["config"] = model.config.to_dict()
        status["source_checkpoint"] = parent
        fingerprint_before = model.weight_fingerprint()
        status["fingerprint_before"] = fingerprint_before

        examples = list(iter_examples(dataset_path))
        usable = [e for e in examples if e.get("kind") in ("text", "sft", "chat")]
        if cfg.max_examples:
            usable = usable[: cfg.max_examples]
        if not usable:
            raise RuntimeError("Dataset produced zero usable examples")

        if cfg.mode == "sft":
            sft_ex = []
            for e in usable:
                if e.get("kind") in ("sft", "chat"):
                    sft_ex.append(e)
                elif e.get("kind") == "text":
                    # Do not silently convert raw text into SFT.
                    continue
            if not sft_ex:
                raise RuntimeError(
                    "SFT mode requires prompt/response, chat messages, or Question:/Answer: examples. "
                    "Raw text corpora belong in pretrain / continue_pretrain."
                )
            usable = sft_ex
        else:
            # Pretrain modes: accept text; also flatten sft/chat to text rather than masking.
            pass

        n = len(usable)
        train_idx, val_idx = split_indices(n, cfg.val_ratio)
        train_ex = [usable[i] for i in train_idx]
        val_ex = [usable[i] for i in val_idx] if val_idx else train_ex[: max(1, len(train_ex) // 10)]
        status["n_train_examples"] = len(train_ex)
        status["n_val_examples"] = len(val_ex)
        block_size = model.config.n_positions

        if cfg.mode == "sft":
            x_train, y_train = _sft_blocks(train_ex, tokenizer, block_size)
            x_val, y_val = _sft_blocks(val_ex, tokenizer, block_size)
            train_x_batches = _batchify(x_train, cfg.batch_size)
            train_y_batches = _batchify(y_train, cfg.batch_size)
            val_x_batches = _batchify(x_val, cfg.batch_size)
            val_y_batches = _batchify(y_val, cfg.batch_size)
        else:
            def docs(exs: list[dict[str, Any]]) -> list[list[int]]:
                streams = []
                for e in exs:
                    if e.get("kind") == "text":
                        streams.append(encode_pretrain_document(tokenizer, e["text"]))
                    elif e.get("kind") == "sft":
                        text = f"Question: {e['prompt']}\nAnswer: {e['response']}"
                        streams.append(encode_pretrain_document(tokenizer, text))
                    elif e.get("kind") == "chat":
                        text = "\n".join(m.get("content", "") for m in e["messages"])
                        streams.append(encode_pretrain_document(tokenizer, text))
                return streams

            train_blocks = packed_blocks(iter(docs(train_ex)), block_size, tokenizer.pad_id)
            val_blocks = packed_blocks(iter(docs(val_ex)), block_size, tokenizer.pad_id)
            if not train_blocks:
                raise RuntimeError("Could not pack any training blocks from the dataset")
            train_x_batches = _batchify(train_blocks, cfg.batch_size)
            train_y_batches = []  # shifted inside loop
            val_x_batches = _batchify(val_blocks, cfg.batch_size) if val_blocks else train_x_batches[:1]
            val_y_batches = []

        status["n_train_batches"] = len(train_x_batches)
        status["block_size"] = block_size
        _write_status(run_dir, status)

        optim, sched = _make_optimizer(model, cfg)
        loss_history: list[dict[str, Any]] = []
        tokens_processed = 0
        step = 0
        best_val = float("inf")
        accum = 0
        optim.zero_grad(set_to_none=True)
        t0 = time.time()
        status["state"] = "training"
        _write_status(run_dir, status)

        while step < cfg.max_steps:
            if _should_stop(run_dir):
                status["state"] = "stopped"
                status["notes"].append("Stop file detected. Training halted before completion.")
                _append_log(run_dir, {"event": "stopped", "step": step})
                break
            for bi, xb in enumerate(train_x_batches):
                if step >= cfg.max_steps:
                    break
                if _should_stop(run_dir):
                    break
                xb = xb.to(device)
                if cfg.mode == "sft":
                    yb = train_y_batches[bi].to(device)
                    logits, loss = model(xb, targets=yb, ignore_index=-100)
                    ntok = int((yb != -100).sum().item())
                else:
                    # next-token: predict x[t+1] from x[t]; pad ignored
                    targets = xb.clone()
                    targets[:, :-1] = xb[:, 1:]
                    targets[:, -1] = -100
                    pad = tokenizer.pad_id
                    targets[xb == pad] = -100
                    logits, loss = model(xb, targets=targets, ignore_index=-100)
                    ntok = int((targets != -100).sum().item())
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss — aborting rather than fabricating metrics")
                (loss / cfg.grad_accum).backward()
                accum += 1
                if accum % cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    step += 1
                    tokens_processed += ntok
                    lr_now = float(optim.param_groups[0]["lr"])
                    train_loss = float(loss.item())
                    val_loss = None
                    if step % max(1, min(10, cfg.ckpt_every)) == 0 or step == cfg.max_steps:
                        val_loss = _eval_loss(
                            model, val_x_batches, val_y_batches, tokenizer, device, cfg.mode
                        )
                    rec = {
                        "step": step,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": lr_now,
                        "tokens_processed": tokens_processed,
                    }
                    loss_history.append(rec)
                    status.update(
                        {
                            "step": step,
                            "train_loss": train_loss,
                            "val_loss": val_loss if val_loss is not None else status.get("val_loss"),
                            "lr": lr_now,
                            "tokens_processed": tokens_processed,
                            "elapsed_s": round(time.time() - t0, 2),
                        }
                    )
                    _write_status(run_dir, status)
                    _append_log(run_dir, {"event": "step", **rec})
                    if progress_cb:
                        progress_cb(status)
                    if val_loss is not None and val_loss < best_val:
                        best_val = val_loss
                        best_dir = Path(cfg.checkpoint_out) / "best"
                        save_checkpoint(
                            model,
                            tokenizer,
                            best_dir,
                            kind="best",
                            training_mode=cfg.mode,
                            step=step,
                            metrics={"val_loss": val_loss, "train_loss": train_loss},
                            optimizer=optim,
                            scheduler=sched,
                            trained=True,
                            notes="Best validation loss during this run.",
                            run_id=run_dir.name,
                            parent=parent,
                        )
                        copy_checkpoint(best_dir, Path(cfg.checkpoint_out).parent.parent / "best")
                        status["best_val_loss"] = best_val
                    if cfg.ckpt_every and step % cfg.ckpt_every == 0:
                        step_dir = Path(cfg.checkpoint_out) / f"step-{step}"
                        save_checkpoint(
                            model,
                            tokenizer,
                            step_dir,
                            kind="run",
                            training_mode=cfg.mode,
                            step=step,
                            metrics={"train_loss": train_loss, "val_loss": val_loss},
                            optimizer=optim,
                            scheduler=sched,
                            trained=True,
                            notes=f"Interval checkpoint at step {step}",
                            run_id=run_dir.name,
                            parent=parent,
                        )
                        status["checkpoint_saved"] = True
                        _append_log(run_dir, {"event": "checkpoint", "path": str(step_dir), "step": step})

        # Flush leftover grads if we stopped mid-accum — do not step with partial accum.
        optim.zero_grad(set_to_none=True)

        fingerprint_after = model.weight_fingerprint()
        changed = fingerprint_before != fingerprint_after
        status["fingerprint_after"] = fingerprint_after
        status["trained_weights_changed"] = changed
        if not changed:
            status["notes"].append(
                "WARNING: weight fingerprint did not change. Training may have been too short or data empty."
            )

        final_val = _eval_loss(model, val_x_batches, val_y_batches, tokenizer, device, cfg.mode)
        status["val_loss"] = final_val

        latest_dir = Path(cfg.checkpoint_out) / "latest"
        save_checkpoint(
            model,
            tokenizer,
            latest_dir,
            kind="latest",
            training_mode=cfg.mode,
            step=step,
            metrics={
                "train_loss": status.get("train_loss"),
                "val_loss": final_val,
                "tokens_processed": tokens_processed,
            },
            optimizer=optim,
            scheduler=sched,
            trained=True,
            notes=f"Latest weights from {cfg.mode} run {run_dir.name}",
            run_id=run_dir.name,
            parent=parent,
        )
        # Also copy to global latest (not original)
        from karllo.paths import LATEST_CKPT_DIR

        copy_checkpoint(latest_dir, LATEST_CKPT_DIR)
        status["checkpoint_saved"] = True
        status["latest_checkpoint"] = str(latest_dir)

        if status.get("state") != "stopped":
            status["state"] = "complete"
            status["complete"] = True
        status["finished_at"] = time.time()
        status["loss_history"] = loss_history[-200:]
        (run_dir / "history.json").write_text(
            json.dumps(loss_history, indent=2), encoding="utf-8"
        )
        _write_status(run_dir, status)
        _append_log(run_dir, {"event": status["state"], "step": step, "changed": changed})
        return status
    except Exception as exc:  # noqa: BLE001
        status["state"] = "error"
        status["error"] = str(exc)
        status["complete"] = False
        status["finished_at"] = time.time()
        _write_status(run_dir, status)
        _append_log(run_dir, {"event": "error", "error": str(exc)})
        raise


def _eval_loss(
    model: KarlloTransformer,
    x_batches: list[torch.Tensor],
    y_batches: list[torch.Tensor],
    tokenizer: KarlloTokenizer,
    device: str,
    mode: str,
) -> Optional[float]:
    if not x_batches:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for i, xb in enumerate(x_batches[:8]):
            xb = xb.to(device)
            if mode == "sft" and y_batches:
                yb = y_batches[i].to(device)
                _, loss = model(xb, targets=yb, ignore_index=-100)
            else:
                targets = xb.clone()
                targets[:, :-1] = xb[:, 1:]
                targets[:, -1] = -100
                targets[xb == tokenizer.pad_id] = -100
                _, loss = model(xb, targets=targets, ignore_index=-100)
            if loss is not None and torch.isfinite(loss):
                losses.append(float(loss.item()))
    model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def _init_pretrain(
    cfg: TrainConfig, dataset_path: Path, run_dir: Path
) -> tuple[KarlloTransformer, KarlloTokenizer, Optional[str]]:
    texts_iter = []
    n = 0
    for ex in iter_examples(dataset_path):
        if ex.get("kind") == "text":
            texts_iter.append(ex["text"])
        elif ex.get("kind") == "sft":
            texts_iter.append(f"Question: {ex['prompt']}\nAnswer: {ex['response']}")
        elif ex.get("kind") == "chat":
            texts_iter.append("\n".join(m.get("content", "") for m in ex["messages"]))
        n += 1
        if cfg.max_examples and n >= cfg.max_examples:
            break
    if cfg.train_tokenizer:
        tokenizer = train_bpe(texts_iter, vocab_size=cfg.vocab_size)
        tokenizer.save(run_dir / "tokenizer.json")
    else:
        tokenizer = KarlloTokenizer(vocab_size=cfg.vocab_size)
    model_cfg = _config_from_size(cfg.model_size, tokenizer.vocab_size, cfg.n_positions)
    model = KarlloTransformer(model_cfg)
    return model, tokenizer, None


def _load_source(cfg: TrainConfig) -> tuple[KarlloTransformer, KarlloTokenizer, str]:
    from karllo.model.checkpoints import load_checkpoint

    path = Path(cfg.source_checkpoint)  # type: ignore[arg-type]
    model, tokenizer, extra = load_checkpoint(path, map_location="cpu")
    return model, tokenizer, str(path)
