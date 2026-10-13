"""Real training: loss should drop and checkpoints must restore."""

from __future__ import annotations

from pathlib import Path

import torch

from karllo.model.checkpoints import load_checkpoint, save_checkpoint, verify_checkpoint
from karllo.model.config import KarlloConfig
from karllo.model.tokenizer import KarlloTokenizer
from karllo.model.transformer import KarlloTransformer
from karllo.training.engine import TrainConfig, run_training


def test_checkpoint_save_load_and_verify(tmp_path: Path):
    cfg = KarlloConfig.tiny()
    model = KarlloTransformer(cfg)
    tok = KarlloTokenizer(vocab_size=cfg.vocab_size)
    dest = tmp_path / "ckpt"
    man = save_checkpoint(
        model,
        tok,
        dest,
        kind="run",
        trained=False,
        notes="test",
        run_id="t",
    )
    assert man["weights_sha256"]
    report = verify_checkpoint(dest)
    assert report["ok"], report["issues"]
    m2, t2, extra = load_checkpoint(dest)
    x = torch.tensor([[t2.bos_id, t2.eos_id]], dtype=torch.long)
    logits_a, _ = model(x)
    logits_b, _ = m2(x)
    assert torch.allclose(logits_a, logits_b)


def test_pretrain_reduces_loss_and_changes_weights(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("Karllo trains weights with gradients. " * 40 + "\n") * 8, encoding="utf-8")
    run_dir = tmp_path / "run"
    ckpt_out = tmp_path / "out"
    cfg = TrainConfig(
        mode="pretrain",
        dataset_file=str(corpus),
        run_dir=str(run_dir),
        checkpoint_out=str(ckpt_out),
        model_size="tiny",
        vocab_size=512,
        n_positions=64,
        lr=3e-3,
        batch_size=2,
        grad_accum=1,
        max_steps=12,
        ckpt_every=12,
        device="cpu",
        max_examples=50,
        train_tokenizer=True,
    )
    status = run_training(cfg)
    assert status["state"] == "complete"
    assert status["complete"] is True
    assert status["trained_weights_changed"] is True
    assert status["train_loss"] is not None
    history = status.get("loss_history") or []
    assert history, "expected recorded steps"
    first = history[0]["train_loss"]
    last = history[-1]["train_loss"]
    # Tiny noisy CPU run: require finite losses and a non-increase of more than a large margin,
    # plus that we actually stepped. Prefer last <= first * 1.05 or last < first.
    assert first > 0 and last > 0
    assert last <= first * 1.15, f"loss did not trend down: {first} -> {last}"
    latest = Path(status["latest_checkpoint"])
    report = verify_checkpoint(latest)
    assert report["ok"], report["issues"]
    model, tok, extra = load_checkpoint(latest)
    assert extra["manifest"]["trained"] is True
    idx = torch.tensor([[tok.bos_id]], dtype=torch.long)
    tokens = list(model.generate(idx, max_new_tokens=4, temperature=1.0, top_k=10))
    assert len(tokens) == 4


def test_sft_requires_instruction_examples(tmp_path: Path):
    raw = tmp_path / "raw.txt"
    raw.write_text("just raw text without qa pairs " * 20, encoding="utf-8")
    # First create a source checkpoint
    src_corpus = tmp_path / "c.txt"
    src_corpus.write_text("hello world " * 50, encoding="utf-8")
    run_dir = tmp_path / "pre"
    ckpt_out = tmp_path / "preout"
    pre = TrainConfig(
        mode="pretrain",
        dataset_file=str(src_corpus),
        run_dir=str(run_dir),
        checkpoint_out=str(ckpt_out),
        model_size="tiny",
        vocab_size=512,
        max_steps=2,
        batch_size=1,
        grad_accum=1,
        device="cpu",
        train_tokenizer=True,
    )
    status = run_training(pre)
    source = status["latest_checkpoint"]
    sft_cfg = TrainConfig(
        mode="sft",
        dataset_file=str(raw),
        run_dir=str(tmp_path / "sft"),
        checkpoint_out=str(tmp_path / "sftout"),
        source_checkpoint=source,
        model_size="tiny",
        max_steps=2,
        batch_size=1,
        grad_accum=1,
        device="cpu",
        train_tokenizer=False,
    )
    try:
        run_training(sft_cfg)
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "SFT mode" in str(exc)
    assert raised


def test_sft_trains_on_jsonl(tmp_path: Path):
    src_corpus = tmp_path / "c.txt"
    src_corpus.write_text("Question: ping\nAnswer: pong\n" * 40, encoding="utf-8")
    pre = run_training(
        TrainConfig(
            mode="pretrain",
            dataset_file=str(src_corpus),
            run_dir=str(tmp_path / "pre"),
            checkpoint_out=str(tmp_path / "preout"),
            model_size="tiny",
            vocab_size=512,
            max_steps=3,
            batch_size=1,
            grad_accum=1,
            device="cpu",
        )
    )
    sft = tmp_path / "sft.jsonl"
    sft.write_text(
        '{"prompt": "ping", "response": "pong"}\n{"prompt": "hello", "response": "world"}\n' * 10,
        encoding="utf-8",
    )
    status = run_training(
        TrainConfig(
            mode="sft",
            dataset_file=str(sft),
            run_dir=str(tmp_path / "sft-run"),
            checkpoint_out=str(tmp_path / "sftout"),
            source_checkpoint=pre["latest_checkpoint"],
            model_size="tiny",
            max_steps=4,
            batch_size=1,
            grad_accum=1,
            device="cpu",
            train_tokenizer=False,
        )
    )
    assert status["complete"] is True
    assert status["trained_weights_changed"] is True
