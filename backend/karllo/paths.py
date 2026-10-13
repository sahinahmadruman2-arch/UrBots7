"""Project filesystem layout. All user content lives under data/ and checkpoints/."""

from __future__ import annotations

import os
from pathlib import Path

# backend/karllo/paths.py -> project root is parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
ROOT = Path(os.environ.get("KARLLO_ROOT", BACKEND_DIR.parent)).resolve()

DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
UPLOADS_DIR = DATA_DIR / "uploads"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
MEMORY_DIR = DATA_DIR / "memory"
RUNS_DIR = DATA_DIR / "runs"
EVAL_DIR = DATA_DIR / "eval"
SEED_DIR = DATA_DIR / "seed"
TOKENIZER_DIR = DATA_DIR / "tokenizer"

CHECKPOINTS_DIR = ROOT / "checkpoints"
ORIGINAL_CKPT_DIR = CHECKPOINTS_DIR / "original"
RUNS_CKPT_DIR = CHECKPOINTS_DIR / "runs"
BEST_CKPT_DIR = CHECKPOINTS_DIR / "best"
LATEST_CKPT_DIR = CHECKPOINTS_DIR / "latest"
ACTIVE_POINTER = CHECKPOINTS_DIR / "active.json"

NOTEBOOKS_DIR = ROOT / "notebooks"


def ensure_dirs() -> None:
    for path in (
        DATA_DIR,
        DATASETS_DIR,
        UPLOADS_DIR,
        CONVERSATIONS_DIR,
        KNOWLEDGE_DIR,
        MEMORY_DIR,
        RUNS_DIR,
        EVAL_DIR,
        SEED_DIR,
        TOKENIZER_DIR,
        CHECKPOINTS_DIR,
        ORIGINAL_CKPT_DIR,
        RUNS_CKPT_DIR,
        BEST_CKPT_DIR,
        LATEST_CKPT_DIR,
        NOTEBOOKS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
