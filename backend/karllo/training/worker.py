"""Training worker process. Run separately from the API so HTTP stays responsive.

Usage:
  python -m karllo.training.worker --config /path/to/run/train_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Ensure backend is on path when launched as a script.
BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from karllo.training.engine import TrainConfig, run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Karllo training worker")
    parser.add_argument("--config", required=True, help="Path to train_config.json")
    args = parser.parse_args()
    path = Path(args.config)
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = TrainConfig(**raw)
    try:
        run_training(cfg)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
