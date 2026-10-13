"""Honest hardware report. Never claims a GPU that is not present."""

from __future__ import annotations

import os
import shutil
from typing import Any

import torch


def hardware_report() -> dict[str, Any]:
    device = "cpu"
    gpu_name = None
    gpu_memory_gb = None
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        gpu_name = "Apple Metal"
    ram_gb = None
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        phys = os.sysconf("SC_PHYS_PAGES")
        ram_gb = round(page * phys / (1024**3), 2)
    except Exception:
        pass
    total, used, free = shutil.disk_usage("/")
    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "ram_gb": ram_gb,
        "disk_free_gb": round(free / (1024**3), 2),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "note": (
            "Training will run on CPU. Expect tiny/small configs and short runs."
            if device == "cpu"
            else f"Training device: {device}."
        ),
    }


def estimate_run(params: int, tokens: int, device: str) -> dict[str, Any]:
    """Very rough wall-time / storage estimates for the operator."""
    bytes_ckpt = params * 4  # fp32
    # Extremely rough tokens/sec guesses.
    tps = 200 if device == "cpu" else 4000
    seconds = tokens / max(tps, 1)
    return {
        "checkpoint_mb": round(bytes_ckpt / (1024 * 1024), 2),
        "rough_seconds": round(seconds, 1),
        "rough_note": (
            "Estimates are approximate for a packed next-token run. "
            "Measure a 20-step pilot before scaling up. Free notebook GPUs are quota-limited."
        ),
        "params": params,
        "tokens": tokens,
        "device": device,
    }
