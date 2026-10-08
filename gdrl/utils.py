"""Seeding, run folders and metric logging. Plumbing."""

from __future__ import annotations

import csv
import random
import re
import sys
import time
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:  # only seed torch if something already imported it
        import torch

        torch.manual_seed(seed)


def make_run_dir(runs_dir: str | Path, name: str, seed: int) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "run"
    run_dir = Path(runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}_s{seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


class RunLogger:
    """Writes every scalar to ``metrics.csv`` and, if available, TensorBoard.

    View curves with:  tensorboard --logdir runs
    """

    def __init__(self, run_dir: Path, tensorboard: bool = True):
        self._file = open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["step", "key", "value"])
        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(run_dir))
            except Exception:  # torch or tensorboard missing: CSV only
                self._tb = None

    def log(self, step: int, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            value = float(value)
            self._csv.writerow([step, key, value])
            if self._tb is not None:
                self._tb.add_scalar(key, value, step)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        if self._tb is not None:
            self._tb.close()
