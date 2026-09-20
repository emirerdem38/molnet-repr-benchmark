#!/usr/bin/env python3
"""Benchmark v5 layout config.

v5 is a clean, flat multi-seed layout. Every seed is a first-class fresh run
(no "primary vs extra" distinction like v4). Both split modes are generated:

    notebooks/seed_{SEED}/{MODE}/{cpu,gpu}/benchmark_{cpu,gpu}_{dataset}.ipynb
    results/seed_{SEED}/{MODE}/{cpu,gpu,combined,splits,...}

with SEED in SEEDS and MODE in SPLIT_MODES.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent

# All five seeds are run fresh from the beginning with the fixed scaffold split.
SEEDS: List[int] = [0, 1, 2, 3, 4]

# Both split protocols are generated so scaffold-vs-random can be compared.
SPLIT_MODES: List[str] = ["scaffold", "random"]

DATASETS: List[str] = [
    "esol",
    "freesolv",
    "lipophilicity",
    "bace",
    "bbbp",
    "tox21",
    "hiv",
]


def seed_results_root(seed: int, mode: str) -> Path:
    return ROOT / "results" / f"seed_{seed}" / mode


def seed_notebook_dir(seed: int, mode: str, device: str) -> Path:
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be cpu or gpu, got {device!r}")
    if mode not in SPLIT_MODES:
        raise ValueError(f"mode must be one of {SPLIT_MODES}, got {mode!r}")
    return ROOT / "notebooks" / f"seed_{seed}" / mode / device


def ensure_layout(seed: int, mode: str) -> Path:
    root = seed_results_root(seed, mode)
    for sub in ("cpu", "gpu", "combined", "splits"):
        (root / sub).mkdir(parents=True, exist_ok=True)
        if sub in ("cpu", "gpu"):
            (root / sub / "hpo").mkdir(parents=True, exist_ok=True)
            (root / sub / "histories").mkdir(parents=True, exist_ok=True)
    return root
