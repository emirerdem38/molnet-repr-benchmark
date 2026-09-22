#!/usr/bin/env python3
"""Paths and constants for the multi-seed MoleculeNet benchmark.

Notebooks and results live under:

    notebooks/seed_{SEED}/{MODE}/{cpu,gpu}/...
    results/seed_{SEED}/{MODE}/{cpu,gpu,combined,splits}/...
"""

from __future__ import annotations

from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent

SEEDS: List[int] = [0, 1, 2, 3, 4]
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
