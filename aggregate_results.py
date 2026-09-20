#!/usr/bin/env python3
"""Aggregate benchmark v5 results into mean ± std tables across seeds.

Reads   results/seed_{SEED}/{MODE}/combined/{dataset}_results.csv
Writes  results/summary/{MODE}/{dataset}_mean_std.csv

Usage (from benchmark_v5/):
  python aggregate_results.py                      # all modes, all seeds
  python aggregate_results.py --modes scaffold
  python aggregate_results.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from v5_config import DATASETS, SEEDS, SPLIT_MODES, ROOT

MEAN_RE = re.compile(r"^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*(?:±|\\pm)?")


def parse_metric(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = MEAN_RE.match(str(value).strip())
    return float(m.group(1)) if m else None


def load_seed_table(seed: int, mode: str, dataset: str):
    path = ROOT / "results" / f"seed_{seed}" / mode / "combined" / f"{dataset}_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "Model" not in df.columns:
        return None
    out = df.copy()
    out["seed"] = seed
    for col in out.columns:
        if col not in ("Model", "seed"):
            out[col] = out[col].map(parse_metric)
    return out


def aggregate(dataset: str, mode: str, seeds):
    frames = [t for s in seeds if (t := load_seed_table(s, mode, dataset)) is not None]
    if not frames:
        return None
    long = pd.concat(frames, ignore_index=True)
    metric_cols = [c for c in long.columns if c not in ("Model", "seed")]
    rows = []
    for model, g in long.groupby("Model", sort=False):
        row = {"Model": model, "n_seeds": int(g["seed"].nunique())}
        for col in metric_cols:
            vals = g[col].dropna()
            if len(vals) == 0:
                row[col] = ""
            elif len(vals) == 1:
                row[col] = f"{vals.iloc[0]:.4f}"
            else:
                row[col] = f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--modes", nargs="+", default=SPLIT_MODES)
    args = parser.parse_args()

    written = 0
    for mode in args.modes:
        out_dir = ROOT / "results" / "summary" / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            table = aggregate(dataset, mode, args.seeds)
            if table is None:
                print(f"[skip] {mode}/{dataset}: no combined CSVs yet")
                continue
            out_path = out_dir / f"{dataset}_mean_std.csv"
            table.to_csv(out_path, index=False)
            print(f"[ok] {out_path.relative_to(ROOT)}")
            written += 1
    print(f"\nWrote {written} summary table(s).")


if __name__ == "__main__":
    main()
