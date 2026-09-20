#!/usr/bin/env python3
"""Merge chunk .pkl files from parallel SLURM runs into one final conformer library."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


def _stats_to_dataframe(stats) -> pd.DataFrame | None:
    """Accept stats as DataFrame (legacy) or dict-of-lists (portable pickle)."""
    if stats is None:
        return None
    if isinstance(stats, pd.DataFrame):
        if len(stats) == 0:
            return None
        return stats
    if isinstance(stats, dict):
        if not stats:
            return None
        return pd.DataFrame(stats)
    raise TypeError(f"Unexpected stats type: {type(stats)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge conformer chunk pickle files")
    parser.add_argument("--dataset", required=True, help="e.g. hiv")
    parser.add_argument("--n_confs", type=int, default=25)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--pattern", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pattern = args.pattern or f"{args.dataset}_conformers_n{args.n_confs}_chunk_*.pkl"
    chunk_paths = sorted(data_dir.glob(pattern))

    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files matching {data_dir / pattern}")

    graphs, valid_idx, stats_parts = [], [], []
    n_total = n_confs = seed = None

    for path in chunk_paths:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        graphs.extend(payload["graphs"])
        valid_idx.extend(payload["valid_idx"])
        stats_df_part = _stats_to_dataframe(payload.get("stats"))
        if stats_df_part is not None:
            stats_parts.append(stats_df_part)
        n_total = payload.get("n_total", n_total)
        n_confs = payload.get("n_confs", n_confs)
        seed = payload.get("seed", seed)
        print(f"  + {path.name}: {len(payload['graphs'])} graphs")

    seen = set()
    graphs_dedup, valid_idx_dedup = [], []
    for g, idx in zip(graphs, valid_idx):
        if idx in seen:
            continue
        seen.add(idx)
        graphs_dedup.append(g)
        valid_idx_dedup.append(idx)

    stats_df = pd.concat(stats_parts, ignore_index=True) if stats_parts else pd.DataFrame()
    if len(stats_df) > 0:
        for col in stats_df.columns:
            if pd.api.types.is_string_dtype(stats_df[col]):
                stats_df[col] = stats_df[col].astype(str)

    out_path = data_dir / f"{args.dataset}_conformers_n{args.n_confs}.pkl"
    payload = {
        "graphs": graphs_dedup,
        "valid_idx": valid_idx_dedup,
        "stats": stats_df,
        "n_confs": n_confs,
        "seed": seed,
        "n_total": n_total,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    print(f"\nMerged {len(chunk_paths)} chunks -> {out_path}")
    print(f"  Graphs: {len(graphs_dedup)} / {n_total}")


if __name__ == "__main__":
    main()
