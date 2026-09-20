#!/usr/bin/env python3
"""Re-save a conformer .pkl with portable dtypes (fixes pandas StringDtype pickle errors)."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from conformer_generation import _pickle_safe_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Path to *_conformers_n25.pkl")
    parser.add_argument("--in-place", action="store_true", default=True,
                        help="Overwrite input file (default)")
    parser.add_argument("--out", type=str, default=None, help="Optional output path")
    args = parser.parse_args()

    src = Path(args.path)
    dst = Path(args.out) if args.out else src

    print(f"pandas {pd.__version__}")
    print(f"Loading {src} ...")

    with open(src, "rb") as f:
        payload = pickle.load(f)

    payload = _pickle_safe_payload(payload)
    n_graphs = len(payload["graphs"])
    n_total = payload.get("n_total", "?")

    with open(dst, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    print(f"Saved {dst}")
    print(f"  graphs={n_graphs}  n_total={n_total}")
    print(f"  stats={'dict' if isinstance(payload.get('stats'), dict) else type(payload.get('stats'))}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        print(
            "\nCould not load pickle — run with the project venv (pandas 3.x), then retry:\n"
            "  ../.venv/bin/python repack_conformer_pkl.py --path YOUR.pkl\n"
            f"\nError: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
