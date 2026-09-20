#!/usr/bin/env python3
"""Check conformer .pkl integrity (complete pickle stream + size).

Detects truncated / partial / corrupt files — the cause of
`UnpicklingError: pickle data was truncated` when loading 3D conformers.

The authoritative test walks the pickle structure with pickletools (standard
library only — no project imports, no custom classes needed), so a file that
ends early is reported as BAD even if its on-disk size looks plausible.

Run from benchmark_v5/ (works through the data/ symlink):
    python verify_conformers.py
Exit code is non-zero if any conformer file is missing or incomplete.
"""

from __future__ import annotations

import pickletools
import sys
from pathlib import Path

DATASETS = ["bace", "bbbp", "esol", "freesolv", "hiv", "lipophilicity", "tox21"]
DATA = Path(__file__).resolve().parent / "data"


def pickle_complete(path: Path) -> tuple[bool, str]:
    """True if the file is a complete pickle stream (ends with STOP)."""
    try:
        last = None
        with open(path, "rb") as f:
            for opcode, _arg, _pos in pickletools.genops(f):
                last = opcode.name
        return (last == "STOP"), ""
    except Exception as exc:  # truncated / corrupt streams raise here
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    print(f"Checking conformer pickles in: {DATA}\n")
    bad = []
    for name in DATASETS:
        path = DATA / f"{name}_conformers_n25.pkl"
        if not path.exists():
            print(f"[MISSING]     {name}")
            bad.append(name)
            continue
        size = path.stat().st_size
        complete, err = pickle_complete(path)
        if complete:
            print(f"[OK]          {name:14} {size:>11} bytes")
        else:
            print(f"[TRUNCATED]   {name:14} {size:>11} bytes  ({err})")
            bad.append(name)

    print()
    if bad:
        print(f"{len(bad)} file(s) need to be re-copied or regenerated: {', '.join(bad)}")
        print("Compare with your local machine:  md5sum data/<name>_conformers_n25.pkl")
        sys.exit(1)
    print("All conformer pickles are complete.")


if __name__ == "__main__":
    main()
