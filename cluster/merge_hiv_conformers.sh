#!/usr/bin/env bash
# Merge SLURM array chunks into data/hiv_conformers_n25.pkl
#
# Run from the repository root/ after all array tasks finished:
#   bash cluster/merge_hiv_conformers.sh

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "Merging in: $ROOT/data/"

if [[ -f "../.venv/bin/activate" ]]; then
  source "../.venv/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi
export PYTHONNOUSERSITE=1

python cluster/merge_conformer_chunks.py \
  --dataset hiv \
  --n_confs 25 \
  --data_dir ./data

echo ""
ls -lh ./data/hiv_conformers_n25.pkl

python - <<'PY'
import pickle
from pathlib import Path
p = Path("data/hiv_conformers_n25.pkl")
with open(p, "rb") as f:
    d = pickle.load(f)
print(f"graphs={len(d['graphs'])}  n_total={d['n_total']}  n_confs={d['n_confs']}")
PY
