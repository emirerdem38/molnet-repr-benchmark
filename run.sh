#!/usr/bin/env bash
# Run benchmark v5 notebooks with papermill (executes + saves outputs in place).
#
# Usage (from benchmark_v5/):
#   bash run.sh <seed> <mode> <device> [dataset]
#
#   seed    : 0 | 1 | 2 | 3 | 4
#   mode    : scaffold | random
#   device  : cpu | gpu
#   dataset : optional single slug (esol, freesolv, lipophilicity, bace,
#             bbbp, tox21, hiv). Omit to run all seven in order.
#
# Examples:
#   bash run.sh 0 scaffold cpu esol      # one dataset
#   bash run.sh 0 scaffold cpu           # all datasets, seed 0, scaffold, CPU
#   bash run.sh 0  random   gpu
#
# Each notebook is resumable: already-finished models are skipped, and metrics /
# HPO params / histories are written under results/seed_<seed>/<mode>/.

set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -lt 3 ]]; then
  sed -n '2,20p' "$0"; exit 1
fi

SEED="$1"; MODE="$2"; DEVICE="$3"; ONLY_DS="${4:-}"

case "$MODE" in scaffold|random) ;; *) echo "mode must be scaffold|random"; exit 1;; esac
case "$DEVICE" in cpu|gpu) ;; *) echo "device must be cpu|gpu"; exit 1;; esac

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYTHON=".venv/bin/python"
  elif [[ -x "../.venv/bin/python" ]]; then PYTHON="../.venv/bin/python"
  else PYTHON="python3"; fi
fi

DATASETS=(esol freesolv lipophilicity bace bbbp tox21 hiv)
if [[ -n "$ONLY_DS" ]]; then DATASETS=("$ONLY_DS"); fi

for ds in "${DATASETS[@]}"; do
  nb="notebooks/seed_${SEED}/${MODE}/${DEVICE}/benchmark_${DEVICE}_${ds}.ipynb"
  out_dir="results/seed_${SEED}/${MODE}/${DEVICE}"
  out="${out_dir}/${ds}_executed.ipynb"
  if [[ ! -f "$nb" ]]; then
    echo "Missing $nb: run: python generate_notebooks.py --seeds ${SEED} --modes ${MODE}"
    exit 1
  fi
  mkdir -p "$out_dir"
  echo "======== seed=${SEED} mode=${MODE} device=${DEVICE} dataset=${ds} ========"
  "$PYTHON" -m papermill "$nb" "$out" --log-output -k thesis-molnet \
    || "$PYTHON" -m papermill "$nb" "$out" --log-output
done

echo "Done: seed=${SEED} mode=${MODE} device=${DEVICE}."
