#!/usr/bin/env bash
# Submit multi-seed GPU jobs for all datasets × given seeds.
#
# Usage (from benchmark_v4/):
#   bash cluster/submit_all_gpu_multiseed.sh
#   bash cluster/submit_all_gpu_multiseed.sh 0 1 2 3
#
# Requires CPU splits already present for each seed/dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATASETS=(esol freesolv lipophilicity bace bbbp tox21 hiv)
SEEDS=("${@:-0 1 2 3}")
# bash empty default quirk: if no args, SEEDS becomes one element "0 1 2 3"
if [[ ${#SEEDS[@]} -eq 1 && "$SEEDS" == *" "* ]]; then
  # shellcheck disable=SC2206
  SEEDS=($SEEDS)
fi
if [[ $# -eq 0 ]]; then
  SEEDS=(0 1 2 3)
fi

echo "Submitting GPU multi-seed jobs…"
echo "  Seeds    : ${SEEDS[*]}"
echo "  Datasets : ${DATASETS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    bash "$SCRIPT_DIR/submit_gpu_multiseed.sh" "$ds" "$seed"
    echo ""
  done
done

echo "All multi-seed GPU jobs submitted. Check: squeue -u \$USER"
