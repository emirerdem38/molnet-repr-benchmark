#!/usr/bin/env bash
# Submit CPU multi-seed jobs for all datasets × given seeds.
#
# Usage:
#   bash cluster/submit_all_cpu_multiseed.sh 0
#   bash cluster/submit_all_cpu_multiseed.sh 0 1 2 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASETS=(esol freesolv lipophilicity bace bbbp tox21 hiv)

if [[ $# -eq 0 ]]; then
  SEEDS=(0)
else
  SEEDS=("$@")
fi

echo "Submitting CPU multi-seed jobs…"
echo "  Seeds    : ${SEEDS[*]}"
echo "  Datasets : ${DATASETS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    bash "$SCRIPT_DIR/submit_cpu_multiseed.sh" "$ds" "$seed"
    echo ""
  done
done

echo "All CPU multi-seed jobs submitted. Check: squeue -u \$USER"
echo "When they finish, submit GPU: bash cluster/submit_all_gpu_multiseed.sh ${SEEDS[*]}"
