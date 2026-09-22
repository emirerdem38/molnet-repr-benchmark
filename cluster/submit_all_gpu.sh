#!/usr/bin/env bash
# Submit GPU jobs for all 7 datasets (parallel: one job per dataset).
#
# Usage (from the repository root/):
#   bash cluster/submit_all_gpu.sh
#
# For sequential runs (one after another), use:
#   bash cluster/submit_all_gpu_sequential.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATASETS=(esol freesolv lipophilicity bace bbbp tox21 hiv)

echo "Submitting ${#DATASETS[@]} GPU jobs (parallel)..."
echo ""

for ds in "${DATASETS[@]}"; do
  bash "$SCRIPT_DIR/submit_gpu.sh" "$ds"
  echo ""
done

echo "All jobs submitted. Check queue: squeue -u \$USER"
