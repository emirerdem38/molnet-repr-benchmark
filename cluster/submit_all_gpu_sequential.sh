#!/usr/bin/env bash
# Submit GPU jobs for all 7 datasets in series (each starts after the previous finishes).
#
# Usage (from the repository root/):
#   bash cluster/submit_all_gpu_sequential.sh
#
# Good when you only have one GPU allocation at a time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
# shellcheck source=rwth_config.sh
source "$SCRIPT_DIR/rwth_config.sh"

mkdir -p cluster/logs

DATASETS=(esol freesolv lipophilicity bace bbbp tox21 hiv)

# Default resources per dataset (must match submit_gpu.sh).
declare -A TIME MEM
TIME[esol]="24:00:00";       MEM[esol]="32G"
TIME[freesolv]="24:00:00";   MEM[freesolv]="32G"
TIME[lipophilicity]="48:00:00"; MEM[lipophilicity]="48G"
TIME[bace]="48:00:00";       MEM[bace]="32G"
TIME[bbbp]="48:00:00";       MEM[bbbp]="32G"
TIME[tox21]="72:00:00";      MEM[tox21]="48G"
TIME[hiv]="96:00:00";        MEM[hiv]="64G"

PREV_JOB=""

for ds in "${DATASETS[@]}"; do
  DEP_FLAG=()
  if [[ -n "$PREV_JOB" ]]; then
    DEP_FLAG=(--dependency=afterok:"$PREV_JOB")
  fi

  echo "Submitting $ds ${DEP_FLAG[*]:-(first in chain)}"
  OUT=$(sbatch \
    "${DEP_FLAG[@]}" \
    --account="$RWTH_ACCOUNT" \
    --partition="$RWTH_PARTITION_GPU" \
    --job-name="bench-gpu-${ds}" \
    --time="${TIME[$ds]}" \
    --mem="${MEM[$ds]}" \
    --mail-user="$RWTH_MAIL_USER" \
    cluster/run_gpu.slurm "$ds")
  echo "$OUT"
  PREV_JOB=$(echo "$OUT" | awk '{print $4}')
done

echo ""
echo "Sequential chain submitted. Last job ID: $PREV_JOB"
echo "Monitor: squeue -u \$USER"
