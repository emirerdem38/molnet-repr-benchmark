#!/usr/bin/env bash
# Submit a single GPU benchmark job.
#
# Usage (from benchmark_v4/):
#   bash cluster/submit_gpu.sh esol
#   bash cluster/submit_gpu.sh hiv 48:00:00 64G
#
# CLAIX max walltime is 48:00:00 — resubmit the same command to continue from
# results/gpu/<dataset>_partial.json (FRESH_RUN=False in the notebook).
#
# Args: DATASET [TIME] [MEM]
# Edit cluster/rwth_config.sh once for --account and mail settings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
# shellcheck source=rwth_config.sh
source "$SCRIPT_DIR/rwth_config.sh"

DATASET="${1:?Usage: bash cluster/submit_gpu.sh <dataset> [time] [mem]
  Datasets: esol freesolv lipophilicity bace bbbp tox21 hiv}"

TIME="${2:-}"
MEM="${3:-}"

# Default walltime / memory per dataset (override via args 2 and 3).
case "$DATASET" in
  esol|freesolv)  DEFAULT_TIME="24:00:00"; DEFAULT_MEM="32G" ;;
  bace|bbbp)      DEFAULT_TIME="48:00:00"; DEFAULT_MEM="32G" ;;
  lipophilicity)  DEFAULT_TIME="48:00:00"; DEFAULT_MEM="48G" ;;
  tox21)          DEFAULT_TIME="48:00:00"; DEFAULT_MEM="48G" ;;
  hiv)            DEFAULT_TIME="48:00:00"; DEFAULT_MEM="64G" ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;
esac

TIME="${TIME:-$DEFAULT_TIME}"
MEM="${MEM:-$DEFAULT_MEM}"

if [[ "$RWTH_ACCOUNT" == "thesXXXX" ]]; then
  echo "WARNING: Set your thesis account in cluster/rwth_config.sh (RWTH_ACCOUNT=thes1234)"
  echo "         Or: export RWTH_ACCOUNT=thes1234 before submitting"
  echo "         Find it: sacctmgr show user \$USER format=account%30"
  echo ""
fi

mkdir -p cluster/logs

JOB_ID=$(sbatch \
  --account="$RWTH_ACCOUNT" \
  --partition="$RWTH_PARTITION_GPU" \
  --job-name="bench-gpu-${DATASET}" \
  --time="$TIME" \
  --mem="$MEM" \
  --mail-user="$RWTH_MAIL_USER" \
  cluster/run_gpu.slurm "$DATASET" \
  | awk '{print $4}')

echo "Submitted bench-gpu-${DATASET}"
echo "  Job ID   : $JOB_ID"
echo "  Account  : $RWTH_ACCOUNT"
echo "  Partition: $RWTH_PARTITION_GPU"
echo "  Time     : $TIME"
echo "  Memory   : $MEM"
echo "  Logs     : cluster/logs/gpu_bench-gpu-${DATASET}_${JOB_ID}.out"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Cancel:   scancel $JOB_ID"
