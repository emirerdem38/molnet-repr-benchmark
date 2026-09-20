#!/usr/bin/env bash
# Submit one multi-seed GPU job.
#
# Usage (from benchmark_v4/):
#   bash cluster/submit_gpu_multiseed.sh esol 0
#   bash cluster/submit_gpu_multiseed.sh hiv 1 48:00:00 64G
#
# Args: DATASET SEED [TIME] [MEM]
# Results → results/multiseed/seed_{SEED}/gpu/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
# shellcheck source=rwth_config.sh
source "$SCRIPT_DIR/rwth_config.sh"

DATASET="${1:?Usage: bash cluster/submit_gpu_multiseed.sh <dataset> <seed> [time] [mem]}"
SEED="${2:?Usage: bash cluster/submit_gpu_multiseed.sh <dataset> <seed> [time] [mem]}"
TIME="${3:-}"
MEM="${4:-}"

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

NOTEBOOK="notebooks/multiseed/seed_${SEED}/gpu/benchmark_gpu_${DATASET}.ipynb"
if [[ ! -f "$NOTEBOOK" ]]; then
  echo "ERROR: Missing notebook: $NOTEBOOK"
  echo "Generate first:  python generate_notebooks.py --multiseed --seeds ${SEED}"
  exit 1
fi

mkdir -p cluster/logs "results/multiseed/seed_${SEED}/"{cpu,gpu,combined,splits}

JOB_ID=$(sbatch \
  --account="$RWTH_ACCOUNT" \
  --partition="$RWTH_PARTITION_GPU" \
  --job-name="ms${SEED}-gpu-${DATASET}" \
  --time="$TIME" \
  --mem="$MEM" \
  --mail-user="$RWTH_MAIL_USER" \
  cluster/run_gpu_multiseed.slurm "$DATASET" "$SEED" \
  | awk '{print $4}')

echo "Submitted multi-seed GPU job"
echo "  Dataset  : $DATASET"
echo "  Seed     : $SEED"
echo "  Job ID   : $JOB_ID"
echo "  Results  : results/multiseed/seed_${SEED}/gpu/"
echo "  Logs     : cluster/logs/"
echo ""
echo "Monitor:  squeue -u \$USER"
