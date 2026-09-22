#!/usr/bin/env bash
# Submit one multi-seed CPU notebook job (tabular models + scaffold split).
#
# Usage (from the repository root/):
#   bash cluster/submit_cpu_multiseed.sh esol 0
#   bash cluster/submit_cpu_multiseed.sh hiv 1 24:00:00 32G
#
# Args: DATASET SEED [TIME] [MEM]
# Partition/account: edit cluster/rwth_config.sh (RWTH_PARTITION_CPU, RWTH_ACCOUNT)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
# shellcheck source=rwth_config.sh
source "$SCRIPT_DIR/rwth_config.sh"

DATASET="${1:?Usage: bash cluster/submit_cpu_multiseed.sh <dataset> <seed> [time] [mem]}"
SEED="${2:?Usage: bash cluster/submit_cpu_multiseed.sh <dataset> <seed> [time] [mem]}"
TIME="${3:-}"
MEM="${4:-}"

case "$DATASET" in
  esol|freesolv)  DEFAULT_TIME="08:00:00"; DEFAULT_MEM="16G" ;;
  bace|bbbp)      DEFAULT_TIME="12:00:00"; DEFAULT_MEM="24G" ;;
  lipophilicity)  DEFAULT_TIME="16:00:00"; DEFAULT_MEM="24G" ;;
  tox21)          DEFAULT_TIME="24:00:00"; DEFAULT_MEM="32G" ;;
  hiv)            DEFAULT_TIME="24:00:00"; DEFAULT_MEM="48G" ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;
esac

TIME="${TIME:-$DEFAULT_TIME}"
MEM="${MEM:-$DEFAULT_MEM}"

NOTEBOOK="notebooks/multiseed/seed_${SEED}/cpu/benchmark_cpu_${DATASET}.ipynb"
if [[ ! -f "$NOTEBOOK" ]]; then
  echo "ERROR: Missing notebook: $NOTEBOOK"
  echo "Generate: python generate_notebooks.py --multiseed --seeds ${SEED}"
  exit 1
fi

if [[ -z "${RWTH_ACCOUNT:-}" ]]; then
  echo "ERROR: Set RWTH_ACCOUNT in cluster/rwth_config.sh or export RWTH_ACCOUNT=ACCOUNT_ID"
  echo "       List accounts: sacctmgr show user \$USER format=account%30"
  exit 1
fi

mkdir -p cluster/logs "results/multiseed/seed_${SEED}/"{cpu,gpu,combined,splits}

JOB_ID=$(sbatch \
  --account="$RWTH_ACCOUNT" \
  --partition="$RWTH_PARTITION_CPU" \
  --job-name="ms${SEED}-cpu-${DATASET}" \
  --time="$TIME" \
  --mem="$MEM" \
  --mail-user="$RWTH_MAIL_USER" \
  cluster/run_cpu_multiseed.slurm "$DATASET" "$SEED" \
  | awk '{print $4}')

echo "Submitted multi-seed CPU job"
echo "  Dataset   : $DATASET"
echo "  Seed      : $SEED"
echo "  Job ID    : $JOB_ID"
echo "  Account   : $RWTH_ACCOUNT"
echo "  Partition : $RWTH_PARTITION_CPU"
echo "  Results   : results/multiseed/seed_${SEED}/cpu/"
echo ""
echo "Monitor: squeue -u \$USER"
