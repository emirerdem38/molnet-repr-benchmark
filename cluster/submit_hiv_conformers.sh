#!/usr/bin/env bash
# Submit HIV conformer generation (parallel array, recommended).
#
# Usage from benchmark_v4/:
#   bash cluster/submit_hiv_conformers.sh
#
# After all array jobs finish:
#   bash cluster/merge_hiv_conformers.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
# shellcheck source=rwth_config.sh
source "$SCRIPT_DIR/rwth_config.sh"

mkdir -p cluster/logs data

if [[ ! -f data/hiv.csv ]]; then
  echo "ERROR: data/hiv.csv not found in $(pwd)/data/"
  exit 1
fi

if [[ "$RWTH_ACCOUNT" == "thesXXXX" ]]; then
  echo "WARNING: Set RWTH_ACCOUNT in cluster/rwth_config.sh before submitting"
  echo ""
fi

JOB_ID=$(sbatch \
  --account="$RWTH_ACCOUNT" \
  --partition="$RWTH_PARTITION_CPU" \
  --mail-user="$RWTH_MAIL_USER" \
  cluster/run_hiv_conformers_array.slurm | awk '{print $4}')

echo "Submitted HIV conformer array (20 chunks)"
echo "  Job ID    : ${JOB_ID}"
echo "  Account   : ${RWTH_ACCOUNT}"
echo "  Partition : ${RWTH_PARTITION_CPU}"
echo "  Chunks    : data/hiv_conformers_n25_chunk_*.pkl"
echo "  Final     : data/hiv_conformers_n25.pkl  (after merge)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "When done: bash cluster/merge_hiv_conformers.sh"
