#!/usr/bin/env bash
# Submit the FreeSolv same-machine smoke timing job.
#
# Usage:
#   bash cluster/submit_smoke_timing.sh
#   bash cluster/submit_smoke_timing.sh freesolv 0 scaffold
#   bash cluster/submit_smoke_timing.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/rwth_config.sh"

DATASET="freesolv"
SEED="0"
MODE="scaffold"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *)
      if [[ "$1" =~ ^(esol|freesolv|lipophilicity|bace|bbbp|tox21|hiv)$ ]]; then
        DATASET="$1"; shift
      elif [[ "$1" =~ ^[0-4]$ ]]; then
        SEED="$1"; shift
      elif [[ "$1" =~ ^(scaffold|random)$ ]]; then
        MODE="$1"; shift
      else
        echo "Unknown arg: $1"; exit 1
      fi
      ;;
  esac
done

mkdir -p cluster/logs results/smoke_timing

CMD=(
  sbatch
  --account="${RWTH_ACCOUNT}"
  --partition="${RWTH_PARTITION_GPU}"
  --mail-user="${RWTH_MAIL_USER}"
  --job-name="v5-smoke-${DATASET}"
  cluster/run_smoke_timing.slurm
  "${DATASET}" "${SEED}" "${MODE}"
)

echo "Submit smoke timing: dataset=${DATASET} seed=${SEED} mode=${MODE}"
echo "Account=${RWTH_ACCOUNT}  Partition=${RWTH_PARTITION_GPU}"
printf '  %q' "${CMD[@]}"; echo

if [[ $DRY -eq 1 ]]; then
  echo "(dry-run: not submitted)"
  exit 0
fi

"${CMD[@]}"
