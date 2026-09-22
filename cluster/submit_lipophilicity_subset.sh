#!/usr/bin/env bash
# Submit Lipophilicity train-subset jobs (scaffold + random, seed 0).
#
# Tabular → CPU partition; deep → GPU partition.
# Frozen HPO params from the full Lipophilicity benchmark (no retuning).
#
# Usage:
#   bash cluster/submit_lipophilicity_subset.sh
#   bash cluster/submit_lipophilicity_subset.sh --account ACCOUNT_ID
#   bash cluster/submit_lipophilicity_subset.sh --track cpu --modes scaffold
#   bash cluster/submit_lipophilicity_subset.sh --dry-run
#
# Env overrides (also set in rwth_config.sh):
#   RWTH_ACCOUNT, RWTH_PARTITION_CPU, RWTH_PARTITION_GPU, RWTH_MAIL_USER

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/rwth_config.sh"

SEED="0"
FRACTIONS="0.125,0.25,0.5,1.0"
TRACK="both"          # cpu | gpu | both
MODES="scaffold,random"
DRY=0
ACCOUNT="${RWTH_ACCOUNT}"
MAIL_USER="${RWTH_MAIL_USER}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) ACCOUNT="$2"; shift 2 ;;
    --track) TRACK="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --fractions) FRACTIONS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${ACCOUNT}" ]]; then
  echo "ERROR: Set RWTH_ACCOUNT in cluster/rwth_config.sh or pass --account ACCOUNT_ID"
  exit 1
fi

mkdir -p cluster/logs results/lipophilicity_subset

IFS=',' read -r -a MODE_ARR <<< "${MODES}"

# Thesis accounts (thes*) on CLAIX are typically GPU-only (c23g / c23g_low).
is_thesis_account=0
if [[ "${ACCOUNT}" == thes* ]]; then
  is_thesis_account=1
fi

submit_one() {
  local track="$1"
  local mode="$2"
  local script part name
  local extra_args=()
  if [[ "$track" == "cpu" ]]; then
    script="cluster/run_lipophilicity_subset_cpu.slurm"
    part="${RWTH_PARTITION_CPU}"
    name="v5-lipo-sub-cpu-${mode}"
    if [[ $is_thesis_account -eq 1 ]]; then
      # thes* cannot use c23ms: run tabular on a GPU node (CPU-only Python).
      part="${RWTH_PARTITION_GPU}"
      extra_args+=(--gres=gpu:1)
    fi
  else
    script="cluster/run_lipophilicity_subset_gpu.slurm"
    part="${RWTH_PARTITION_GPU}"
    name="v5-lipo-sub-gpu-${mode}"
  fi

  local cmd=(
    sbatch
    --account="${ACCOUNT}"
    --partition="${part}"
    --mail-user="${MAIL_USER}"
    --job-name="${name}"
    "${extra_args[@]}"
    "${script}"
    "${mode}" "${SEED}" "${FRACTIONS}"
  )

  echo "→ track=${track}  mode=${mode}  account=${ACCOUNT}  partition=${part}"
  printf '  %q' "${cmd[@]}"; echo

  if [[ $DRY -eq 1 ]]; then
    echo "  (dry-run)"
  else
    "${cmd[@]}"
  fi
}

echo "Lipophilicity subset submit"
echo "  account=${ACCOUNT}  seed=${SEED}  fractions=${FRACTIONS}"
echo "  track=${TRACK}  modes=${MODES}"
echo

for mode in "${MODE_ARR[@]}"; do
  mode="$(echo "$mode" | xargs)"
  [[ -z "$mode" ]] && continue
  case "$TRACK" in
    cpu)  submit_one cpu "$mode" ;;
    gpu)  submit_one gpu "$mode" ;;
    both)
      submit_one cpu "$mode"
      submit_one gpu "$mode"
      ;;
    *) echo "Unknown --track ${TRACK} (use cpu|gpu|both)"; exit 1 ;;
  esac
done

echo
echo "Done submitting. Monitor with: squeue -u \$USER"
echo "Results: results/lipophilicity_subset/"
