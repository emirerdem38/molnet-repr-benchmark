#!/usr/bin/env bash
# Benchmark v5 — modular SLURM submitter.
#
# Pick a device (cpu/gpu), split mode (scaffold/random), seed, and one or all
# datasets. Reads cluster/rwth_config.sh for --account / --partition / --mail so
# nothing is hardcoded. Results are written automatically under
# results/seed_<seed>/<mode>/.
#
# Usage:
#   bash cluster/submit.sh --device cpu --seed 0 --mode scaffold esol
#   bash cluster/submit.sh -d gpu -s 0 -m random esol
#   bash cluster/submit.sh -d cpu -s 0 -m scaffold --all      # all 7 datasets
#   bash cluster/submit.sh -d cpu -s 0 --all                  # mode defaults to scaffold
#   bash cluster/submit.sh -d gpu -s 2 -m scaffold hiv --dry-run
#   bash cluster/submit.sh -d gpu -s 1 -m random hiv --time 7-00:00:00 --mem 64G
#   bash cluster/submit.sh -d gpu -s 2 -m random hiv --account rwth2003 --after-dmpnn --time 3-00:00:00
#
# Flags:
#   -d, --device   cpu | gpu                 (required)
#   -s, --seed     0 | 1 | 2 | 3 | 4         (required)
#   -m, --mode     scaffold | random         (default: scaffold)
#   -a, --all      submit all datasets for that seed/mode/device
#       --time     SLURM walltime (e.g. 7-00:00:00); rwth2175 max is 7 days
#       --mem      SLURM memory (e.g. 64G)
#       --account  override RWTH_ACCOUNT from rwth_config.sh (e.g. rwth2003)
#       --skip-models  pipe-separated model names to skip, e.g. 'GIN (2D)|D-MPNN (2D)'
#       --after-dmpnn  shortcut: skip GIN (2D) and D-MPNN (2D); run GIN (3D)+SchNet+LSTM
#       --dry-run  print the sbatch command(s) without submitting
#   <dataset>      one of: esol freesolv lipophilicity bace bbbp tox21 hiv
#
# After uploading the folder to the cluster, do these once:
#   1) bash cluster/setup_cluster_env.sh          # build the venv
#   2) edit cluster/rwth_config.sh                # set RWTH_ACCOUNT=thes...
#   3) bash cluster/submit.sh -d cpu -s 0 -m scaffold --all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

ALL_DATASETS=(esol freesolv lipophilicity bace bbbp tox21 hiv)
VALID_SEEDS=(0 1 2 3 4)

DEVICE=""; SEED=""; MODE="scaffold"; RUN_ALL=0; DRY=0
DATASET=""
TIME=""; MEM=""
ACCOUNT_OVERRIDE=""
SKIP_MODELS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--device) DEVICE="$2"; shift 2 ;;
    -s|--seed)   SEED="$2";   shift 2 ;;
    -m|--mode)   MODE="$2";   shift 2 ;;
    --time)      TIME="$2";   shift 2 ;;
    --mem)       MEM="$2";    shift 2 ;;
    --account)   ACCOUNT_OVERRIDE="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS="$2"; shift 2 ;;
    --after-dmpnn) SKIP_MODELS="GIN (2D)|D-MPNN (2D)"; shift ;;
    -a|--all)    RUN_ALL=1;   shift ;;
    --dry-run)   DRY=1;       shift ;;
    -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
    -*)          echo "Unknown flag: $1"; exit 1 ;;
    *)           DATASET="$1"; shift ;;
  esac
done

# --- validate --------------------------------------------------------------
[[ -z "$DEVICE" ]] && { echo "ERROR: --device cpu|gpu is required"; exit 1; }
[[ -z "$SEED"   ]] && { echo "ERROR: --seed <0-4> is required"; exit 1; }
case "$DEVICE" in cpu|gpu) ;; *) echo "ERROR: device must be cpu|gpu"; exit 1;; esac
case "$MODE"   in scaffold|random) ;; *) echo "ERROR: mode must be scaffold|random"; exit 1;; esac
if ! printf '%s\n' "${VALID_SEEDS[@]}" | grep -qx "$SEED"; then
  echo "ERROR: seed must be one of: ${VALID_SEEDS[*]}"; exit 1
fi
if [[ $RUN_ALL -eq 0 && -z "$DATASET" ]]; then
  echo "ERROR: give a <dataset> or use --all"; exit 1
fi
if [[ $RUN_ALL -eq 0 ]]; then
  if ! printf '%s\n' "${ALL_DATASETS[@]}" | grep -qx "$DATASET"; then
    echo "ERROR: unknown dataset '$DATASET' (valid: ${ALL_DATASETS[*]})"; exit 1
  fi
fi

# --- cluster config --------------------------------------------------------
CFG="cluster/rwth_config.sh"
[[ -f "$CFG" ]] && source "$CFG"
ACCOUNT="${ACCOUNT_OVERRIDE:-${RWTH_ACCOUNT:-thesXXXX}}"
MAIL="${RWTH_MAIL_USER:-emir.erdem@rwth-aachen.de}"
if [[ "$DEVICE" == "cpu" ]]; then
  PARTITION="${RWTH_PARTITION_CPU:-c23ms}"
  SLURM_FILE="cluster/run_cpu.slurm"
else
  PARTITION="${RWTH_PARTITION_GPU:-c23g}"
  SLURM_FILE="cluster/run_gpu.slurm"
fi
if [[ "$ACCOUNT" == "thesXXXX" ]]; then
  echo "WARNING: RWTH_ACCOUNT is still the placeholder 'thesXXXX'."
  echo "         Edit cluster/rwth_config.sh before real submission."
fi

mkdir -p cluster/logs

# --- submit ----------------------------------------------------------------
if [[ $RUN_ALL -eq 1 ]]; then
  TARGETS=("${ALL_DATASETS[@]}")
else
  TARGETS=("$DATASET")
fi

for ds in "${TARGETS[@]}"; do
  JOB_NAME="v5-${DEVICE}-${MODE}-s${SEED}-${ds}"
  if [[ -n "$SKIP_MODELS" ]]; then
    # shorter tag so job name stays readable
    JOB_NAME="${JOB_NAME}-postdmpnn"
  fi
  GRES_FLAG=()
  [[ "$DEVICE" == "gpu" ]] && GRES_FLAG=(--gres=gpu:1)
  CMD=(sbatch
    --account="$ACCOUNT"
    --partition="$PARTITION"
    --mail-user="$MAIL"
    --job-name="$JOB_NAME"
    "${GRES_FLAG[@]}")
  [[ -n "$TIME" ]] && CMD+=(--time="$TIME")
  [[ -n "$MEM" ]] && CMD+=(--mem="$MEM")
  # export skip list into the batch job environment
  if [[ "$SKIP_MODELS" == "GIN (2D)|D-MPNN (2D)" ]]; then
    # Prefer flag without spaces/parens (SLURM --export is fragile with those).
    CMD+=(--export=ALL,BENCH_AFTER_DMPNN=1)
  elif [[ -n "$SKIP_MODELS" ]]; then
    CMD+=(--export=ALL,BENCH_SKIP_MODELS="${SKIP_MODELS}")
  fi
  CMD+=("$SLURM_FILE" "$ds" "$SEED" "$MODE")
  if [[ $DRY -eq 1 ]]; then
    echo "[dry-run] ${CMD[*]}"
    [[ -n "$SKIP_MODELS" ]] && echo "         BENCH_SKIP_MODELS=${SKIP_MODELS}"
  else
    echo "Submitting: $JOB_NAME  account=$ACCOUNT  skip=${SKIP_MODELS:-(none)}"
    "${CMD[@]}"
  fi
done
