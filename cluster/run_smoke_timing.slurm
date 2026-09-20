#!/usr/local_rwth/bin/zsh
#SBATCH --account=rwth2175
#SBATCH --partition=c23g
#SBATCH --job-name=v5-smoke-timing
#SBATCH --cpus-per-task=8
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err
#SBATCH --mail-user=emir.erdem@rwth-aachen.de
#SBATCH --mail-type=END,FAIL
#
# Same-machine smoke timing for all 12 models on FreeSolv (reduced HPO).
# Prefer: bash cluster/submit_smoke_timing.sh

set -euo pipefail

DATASET="${1:-freesolv}"
SEED="${2:-0}"
MODE="${3:-scaffold}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs results/smoke_timing

source cluster/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Smoke timing — dataset=${DATASET} seed=${SEED} mode=${MODE}"
echo "Host      : $(hostname)"
echo "Start     : $(date -Iseconds)"
echo "Job ID    : ${SLURM_JOB_ID:-n/a}"
echo "Partition : ${SLURM_JOB_PARTITION:-n/a}"
echo "============================================================"

python - <<'PY'
import torch
print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
PY

python smoke_timing_benchmark.py \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --mode "${MODE}"

echo "End: $(date -Iseconds)"
echo "Results under results/smoke_timing/"
ls -la results/smoke_timing/ || true
