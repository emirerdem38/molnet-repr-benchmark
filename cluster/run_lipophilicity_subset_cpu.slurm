#!/usr/local_rwth/bin/zsh
#SBATCH --job-name=v5-lipo-sub-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#
# Tabular Lipophilicity subset benchmark (no HPO).
# Prefer: bash cluster/submit_lipophilicity_subset.sh

set -euo pipefail

MODE="${1:-scaffold}"
SEED="${2:-0}"
FRACTIONS="${3:-0.125,0.25,0.5,1.0}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs results/lipophilicity_subset

source cluster/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Lipophilicity subset CPU — mode=${MODE} seed=${SEED}"
echo "fractions=${FRACTIONS}"
echo "Host      : $(hostname)"
echo "Account   : ${SLURM_JOB_ACCOUNT:-n/a}"
echo "Start     : $(date -Iseconds)"
echo "Job ID    : ${SLURM_JOB_ID:-n/a}"
echo "============================================================"

python lipophilicity_subset_benchmark.py \
  --mode "${MODE}" \
  --seed "${SEED}" \
  --models tabular \
  --fractions "${FRACTIONS}" \
  --device cpu

echo "End: $(date -Iseconds)"
ls -la results/lipophilicity_subset/ || true
