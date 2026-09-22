#!/usr/local_rwth/bin/zsh
#SBATCH --account=ACCOUNT_ID
#SBATCH --partition=c23ms
#SBATCH --job-name=bench-ms-cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=cluster/logs/cpu_ms_%x_%j.out
#SBATCH --error=cluster/logs/cpu_ms_%x_%j.err
#SBATCH --mail-user=name@example.com
#SBATCH --mail-type=END,FAIL

# Multi-seed CPU notebook (tabular + scaffold split).
#   bash cluster/submit_cpu_multiseed.sh esol 0

set -euo pipefail

DATASET="${1:?Usage: sbatch cluster/run_cpu_multiseed.slurm <dataset> <seed>}"
SEED="${2:?Usage: sbatch cluster/run_cpu_multiseed.slurm <dataset> <seed>}"

VALID_DATASETS="esol freesolv lipophilicity bace bbbp tox21 hiv"
if ! echo "$VALID_DATASETS" | tr ' ' '\n' | grep -qx "$DATASET"; then
  echo "Unknown dataset: $DATASET"
  exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs \
  "results/multiseed/seed_${SEED}/splits" \
  "results/multiseed/seed_${SEED}/cpu/hpo" \
  "results/multiseed/seed_${SEED}/cpu/histories" \
  "results/multiseed/seed_${SEED}/combined"

source cluster/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python -m pip install -q papermill ipykernel optuna scipy matplotlib seaborn scikit-learn xgboost 2>/dev/null || true
python -m ipykernel install --user --name thesis-molnet --display-name "Python (Thesis molnet)" 2>/dev/null || true

NOTEBOOK="notebooks/multiseed/seed_${SEED}/cpu/benchmark_cpu_${DATASET}.ipynb"
OUTPUT="results/multiseed/seed_${SEED}/cpu/${DATASET}_executed.ipynb"

echo "============================================================"
echo "CPU multi-seed: ${DATASET}  seed=${SEED}"
echo "Host      : $(hostname)"
echo "Start     : $(date -Iseconds)"
echo "Notebook  : ${NOTEBOOK}"
echo "============================================================"

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "ERROR: Notebook not found: $NOTEBOOK"
  exit 1
fi

papermill "$NOTEBOOK" "$OUTPUT" --log-output -k thesis-molnet

echo "============================================================"
echo "Done      : $(date -Iseconds)"
echo "Partial   : results/multiseed/seed_${SEED}/cpu/${DATASET}_partial.json"
echo "Split     : results/multiseed/seed_${SEED}/splits/${DATASET}_scaffold_splits.pkl"
echo "============================================================"
