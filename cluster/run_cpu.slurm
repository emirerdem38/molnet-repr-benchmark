#!/usr/local_rwth/bin/zsh
#SBATCH --account=thesXXXX
#SBATCH --partition=c23ms
#SBATCH --job-name=v5-cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err
#SBATCH --mail-user=emir.erdem@rwth-aachen.de
#SBATCH --mail-type=END,FAIL
#
# Benchmark v5 — CPU track (tabular models) for one dataset / seed / split mode.
#
# Preferred: submit via the wrapper (fills --account/--partition from rwth_config.sh):
#   bash cluster/submit.sh --device cpu --seed 0 --mode scaffold esol
#
# Direct sbatch (edit the #SBATCH --account above first):
#   sbatch cluster/run_cpu.slurm <dataset> <seed> [mode]
#   sbatch cluster/run_cpu.slurm esol 0 scaffold

set -euo pipefail

DATASET="${1:?Usage: sbatch cluster/run_cpu.slurm <dataset> <seed> [scaffold|random]}"
SEED="${2:?Usage: sbatch cluster/run_cpu.slurm <dataset> <seed> [scaffold|random]}"
MODE="${3:-scaffold}"

VALID_DATASETS="esol freesolv lipophilicity bace bbbp tox21 hiv"
if ! echo "$VALID_DATASETS" | tr ' ' '\n' | grep -qx "$DATASET"; then
  echo "Unknown dataset: $DATASET  (valid: $VALID_DATASETS)"; exit 1
fi
if [[ "$MODE" != "scaffold" && "$MODE" != "random" ]]; then
  echo "Unknown mode: $MODE  (valid: scaffold random)"; exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

RESULTS_ROOT="results/seed_${SEED}/${MODE}"
mkdir -p cluster/logs \
  "${RESULTS_ROOT}/splits" \
  "${RESULTS_ROOT}/cpu/hpo" \
  "${RESULTS_ROOT}/cpu/histories" \
  "${RESULTS_ROOT}/combined"

source cluster/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# Make sure the Jupyter kernel the notebooks reference exists on this node.
python -m ipykernel install --user --name thesis-molnet \
  --display-name "Python (Thesis molnet)" 2>/dev/null || true

NOTEBOOK="notebooks/seed_${SEED}/${MODE}/cpu/benchmark_cpu_${DATASET}.ipynb"
OUTPUT="${RESULTS_ROOT}/cpu/${DATASET}_executed.ipynb"

echo "============================================================"
echo "Benchmark v5 CPU — dataset=${DATASET} seed=${SEED} mode=${MODE}"
echo "Host      : $(hostname)"
echo "Start     : $(date -Iseconds)"
echo "Notebook  : ${NOTEBOOK}"
echo "Results   : ${RESULTS_ROOT}/"
echo "============================================================"

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "ERROR: Notebook not found: $NOTEBOOK"
  echo "Generate it with: python generate_notebooks.py --seeds ${SEED} --modes ${MODE}"
  exit 1
fi

papermill "$NOTEBOOK" "$OUTPUT" --log-output -k thesis-molnet

echo "============================================================"
echo "Done      : $(date -Iseconds)"
echo "Metrics   : ${RESULTS_ROOT}/cpu/${DATASET}_partial.json"
echo "HPO params: ${RESULTS_ROOT}/cpu/hpo/${DATASET}/"
echo "Split     : ${RESULTS_ROOT}/splits/${DATASET}_${MODE}_splits.pkl"
echo "============================================================"
