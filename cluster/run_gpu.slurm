#!/usr/local_rwth/bin/zsh
#SBATCH --account=rwth2175
#SBATCH --partition=c23g
#SBATCH --job-name=v5-gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=cluster/logs/%x_%j.out
#SBATCH --error=cluster/logs/%x_%j.err
#SBATCH --mail-user=emir.erdem@rwth-aachen.de
#SBATCH --mail-type=END,FAIL
#
# Benchmark v5 — GPU track (GIN / D-MPNN / SchNet / LSTM) for one dataset / seed / mode.
#
# Preferred: submit via the wrapper (fills --account/--partition/--gres from rwth_config.sh):
#   bash cluster/submit.sh --device gpu --seed 0 --mode scaffold esol
#
# Direct sbatch (edit the #SBATCH --account above first):
#   sbatch cluster/run_gpu.slurm <dataset> <seed> [mode]
#   sbatch cluster/run_gpu.slurm esol 0 scaffold
#
# The GPU notebook creates the split itself if the CPU job has not run yet.
# Run the CPU job first if you want the merged combined/ table in the same pass.

set -euo pipefail

DATASET="${1:?Usage: sbatch cluster/run_gpu.slurm <dataset> <seed> [scaffold|random]}"
SEED="${2:?Usage: sbatch cluster/run_gpu.slurm <dataset> <seed> [scaffold|random]}"
MODE="${3:-scaffold}"
# Optional 4th arg or env BENCH_SKIP_MODELS: pipe-separated model names to skip
# e.g. 'GIN (2D)|D-MPNN (2D)' → only run GIN (3D), SchNet, SMILES LSTM
if [[ $# -ge 4 && -n "${4:-}" ]]; then
  export BENCH_SKIP_MODELS="$4"
fi
export BENCH_SKIP_MODELS="${BENCH_SKIP_MODELS:-}"

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
  "${RESULTS_ROOT}/cpu" \
  "${RESULTS_ROOT}/gpu/hpo" \
  "${RESULTS_ROOT}/gpu/histories" \
  "${RESULTS_ROOT}/combined"

source cluster/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python -m ipykernel install --user --name thesis-molnet \
  --display-name "Python (Thesis molnet)" 2>/dev/null || true

NOTEBOOK="notebooks/seed_${SEED}/${MODE}/gpu/benchmark_gpu_${DATASET}.ipynb"
OUTPUT="${RESULTS_ROOT}/gpu/${DATASET}_executed.ipynb"
CPU_PARTIAL="${RESULTS_ROOT}/cpu/${DATASET}_partial.json"

echo "============================================================"
echo "Benchmark v5 GPU — dataset=${DATASET} seed=${SEED} mode=${MODE}"
echo "Host      : $(hostname)"
echo "Start     : $(date -Iseconds)"
echo "Notebook  : ${NOTEBOOK}"
echo "Results   : ${RESULTS_ROOT}/"
if [[ "${BENCH_AFTER_DMPNN:-}" == "1" ]]; then
  echo "Skip      : AFTER_DMPNN (GIN 2D + D-MPNN)"
elif [[ -n "${BENCH_SKIP_MODELS:-}" ]]; then
  echo "Skip      : ${BENCH_SKIP_MODELS}"
else
  echo "Skip      : (none — run all missing models)"
fi
echo "============================================================"

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "ERROR: Notebook not found: $NOTEBOOK"
  echo "Generate it with: python generate_notebooks.py --seeds ${SEED} --modes ${MODE}"
  exit 1
fi

if [[ ! -f "$CPU_PARTIAL" ]]; then
  echo "NOTE: CPU results missing ($CPU_PARTIAL)."
  echo "      GPU models will still run; the combined/ table needs the CPU job too."
fi

# Fail fast if CUDA / SchNet radius-graph backends aren't available on this node.
python - <<'PY'
import sys, torch
print("PyTorch   :", torch.__version__)
print("CUDA avail:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA not available on this node"); sys.exit(1)
print("GPU       :", torch.cuda.get_device_name(0))
PY

python - <<'PY'
import torch
import torch_geometric.typing as pyg_typing
from mol_repr_utils import build_schnet

if pyg_typing.WITH_RADIUS:
    print("pyg-lib OK (SchNet radius_graph)")
else:
    from torch_cluster import radius_graph  # noqa: F401
    print("torch-cluster OK (SchNet fallback)")

device = torch.device("cuda")
z = torch.tensor([6, 6, 8], dtype=torch.long, device=device)
pos = torch.randn(3, 3, device=device)
batch = torch.zeros(3, dtype=torch.long, device=device)
build_schnet(1).to(device)(z, pos, batch)
print("SchNet forward OK")
PY

nvidia-smi -L 2>/dev/null || echo "(nvidia-smi not available)"

papermill "$NOTEBOOK" "$OUTPUT" --log-output -k thesis-molnet

echo "============================================================"
echo "Done      : $(date -Iseconds)"
echo "Metrics   : ${RESULTS_ROOT}/gpu/${DATASET}_partial.json"
echo "Combined  : ${RESULTS_ROOT}/combined/${DATASET}_results.csv (if CPU job also done)"
echo "============================================================"
