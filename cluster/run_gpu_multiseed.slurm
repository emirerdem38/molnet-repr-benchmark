#!/usr/local_rwth/bin/zsh
#SBATCH --account=ACCOUNT_ID
#SBATCH --partition=c23g
#SBATCH --job-name=bench-ms-gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=cluster/logs/gpu_ms_%x_%j.out
#SBATCH --error=cluster/logs/gpu_ms_%x_%j.err
#SBATCH --mail-user=name@example.com
#SBATCH --mail-type=END,FAIL

# Multi-seed GPU notebook via papermill.
#   sbatch cluster/run_gpu_multiseed.slurm <dataset> <seed>
#   bash cluster/submit_gpu_multiseed.sh esol 0

set -euo pipefail

DATASET="${1:?Usage: sbatch cluster/run_gpu_multiseed.slurm <dataset> <seed>}"
SEED="${2:?Usage: sbatch cluster/run_gpu_multiseed.slurm <dataset> <seed>}"

VALID_DATASETS="esol freesolv lipophilicity bace bbbp tox21 hiv"
if ! echo "$VALID_DATASETS" | tr ' ' '\n' | grep -qx "$DATASET"; then
  echo "Unknown dataset: $DATASET"
  exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs \
  "results/multiseed/seed_${SEED}/splits" \
  "results/multiseed/seed_${SEED}/cpu" \
  "results/multiseed/seed_${SEED}/gpu/histories" \
  "results/multiseed/seed_${SEED}/gpu/hpo" \
  "results/multiseed/seed_${SEED}/combined"

source cluster/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

python -m pip install -q papermill ipykernel optuna scipy matplotlib seaborn 2>/dev/null || true
python -m ipykernel install --user --name thesis-molnet --display-name "Python (Thesis molnet)" 2>/dev/null || true

NOTEBOOK="notebooks/multiseed/seed_${SEED}/gpu/benchmark_gpu_${DATASET}.ipynb"
OUTPUT="results/multiseed/seed_${SEED}/gpu/${DATASET}_executed.ipynb"
SPLIT_PKL="results/multiseed/seed_${SEED}/splits/${DATASET}_scaffold_splits.pkl"
CPU_PARTIAL="results/multiseed/seed_${SEED}/cpu/${DATASET}_partial.json"

echo "============================================================"
echo "GPU multi-seed: ${DATASET}  seed=${SEED}"
echo "Host      : $(hostname)"
echo "Start     : $(date -Iseconds)"
echo "Notebook  : ${NOTEBOOK}"
echo "Output    : ${OUTPUT}"
echo "============================================================"

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "ERROR: Notebook not found: $NOTEBOOK"
  echo "Run: python generate_notebooks.py --multiseed --seeds ${SEED}"
  exit 1
fi

if [[ ! -f "$SPLIT_PKL" ]]; then
  echo "ERROR: Scaffold split missing: $SPLIT_PKL"
  echo "Run the CPU multi-seed notebook for this dataset/seed first (creates the split)."
  exit 1
fi

if [[ ! -f "$CPU_PARTIAL" ]]; then
  echo "WARNING: CPU partial missing: $CPU_PARTIAL"
  echo "GPU will still run; combined merge needs CPU results later."
fi

python - <<'PY'
import sys
import torch
print("PyTorch   :", torch.__version__)
print("CUDA avail:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)
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

papermill "$NOTEBOOK" "$OUTPUT" --log-output -k thesis-molnet

echo "============================================================"
echo "Done      : $(date -Iseconds)"
echo "Partial   : results/multiseed/seed_${SEED}/gpu/${DATASET}_partial.json"
echo "============================================================"
