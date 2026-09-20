#!/usr/bin/env bash
# One-time Mac (Apple Silicon) setup for benchmark v4 GPU/CPU notebooks.
#
# Usage (from benchmark_v4/):
#   bash cluster/setup_mac_env.sh
#
# Creates the venv with Python 3.12, NumPy 2.x, RDKit, PyTorch (CPU/MPS), PyG.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/../.venv}"

echo "Benchmark v4 root: $ROOT"
echo "Target venv    : $VENV_DIR"

# Python 3.12 + NumPy 2.x is the tested combo (RDKit 2026 supports NumPy 2;
# scipy 1.18 / scikit-learn 1.9 require it).
PYTHON=""
for candidate in /opt/homebrew/bin/python3.12 python3.12 "${PYTHON_BIN:-}" python3; do
  [[ -z "$candidate" ]] && continue
  if command -v "$candidate" &>/dev/null; then
  PYVER="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$PYVER" == "3.12" ]] || [[ "$PYVER" == "3.11" ]]; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "ERROR: Need Python 3.11 or 3.12 (e.g. brew install python@3.12)."
  exit 1
fi
echo "Using Python   : $(command -v "$PYTHON") ($("$PYTHON" --version))"

if [[ -d "$VENV_DIR" ]]; then
  echo ""
  echo "Removing existing venv (required for a clean NumPy 2.x + RDKit build): $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip wheel setuptools

python -m pip install -c cluster/constraints.txt \
  "numpy>=2.0" \
  pandas pytz python-dateutil \
  scipy scikit-learn xgboost \
  rdkit \
  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
  "torch-geometric==2.6.1" \
  optuna papermill ipykernel jupyter notebook \
  tqdm matplotlib seaborn

python -m pip install --force-reinstall --no-cache-dir "rdkit>=2024.3.1"

# PyG extensions (SchNet / torch-cluster)
WHEEL_TAG=$(python - <<'PY'
import torch
ver = torch.__version__.split("+")
torch_v = ver[0]
cuda = ver[1] if len(ver) > 1 else "cpu"
print(f"torch-{torch_v}+{cuda}")
PY
)
WHEEL_URL="https://data.pyg.org/whl/${WHEEL_TAG}.html"
echo "Installing PyG extensions from ${WHEEL_URL} ..."
if ! python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster -f "${WHEEL_URL}"; then
  echo "WARNING: Prebuilt PyG wheels failed; trying torch-cluster only ..."
  python -m pip install torch-cluster -f "${WHEEL_URL}" || {
    echo "ERROR: Could not install torch-cluster. SchNet will fail until this is fixed."
    exit 1
  }
fi

python -m ipykernel install --user \
  --name thesis-molnet \
  --display-name "Python (Thesis molnet)"

mkdir -p results/splits results/cpu results/gpu/histories results/gpu/hpo results/combined

echo ""
echo "=== Smoke test ==="
python - <<'PY'
import numpy
import pandas as pd
import pytz
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import torch_geometric
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import SchNet

assert numpy.__version__.startswith("2."), numpy.__version__
mol = Chem.MolFromSmiles("CCO")
assert mol is not None

pos = torch.randn(10, 3)
radius_graph(pos, r=5.0)
m = SchNet(hidden_channels=96, num_filters=96, num_interactions=3, num_gaussians=50, cutoff=10.0)
m(torch.tensor([6, 6, 8], dtype=torch.long), torch.randn(3, 3))

print("numpy  ", numpy.__version__)
print("pandas ", pd.__version__)
print("pytz   ", pytz.__version__)
print("torch  ", torch.__version__)
print("PyG    ", torch_geometric.__version__)
print("MPS    ", torch.backends.mps.is_available())
print("RDKit  OK")
print("SchNet / radius_graph OK")
PY

echo ""
echo "=== Setup complete ==="
echo "Activate:"
echo "  cd \"$ROOT\""
echo "  source \"$VENV_DIR/bin/activate\""
echo "  export PYTHONNOUSERSITE=1"
echo ""
echo "Run a GPU notebook (from benchmark_v4/):"
echo "  jupyter notebook notebooks/gpu/benchmark_gpu_bbbp.ipynb"
echo ""
echo "Or papermill:"
echo "  papermill notebooks/gpu/benchmark_gpu_bbbp.ipynb results/gpu/bbbp_executed.ipynb -k thesis-molnet"
