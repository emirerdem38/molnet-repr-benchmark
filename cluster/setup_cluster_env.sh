#!/usr/bin/env bash
# One-time cluster setup for the benchmark (conformers + GPU runs).
# Run from the repository root on the cluster:
#   bash cluster/setup_cluster_env.sh
#   # or into an in-folder venv:  VENV_DIR=./.venv bash cluster/setup_cluster_env.sh

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "Repository root: $ROOT"

# Optional: load a recent Python module if the site provides one.
if command -v module &>/dev/null; then
  module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
  module load Python/3.10 2>/dev/null || true
fi

# Default: venv inside the repository (override with VENV_DIR=...)
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
else
  echo "Using existing venv at $VENV_DIR"
  echo "(If this venv was copied from another OS, delete it and rerun: rm -rf \"$VENV_DIR\")"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip wheel

# Core benchmark stack (adjust versions if your cluster module system provides torch).
# NumPy 2.x is REQUIRED by scipy 1.18 / scikit-learn 1.9 and works with RDKit 2026.
python -m pip install -c cluster/constraints.txt \
  "numpy>=2.0" \
  pandas pytz python-dateutil scipy scikit-learn xgboost \
  rdkit \
  torch torchvision torchaudio \
  torch-geometric \
  optuna papermill ipykernel jupyter \
  tqdm matplotlib seaborn

python -m ipykernel install --user \
  --name thesis-molnet \
  --display-name "Python (Thesis molnet)"

mkdir -p cluster/logs results/splits results/cpu results/gpu results/combined

python -c "
import numpy
assert numpy.__version__.startswith('2.'), f'Need numpy 2.x, got {numpy.__version__}'
import pytz
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
print('numpy', numpy.__version__)
print('pytz', pytz.__version__)
print('pandas', pd.__version__)
print('RDKit OK')
print('PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available())
"

echo ""
echo "Setup complete. Venv: $VENV_DIR"
echo "Activate with:  source $VENV_DIR/bin/activate"
echo ""
echo "Resubmit conformers:"
echo "  bash cluster/submit_hiv_conformers.sh"
