#!/usr/bin/env bash
# Lightweight env for conformer jobs (CPU). Faster than the full GPU setup.
#
# From the repository root on the cluster:
#   bash cluster/setup_conformers_env.sh
#
# If a .venv was copied from another OS, delete it first and recreate:
#   rm -rf .venv && VENV_DIR=./.venv bash cluster/setup_conformers_env.sh

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
CONSTRAINTS="$ROOT/cluster/constraints.txt"

if command -v module &>/dev/null; then
  module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
  module load Python/3.10 2>/dev/null || true
fi

echo "Repository root: $ROOT"
echo "Venv target:     $VENV_DIR"
echo "Python:          $(command -v python3) ($(python3 -V 2>&1))"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "Creating venv..."
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip wheel

# 1) NumPy 1.x + RDKit first (RDKit must compile against numpy 1.x)
python -m pip install -c "$CONSTRAINTS" -r cluster/requirements-conformers.txt
python -m pip install --force-reinstall --no-cache-dir rdkit-pypi

# 2) PyTorch stack (may try to upgrade numpy → constraints + final re-pin)
python -m pip install -c "$CONSTRAINTS" \
  torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c "$CONSTRAINTS" torch-geometric

# 3) Force numpy 1.26.4 last: torch-geometric often ignores constraints
python -m pip install "numpy==1.26.4" --force-reinstall --no-deps

python -c "
import numpy
import pandas, pytz, tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator
import torch
from torch_geometric.data import Data
assert numpy.__version__ == '1.26.4', numpy.__version__
print('OK: numpy', numpy.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__)
print('RDKit deep import OK')
"

echo ""
echo "Conformer env ready: $VENV_DIR"
echo "Resubmit: bash cluster/submit_hiv_conformers.sh"
