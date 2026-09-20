#!/usr/bin/env bash
# Fix a benchmark v5 venv that has numpy 1.x (too old for scipy 1.18 / sklearn
# 1.9 -> "module 'numpy' has no attribute 'long'"). RDKit 2026 works with numpy
# 2.x, so this upgrades numpy to 2.x to match the rest of the stack.
#
# Run from benchmark_v5/:
#   bash cluster/fix_numpy_rdkit.sh
#   VENV_DIR=./.venv bash cluster/fix_numpy_rdkit.sh   # for an in-folder venv

set -euo pipefail

cd "$(dirname "$0")/.."
# Prefer an in-folder venv, then the parent one.
VENV_DIR="${VENV_DIR:-}"
if [[ -z "$VENV_DIR" ]]; then
  if [[ -f "./.venv/bin/activate" ]]; then VENV_DIR="./.venv"
  else VENV_DIR="$(cd .. && pwd)/.venv"; fi
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: No venv at $VENV_DIR — run: bash cluster/setup_cluster_env.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

echo "Before:"
python -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || echo "numpy: not installed"

echo ""
echo "Upgrading numpy to 2.x (compatible with scipy/scikit-learn and RDKit 2026)..."
python -m pip install --upgrade pip
python -m pip install "numpy>=2.0"

echo ""
echo "After:"
python -c "
import numpy
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator
import scipy, sklearn
assert numpy.__version__.startswith('2.'), numpy.__version__
print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| sklearn', sklearn.__version__)
print('RDKit OK — AllChem, Descriptors, rdFingerprintGenerator')
"
