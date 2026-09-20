#!/usr/bin/env bash
# PyG extensions for SchNet (radius_graph).
#
# PyG 2.6+ needs pyg-lib>=0.6.0. torch-scatter/sparse/cluster are optional
# (often no prebuilt Mac wheels for every torch build — SchNet does not need them
# if pyg-lib is installed).
#
# Run from benchmark_v5/ (after torch is in the venv):
#   bash cluster/install_pyg_extensions.sh
# On RWTH GPU nodes, run after install_cuda_torch.sh.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV_DIR=""
for _v in "${VENV_DIR:-}" "${ROOT}/.venv" "${ROOT}/../.venv"; do
  [[ -z "$_v" ]] && continue
  if [[ -f "${_v}/bin/activate" ]]; then
    VENV_DIR="$_v"
    break
  fi
done
if [[ -z "$VENV_DIR" ]]; then
  echo "ERROR: No venv found. Tried: ${ROOT}/.venv and ${ROOT}/../.venv"
  exit 1
fi
echo "Using venv: $VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

WHEEL_TAG=$(python - <<'PY'
import torch
ver = torch.__version__.split("+")
torch_v = ver[0]
cuda = ver[1] if len(ver) > 1 else "cpu"
print(f"torch-{torch_v}+{cuda}")
PY
)

WHEEL_URL="https://data.pyg.org/whl/${WHEEL_TAG}.html"
echo "PyG wheel index: ${WHEEL_URL}"
python -m pip install --upgrade pip

# ── Required for SchNet on PyG 2.6+ ─────────────────────────────────────────
echo "Installing pyg-lib (required for SchNet) ..."
python -m pip install pyg-lib -f "${WHEEL_URL}"

# ── Optional extras (skip if no prebuilt wheel — avoids source builds on Mac) ─
for pkg in torch-scatter torch-sparse torch-cluster; do
  echo "Optional: ${pkg} ..."
  if python -m pip install "${pkg}" -f "${WHEEL_URL}" --only-binary=:all: 2>/dev/null; then
    echo "  ${pkg} OK"
  else
    echo "  ${pkg} skipped (no prebuilt wheel for ${WHEEL_TAG})"
  fi
done

# NumPy must stay 2.x (scipy 1.18 / scikit-learn 1.9 require it; RDKit 2026 is fine).
# Do NOT downgrade numpy here — a --no-deps downgrade leaves scipy expecting numpy 2
# and breaks imports with "module 'numpy' has no attribute 'long'".
python -c "import numpy; assert numpy.__version__.startswith('2.'), \
  f'numpy {numpy.__version__} is too old; run: bash cluster/fix_numpy_rdkit.sh'"

python - <<'PY'
import torch
import torch_geometric.typing as pyg_typing
from mol_repr_utils import build_schnet

print("torch", torch.__version__)
if pyg_typing.WITH_RADIUS:
    print("pyg-lib OK — PyG radius_graph")
elif pyg_typing.WITH_PYG_LIB:
    print("pyg-lib imported but radius op missing — check pyg-lib version")
else:
    from torch_cluster import radius_graph  # noqa: F401
    print("torch-cluster OK — SchNet fallback radius graph")

device = "cuda" if torch.cuda.is_available() else "cpu"
z = torch.tensor([6, 6, 8], dtype=torch.long, device=device)
pos = torch.randn(3, 3, device=device)
batch = torch.zeros(3, dtype=torch.long, device=device)
build_schnet(1).to(device)(z, pos, batch)
print("SchNet forward OK (build_schnet)")
PY

echo "PyG extensions ready."
