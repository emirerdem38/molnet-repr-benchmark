#!/usr/bin/env bash
# Install CUDA PyTorch into the benchmark v4 venv (required for GPU SLURM jobs).
# Conformer setup uses CPU-only torch: run this before submit_gpu.sh.
#
# From the repository root/:
#   bash cluster/install_cuda_torch.sh

set -euo pipefail

cd "$(dirname "$0")/.."
VENV_DIR="${VENV_DIR:-$(cd .. && pwd)/.venv}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip

# H100 on RWTH: CUDA 12.x wheels (adjust cu124 → cu121 if your node uses older CUDA).
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -m pip install -c cluster/constraints.txt torch-geometric
# Keep numpy 2.x (constraints.txt enforces >=2.0); do NOT downgrade it.

bash cluster/install_pyg_extensions.sh

python - <<'PY'
import numpy
import torch
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit(
        "CUDA still not available: load a CUDA module (module avail CUDA) and retry, "
        "or ask RWTH support for the correct PyTorch module on c23g."
    )
PY

echo "CUDA PyTorch ready."
