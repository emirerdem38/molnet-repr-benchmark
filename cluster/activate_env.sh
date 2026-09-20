# Source from benchmark_v4/ SLURM scripts (after cd to project root).
#   source cluster/activate_env.sh
#
# Prevents ~/.local/site-packages from breaking imports (e.g. pandas without pytz).

export PYTHONNOUSERSITE=1

_BENCH_ROOT="$(pwd)"
_VENV_DIR=""

for _v in "${VENV_DIR:-}" "${_BENCH_ROOT}/../.venv" "${_BENCH_ROOT}/.venv"; do
  [[ -z "$_v" ]] && continue
  if [[ -f "${_v}/bin/activate" ]]; then
    _VENV_DIR="$_v"
    break
  fi
done

if [[ -z "$_VENV_DIR" ]]; then
  echo "ERROR: No Python venv found for benchmark_v4."
  echo "  cwd: ${_BENCH_ROOT}"
  echo "  Tried: \${VENV_DIR}, ../.venv, .venv"
  echo ""
  echo "On the cluster, run once:"
  echo "  bash cluster/setup_cluster_env.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "${_VENV_DIR}/bin/activate"

echo "Python    : $(command -v python)"
echo "Version   : $(python -V 2>&1)"
python -c "
import numpy
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator
ver = numpy.__version__
if not ver.startswith('2.'):
    raise SystemExit(f'FATAL: numpy {ver} is too old for scipy/scikit-learn here (need 2.x) — run: bash cluster/fix_numpy_rdkit.sh')
import pandas
print('Imports OK: numpy', ver, '| pandas', pandas.__version__, '| rdkit AllChem+Descriptors')
" || exit 1
