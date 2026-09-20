#!/usr/local_rwth/bin/zsh
#SBATCH --account=thesXXXX
#SBATCH --partition=c23ms
#SBATCH --job-name=hiv-conf-v4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2540M
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-19
#SBATCH --output=cluster/logs/hiv_conformers_%A_%a.out
#SBATCH --error=cluster/logs/hiv_conformers_%A_%a.err
#SBATCH --mail-user=emir.erdem@rwth-aachen.de
#SBATCH --mail-type=END,FAIL

# Parallel HIV conformer generation (CPU only, no GPU needed).
# Chunks → data/hiv_conformers_n25_chunk_*.pkl
# Then: bash cluster/merge_hiv_conformers.sh → data/hiv_conformers_n25.pkl

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs data

source cluster/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Chunk on *preprocessed* valid molecules (HIV CSV has ~41k rows → ~21k valid SMILES).
N_MOLECULES=$(python -c "
import pandas as pd
from pathlib import Path
from mol_repr_utils import preprocess_dataset
df = pd.read_csv(Path('data') / 'hiv.csv')
smiles, _ = preprocess_dataset(df, 'hiv')
print(len(smiles))
")
N_CHUNKS=20
CHUNK_SIZE=$(( (N_MOLECULES + N_CHUNKS - 1) / N_CHUNKS ))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$(( TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
if (( END > N_MOLECULES )); then
  END=$N_MOLECULES
fi

echo "HIV conformers — array task ${TASK_ID}  [${START}:${END}) of ${N_MOLECULES} valid"
echo "Host: $(hostname)  |  $(date -Iseconds)"

python conformer_generation.py \
  --dataset hiv \
  --n_confs 25 \
  --data_dir ./data \
  --seed 42 \
  --start_idx "${START}" \
  --end_idx "${END}" \
  --chunk_tag "$(printf '%03d' "${TASK_ID}")" \
  --no_plot

echo "Chunk: data/hiv_conformers_n25_chunk_$(printf '%03d' "${TASK_ID}").pkl"
echo "Done: $(date -Iseconds)"
