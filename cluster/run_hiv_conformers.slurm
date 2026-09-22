#!/usr/local_rwth/bin/zsh
#SBATCH --account=ACCOUNT_ID
#SBATCH --partition=c23ms
#SBATCH --job-name=hiv-conf-v4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2540M
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=cluster/logs/hiv_conformers_%j.out
#SBATCH --error=cluster/logs/hiv_conformers_%j.err
#SBATCH --mail-user=name@example.com
#SBATCH --mail-type=END,FAIL

# Single-job HIV conformer generation → data/hiv_conformers_n25.pkl

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p cluster/logs data

source cluster/activate_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "HIV conformers (single job): $(date -Iseconds)"

python conformer_generation.py \
  --dataset hiv \
  --n_confs 25 \
  --data_dir ./data \
  --seed 42 \
  --no_plot

ls -lh data/hiv_conformers_n25.pkl
echo "Done: $(date -Iseconds)"
