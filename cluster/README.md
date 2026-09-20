# Benchmark v4 — Cluster GPU runs

Run the five GPU models per dataset (GIN 2D, D-MPNN, GIN 3D, SchNet, SMILES LSTM) via **papermill** on RWTH SLURM.

## Prerequisites (upload from Mac)

From your Mac, upload/sync at least:

```
molnet_esol_project/
├── .venv/                          # or run setup_cluster_env.sh on cluster
└── benchmark_v4/
    ├── *.py, conformer_generation.py
    ├── data/                         # CSVs + *_conformers_n25.pkl
    ├── notebooks/gpu/*.ipynb
    ├── results/splits/               # REQUIRED — from CPU runs
    │   └── {dataset}_scaffold_splits.pkl  (×7)
    └── results/cpu/                  # for combined CSV merge
        └── {dataset}_partial.json    (×7)
```

**Critical:** GPU notebooks reuse **CPU scaffold splits**. Without `results/splits/*.pkl`, jobs will exit immediately.

**3D models** need conformer pickles, e.g. `data/esol_conformers_n25.pkl`. HIV: `data/hiv_conformers_n25.pkl`.

### Generate HIV conformers on cluster

**Do not upload `.venv` from your Mac** — recreate it on the cluster (Linux + different Python).

```bash
cd benchmark_v4

# Conformers only (recommended first — faster):
bash cluster/setup_conformers_env.sh

# Or full stack (GPU notebooks too):
# bash cluster/setup_cluster_env.sh

bash cluster/submit_hiv_conformers.sh
# wait for array → bash cluster/merge_hiv_conformers.sh
```

Quick manual fix if venv is empty:

```bash
source ../.venv/bin/activate
export PYTHONNOUSERSITE=1
pip install -r cluster/requirements-conformers.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

If you see `No module named 'pytz'` or `No module named 'pandas'`, the job used `~/.local` Python or an empty venv — run setup above and resubmit.

**NumPy error** (`module 'numpy' has no attribute 'long'`): the venv has numpy 1.x,
which is too old for scipy 1.18 / scikit-learn 1.9. RDKit 2026 supports numpy 2.x,
so upgrade rather than downgrade:

```bash
bash cluster/fix_numpy_rdkit.sh
# or manually:
pip install "numpy>=2.0"
```

**Empty slice / array tasks 11–19 failed:** The old script chunked 41k CSV rows, but HIV has only ~21k valid molecules after preprocessing. Tasks 0–10 cover the full set; tasks 11–19 were empty. If those chunks finished, merge without resubmitting:

```bash
bash cluster/merge_hiv_conformers.sh
```

Re-upload `cluster/run_hiv_conformers_array.slurm` for future runs (now counts valid molecules automatically).

Single-job alternative (slower): `sbatch cluster/run_hiv_conformers.slurm`


## RWTH account (required once)

Thesis accounts look like `thes1234` (not `rwth2003`). I don't have yours in this repo — find it on the cluster:

```bash
sacctmgr show user $USER format=account%30
```

Then edit **`cluster/rwth_config.sh`**:

```bash
export RWTH_ACCOUNT="thes1234"   # your actual ID
```

Submit scripts pass `--account` from that file (overrides `#SBATCH --account=thesXXXX` in the `.slurm` files).

## One-time setup on cluster

```bash
cd ~/path/to/molnet_esol_project/benchmark_v4
bash cluster/setup_cluster_env.sh
```

If PyTorch CUDA wheels fail, load your cluster's PyTorch module first, then re-run pip for the remaining packages.

## Submit jobs

All commands run from **`benchmark_v4/`**:

```bash
# Single dataset
bash cluster/submit_gpu.sh esol
bash cluster/submit_gpu.sh hiv              # 96h, 64G RAM

# All 7 datasets in parallel (needs 7 GPU slots)
bash cluster/submit_all_gpu.sh

# All 7 in sequence (one GPU at a time)
bash cluster/submit_all_gpu_sequential.sh

# Per-dataset shortcuts
bash cluster/jobs/submit_esol.sh
bash cluster/jobs/submit_hiv.sh
```

Or directly (set account first):

```bash
sbatch --account=thes1234 --partition=c23g cluster/run_gpu.slurm tox21
```

## Default resources

SLURM headers follow RWTH format (`#!/usr/local_rwth/bin/zsh`, `--nodes=1`, mail on END/FAIL). GPU jobs use `--mem=32G` (c23g sets `mem-per-gpu` by default — do not combine with `--mem-per-cpu`).

| Dataset       | Partition | Walltime | Memory (8 CPU) |
|---------------|-----------|----------|----------------|
| ESOL          | c23g      | 24 h     | 32G            |
| FreeSolv      | c23g      | 24 h     | 32G            |
| Lipophilicity | c23g      | 48 h     | 48G            |
| BACE          | c23g      | 48 h     | 32G            |
| BBBP          | c23g      | 48 h     | 32G            |
| Tox21         | c23g      | 72 h     | 48G            |
| HIV           | c23g      | 96 h     | 64G            |

HIV conformers (CPU): partition `c23ms`, `--mem-per-cpu=2540M`.

Override: `bash cluster/submit_gpu.sh hiv 120:00:00 64G`

Edit `cluster/rwth_config.sh` for account, partition, and mail.

## Monitor

```bash
squeue -u $USER
tail -f cluster/logs/gpu_bench-gpu-esol_<JOBID>.out
```

## Outputs

| Path | Content |
|------|---------|
| `results/gpu/{dataset}_partial.json` | GPU model metrics (resume-safe) |
| `results/gpu/{dataset}_executed.ipynb` | Full executed notebook |
| `results/combined/{dataset}_results.csv` | CPU + GPU merged (if CPU partial uploaded) |
| `results/gpu/hpo/{dataset}/` | Optuna best params |
| `results/gpu/histories/` | Training curves |

Notebooks use `FRESH_RUN = False` — interrupted jobs resume from partial JSON.

## Suggested order

Same as CPU: **ESOL → FreeSolv → Lipophilicity → BACE → BBBP → Tox21 → HIV**

For a first smoke test, submit **ESOL only** before launching HIV.
