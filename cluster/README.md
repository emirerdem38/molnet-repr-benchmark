# Cluster helpers (optional)

SLURM helpers for running the benchmark on an HPC cluster (examples use
RWTH CLAIX naming). Local runs do not need this folder.

## Prerequisites

Sync the repository to the cluster, including:

```
molnet-repr-benchmark/
├── *.py, run.sh, generate_notebooks.py
├── data/                 # MoleculeNet CSVs and conformer pickles
├── notebooks/            # CPU/GPU notebooks
├── results/              # optional: existing splits / partials to resume
└── cluster/              # this folder
```

Do not copy a local `.venv` to the cluster. Recreate the environment on Linux
(see setup below). macOS or Windows wheels will not work there.

**Splits:** GPU notebooks reuse scaffold or random splits written by the CPU
track under `results/seed_<seed>/<mode>/splits/`. Without those pickles, GPU
jobs exit early unless they recreate the split themselves.

**3D models** need conformer pickles such as `data/esol_conformers_n25.pkl`
(and the matching file for each dataset).

### HIV conformers on the cluster

```bash
cd molnet-repr-benchmark
bash cluster/setup_conformers_env.sh
bash cluster/submit_hiv_conformers.sh
# after the array finishes:
bash cluster/merge_hiv_conformers.sh
```

Manual repair if the venv is incomplete:

```bash
source .venv/bin/activate   # or the path used by setup_*.sh
export PYTHONNOUSERSITE=1
pip install -r cluster/requirements-conformers.txt
pip install torch -i https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

If jobs report missing `pytz` or `pandas`, they likely used a user site-packages
install or an empty venv. Rerun the setup script and resubmit.

**NumPy / RDKit mismatch** (`numpy` has no attribute `long`): upgrade NumPy
rather than downgrade the stack:

```bash
bash cluster/fix_numpy_rdkit.sh
# or: pip install "numpy>=2.0"
```

## SLURM account (set once)

Set the computing account that `sbatch` should charge. On many sites:

```bash
sacctmgr show user $USER format=account%30
```

Edit `cluster/rwth_config.sh`:

```bash
export RWTH_ACCOUNT="ACCOUNT_ID"
export RWTH_MAIL_USER="name@example.com"
```

Submit wrappers read that file and pass the account to `sbatch`.

## One-time environment setup

```bash
cd molnet-repr-benchmark
bash cluster/setup_cluster_env.sh
```

If CUDA PyTorch wheels fail, load the site module for PyTorch first, then finish
with pip for the remaining packages. Optional CUDA extras:

```bash
bash cluster/install_cuda_torch.sh
bash cluster/install_pyg_extensions.sh
```

## Submit jobs

From the repository root:

```bash
# One dataset (device, seed, split mode)
bash cluster/submit.sh -d cpu -s 0 -m scaffold esol
bash cluster/submit.sh -d gpu -s 0 -m scaffold hiv

# All seven datasets for that seed / mode / device
bash cluster/submit.sh -d cpu -s 0 -m scaffold --all
bash cluster/submit.sh -d gpu -s 0 -m scaffold --all

# Preview without submitting
bash cluster/submit.sh -d gpu -s 0 -m scaffold hiv --dry-run
```

See `cluster/SUBMIT_V5.md` for a longer walkthrough.

Direct `sbatch` (after setting the account):

```bash
sbatch --account=ACCOUNT_ID --partition=c23g cluster/run_gpu.slurm tox21
```

## Default resources (example)

Headers follow a typical RWTH layout (`#!/usr/local_rwth/bin/zsh`, mail on
END/FAIL). Adjust partitions and memory for the local site. Example GPU table:

| Dataset       | Partition | Walltime | Memory |
|---------------|-----------|----------|--------|
| ESOL          | c23g      | 24 h     | 32G    |
| FreeSolv      | c23g      | 24 h     | 32G    |
| Lipophilicity | c23g      | 48 h     | 48G    |
| BACE          | c23g      | 48 h     | 32G    |
| BBBP          | c23g      | 48 h     | 32G    |
| Tox21         | c23g      | 72 h     | 48G    |
| HIV           | c23g      | 96 h     | 64G    |

Conformer jobs often use a CPU partition (example: `c23ms`). Override time and
memory when calling the submit helpers, and edit `cluster/rwth_config.sh` for
account, partition, and mail.

## Monitor

```bash
squeue -u $USER
tail -f cluster/logs/*.out
```

## Outputs

| Path | Content |
|------|---------|
| `results/seed_*/.../cpu/` or `gpu/` | Metrics, HPO, histories |
| `results/seed_*/.../combined/` | Merged tables and plots when both tracks finished |
| `results/seed_*/.../splits/` | Split pickles |

Notebooks use resume-friendly partial JSON when `FRESH_RUN` is false.

## Suggested order

Run datasets roughly small to large:
**ESOL, FreeSolv, Lipophilicity, BACE, BBBP, Tox21, HIV**.

For a first check, submit ESOL alone before launching HIV.
