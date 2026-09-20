# Submitting benchmark v5 jobs on SLURM (RWTH)

Modular submission: pick **device** (cpu/gpu), **split** (scaffold/random),
**seed** (0–4), and **one or all datasets**. Results save automatically under
`results/seed_<seed>/<mode>/`.

## Before you upload

`benchmark_v5` is not fully self-contained by default — two things point outside it:

- **`data/` is a symlink** to `../benchmark_v3/data` (datasets + precomputed
  conformers). GPU/3D jobs need the conformer pickles.
- **`setup_cluster_env.sh` builds the venv at `../.venv`** (shared with v4).

Two ways to handle it:

1. **Upload the whole `molnet_esol_project/`** (simplest — symlink and `../.venv`
   both resolve), then `cd benchmark_v5` on the cluster and proceed. Don't upload
   any local `.venv` (macOS binaries won't run on Linux — rebuild on the cluster).
2. **Upload only `benchmark_v5/`** — first make it self-contained: replace the
   `data` symlink with the real folder (`rm data && cp -R ../benchmark_v3/data data`)
   and build the venv inside it (`VENV_DIR=./.venv bash cluster/setup_cluster_env.sh`).
   `activate_env.sh` finds `./.venv` automatically.

## One-time setup on the cluster

```bash
cd molnet_esol_project/benchmark_v5

# 1) build the venv + install deps (rdkit, torch, PyG, xgboost, papermill, ...)
bash cluster/setup_cluster_env.sh
#    GPU node CUDA build of torch + PyG extensions, if needed:
#    bash cluster/install_cuda_torch.sh && bash cluster/install_pyg_extensions.sh

# 2) set your account (partition/mail already default to c23ms / c23g)
nano cluster/rwth_config.sh          # RWTH_ACCOUNT=thes1234
```

Find your account with `sacctmgr show user $USER format=account%30`.

## Submitting

```bash
# one dataset
bash cluster/submit.sh --device cpu --seed 0 --mode scaffold esol
bash cluster/submit.sh -d gpu -s 0 -m random esol

# all seven datasets for a seed+mode+device
bash cluster/submit.sh -d cpu -s 0 -m scaffold --all
bash cluster/submit.sh -d gpu -s 0 -m scaffold --all

# preview the sbatch commands without submitting
bash cluster/submit.sh -d gpu -s 2 -m scaffold hiv --dry-run
```

Flags: `-d/--device cpu|gpu` (required), `-s/--seed 0-4` (required),
`-m/--mode scaffold|random` (default scaffold), `-a/--all`, `--dry-run`.

The wrapper reads `cluster/rwth_config.sh` and passes `--account`, `--partition`
(c23ms for cpu, c23g for gpu), `--gres=gpu:1` (gpu only), and a descriptive
`--job-name` like `v5-gpu-scaffold-s0-esol`.

## Recommended order

For a given seed + mode, run **CPU first, then GPU** on each dataset: the CPU job
writes the tabular metrics and the shared split, and the GPU job merges both into
`results/seed_<seed>/<mode>/combined/`. (The GPU job will still run alone — it
recreates the split if needed — but the combined table needs the CPU results.)

A full seed, both splits:

```bash
for m in scaffold random; do
  bash cluster/submit.sh -d cpu -s 0 -m "$m" --all
  bash cluster/submit.sh -d gpu -s 0 -m "$m" --all
done
```

## Monitoring & output

```bash
squeue --me
tail -f cluster/logs/v5-cpu-scaffold-s0-esol_<jobid>.out
```

Per job you get: `results/seed_<seed>/<mode>/{cpu,gpu}/<dataset>_partial.json`
(metrics), `.../hpo/<dataset>/*.json` (HPO params), `.../histories/` (training
curves), `.../combined/<dataset>_results.csv` + plots, and the split pickle in
`.../splits/`. Aggregate across seeds afterwards with
`python aggregate_results.py`.

## Notes

- Split work across machines by seed so `results/seed_N/` never collide.
- `run_cpu.slurm` / `run_gpu.slurm` also accept direct `sbatch` calls
  (`sbatch cluster/run_cpu.slurm esol 0 scaffold`) if you edit the `#SBATCH
  --account` line, but the wrapper is easier.
- The older `run_*_multiseed.slurm` / `submit_*` scripts in this folder are the
  legacy v4 (pre-flat-layout) versions and the conformer-generation jobs — ignore
  them for v5 benchmark runs.
```
