# Submitting benchmark jobs on SLURM

Modular submission: choose **device** (cpu/gpu), **split** (scaffold/random),
**seed** (0-4), and one dataset or all seven. Results are written under
`results/seed_<seed>/<mode>/`.

## Before uploading

This repository is meant to be self-contained. Ensure:

- **`data/`** holds MoleculeNet CSVs and, for 3D models, conformer pickles.
- The Python environment is built **on the cluster** (do not upload a local
  `.venv` from another OS).

```bash
cd molnet-repr-benchmark
bash cluster/setup_cluster_env.sh
# Optional CUDA builds:
# bash cluster/install_cuda_torch.sh && bash cluster/install_pyg_extensions.sh
```

Set the SLURM account and mail in `cluster/rwth_config.sh`:

```bash
export RWTH_ACCOUNT="ACCOUNT_ID"
export RWTH_MAIL_USER="name@example.com"
```

List accounts with `sacctmgr show user $USER format=account%30` if unsure.

## Submitting

```bash
# One dataset
bash cluster/submit.sh --device cpu --seed 0 --mode scaffold esol
bash cluster/submit.sh -d gpu -s 0 -m random esol

# All seven datasets for a seed + mode + device
bash cluster/submit.sh -d cpu -s 0 -m scaffold --all
bash cluster/submit.sh -d gpu -s 0 -m scaffold --all

# Preview sbatch commands without submitting
bash cluster/submit.sh -d gpu -s 2 -m scaffold hiv --dry-run
```

Flags: `-d/--device cpu|gpu` (required), `-s/--seed 0-4` (required),
`-m/--mode scaffold|random` (default scaffold), `-a/--all`, `--dry-run`.

The wrapper reads `cluster/rwth_config.sh` and passes account, partition
(c23ms for cpu, c23g for gpu in the defaults), `--gres=gpu:1` for GPU jobs,
and a job name such as `v5-gpu-scaffold-s0-esol`.

## Recommended order

For a given seed and mode, run **CPU first, then GPU** per dataset. The CPU job
writes tabular metrics and the shared split; the GPU job can merge both into
`results/seed_<seed>/<mode>/combined/`. GPU-only runs still work (they recreate
the split if needed), but the combined table needs CPU results.

Full seed, both splits:

```bash
for m in scaffold random; do
  bash cluster/submit.sh -d cpu -s 0 -m "$m" --all
  bash cluster/submit.sh -d gpu -s 0 -m "$m" --all
done
```

## Monitoring and output

```bash
squeue -u $USER
tail -f cluster/logs/v5-cpu-scaffold-s0-esol_<jobid>.out
```

Each job writes under `results/seed_<seed>/<mode>/{cpu,gpu}/` (metrics, HPO,
histories) and may update `combined/` and `splits/`. Aggregate across seeds with
`python aggregate_results.py`.

## Notes

- Split work across machines by seed so `results/seed_N/` directories do not collide.
- `run_cpu.slurm` / `run_gpu.slurm` accept direct `sbatch` if the `#SBATCH`
  account line is edited; the wrapper is usually simpler.
- Older `run_*_multiseed.slurm` / related scripts are legacy helpers (including
  conformer generation). Prefer `submit.sh` for the main benchmark.
