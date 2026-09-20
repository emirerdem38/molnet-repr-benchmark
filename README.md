# MoleculeNet representation benchmark

Code for the M.Sc. thesis comparing molecular representations on seven
MoleculeNet datasets (ESOL, FreeSolv, Lipophilicity, BACE, BBBP, Tox21, HIV).

Twelve setups are run under the same splits and HPO budget:

- classical models on Morgan fingerprints, RDKit descriptors, or both (RF, SVM, XGBoost)
- GIN (2D / 3D), D-MPNN, SchNet, SMILES LSTM

Scaffold splitting is the main protocol; random splits are included for comparison.
Each configuration is repeated over seeds 0–4.

## Layout

```
*.py, run.sh          shared code and runner
notebooks/            per-seed CPU/GPU notebooks (regenerate with generate_notebooks.py)
results/summary/      mean ± std tables used in the thesis
data/                 put MoleculeNet CSVs / conformers here (see data/README.md)
cluster/              optional SLURM helpers (RWTH-style)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On a CUDA machine, install the matching PyTorch build first
(https://pytorch.org), then the rest of `requirements.txt`.
For SchNet / PyG extras on cluster or Mac, see `cluster/`.

## Run

Generate notebooks (only if you change the generator or config):

```bash
python generate_notebooks.py
# or: python generate_notebooks.py --seeds 0 --modes scaffold
```

Execute one track:

```bash
bash run.sh 0 scaffold cpu esol     # one dataset
bash run.sh 0 scaffold cpu          # all datasets, tabular
bash run.sh 0 scaffold gpu          # deep models (also writes combined/)
```

Aggregate when seeds are finished:

```bash
python aggregate_results.py
```

Extra studies from the thesis:

```bash
python smoke_timing_benchmark.py
python lipophilicity_subset_benchmark.py
```

## Results in this repo

`results/summary/` holds the aggregated CSVs (and compute-time tables) that
back the thesis numbers. Full per-seed HPO dumps, histories, and raw run folders
are large and are not mirrored here; regenerate them with `run.sh` or ask if you
need a specific artifact.

## Thesis

Comparative Study of Molecular Representations for Machine Learning-Based
Prediction of Pharmaceutical Properties (RWTH Aachen University).

## License

MIT
