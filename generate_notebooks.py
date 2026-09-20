#!/usr/bin/env python3
"""Generate benchmark v5 notebooks (flat 5-seed layout, scaffold + random splits).

For every (seed, split_mode, device, dataset) this writes:

    notebooks/seed_{SEED}/{MODE}/{cpu,gpu}/benchmark_{cpu,gpu}_{dataset}.ipynb

which reads/writes its own isolated results under:

    results/seed_{SEED}/{MODE}/{cpu,gpu,combined,splits,...}

Usage (from benchmark_v5/):
    python generate_notebooks.py                       # all seeds, both modes
    python generate_notebooks.py --seeds 0 4           # subset of seeds
    python generate_notebooks.py --modes scaffold      # scaffold only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "benchmark_v2" / "notebooks"))
from generate_dataset_notebooks import DATASETS  # noqa: E402
from v5_config import SEEDS, SPLIT_MODES, ensure_layout, seed_notebook_dir  # noqa: E402

try:
    # Preferred: single source of truth in hpo_config (needs scipy at import).
    from hpo_config import torch_hpo_settings, torch_hpo_tier  # noqa: E402
except ModuleNotFoundError:
    # Fallback for environments without scipy (notebook generation only; the
    # notebooks themselves bake these numbers in as literals). Keep in sync with
    # hpo_config._DATASET_HPO_TIER / torch_hpo_settings.
    _TIER = {
        "esol": "standard", "freesolv": "standard", "bace": "standard",
        "bbbp": "standard", "lipophilicity": "large", "tox21": "large",
        "hiv": "xlarge",
    }
    _SETTINGS = {"standard": (20, 80), "large": (10, 80), "xlarge": (10, 50)}

    def torch_hpo_tier(dataset: str) -> str:
        return _TIER.get(dataset, "standard")

    def torch_hpo_settings(dataset: str):
        return _SETTINGS[torch_hpo_tier(dataset)]


# ── cell helpers ─────────────────────────────────────────────────────────────
def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [dedent(text).strip() + "\n"]}


def code(text: str) -> dict:
    lines = dedent(text).strip("\n").splitlines()
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + "\n" for l in lines]}


def setup_cell(mode: str) -> dict:
    """mode here is the DEVICE (cpu/gpu). Paths are set in the constants cell."""
    return code(f"""
        # NOTE: autoreload is intentionally NOT enabled. With C-extension packages
        # (numpy, torch, torch_geometric) `%autoreload 2` reloads them mid-run and
        # corrupts their state (numpy RecursionError / torch_geometric circular
        # import). If you edit the benchmark modules, restart the kernel instead.
        import os
        # Avoid XGBoost/OpenMP segfaults in Jupyter on macOS (OMP Error #179).
        for _omp_var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[_omp_var] = "1"

        import sys
        from pathlib import Path

        ROOT = Path(".").resolve()
        for _ in range(6):
            if (ROOT / "mol_repr_utils.py").exists():
                break
            ROOT = ROOT.parent
        else:
            raise FileNotFoundError("Could not find benchmark_v5 root (mol_repr_utils.py)")
        sys.path.insert(0, str(ROOT))

        from mol_repr_utils import *
        from split_utils import (
            get_or_create_splits, subset_array, subset_list, subset_graphs_by_valid_idx,
            check_split_proportions,
        )
        from conformer_generation import load_conformer_graphs
        from hpo_utils import (
            hpo_tabular, hpo_gnn, hpo_schnet, hpo_lstm,
            save_hpo_params, hpo_params_path,
        )
        from hpo_config import SKLEARN_HPO_N_ITER

        DATA_DIR = ROOT / "data"

        DEVICE = get_device()
        print(f"Benchmark v5 ({mode.upper()}) code root: {{ROOT}}")
        print(f"Device: {{DEVICE}}")
        print("Result paths are set in the next cell from SPLIT_SEED / SPLIT_MODE.")
    """)


def constants_cell(cfg: dict, device: str, *, seed: int, split_mode: str) -> dict:
    gpu_lines = ""
    if device == "gpu":
        hpo_trials, hpo_epochs = torch_hpo_settings(cfg["slug"])
        tier = torch_hpo_tier(cfg["slug"])
        gpu_lines = f"""
        HPO_EPOCHS = {hpo_epochs}  # Optuna trial length; GNN_EPOCHS for final refit
        HPO_N_TRIALS = {hpo_trials}
        print(f"GPU HPO tier: {tier} — {{HPO_N_TRIALS}} trials, {{HPO_EPOCHS}} epochs/trial, batch={{BATCH_SIZE}}")
"""
    path_block = f"""
        RESULTS_ROOT = ROOT / "results" / f"seed_{{SPLIT_SEED}}" / SPLIT_MODE
        RESULTS_DIR = RESULTS_ROOT / "{device}"
        SPLITS_DIR = RESULTS_ROOT / "splits"
        HPO_DIR = RESULTS_DIR / "hpo"
        HISTORY_DIR = RESULTS_DIR / "histories"
        for d in (RESULTS_DIR, SPLITS_DIR, HPO_DIR, HISTORY_DIR):
            d.mkdir(parents=True, exist_ok=True)
        print(f"Results → {{RESULTS_ROOT}}")
"""
    return code(f"""
        DATASET_SLUG = "{cfg['slug']}"
        TASK_TYPE = "{cfg['task_type']}"
        OUT_CHANNELS = {cfg['out_channels']}
        BATCH_SIZE = {cfg['batch_size']}
        GNN_EPOCHS = {cfg['gnn_epochs']}
        LSTM_EPOCHS = {cfg['lstm_epochs']}
        TOX21_TABULAR = {cfg['tox21_tabular']}
{gpu_lines}
        SPLIT_MODE = "{split_mode}"   # "scaffold" or "random"
        SPLIT_SEED = {seed}
{path_block}
        PARTIAL_PATH = RESULTS_DIR / f"{{DATASET_SLUG}}_partial.json"
        FRESH_RUN = False
        HPO_N_ITER = SKLEARN_HPO_N_ITER
        # Optional skip list (pipe-separated names), and/or BENCH_AFTER_DMPNN=1
        # which skips GIN (2D) + D-MPNN (2D) so only GIN (3D)/SchNet/LSTM run.
        SKIP_MODELS = {{
            m.strip()
            for m in os.environ.get("BENCH_SKIP_MODELS", "").split("|")
            if m.strip()
        }}
        if os.environ.get("BENCH_AFTER_DMPNN", "").strip().lower() in {{"1", "true", "yes"}}:
            SKIP_MODELS |= {{"GIN (2D)", "D-MPNN (2D)"}}
        if SKIP_MODELS:
            print(f"Skipping models: {{sorted(SKIP_MODELS)}}")

        if FRESH_RUN:
            reset_dataset_run(
                DATASET_SLUG, RESULTS_DIR,
                results_root=RESULTS_ROOT, split_mode=SPLIT_MODE,
            )
            print("Fresh run — cleared partial results and splits.")

        results = load_partial_results(PARTIAL_PATH)
        if results:
            print(f"Resuming {{len(results)}} saved model(s): {{', '.join(order_results(results).keys())}}")
        print(f"SPLIT_SEED={{SPLIT_SEED}}  mode={{SPLIT_MODE}}")
    """)


def split_cell() -> dict:
    return code("""
        split_data = get_or_create_splits(
            smiles, RESULTS_ROOT, DATASET_SLUG,
            mode=SPLIT_MODE, seed=SPLIT_SEED, force=FRESH_RUN,
        )
        train_idx = split_data["train_idx"]
        val_idx = split_data["val_idx"]
        test_idx = split_data["test_idx"]
        # Guard: v4's malformed-split failure mode can never pass silently now.
        check_split_proportions(train_idx, val_idx, test_idx, raise_on_fail=True)
        print(split_data["summary"])
    """)


def cpu_tabular_cell(model_key: str, x_var: str, kind: str, tox21: bool) -> dict:
    return code(f"""
        MODEL_KEY = "{model_key}"
        results = load_partial_results(PARTIAL_PATH)
        if MODEL_KEY in results or MODEL_KEY in SKIP_MODELS:
            why = "already in partial" if MODEL_KEY in results else "BENCH_SKIP_MODELS"
            print(f"Skipping {{MODEL_KEY}} ({{why}})")
        else:
            X_tr = subset_array({x_var}, train_idx)
            X_te = subset_array({x_var}, test_idx)
            y_tr = subset_array(y, train_idx)
            y_te = subset_array(y, test_idx)
            print(f"→ {{MODEL_KEY}} (HPO on train={{len(train_idx)}}, test={{len(test_idx)}}) …")
            best, metrics = hpo_tabular(
                X_tr, y_tr, TASK_TYPE, model_kind="{kind}",
                tox21={tox21}, n_iter=HPO_N_ITER, seed=SPLIT_SEED,
                X_test=X_te, y_test=y_te,
            )
            save_hpo_params(
                hpo_params_path(HPO_DIR, MODEL_KEY, DATASET_SLUG),
                best,
                dataset=DATASET_SLUG,
                model=MODEL_KEY,
                seed=SPLIT_SEED,
            )
            save_model_result(results, MODEL_KEY, metrics, PARTIAL_PATH)
            print(f"   best params: {{best}}")
            print(f"   test metrics: {{metrics}}")
    """)


def gpu_splits_cell() -> str:
    return dedent("""
        graphs2d_all = smiles_to_graphs(smiles, y)
        graphs_dmpnn_all = smiles_to_dmpnn_graphs(smiles, y)
        X_tok_all, vocab = tokenize_smiles(smiles)

        graphs3d_all, valid_idx_3d, _ = load_conformer_graphs(
            DATASET_SLUG, data_dir=DATA_DIR, n_confs=25
        )
        graphs3d_all = ensure_atomic_numbers_list(graphs3d_all)
        print(f"3D graphs loaded: {len(graphs3d_all)} / {len(smiles)} (valid_idx mapping)")

        split_data = get_or_create_splits(
            smiles, RESULTS_ROOT, DATASET_SLUG,
            mode=SPLIT_MODE, seed=SPLIT_SEED, force=FRESH_RUN,
        )
        train_idx = split_data["train_idx"]
        val_idx = split_data["val_idx"]
        test_idx = split_data["test_idx"]
        check_split_proportions(train_idx, val_idx, test_idx, raise_on_fail=True)
        print(split_data["summary"])

        tr2d = subset_list(graphs2d_all, train_idx)
        va2d = subset_list(graphs2d_all, val_idx)
        te2d = subset_list(graphs2d_all, test_idx)

        tr_dmpnn = subset_list(graphs_dmpnn_all, train_idx)
        va_dmpnn = subset_list(graphs_dmpnn_all, val_idx)
        te_dmpnn = subset_list(graphs_dmpnn_all, test_idx)

        tr3d = subset_graphs_by_valid_idx(graphs3d_all, valid_idx_3d, train_idx)
        va3d = subset_graphs_by_valid_idx(graphs3d_all, valid_idx_3d, val_idx)
        te3d = subset_graphs_by_valid_idx(graphs3d_all, valid_idx_3d, test_idx)

        tr_ds = SMILESDataset(subset_array(X_tok_all, train_idx), subset_array(y, train_idx))
        va_ds = SMILESDataset(subset_array(X_tok_all, val_idx), subset_array(y, val_idx))
        te_ds = SMILESDataset(subset_array(X_tok_all, test_idx), subset_array(y, test_idx))
    """).strip()


def gpu_gnn_cell(model_key: str, arch: str, dim: str) -> dict:
    return code(f"""
        MODEL_KEY = "{model_key}"
        tr = tr3d if "{dim}" == "3D" else tr2d
        va = va3d if "{dim}" == "3D" else va2d
        te = te3d if "{dim}" == "3D" else te2d
        results = load_partial_results(PARTIAL_PATH)
        if MODEL_KEY in results or MODEL_KEY in SKIP_MODELS:
            why = "already in partial" if MODEL_KEY in results else "BENCH_SKIP_MODELS"
            print(f"Skipping {{MODEL_KEY}} ({{why}})")
        else:
            print(f"→ {{MODEL_KEY}} (Optuna {{HPO_N_TRIALS}} trials, {{SPLIT_MODE}} split) …")
            best, history = hpo_gnn(
                tr, va, TASK_TYPE, "{arch}", "{dim}", OUT_CHANNELS,
                n_trials=HPO_N_TRIALS, epochs=HPO_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
                seed=SPLIT_SEED,
            )
            save_hpo_params(
                hpo_params_path(HPO_DIR, MODEL_KEY, DATASET_SLUG),
                best,
                dataset=DATASET_SLUG,
                model=MODEL_KEY,
                seed=SPLIT_SEED,
            )
            from mol_repr_utils import eval_gnn, PyGLoader, build_gnn, train_gnn
            model = build_gnn("{arch}", "{dim}", OUT_CHANNELS,
                              hidden=best["hidden"], num_layers=best["num_layers"], dropout=best["dropout"])
            train_gnn(model, tr, va, TASK_TYPE, epochs=GNN_EPOCHS, lr=best["lr"],
                      batch_size=BATCH_SIZE, device=DEVICE, verbose=False)
            test_metrics = eval_gnn(model, PyGLoader(te, batch_size=BATCH_SIZE), DEVICE, TASK_TYPE)
            metrics = {{k: (v, 0.0) for k, v in test_metrics.items()}}
            hist_name = "{model_key}".replace(" ", "_").replace("(", "").replace(")", "")
            save_model_result(results, MODEL_KEY, metrics, PARTIAL_PATH,
                              history=history, history_path=HISTORY_DIR / f"{{DATASET_SLUG}}_{{hist_name}}.json")
            print(f"   {{SPLIT_MODE}} test: {{test_metrics}}")
    """)


def gpu_dmpnn_cell() -> dict:
    return code("""
        MODEL_KEY = "D-MPNN (2D)"
        results = load_partial_results(PARTIAL_PATH)
        if MODEL_KEY in results or MODEL_KEY in SKIP_MODELS:
            why = "already in partial" if MODEL_KEY in results else "BENCH_SKIP_MODELS"
            print(f"Skipping {MODEL_KEY} ({why})")
        else:
            print(f"→ {MODEL_KEY} (Optuna {HPO_N_TRIALS} trials, {SPLIT_MODE} split) …")
            best, history = hpo_gnn(
                tr_dmpnn, va_dmpnn, TASK_TYPE, "DMPNN", "2D", OUT_CHANNELS,
                n_trials=HPO_N_TRIALS, epochs=HPO_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
                seed=SPLIT_SEED,
            )
            save_hpo_params(
                hpo_params_path(HPO_DIR, MODEL_KEY, DATASET_SLUG),
                best,
                dataset=DATASET_SLUG,
                model=MODEL_KEY,
                seed=SPLIT_SEED,
            )
            from mol_repr_utils import eval_gnn, PyGLoader, build_gnn, train_gnn
            model = build_gnn("DMPNN", "2D", OUT_CHANNELS,
                              hidden=best["hidden"], num_layers=best["num_layers"], dropout=best["dropout"])
            train_gnn(model, tr_dmpnn, va_dmpnn, TASK_TYPE, epochs=GNN_EPOCHS, lr=best["lr"],
                      batch_size=BATCH_SIZE, device=DEVICE, verbose=False)
            test_metrics = eval_gnn(model, PyGLoader(te_dmpnn, batch_size=BATCH_SIZE), DEVICE, TASK_TYPE)
            metrics = {k: (v, 0.0) for k, v in test_metrics.items()}
            save_model_result(results, MODEL_KEY, metrics, PARTIAL_PATH,
                              history=history, history_path=HISTORY_DIR / f"{DATASET_SLUG}_dmpnn.json")
            print(f"   {SPLIT_MODE} test: {test_metrics}")
    """)


def _schnet_cell() -> str:
    return dedent("""
        MODEL_KEY = "SchNet (3D coords)"
        results = load_partial_results(PARTIAL_PATH)
        if MODEL_KEY in results or MODEL_KEY in SKIP_MODELS:
            why = "already in partial" if MODEL_KEY in results else "BENCH_SKIP_MODELS"
            print(f"Skipping {MODEL_KEY} ({why})")
        else:
            print(f"→ {MODEL_KEY} (Optuna {HPO_N_TRIALS} trials, {SPLIT_MODE} split) …")
            best, history = hpo_schnet(
                tr3d, va3d, TASK_TYPE, OUT_CHANNELS,
                n_trials=HPO_N_TRIALS, epochs=HPO_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
                seed=SPLIT_SEED,
            )
            save_hpo_params(
                hpo_params_path(HPO_DIR, MODEL_KEY, DATASET_SLUG),
                best,
                dataset=DATASET_SLUG,
                model=MODEL_KEY,
                seed=SPLIT_SEED,
            )
            from mol_repr_utils import eval_schnet, PyGLoader, build_schnet, train_schnet
            model = build_schnet(
                OUT_CHANNELS,
                hidden_channels=best["hidden_channels"],
                num_filters=best["num_filters"],
                num_interactions=best["num_interactions"],
                num_gaussians=best["num_gaussians"],
                cutoff=best["cutoff"],
            )
            train_schnet(model, tr3d, va3d, TASK_TYPE, epochs=GNN_EPOCHS, lr=best["lr"],
                         batch_size=BATCH_SIZE, device=DEVICE, verbose=False)
            te3d_schnet = ensure_atomic_numbers_list(te3d)
            test_metrics = eval_schnet(model, PyGLoader(te3d_schnet, batch_size=BATCH_SIZE), DEVICE, TASK_TYPE)
            metrics = {k: (v, 0.0) for k, v in test_metrics.items()}
            save_model_result(results, MODEL_KEY, metrics, PARTIAL_PATH,
                              history=history, history_path=HISTORY_DIR / f"{DATASET_SLUG}_schnet.json")
            print(f"   {SPLIT_MODE} test: {test_metrics}")
    """).strip()


def _lstm_cell() -> str:
    return dedent("""
        MODEL_KEY = "SMILES LSTM"
        results = load_partial_results(PARTIAL_PATH)
        if MODEL_KEY in results or MODEL_KEY in SKIP_MODELS:
            why = "already in partial" if MODEL_KEY in results else "BENCH_SKIP_MODELS"
            print(f"Skipping {MODEL_KEY} ({why})")
        else:
            print(f"→ {MODEL_KEY} (Optuna {HPO_N_TRIALS} trials, {SPLIT_MODE} split) …")
            best, history = hpo_lstm(
                tr_ds, va_ds, TASK_TYPE, len(vocab), OUT_CHANNELS,
                n_trials=HPO_N_TRIALS, epochs=HPO_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
                seed=SPLIT_SEED,
            )
            save_hpo_params(
                hpo_params_path(HPO_DIR, MODEL_KEY, DATASET_SLUG),
                best,
                dataset=DATASET_SLUG,
                model=MODEL_KEY,
                seed=SPLIT_SEED,
            )
            from mol_repr_utils import SmilesLSTM, eval_lstm, TorchLoader, train_lstm
            model = SmilesLSTM(len(vocab), embed_dim=best["embed_dim"], hidden_dim=best["hidden_dim"],
                               n_layers=best["n_layers"], out_channels=OUT_CHANNELS, dropout=best["dropout"])
            train_lstm(model, tr_ds, va_ds, TASK_TYPE, epochs=LSTM_EPOCHS, lr=best["lr"],
                       batch_size=BATCH_SIZE, device=DEVICE, verbose=False)
            test_metrics = eval_lstm(model, TorchLoader(te_ds, batch_size=BATCH_SIZE), DEVICE, TASK_TYPE)
            metrics = {k: (v, 0.0) for k, v in test_metrics.items()}
            save_model_result(results, MODEL_KEY, metrics, PARTIAL_PATH,
                              history=history, history_path=HISTORY_DIR / f"{DATASET_SLUG}_lstm.json")
            print(f"   {SPLIT_MODE} test: {test_metrics}")
    """).strip()


# ── notebook builders ────────────────────────────────────────────────────────
def build_cpu_notebook(cfg: dict, *, seed: int, split_mode: str) -> dict:
    minimize = "True" if cfg["task_type"] == "regression" else "False"
    cells = [
        md(
            f"# Benchmark v5 — CPU — {cfg['display']} — {split_mode} split, seed {seed}\n\n"
            f"{cfg['description']}\n\n"
            f"**{split_mode.capitalize()} split** (80/10/10), seed **{seed}**. "
            "Tabular HPO on train; metrics on the held-out test set.\n\n"
            f"Results go to `results/seed_{seed}/{split_mode}/`."
        ),
        setup_cell("cpu"),
        constants_cell(cfg, "cpu", seed=seed, split_mode=split_mode),
        md("### Load data"),
        code(f"import numpy as np\nimport pandas as pd\n\n{cfg['load_code']}\nsmiles, y = filter_bonded_molecules(smiles, y)\nprint(f'After bond filter: {{len(smiles)}} molecules')"),
        md(f"### {split_mode.capitalize()} split"),
        split_cell(),
        md("### Representations"),
        code("X_morgan = smiles_to_morgan(smiles)\nX_desc = smiles_to_descriptors(smiles)\nX_combined = smiles_to_combined(smiles)\nprint(X_morgan.shape, X_desc.shape, X_combined.shape)"),
        md("### Morgan + RF"), cpu_tabular_cell("Morgan + RF", "X_morgan", "rf", cfg["tox21_tabular"]),
        md("### Morgan + SVM"), cpu_tabular_cell("Morgan + SVM", "X_morgan", "svm", cfg["tox21_tabular"]),
        md("### Morgan + XGBoost"), cpu_tabular_cell("Morgan + XGBoost", "X_morgan", "xgb", cfg["tox21_tabular"]),
        md("### Descriptors + RF"), cpu_tabular_cell("Descriptors + RF", "X_desc", "rf", cfg["tox21_tabular"]),
        md("### Descriptors + SVM"), cpu_tabular_cell("Descriptors + SVM", "X_desc", "svm", cfg["tox21_tabular"]),
        md("### Descriptors + XGBoost"), cpu_tabular_cell("Descriptors + XGBoost", "X_desc", "xgb", cfg["tox21_tabular"]),
        md("### Combined + XGBoost"), cpu_tabular_cell("Combined + XGBoost", "X_combined", "xgb", cfg["tox21_tabular"]),
        md("### Finalize CPU track"),
        code(
            f'finalize_dataset_results(DATASET_SLUG, PARTIAL_PATH, RESULTS_DIR, TASK_TYPE, '
            f'plot_title=f"{cfg["plot_title"]} (CPU, {{SPLIT_MODE}}, seed={{SPLIT_SEED}})", '
            f'ylabel="{cfg["ylabel"]}", minimize={minimize})'
        ),
    ]
    return _nb(cells)


def build_gpu_notebook(cfg: dict, *, seed: int, split_mode: str) -> dict:
    minimize = "True" if cfg["task_type"] == "regression" else "False"
    cells = [
        md(
            f"# Benchmark v5 — GPU — {cfg['display']} — {split_mode} split, seed {seed}\n\n"
            f"{cfg['description']}\n\n"
            f"**{split_mode.capitalize()} split**, seed **{seed}** — GIN, D-MPNN, SchNet, LSTM with Optuna HPO.\n\n"
            f"Results go to `results/seed_{seed}/{split_mode}/`."
        ),
        setup_cell("gpu"),
        constants_cell(cfg, "gpu", seed=seed, split_mode=split_mode),
        md("### Load data"),
        code(f"import numpy as np\nimport pandas as pd\n\n{cfg['load_code']}\nsmiles, y = filter_bonded_molecules(smiles, y)\nprint(f'After bond filter: {{len(smiles)}} molecules')"),
        md(f"### Representations & {split_mode} split"),
        code(gpu_splits_cell()),
        md("### GIN (2D)"), gpu_gnn_cell("GIN (2D)", "GIN", "2D"),
        md("### D-MPNN (2D)\nBond-centric directed MPNN (Yang et al. 2019)."),
        gpu_dmpnn_cell(),
        md("### GIN (3D)"), gpu_gnn_cell("GIN (3D)", "GIN", "3D"),
        md("### SchNet (3D coords)\nUses atomic numbers + xyz from min-energy n=25 conformer."),
        code(_schnet_cell()),
        md("### SMILES LSTM\nCanonical isomeric SMILES tokens."),
        code(_lstm_cell()),
        md("### Finalize GPU + merge"),
        code(f"""
            finalize_dataset_results(DATASET_SLUG, PARTIAL_PATH, RESULTS_DIR, TASK_TYPE,
                plot_title=f"{cfg['plot_title']} (GPU, {{SPLIT_MODE}}, seed={{SPLIT_SEED}})", ylabel="{cfg['ylabel']}", minimize={minimize})
            cpu_partial = RESULTS_ROOT / "cpu" / f"{{DATASET_SLUG}}_partial.json"
            if cpu_partial.exists():
                merge_combined_results(DATASET_SLUG, cpu_partial, PARTIAL_PATH, RESULTS_ROOT / "combined",
                    TASK_TYPE, plot_title=f"{cfg['plot_title']} (all models, {{SPLIT_MODE}}, seed={{SPLIT_SEED}})", ylabel="{cfg['ylabel']}", minimize={minimize})
            else:
                print("Run CPU notebook first for combined results.")
        """),
    ]
    return _nb(cells)


def _nb(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python (Thesis molnet)", "language": "python", "name": "thesis-molnet"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Seeds to generate (default: {SEEDS})")
    parser.add_argument("--modes", nargs="+", default=None,
                        help=f"Split modes to generate (default: {SPLIT_MODES})")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(SEEDS)
    modes = args.modes if args.modes is not None else list(SPLIT_MODES)

    for seed in seeds:
        for mode in modes:
            if mode not in SPLIT_MODES:
                raise SystemExit(f"Unknown mode {mode!r}; choose from {SPLIT_MODES}")
            ensure_layout(seed, mode)
            cpu_dir = seed_notebook_dir(seed, mode, "cpu")
            gpu_dir = seed_notebook_dir(seed, mode, "gpu")
            cpu_dir.mkdir(parents=True, exist_ok=True)
            gpu_dir.mkdir(parents=True, exist_ok=True)
            for cfg in DATASETS:
                (cpu_dir / f"benchmark_cpu_{cfg['slug']}.ipynb").write_text(
                    json.dumps(build_cpu_notebook(cfg, seed=seed, split_mode=mode), indent=1) + "\n"
                )
                (gpu_dir / f"benchmark_gpu_{cfg['slug']}.ipynb").write_text(
                    json.dumps(build_gpu_notebook(cfg, seed=seed, split_mode=mode), indent=1) + "\n"
                )
            print(f"[seed {seed} / {mode}] {len(DATASETS)} datasets → {cpu_dir.parent.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
