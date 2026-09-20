"""Hyperparameter search spaces for benchmark v5 (uniform protocol across models)."""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform

# ── Tabular (RandomizedSearchCV) ─────────────────────────────────────────────

# Bare estimator param names — hpo_utils adds the "model__" pipeline prefix.
RF_PARAM_DIST = {
    "n_estimators": randint(100, 301),
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", 0.5],
}

SVM_PARAM_DIST = {
    "C": loguniform(1e-2, 1e2),
    "gamma": ["scale", "auto"],
}

XGB_PARAM_DIST = {
    "n_estimators": randint(100, 301),
    "max_depth": randint(3, 11),
    "learning_rate": loguniform(1e-3, 3e-1),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 6),
}

SKLEARN_HPO_N_ITER = 15
SKLEARN_HPO_CV = 5

# XGBoost OpenMP can segfault Jupyter on macOS (OMP Error #179); use 1 thread.
XGB_N_JOBS = 1

# RBF SVM does not scale well: fit cost grows ~O(n^2)-O(n^3), so HPO on a few
# thousand molecules with 5-fold CV x 15 candidates can run for hours. HPO uses a
# stratified subsample capped here, then the final model is refit on the full
# training split. Lowered 5000 -> 2000 in v5 so lipophilicity (~3.4k train) and
# other mid-size sets subsample for HPO instead of grinding on the full set.
SVM_HPO_MAX_TRAIN = 2000

# Parallelism for the tabular RandomizedSearchCV. SVM candidates are independent
# and single-threaded, so parallelize the search (-1 = all cores). RF/XGB already
# parallelize inside each fit, so their search stays single-process to avoid
# oversubscription and the macOS OpenMP segfault.
SVM_HPO_SEARCH_N_JOBS = -1

# ── PyTorch (Optuna) ─────────────────────────────────────────────────────────

GNN_HPO_SPACE = {
    "hidden": [128, 192, 256],
    "num_layers": [3, 4, 5],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
}

SCHNET_HPO_SPACE = {
    "hidden_channels": [96, 128, 192],
    "num_filters": [96, 128, 192],
    "num_interactions": [3, 4, 5, 6],
    "num_gaussians": [25, 50],
    "cutoff": [5.0, 10.0],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
}

LSTM_HPO_SPACE = {
    "embed_dim": [64, 128],
    "hidden_dim": [128, 192, 256],
    "n_layers": [1, 2],
    "dropout": [0.2, 0.3, 0.4],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3],
}

TORCH_HPO_N_TRIALS = 20
TORCH_HPO_EPOCHS = 80
TORCH_HPO_PATIENCE = 15

# ── GPU Optuna budget by dataset size (benchmark v4 tiers) ───────────────────
# Standard : ESOL, FreeSolv, BACE, BBBP  — 20 trials, 80 HPO epochs, batch 32
# Large    : Lipophilicity, Tox21         — 10 trials, 80 HPO epochs, batch 64–128
# X-large  : HIV                          — 10 trials, 50 HPO epochs, batch 128
#   (if HIV pilot fails: set TORCH_HPO_N_TRIALS_XLARGE = 5 in constants / notebook)

TORCH_HPO_N_TRIALS_STANDARD = 20
TORCH_HPO_EPOCHS_STANDARD = 80

TORCH_HPO_N_TRIALS_LARGE = 10
TORCH_HPO_EPOCHS_LARGE = 80

TORCH_HPO_N_TRIALS_XLARGE = 10
TORCH_HPO_EPOCHS_XLARGE = 50

_DATASET_HPO_TIER: dict[str, str] = {
    "esol": "standard",
    "freesolv": "standard",
    "bace": "standard",
    "bbbp": "standard",
    "lipophilicity": "large",
    "tox21": "large",
    "hiv": "xlarge",
}


def torch_hpo_settings(dataset: str) -> tuple[int, int]:
    """Return (n_trials, hpo_epochs) for a dataset slug."""
    tier = _DATASET_HPO_TIER.get(dataset, "standard")
    if tier == "large":
        return TORCH_HPO_N_TRIALS_LARGE, TORCH_HPO_EPOCHS_LARGE
    if tier == "xlarge":
        return TORCH_HPO_N_TRIALS_XLARGE, TORCH_HPO_EPOCHS_XLARGE
    return TORCH_HPO_N_TRIALS_STANDARD, TORCH_HPO_EPOCHS_STANDARD


def torch_hpo_tier(dataset: str) -> str:
    return _DATASET_HPO_TIER.get(dataset, "standard")

# Default trained architecture (after HPO, or fallback if HPO skipped)
GNN_DEFAULT = {"hidden": 256, "num_layers": 4, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4}
SCHNET_DEFAULT = {
    "hidden_channels": 128,
    "num_filters": 128,
    "num_interactions": 4,
    "num_gaussians": 50,
    "cutoff": 10.0,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}
LSTM_DEFAULT = {
    "embed_dim": 64,
    "hidden_dim": 256,
    "n_layers": 2,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
}
