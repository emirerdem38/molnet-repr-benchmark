"""Hyperparameter optimization helpers for benchmark v5."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import optuna
from sklearn.model_selection import RandomizedSearchCV

from hpo_config import (
    GNN_DEFAULT,
    GNN_HPO_SPACE,
    LSTM_DEFAULT,
    LSTM_HPO_SPACE,
    RF_PARAM_DIST,
    SCHNET_DEFAULT,
    SCHNET_HPO_SPACE,
    SKLEARN_HPO_CV,
    SKLEARN_HPO_N_ITER,
    SVM_HPO_MAX_TRAIN,
    SVM_HPO_SEARCH_N_JOBS,
    SVM_PARAM_DIST,
    XGB_PARAM_DIST,
    XGB_N_JOBS,
    TORCH_HPO_EPOCHS,
    TORCH_HPO_N_TRIALS,
    TORCH_HPO_PATIENCE,
)
def _pipeline_param_dist(estimator_params: Dict[str, Any]) -> Dict[str, Any]:
    """Map estimator hyperparams to Pipeline step ``model`` (idempotent on ``model__``)."""
    out: Dict[str, Any] = {}
    for key, value in estimator_params.items():
        if key.startswith("model__"):
            out[key] = value
        else:
            out[f"model__{key}"] = value
    return out


from mol_repr_utils import (
    build_gnn,
    build_rf_pipeline,
    build_svm_pipeline,
    build_xgb_pipeline,
    build_schnet,
    cv_sklearn,
    prepare_schnet_graphs,
    evaluate_sklearn_test,
    get_device,
    set_seed,
    train_gnn,
    train_lstm,
    train_schnet,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _safe_model_key(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def hpo_params_path(
    hpo_dir: Union[str, Path],
    model_key: str,
    dataset_slug: str,
) -> Path:
    """One JSON per (dataset, model), e.g. results/cpu/hpo/esol/Morgan_+_RF_hpo.json."""
    return Path(hpo_dir) / dataset_slug / f"{_safe_model_key(model_key)}_hpo.json"


def save_hpo_params(
    path: Union[str, Path],
    params: Dict[str, Any],
    *,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    seed: int = 42,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "dataset": dataset or path.parent.name,
        "model": model,
        "seed": seed,
        "params": params,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_hpo_params(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "params" in data:
        return data["params"]
    return data


def _subsample_train_for_hpo(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    max_n: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a reproducible train subsample for large-sample SVM HPO."""
    n = len(y)
    if n <= max_n:
        return X, y
    if task_type == "classification":
        from sklearn.model_selection import StratifiedShuffleSplit

        split = StratifiedShuffleSplit(n_splits=1, train_size=max_n, random_state=seed)
        train_idx, _ = next(split.split(X, y))
        return X[train_idx], y[train_idx]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return X[idx], y[idx]


def _sample_hyperparameter(dist: Any, rng: np.random.Generator) -> Any:
    """Sample one value from a list or scipy distribution (correct int/float types)."""
    if isinstance(dist, list):
        return dist[int(rng.integers(0, len(dist)))]
    val = dist.rvs(random_state=rng)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    return val


def tox21_cv_auc(X, y, model, n_splits=5, seed=42) -> Tuple[float, float]:
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import warnings

    fold_aucs = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        task_aucs = []
        for t in range(y.shape[1]):
            tr_mask = ~np.isnan(y_tr[:, t])
            te_mask = ~np.isnan(y_te[:, t])
            if tr_mask.sum() < 10 or te_mask.sum() < 5:
                continue
            if len(np.unique(y_tr[tr_mask, t])) < 2 or len(np.unique(y_te[te_mask, t])) < 2:
                continue
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("m", model),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_tr[tr_mask], y_tr[tr_mask, t])
                proba = (
                    pipe.predict_proba(X_te[te_mask])[:, 1]
                    if hasattr(pipe.named_steps["m"], "predict_proba")
                    else pipe.decision_function(X_te[te_mask])
                )
                task_aucs.append(roc_auc_score(y_te[te_mask, t], proba))
        if task_aucs:
            fold_aucs.append(float(np.mean(task_aucs)))
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def _tox21_random_search(
    X,
    y,
    builder: Callable,
    param_dist: Dict,
    n_iter: int,
    seed: int,
    *,
    subsample_for_hpo: bool = False,
) -> Tuple[Dict[str, Any], Tuple[float, float]]:
    X_hpo, y_hpo = X, y
    if subsample_for_hpo:
        X_hpo, y_hpo = _subsample_train_for_hpo(
            X, y, "multilabel_classification", SVM_HPO_MAX_TRAIN, seed
        )
        logger.info(
            "Tox21 SVM HPO subsample: %d / %d train molecules",
            len(y_hpo),
            len(y),
        )
    rng = np.random.default_rng(seed)
    keys = list(param_dist.keys())
    best_score, best_params = -np.inf, None
    for _ in range(n_iter):
        params = {k: _sample_hyperparameter(v, rng) for k, v in param_dist.items()}
        model = builder(**params)
        mean, std = tox21_cv_auc(X_hpo, y_hpo, model, n_splits=SKLEARN_HPO_CV, seed=seed)
        if mean > best_score:
            best_score, best_params = mean, dict(params)
    assert best_params is not None
    final = tox21_cv_auc(X_hpo, y_hpo, builder(**best_params), n_splits=SKLEARN_HPO_CV, seed=seed)
    return best_params, final


def _tox21_evaluate_test(
    builder: Callable,
    best: Dict[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
) -> Dict[str, Tuple[float, float]]:
    """Per-task train/test evaluation with NaN masking (Tox21 missing labels)."""
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import warnings

    model = builder(**best)
    aucs, prcs = [], []
    for t in range(y_train.shape[1]):
        tr_mask = ~np.isnan(y_train[:, t])
        te_mask = ~np.isnan(y_test[:, t])
        if tr_mask.sum() < 10 or te_mask.sum() < 5:
            continue
        if len(np.unique(y_train[tr_mask, t])) < 2 or len(np.unique(y_test[te_mask, t])) < 2:
            continue
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", model),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train[tr_mask], y_train[tr_mask, t])
            proba = (
                pipe.predict_proba(X_test[te_mask])[:, 1]
                if hasattr(pipe.named_steps["m"], "predict_proba")
                else pipe.decision_function(X_test[te_mask])
            )
        aucs.append(roc_auc_score(y_test[te_mask, t], proba))
        prcs.append(average_precision_score(y_test[te_mask, t], proba))
    return {
        "ROC-AUC": (float(np.mean(aucs)) if aucs else float("nan"), 0.0),
        "PRC-AUC": (float(np.mean(prcs)) if prcs else float("nan"), 0.0),
    }


def _tox21_tabular_result(
    builder: Callable,
    best: Dict[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
    cv_metrics: Tuple[float, float],
) -> Dict[str, Tuple[float, float]]:
    """Return scaffold test metrics for Tox21 when test set provided, else CV metrics."""
    if X_test is None or y_test is None:
        return {"ROC-AUC": cv_metrics}
    return _tox21_evaluate_test(builder, best, X_train, y_train, X_test, y_test)


def hpo_tabular(
    X,
    y,
    task_type: str,
    model_kind: str,
    *,
    tox21: bool = False,
    n_iter: int = SKLEARN_HPO_N_ITER,
    seed: int = 42,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[float, float]]]:
    """HPO on train set; optional scaffold test evaluation when X_test/y_test given."""
    from mol_repr_utils import evaluate_sklearn_test

    if model_kind == "rf":
        pipe = build_rf_pipeline(task_type, random_state=seed)
        param_dist = _pipeline_param_dist(RF_PARAM_DIST)
        if tox21:
            def builder(**kw):
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(random_state=seed, n_jobs=-1, **kw)
            flat = {k.replace("model__", ""): v for k, v in param_dist.items()}
            best, metrics_tuple = _tox21_random_search(
                X, y, builder, flat, n_iter=n_iter, seed=seed
            )
            return best, _tox21_tabular_result(builder, best, X, y, X_test, y_test, metrics_tuple)
    elif model_kind == "svm":
        pipe = build_svm_pipeline(task_type, random_state=seed)
        param_dist = _pipeline_param_dist(SVM_PARAM_DIST)
        if tox21:
            def builder(**kw):
                from sklearn.svm import SVC
                return SVC(probability=True, kernel="rbf", random_state=seed, **kw)
            flat = {k.replace("model__", ""): v for k, v in param_dist.items()}
            best, metrics_tuple = _tox21_random_search(
                X, y, builder, flat, n_iter=n_iter, seed=seed,
                subsample_for_hpo=len(y) > SVM_HPO_MAX_TRAIN,
            )
            return best, _tox21_tabular_result(builder, best, X, y, X_test, y_test, metrics_tuple)
    elif model_kind == "xgb":
        pipe = build_xgb_pipeline(task_type, random_state=seed)
        param_dist = _pipeline_param_dist(XGB_PARAM_DIST)
        if tox21:
            def builder(**kw):
                from xgboost import XGBClassifier
                defaults = dict(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=XGB_N_JOBS,
                    tree_method="hist",
                    verbosity=0,
                )
                defaults.update(kw)
                return XGBClassifier(**defaults)
            flat = {k.replace("model__", ""): v for k, v in param_dist.items()}
            best, metrics_tuple = _tox21_random_search(X, y, builder, flat, n_iter=n_iter, seed=seed)
            return best, _tox21_tabular_result(builder, best, X, y, X_test, y_test, metrics_tuple)
    else:
        raise ValueError(f"Unknown tabular model_kind: {model_kind!r}")

    if not tox21:
        X_hpo, y_hpo = X, y
        svm_subsampled = model_kind == "svm" and len(y) > SVM_HPO_MAX_TRAIN
        if svm_subsampled:
            X_hpo, y_hpo = _subsample_train_for_hpo(X, y, task_type, SVM_HPO_MAX_TRAIN, seed)
            logger.info(
                "SVM HPO subsample: %d / %d train molecules",
                len(y_hpo),
                len(y),
            )
        # SVM candidates are independent + single-threaded -> parallelize the
        # search. RF/XGB parallelize inside each fit, so keep their search serial.
        search_n_jobs = SVM_HPO_SEARCH_N_JOBS if model_kind == "svm" else 1
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=SKLEARN_HPO_CV,
            random_state=seed,
            n_jobs=search_n_jobs,
            scoring="roc_auc" if task_type == "classification" else "neg_root_mean_squared_error",
        )
        search.fit(X_hpo, y_hpo)
        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        if svm_subsampled:
            final_estimator = build_svm_pipeline(task_type, random_state=seed, **best)
            final_estimator.fit(X, y)
        else:
            final_estimator = search.best_estimator_
        if X_test is not None and y_test is not None:
            metrics = evaluate_sklearn_test(
                final_estimator, X, y, X_test, y_test, task_type
            )
        else:
            metrics = cv_sklearn(
                final_estimator, X, y, task_type,
                n_splits=SKLEARN_HPO_CV, random_state=seed,
            )
        return best, metrics

    raise RuntimeError("unreachable")


def _gnn_val_score(model, val_graphs, device, task_type) -> float:
    from mol_repr_utils import eval_gnn, PyGLoader
    loader = PyGLoader(val_graphs, batch_size=64, shuffle=False)
    metrics = eval_gnn(model, loader, device, task_type)
    if task_type == "regression":
        return metrics["RMSE"]
    return -metrics.get("ROC-AUC", metrics.get("Mean ROC-AUC", 0.0))


def _optuna_trial_callback(label: str, n_trials: int):
    """Log each finished Optuna trial (training inside trials uses verbose=False)."""

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        val = trial.value
        best = study.best_value if study.best_trial is not None else None
        val_s = f"{val:.6f}" if val is not None else "FAILED"
        best_s = f"{best:.6f}" if best is not None else "n/a"
        logger.info(
            "%s — trial %d/%d finished | val=%s | best=%s",
            label,
            trial.number + 1,
            n_trials,
            val_s,
            best_s,
        )
        sys.stdout.flush()

    return _cb


def hpo_gnn(
    train_graphs,
    val_graphs,
    task_type: str,
    arch: str,
    dim: str,
    out_channels: int,
    *,
    n_trials: int = TORCH_HPO_N_TRIALS,
    epochs: int = TORCH_HPO_EPOCHS,
    batch_size: int = 32,
    seed: int = 42,
    device=None,
) -> Tuple[Dict[str, Any], Dict]:
    device = device or get_device()
    set_seed(seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden": trial.suggest_categorical("hidden", GNN_HPO_SPACE["hidden"]),
            "num_layers": trial.suggest_categorical("num_layers", GNN_HPO_SPACE["num_layers"]),
            "dropout": trial.suggest_categorical("dropout", GNN_HPO_SPACE["dropout"]),
            "lr": trial.suggest_categorical("lr", GNN_HPO_SPACE["lr"]),
            "weight_decay": trial.suggest_categorical("weight_decay", GNN_HPO_SPACE["weight_decay"]),
        }
        logger.info(
            "%s %s — starting trial %d/%d (layers=%s, hidden=%s)",
            arch, dim, trial.number + 1, n_trials, params["num_layers"], params["hidden"],
        )
        sys.stdout.flush()
        model = build_gnn(
            arch, dim, out_channels,
            hidden=params["hidden"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
        )
        train_gnn(
            model, train_graphs, val_graphs, task_type,
            epochs=epochs, lr=params["lr"], batch_size=batch_size,
            device=device, verbose=False,
            patience=TORCH_HPO_PATIENCE, weight_decay=params["weight_decay"],
        )
        return _gnn_val_score(model, val_graphs, device, task_type)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    label = f"{arch} {dim} HPO"
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_optuna_trial_callback(label, n_trials)],
    )
    best = study.best_params or dict(GNN_DEFAULT)
    model = build_gnn(
        arch, dim, out_channels,
        hidden=best["hidden"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
    )
    history = train_gnn(
        model, train_graphs, val_graphs, task_type,
        epochs=epochs, lr=best["lr"], batch_size=batch_size,
        device=device, verbose=True,
        weight_decay=best.get("weight_decay", 1e-4),
    )
    return best, history


def hpo_schnet(
    train_graphs,
    val_graphs,
    task_type: str,
    out_channels: int,
    *,
    n_trials: int = TORCH_HPO_N_TRIALS,
    epochs: int = TORCH_HPO_EPOCHS,
    batch_size: int = 32,
    seed: int = 42,
    device=None,
) -> Tuple[Dict[str, Any], Dict]:
    device = device or get_device()
    train_graphs = prepare_schnet_graphs(train_graphs)
    val_graphs = prepare_schnet_graphs(val_graphs)
    set_seed(seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden_channels": trial.suggest_categorical("hidden_channels", SCHNET_HPO_SPACE["hidden_channels"]),
            "num_filters": trial.suggest_categorical("num_filters", SCHNET_HPO_SPACE["num_filters"]),
            "num_interactions": trial.suggest_categorical("num_interactions", SCHNET_HPO_SPACE["num_interactions"]),
            "num_gaussians": trial.suggest_categorical("num_gaussians", SCHNET_HPO_SPACE["num_gaussians"]),
            "cutoff": trial.suggest_categorical("cutoff", SCHNET_HPO_SPACE["cutoff"]),
            "lr": trial.suggest_categorical("lr", SCHNET_HPO_SPACE["lr"]),
            "weight_decay": trial.suggest_categorical("weight_decay", SCHNET_HPO_SPACE["weight_decay"]),
        }
        logger.info("SchNet — starting trial %d/%d", trial.number + 1, n_trials)
        sys.stdout.flush()
        model = build_schnet(out_channels=out_channels, **{k: v for k, v in params.items() if k not in ("lr", "weight_decay")})
        train_schnet(
            model, train_graphs, val_graphs, task_type,
            epochs=epochs, lr=params["lr"], batch_size=batch_size,
            device=device, verbose=False,
            patience=TORCH_HPO_PATIENCE, weight_decay=params["weight_decay"],
        )
        from mol_repr_utils import eval_schnet, PyGLoader
        loader = PyGLoader(val_graphs, batch_size=batch_size, shuffle=False)
        metrics = eval_schnet(model, loader, device, task_type)
        if task_type == "regression":
            return metrics["RMSE"]
        return -metrics.get("ROC-AUC", metrics.get("Mean ROC-AUC", 0.0))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_optuna_trial_callback("SchNet HPO", n_trials)],
    )
    best = study.best_params or dict(SCHNET_DEFAULT)
    model = build_schnet(
        out_channels=out_channels,
        hidden_channels=best["hidden_channels"],
        num_filters=best["num_filters"],
        num_interactions=best["num_interactions"],
        num_gaussians=best["num_gaussians"],
        cutoff=best["cutoff"],
    )
    history = train_schnet(
        model, train_graphs, val_graphs, task_type,
        epochs=epochs, lr=best["lr"], batch_size=batch_size,
        device=device, verbose=True,
        weight_decay=best.get("weight_decay", 1e-4),
    )
    return best, history


def hpo_lstm(
    train_ds,
    val_ds,
    task_type: str,
    vocab_size: int,
    out_channels: int,
    *,
    n_trials: int = TORCH_HPO_N_TRIALS,
    epochs: int = TORCH_HPO_EPOCHS,
    batch_size: int = 32,
    seed: int = 42,
    device=None,
) -> Tuple[Dict[str, Any], Dict]:
    from mol_repr_utils import SmilesLSTM, eval_lstm, TorchLoader
    device = device or get_device()
    set_seed(seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "embed_dim": trial.suggest_categorical("embed_dim", LSTM_HPO_SPACE["embed_dim"]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", LSTM_HPO_SPACE["hidden_dim"]),
            "n_layers": trial.suggest_categorical("n_layers", LSTM_HPO_SPACE["n_layers"]),
            "dropout": trial.suggest_categorical("dropout", LSTM_HPO_SPACE["dropout"]),
            "lr": trial.suggest_categorical("lr", LSTM_HPO_SPACE["lr"]),
            "weight_decay": trial.suggest_categorical("weight_decay", LSTM_HPO_SPACE["weight_decay"]),
        }
        model = SmilesLSTM(
            vocab_size=vocab_size,
            embed_dim=params["embed_dim"],
            hidden_dim=params["hidden_dim"],
            n_layers=params["n_layers"],
            out_channels=out_channels,
            dropout=params["dropout"],
        )
        logger.info("LSTM — starting trial %d/%d", trial.number + 1, n_trials)
        sys.stdout.flush()
        train_lstm(
            model, train_ds, val_ds, task_type,
            epochs=epochs, lr=params["lr"], batch_size=batch_size,
            device=device, verbose=False,
            patience=TORCH_HPO_PATIENCE, weight_decay=params["weight_decay"],
        )
        loader = TorchLoader(val_ds, batch_size=batch_size, shuffle=False)
        metrics = eval_lstm(model, loader, device, task_type)
        if task_type == "regression":
            return metrics["RMSE"]
        return -metrics.get("ROC-AUC", metrics.get("Mean ROC-AUC", 0.0))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_optuna_trial_callback("LSTM HPO", n_trials)],
    )
    best = study.best_params or dict(LSTM_DEFAULT)
    model = SmilesLSTM(
        vocab_size=vocab_size,
        embed_dim=best["embed_dim"],
        hidden_dim=best["hidden_dim"],
        n_layers=best["n_layers"],
        out_channels=out_channels,
        dropout=best["dropout"],
    )
    history = train_lstm(
        model, train_ds, val_ds, task_type,
        epochs=epochs, lr=best["lr"], batch_size=batch_size,
        device=device, verbose=True,
        weight_decay=best.get("weight_decay", 1e-4),
    )
    return best, history
