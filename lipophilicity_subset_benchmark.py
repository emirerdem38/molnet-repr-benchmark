#!/usr/bin/env python3
"""Lipophilicity train-subset benchmark (no HPO) for Discussion / Outlook.

Uses fixed scaffold/random splits and previously saved best hyperparameters.
Subsamples the train set only; val/test stay fixed. Single seed by default.

Purpose: data-efficiency illustration — can a smaller labeled train subset
recover near-full-data test metrics under the same HPO params?

Example:
  python lipophilicity_subset_benchmark.py --mode scaffold --seed 0 --models tabular
  python lipophilicity_subset_benchmark.py --mode random --seed 0 --models deep \\
      --fractions 0.125,0.25,0.5,1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

DATASET = "lipophilicity"
DEFAULT_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
FINAL_EPOCHS = 100
BATCH_SIZE = 64

TABULAR_SPECS = [
    ("Morgan + RF", "rf", "morgan"),
    ("Morgan + SVM", "svm", "morgan"),
    ("Morgan + XGBoost", "xgb", "morgan"),
    ("Descriptors + RF", "rf", "desc"),
    ("Descriptors + SVM", "svm", "desc"),
    ("Descriptors + XGBoost", "xgb", "desc"),
    ("Combined + XGBoost", "xgb", "combined"),
]
DEEP_NAMES = {
    "GIN (2D)",
    "D-MPNN (2D)",
    "GIN (3D)",
    "SchNet (3D coords)",
    "SMILES LSTM",
}


def _human(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_fractions(text: str) -> List[float]:
    out: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        f = float(part)
        if not (0.0 < f <= 1.0):
            raise argparse.ArgumentTypeError(f"fraction must be in (0,1], got {f}")
        out.append(f)
    if not out:
        raise argparse.ArgumentTypeError("at least one fraction required")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="scaffold", choices=["scaffold", "random"])
    p.add_argument("--seed", type=int, default=0, help="Split + HPO lookup seed")
    p.add_argument(
        "--subset-seed",
        type=int,
        default=0,
        help="RNG seed for train subsample (independent of split seed)",
    )
    p.add_argument(
        "--fractions",
        type=_parse_fractions,
        default=list(DEFAULT_FRACTIONS),
        help="Comma-separated train fractions, e.g. 0.125,0.25,0.5,1.0",
    )
    p.add_argument(
        "--models",
        default="all",
        help="all | tabular | deep | comma-separated model names",
    )
    p.add_argument("--epochs", type=int, default=FINAL_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "lipophilicity_subset",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for deep models (SchNet uses CPU if radius_graph fails on MPS)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run rows even if already present with note=ok",
    )
    return p.parse_args()


def _wanted_models(spec: str) -> Set[str]:
    tabular = {m for m, _, _ in TABULAR_SPECS}
    if spec == "all":
        return tabular | DEEP_NAMES
    if spec == "tabular":
        return tabular
    if spec == "deep":
        return set(DEEP_NAMES)
    return {m.strip() for m in spec.split(",") if m.strip()}


def _stem(mode: str, seed: int) -> str:
    return f"lipophilicity_seed{seed}_{mode}_subset"


def _load_existing(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _row_key(fraction: float, model: str) -> Tuple[str, str]:
    return (f"{fraction:.6g}", model)


def _done_keys(rows: Sequence[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    for r in rows:
        if r.get("note") == "ok":
            done.add(_row_key(float(r["fraction"]), r["model"]))
    return done


def _metrics_flat(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Flatten evaluate_* / sklearn metric dicts to scalar columns."""
    out: Dict[str, Optional[float]] = {
        "RMSE": None,
        "MAE": None,
        "R2": None,
        "ROC_AUC": None,
        "PRC_AUC": None,
    }
    for key, val in metrics.items():
        if isinstance(val, (tuple, list)) and val:
            num = float(val[0])
        else:
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
        k = key.replace("-", "_").upper()
        if k in out:
            out[k] = num
        elif key == "ROC-AUC":
            out["ROC_AUC"] = num
        elif key == "PRC-AUC":
            out["PRC_AUC"] = num
    return out


def _subsample_train(train_idx, fraction: float, subset_seed: int):
    import numpy as np

    train_idx = np.asarray(train_idx)
    n = len(train_idx)
    k = max(1, int(round(fraction * n)))
    k = min(k, n)
    if k == n:
        return train_idx.copy(), k
    rng = np.random.default_rng(subset_seed)
    chosen = rng.choice(train_idx, size=k, replace=False)
    return np.sort(chosen), k


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from conformer_generation import load_conformer_graphs
    from hpo_utils import hpo_params_path, load_hpo_params
    from mol_repr_utils import (
        SMILESDataset,
        SmilesLSTM,
        PyGLoader,
        TorchLoader,
        build_gnn,
        build_rf_pipeline,
        build_schnet,
        build_svm_pipeline,
        build_xgb_pipeline,
        ensure_atomic_numbers_list,
        eval_gnn,
        eval_lstm,
        eval_schnet,
        evaluate_sklearn_test,
        filter_bonded_molecules,
        get_device,
        load_dataset,
        prepare_schnet_graphs,
        preprocess_dataset,
        set_seed,
        smiles_to_combined,
        smiles_to_descriptors,
        smiles_to_dmpnn_graphs,
        smiles_to_graphs,
        smiles_to_morgan,
        tokenize_smiles,
        train_gnn,
        train_lstm,
        train_schnet,
    )
    from split_utils import (
        check_split_proportions,
        get_or_create_splits,
        subset_array,
        subset_graphs_by_valid_idx,
        subset_list,
    )

    import numpy as np
    import torch

    mode = args.mode
    seed = args.seed
    subset_seed = args.subset_seed
    fractions = list(args.fractions)
    wanted = _wanted_models(args.models)
    data_dir = ROOT / "data"
    results_root = ROOT / "results" / f"seed_{seed}" / mode
    hpo_cpu = results_root / "cpu" / "hpo"
    hpo_gpu = results_root / "gpu" / "hpo"
    (results_root / "splits").mkdir(parents=True, exist_ok=True)

    task_type = "regression"
    out_channels = 1

    if args.device == "auto":
        device = get_device()
        if str(device) == "cpu" and torch.backends.mps.is_available():
            device = torch.device("mps")
    else:
        device = torch.device(args.device)

    host_info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
    }

    stem = _stem(mode, seed)
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    rows = _load_existing(csv_path)
    done = set() if args.force else _done_keys(rows)

    print("=" * 72)
    print("Lipophilicity subset benchmark (fixed HPO params, no retuning)")
    print(f"mode={mode}  seed={seed}  subset_seed={subset_seed}")
    print(f"fractions={fractions}")
    print(f"models={sorted(wanted)}")
    print(f"host={host_info['hostname']}  device={host_info['device']}")
    print(f"out={csv_path}")
    print("=" * 72)

    t_prep0 = time.perf_counter()
    df = load_dataset(DATASET, cache_dir=str(data_dir))
    smiles, y = preprocess_dataset(df, DATASET)
    smiles, y = filter_bonded_molecules(smiles, y)

    need_tabular = any(m in wanted for m, _, _ in TABULAR_SPECS)
    need_2d = "GIN (2D)" in wanted
    need_dmpnn = "D-MPNN (2D)" in wanted
    need_3d = ("GIN (3D)" in wanted) or ("SchNet (3D coords)" in wanted)
    need_lstm = "SMILES LSTM" in wanted

    X_morgan = X_desc = X_combined = None
    if need_tabular:
        X_morgan = smiles_to_morgan(smiles)
        X_desc = smiles_to_descriptors(smiles)
        X_combined = smiles_to_combined(smiles)

    graphs2d_all = smiles_to_graphs(smiles, y) if need_2d else None
    graphs_dmpnn_all = smiles_to_dmpnn_graphs(smiles, y) if need_dmpnn else None
    X_tok_all, vocab = (tokenize_smiles(smiles) if need_lstm else (None, None))

    graphs3d_all = valid_idx_3d = None
    if need_3d:
        graphs3d_all, valid_idx_3d, _ = load_conformer_graphs(
            DATASET, data_dir=str(data_dir), n_confs=25
        )
        graphs3d_all = ensure_atomic_numbers_list(graphs3d_all)

    split_data = get_or_create_splits(
        smiles, results_root, DATASET, mode=mode, seed=seed, force=False,
    )
    train_idx_full = split_data["train_idx"]
    val_idx = split_data["val_idx"]
    test_idx = split_data["test_idx"]
    check_split_proportions(train_idx_full, val_idx, test_idx, raise_on_fail=True)

    feat_map = {"morgan": X_morgan, "desc": X_desc, "combined": X_combined}
    builders = {
        "rf": build_rf_pipeline,
        "svm": build_svm_pipeline,
        "xgb": build_xgb_pipeline,
    }

    prep_s = time.perf_counter() - t_prep0
    print(
        f"Prep done in {_human(prep_s)}  "
        f"(n={len(smiles)}, full_train={len(train_idx_full)}, "
        f"val={len(val_idx)}, test={len(test_idx)})"
    )

    fieldnames = [
        "dataset", "mode", "seed", "subset_seed", "fraction", "n_train",
        "n_val", "n_test", "model", "device", "RMSE", "MAE", "R2",
        "ROC_AUC", "PRC_AUC", "seconds", "human", "note", "timestamp_utc",
        "hpo_path",
    ]

    def flush() -> None:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        payload = {
            "protocol": {
                "name": "lipophilicity_subset_benchmark",
                "dataset": DATASET,
                "mode": mode,
                "seed": seed,
                "subset_seed": subset_seed,
                "fractions": fractions,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "purpose": (
                    "Train-subset data-efficiency study with frozen HPO params "
                    "from the full-train benchmark. Val/test fixed."
                ),
            },
            "host": host_info,
            "prep_seconds": round(prep_s, 3),
            "rows": rows,
        }
        json_path.write_text(json.dumps(payload, indent=2))

    def record(
        *,
        fraction: float,
        n_train: int,
        model: str,
        device_kind: str,
        seconds: float,
        metrics: Optional[Dict[str, Any]] = None,
        note: str = "ok",
        hpo_path: str = "",
    ) -> None:
        flat = _metrics_flat(metrics or {})
        row = {
            "dataset": DATASET,
            "mode": mode,
            "seed": seed,
            "subset_seed": subset_seed,
            "fraction": fraction,
            "n_train": n_train,
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "model": model,
            "device": device_kind,
            "RMSE": flat["RMSE"],
            "MAE": flat["MAE"],
            "R2": flat["R2"],
            "ROC_AUC": flat["ROC_AUC"],
            "PRC_AUC": flat["PRC_AUC"],
            "seconds": round(seconds, 3),
            "human": _human(seconds),
            "note": note,
            "timestamp_utc": _now(),
            "hpo_path": hpo_path,
        }
        # Replace prior row for same (fraction, model) if re-running
        rows[:] = [
            r for r in rows
            if not (
                abs(float(r.get("fraction", -1)) - fraction) < 1e-12
                and r.get("model") == model
            )
        ]
        rows.append(row)
        score = flat["RMSE"] if flat["RMSE"] is not None else flat["ROC_AUC"]
        print(
            f"  ✓ {model:22s}  frac={fraction:<5}  n={n_train:<5}  "
            f"score={score}  {_human(seconds):>10s}  {note}"
        )
        flush()

    for fraction in fractions:
        sub_train, n_train = _subsample_train(train_idx_full, fraction, subset_seed)
        print("-" * 72)
        print(f"Fraction {fraction} → n_train={n_train} / {len(train_idx_full)}")

        set_seed(seed)

        # ---- tabular ----
        for model_key, kind, feat_name in TABULAR_SPECS:
            if model_key not in wanted:
                continue
            if _row_key(fraction, model_key) in done:
                print(f"  · skip {model_key} (already ok)")
                continue
            X = feat_map[feat_name]
            hpo_file = hpo_params_path(hpo_cpu, model_key, DATASET)
            best = load_hpo_params(hpo_file)
            if best is None:
                record(
                    fraction=fraction, n_train=n_train, model=model_key,
                    device_kind="cpu", seconds=0.0,
                    note=f"ERROR: missing HPO params at {hpo_file}",
                    hpo_path=str(hpo_file),
                )
                continue
            print(f"→ {model_key} …")
            t0 = time.perf_counter()
            try:
                pipe = builders[kind](task_type, random_state=seed, **best)
                metrics = evaluate_sklearn_test(
                    pipe,
                    subset_array(X, sub_train),
                    subset_array(y, sub_train),
                    subset_array(X, test_idx),
                    subset_array(y, test_idx),
                    task_type,
                )
                record(
                    fraction=fraction, n_train=n_train, model=model_key,
                    device_kind="cpu", seconds=time.perf_counter() - t0,
                    metrics=metrics, hpo_path=str(hpo_file),
                )
            except Exception as e:
                record(
                    fraction=fraction, n_train=n_train, model=model_key,
                    device_kind="cpu", seconds=time.perf_counter() - t0,
                    note=f"ERROR: {e}", hpo_path=str(hpo_file),
                )

        def run_gnn(model_key: str, arch: str, dim: str, tr, va, te) -> None:
            if model_key not in wanted:
                return
            if _row_key(fraction, model_key) in done:
                print(f"  · skip {model_key} (already ok)")
                return
            hpo_file = hpo_params_path(hpo_gpu, model_key, DATASET)
            best = load_hpo_params(hpo_file)
            device_kind = "gpu" if host_info["cuda_available"] else str(device)
            if best is None:
                record(
                    fraction=fraction, n_train=n_train, model=model_key,
                    device_kind=device_kind, seconds=0.0,
                    note=f"ERROR: missing HPO params at {hpo_file}",
                    hpo_path=str(hpo_file),
                )
                return
            print(f"→ {model_key} …")
            t0 = time.perf_counter()
            try:
                model = build_gnn(
                    arch, dim, out_channels,
                    hidden=best["hidden"],
                    num_layers=best["num_layers"],
                    dropout=best["dropout"],
                )
                train_gnn(
                    model, tr, va, task_type,
                    epochs=args.epochs,
                    lr=best["lr"],
                    batch_size=args.batch_size,
                    device=device,
                    verbose=False,
                    patience=None,
                    weight_decay=best.get("weight_decay", 1e-4),
                )
                metrics = eval_gnn(
                    model,
                    PyGLoader(te, batch_size=args.batch_size, shuffle=False),
                    device,
                    task_type,
                )
                record(
                    fraction=fraction, n_train=n_train, model=model_key,
                    device_kind=device_kind, seconds=time.perf_counter() - t0,
                    metrics=metrics, hpo_path=str(hpo_file),
                )
            except Exception as e:
                record(
                    fraction=fraction, n_train=n_train, model=model_key,
                    device_kind=device_kind, seconds=time.perf_counter() - t0,
                    note=f"ERROR: {e}", hpo_path=str(hpo_file),
                )

        if need_2d:
            run_gnn(
                "GIN (2D)", "GIN", "2D",
                subset_list(graphs2d_all, sub_train),
                subset_list(graphs2d_all, val_idx),
                subset_list(graphs2d_all, test_idx),
            )
        if need_dmpnn:
            run_gnn(
                "D-MPNN (2D)", "DMPNN", "2D",
                subset_list(graphs_dmpnn_all, sub_train),
                subset_list(graphs_dmpnn_all, val_idx),
                subset_list(graphs_dmpnn_all, test_idx),
            )
        if "GIN (3D)" in wanted:
            run_gnn(
                "GIN (3D)", "GIN", "3D",
                subset_graphs_by_valid_idx(graphs3d_all, valid_idx_3d, sub_train),
                subset_graphs_by_valid_idx(graphs3d_all, valid_idx_3d, val_idx),
                subset_graphs_by_valid_idx(graphs3d_all, valid_idx_3d, test_idx),
            )

        if "SchNet (3D coords)" in wanted:
            if _row_key(fraction, "SchNet (3D coords)") in done:
                print("  · skip SchNet (3D coords) (already ok)")
            else:
                hpo_file = hpo_params_path(hpo_gpu, "SchNet (3D coords)", DATASET)
                best = load_hpo_params(hpo_file)
                # Prefer CUDA; fall back to CPU (MPS often fails on radius_graph).
                schnet_device = (
                    torch.device("cuda") if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                device_kind = "gpu" if schnet_device.type == "cuda" else "cpu"
                if best is None:
                    record(
                        fraction=fraction, n_train=n_train,
                        model="SchNet (3D coords)", device_kind=device_kind,
                        seconds=0.0,
                        note=f"ERROR: missing HPO params at {hpo_file}",
                        hpo_path=str(hpo_file),
                    )
                else:
                    print("→ SchNet (3D coords) …")
                    t0 = time.perf_counter()
                    try:
                        tr3d = subset_graphs_by_valid_idx(
                            graphs3d_all, valid_idx_3d, sub_train
                        )
                        va3d = subset_graphs_by_valid_idx(
                            graphs3d_all, valid_idx_3d, val_idx
                        )
                        te3d = subset_graphs_by_valid_idx(
                            graphs3d_all, valid_idx_3d, test_idx
                        )
                        model = build_schnet(
                            out_channels=out_channels,
                            hidden_channels=best["hidden_channels"],
                            num_filters=best["num_filters"],
                            num_interactions=best["num_interactions"],
                            num_gaussians=best["num_gaussians"],
                            cutoff=best["cutoff"],
                        )
                        tr_s = prepare_schnet_graphs(tr3d)
                        va_s = prepare_schnet_graphs(va3d)
                        te_s = prepare_schnet_graphs(te3d)
                        train_schnet(
                            model, tr_s, va_s, task_type,
                            epochs=args.epochs,
                            lr=best["lr"],
                            batch_size=args.batch_size,
                            device=schnet_device,
                            verbose=False,
                            patience=None,
                            weight_decay=best.get("weight_decay", 1e-4),
                        )
                        metrics = eval_schnet(
                            model,
                            PyGLoader(te_s, batch_size=args.batch_size, shuffle=False),
                            schnet_device,
                            task_type,
                        )
                        record(
                            fraction=fraction, n_train=n_train,
                            model="SchNet (3D coords)", device_kind=device_kind,
                            seconds=time.perf_counter() - t0,
                            metrics=metrics, hpo_path=str(hpo_file),
                        )
                    except Exception as e:
                        record(
                            fraction=fraction, n_train=n_train,
                            model="SchNet (3D coords)", device_kind=device_kind,
                            seconds=time.perf_counter() - t0,
                            note=f"ERROR: {e}", hpo_path=str(hpo_file),
                        )

        if "SMILES LSTM" in wanted:
            if _row_key(fraction, "SMILES LSTM") in done:
                print("  · skip SMILES LSTM (already ok)")
            else:
                hpo_file = hpo_params_path(hpo_gpu, "SMILES LSTM", DATASET)
                best = load_hpo_params(hpo_file)
                device_kind = "gpu" if host_info["cuda_available"] else str(device)
                if best is None:
                    record(
                        fraction=fraction, n_train=n_train, model="SMILES LSTM",
                        device_kind=device_kind, seconds=0.0,
                        note=f"ERROR: missing HPO params at {hpo_file}",
                        hpo_path=str(hpo_file),
                    )
                else:
                    print("→ SMILES LSTM …")
                    t0 = time.perf_counter()
                    try:
                        tr_lstm = SMILESDataset(
                            subset_array(X_tok_all, sub_train),
                            subset_array(y, sub_train),
                        )
                        va_lstm = SMILESDataset(
                            subset_array(X_tok_all, val_idx),
                            subset_array(y, val_idx),
                        )
                        te_lstm = SMILESDataset(
                            subset_array(X_tok_all, test_idx),
                            subset_array(y, test_idx),
                        )
                        model = SmilesLSTM(
                            vocab_size=len(vocab),
                            embed_dim=best["embed_dim"],
                            hidden_dim=best["hidden_dim"],
                            n_layers=best["n_layers"],
                            out_channels=out_channels,
                            dropout=best["dropout"],
                        )
                        train_lstm(
                            model, tr_lstm, va_lstm, task_type,
                            epochs=args.epochs,
                            lr=best["lr"],
                            batch_size=args.batch_size,
                            device=device,
                            verbose=False,
                            patience=None,
                            weight_decay=best.get("weight_decay", 1e-4),
                        )
                        metrics = eval_lstm(
                            model,
                            TorchLoader(
                                te_lstm, batch_size=args.batch_size, shuffle=False
                            ),
                            device,
                            task_type,
                        )
                        record(
                            fraction=fraction, n_train=n_train, model="SMILES LSTM",
                            device_kind=device_kind,
                            seconds=time.perf_counter() - t0,
                            metrics=metrics, hpo_path=str(hpo_file),
                        )
                    except Exception as e:
                        record(
                            fraction=fraction, n_train=n_train, model="SMILES LSTM",
                            device_kind=device_kind,
                            seconds=time.perf_counter() - t0,
                            note=f"ERROR: {e}", hpo_path=str(hpo_file),
                        )

    flush()
    ok = [r for r in rows if r.get("note") == "ok"]
    print("=" * 72)
    print(f"Done. Wrote {csv_path}  ({len(ok)} ok rows / {len(rows)} total)")


if __name__ == "__main__":
    main()
