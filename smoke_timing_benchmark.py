#!/usr/bin/env python3
"""Same-machine smoke timing benchmark for thesis cost comparison.

Runs all 12 models on FreeSolv (smallest dataset by default) with a reduced
HPO budget on one host, and writes wall-clock times + a relative cost index.

This does NOT replace the full multi-seed accuracy tables. It only produces
comparable relative training costs under a fixed short protocol.

Example:
  python smoke_timing_benchmark.py --dataset freesolv --seed 0 --mode scaffold
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
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Smoke HPO budget (short, but still search + final fit)
SMOKE_SKLEARN_N_ITER = 3
SMOKE_TORCH_N_TRIALS = 3
SMOKE_TORCH_HPO_EPOCHS = 20
SMOKE_TORCH_FINAL_EPOCHS = 40
SMOKE_BATCH_SIZE = 32


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="freesolv", choices=[
        "esol", "freesolv", "lipophilicity", "bace", "bbbp", "tox21", "hiv",
    ])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="scaffold", choices=["scaffold", "random"])
    p.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys, or 'all' / 'tabular' / 'deep'",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "smoke_timing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from conformer_generation import load_conformer_graphs
    from hpo_utils import hpo_gnn, hpo_lstm, hpo_schnet, hpo_tabular
    from mol_repr_utils import (
        SMILESDataset,
        SmilesLSTM,
        PyGLoader,
        TorchLoader,
        build_gnn,
        build_schnet,
        ensure_atomic_numbers_list,
        eval_gnn,
        eval_lstm,
        eval_schnet,
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

    dataset = args.dataset
    seed = args.seed
    mode = args.mode
    data_dir = ROOT / "data"
    results_root = ROOT / "results" / f"seed_{seed}" / mode
    (results_root / "splits").mkdir(parents=True, exist_ok=True)

    task_type = "classification" if dataset in {"bace", "bbbp", "tox21", "hiv"} else "regression"
    out_channels = 12 if dataset == "tox21" else 1
    tox21 = dataset == "tox21"

    device = get_device()
    # Prefer Apple GPU for this local smoke timing run when CUDA is absent.
    try:
        import torch

        if str(device) == "cpu" and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS for deep models")
    except Exception:
        pass
    host_info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "device": str(device),
        "cuda_available": False,
        "cuda_name": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    try:
        import torch

        host_info["cuda_available"] = bool(torch.cuda.is_available())
        host_info["torch"] = torch.__version__
        if torch.cuda.is_available():
            host_info["cuda_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    print("=" * 72)
    print("Smoke timing benchmark")
    print(f"dataset={dataset}  mode={mode}  seed={seed}")
    print(f"host={host_info['hostname']}  device={host_info['device']}  gpu={host_info.get('cuda_name')}")
    print(
        f"budget: sklearn n_iter={SMOKE_SKLEARN_N_ITER}; "
        f"torch trials={SMOKE_TORCH_N_TRIALS}, "
        f"hpo_epochs={SMOKE_TORCH_HPO_EPOCHS}, final_epochs={SMOKE_TORCH_FINAL_EPOCHS}"
    )
    print("=" * 72)

    t_prep0 = time.perf_counter()
    df = load_dataset(dataset, cache_dir=str(data_dir))
    smiles, y = preprocess_dataset(df, dataset)
    smiles, y = filter_bonded_molecules(smiles, y)

    X_morgan = smiles_to_morgan(smiles)
    X_desc = smiles_to_descriptors(smiles)
    X_combined = smiles_to_combined(smiles)

    graphs2d_all = smiles_to_graphs(smiles, y)
    graphs_dmpnn_all = smiles_to_dmpnn_graphs(smiles, y)
    X_tok_all, vocab = tokenize_smiles(smiles)

    graphs3d_all, valid_idx_3d, _ = load_conformer_graphs(
        dataset, data_dir=str(data_dir), n_confs=25
    )
    graphs3d_all = ensure_atomic_numbers_list(graphs3d_all)

    split_data = get_or_create_splits(
        smiles, results_root, dataset, mode=mode, seed=seed, force=False,
    )
    train_idx = split_data["train_idx"]
    val_idx = split_data["val_idx"]
    test_idx = split_data["test_idx"]
    check_split_proportions(train_idx, val_idx, test_idx, raise_on_fail=True)

    tr2d = subset_list(graphs2d_all, train_idx)
    va2d = subset_list(graphs2d_all, val_idx)
    te2d = subset_list(graphs2d_all, test_idx)
    tr_dmpnn = subset_list(graphs_dmpnn_all, train_idx)
    va_dmpnn = subset_list(graphs_dmpnn_all, val_idx)
    te_dmpnn = subset_list(graphs_dmpnn_all, test_idx)
    tr3d = subset_graphs_by_valid_idx(graphs3d_all, valid_idx_3d, train_idx)
    va3d = subset_graphs_by_valid_idx(graphs3d_all, valid_idx_3d, val_idx)
    te3d = subset_graphs_by_valid_idx(graphs3d_all, valid_idx_3d, test_idx)

    tr_lstm = SMILESDataset(subset_array(X_tok_all, train_idx), subset_array(y, train_idx))
    va_lstm = SMILESDataset(subset_array(X_tok_all, val_idx), subset_array(y, val_idx))
    te_lstm = SMILESDataset(subset_array(X_tok_all, test_idx), subset_array(y, test_idx))

    prep_s = time.perf_counter() - t_prep0
    print(f"Prep done in {_human(prep_s)}  (n={len(smiles)}, train={len(train_idx)}, 3D train={len(tr3d)})")

    tabular_specs = [
        ("Morgan + RF", "rf", X_morgan),
        ("Morgan + SVM", "svm", X_morgan),
        ("Morgan + XGBoost", "xgb", X_morgan),
        ("Descriptors + RF", "rf", X_desc),
        ("Descriptors + SVM", "svm", X_desc),
        ("Descriptors + XGBoost", "xgb", X_desc),
        ("Combined + XGBoost", "xgb", X_combined),
    ]
    deep_names = {
        "GIN (2D)", "D-MPNN (2D)", "GIN (3D)", "SchNet (3D coords)", "SMILES LSTM",
    }

    if args.models == "all":
        wanted = {m for m, _, _ in tabular_specs} | deep_names
    elif args.models == "tabular":
        wanted = {m for m, _, _ in tabular_specs}
    elif args.models == "deep":
        wanted = set(deep_names)
    else:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}

    rows: List[Dict[str, Any]] = []
    set_seed(seed)
    use_gpu_label = "gpu" if host_info["cuda_available"] else "cpu"

    def flush() -> Path:
        return _flush(rows, out_dir, dataset, seed, mode, host_info, prep_s)

    def record(model: str, device_kind: str, seconds: float, note: str = "ok") -> None:
        row = {
            "dataset": dataset,
            "mode": mode,
            "seed": seed,
            "model": model,
            "device": device_kind,
            "seconds": round(seconds, 3),
            "hours": round(seconds / 3600.0, 6),
            "human": _human(seconds),
            "relative_to_fastest": None,
            "note": note,
            "timestamp_utc": _now(),
        }
        rows.append(row)
        print(f"  ✓ {model:22s}  {_human(seconds):>12s}  ({device_kind}) {note}")
        flush()

    for model_key, kind, X in tabular_specs:
        if model_key not in wanted:
            continue
        X_tr = subset_array(X, train_idx)
        X_te = subset_array(X, test_idx)
        y_tr = subset_array(y, train_idx)
        y_te = subset_array(y, test_idx)
        print(f"→ {model_key} …")
        t0 = time.perf_counter()
        try:
            hpo_tabular(
                X_tr, y_tr, task_type, model_kind=kind,
                tox21=tox21, n_iter=SMOKE_SKLEARN_N_ITER, seed=seed,
                X_test=X_te, y_test=y_te,
            )
            record(model_key, "cpu", time.perf_counter() - t0)
        except Exception as e:
            record(model_key, "cpu", time.perf_counter() - t0, note=f"ERROR: {e}")

    def run_gnn(model_key: str, arch: str, dim: str, tr, va, te) -> None:
        if model_key not in wanted:
            return
        print(f"→ {model_key} …")
        t0 = time.perf_counter()
        try:
            best, _ = hpo_gnn(
                tr, va, task_type, arch, dim, out_channels,
                n_trials=SMOKE_TORCH_N_TRIALS,
                epochs=SMOKE_TORCH_HPO_EPOCHS,
                batch_size=SMOKE_BATCH_SIZE,
                seed=seed,
                device=device,
            )
            model = build_gnn(
                arch, dim, out_channels,
                hidden=best["hidden"],
                num_layers=best["num_layers"],
                dropout=best["dropout"],
            )
            train_gnn(
                model, tr, va, task_type,
                epochs=SMOKE_TORCH_FINAL_EPOCHS,
                lr=best["lr"],
                batch_size=SMOKE_BATCH_SIZE,
                device=device,
                verbose=False,
                weight_decay=best.get("weight_decay", 1e-4),
            )
            eval_gnn(
                model,
                PyGLoader(te, batch_size=SMOKE_BATCH_SIZE, shuffle=False),
                device,
                task_type,
            )
            record(model_key, use_gpu_label, time.perf_counter() - t0)
        except Exception as e:
            record(model_key, use_gpu_label, time.perf_counter() - t0, note=f"ERROR: {e}")

    run_gnn("GIN (2D)", "GIN", "2D", tr2d, va2d, te2d)
    run_gnn("D-MPNN (2D)", "DMPNN", "2D", tr_dmpnn, va_dmpnn, te_dmpnn)
    run_gnn("GIN (3D)", "GIN", "3D", tr3d, va3d, te3d)

    if "SchNet (3D coords)" in wanted:
        print("→ SchNet (3D coords) …")
        import torch
        # PyG radius_graph used by SchNet must run on CPU on this Mac build.
        schnet_device = torch.device("cpu")
        t0 = time.perf_counter()
        try:
            best, _ = hpo_schnet(
                tr3d, va3d, task_type, out_channels,
                n_trials=SMOKE_TORCH_N_TRIALS,
                epochs=SMOKE_TORCH_HPO_EPOCHS,
                batch_size=SMOKE_BATCH_SIZE,
                seed=seed,
                device=schnet_device,
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
                epochs=SMOKE_TORCH_FINAL_EPOCHS,
                lr=best["lr"],
                batch_size=SMOKE_BATCH_SIZE,
                device=schnet_device,
                verbose=False,
                weight_decay=best.get("weight_decay", 1e-4),
            )
            eval_schnet(
                model,
                PyGLoader(te_s, batch_size=SMOKE_BATCH_SIZE, shuffle=False),
                schnet_device,
                task_type,
            )
            record("SchNet (3D coords)", "cpu", time.perf_counter() - t0)
        except Exception as e:
            record("SchNet (3D coords)", "cpu", time.perf_counter() - t0, note=f"ERROR: {e}")


    if "SMILES LSTM" in wanted:
        print("→ SMILES LSTM …")
        t0 = time.perf_counter()
        try:
            best, _ = hpo_lstm(
                tr_lstm, va_lstm, task_type, len(vocab), out_channels,
                n_trials=SMOKE_TORCH_N_TRIALS,
                epochs=SMOKE_TORCH_HPO_EPOCHS,
                batch_size=SMOKE_BATCH_SIZE,
                seed=seed,
                device=device,
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
                epochs=SMOKE_TORCH_FINAL_EPOCHS,
                lr=best["lr"],
                batch_size=SMOKE_BATCH_SIZE,
                device=device,
                verbose=False,
                weight_decay=best.get("weight_decay", 1e-4),
            )
            eval_lstm(
                model,
                TorchLoader(te_lstm, batch_size=SMOKE_BATCH_SIZE, shuffle=False),
                device,
                task_type,
            )
            record("SMILES LSTM", use_gpu_label, time.perf_counter() - t0)
        except Exception as e:
            record("SMILES LSTM", use_gpu_label, time.perf_counter() - t0, note=f"ERROR: {e}")

    ok = [r for r in rows if r["note"] == "ok" and r["seconds"] > 0]
    if ok:
        base = min(r["seconds"] for r in ok)
        for r in rows:
            if r["note"] == "ok":
                r["relative_to_fastest"] = round(r["seconds"] / base, 3)

    meta_path = flush()
    print("=" * 72)
    print(f"Done. Wrote {meta_path}")
    if ok:
        print("Relative costs (fastest successful model = 1.0):")
        for r in sorted(ok, key=lambda x: x["seconds"]):
            print(f"  {r['relative_to_fastest']:8.2f}×  {r['model']}  ({r['human']})")


def _flush(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    dataset: str,
    seed: int,
    mode: str,
    host_info: Dict[str, Any],
    prep_s: float,
) -> Path:
    stem = f"{dataset}_seed{seed}_{mode}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    fieldnames = [
        "dataset", "mode", "seed", "model", "device", "seconds", "hours",
        "human", "relative_to_fastest", "note", "timestamp_utc",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    payload = {
        "protocol": {
            "name": "smoke_timing_benchmark",
            "sklearn_n_iter": SMOKE_SKLEARN_N_ITER,
            "torch_n_trials": SMOKE_TORCH_N_TRIALS,
            "torch_hpo_epochs": SMOKE_TORCH_HPO_EPOCHS,
            "torch_final_epochs": SMOKE_TORCH_FINAL_EPOCHS,
            "batch_size": SMOKE_BATCH_SIZE,
            "purpose": (
                "Same-machine relative wall-clock costs with a reduced HPO budget. "
                "Not used for accuracy tables in the thesis."
            ),
        },
        "host": host_info,
        "prep_seconds": round(prep_s, 3),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    return json_path


if __name__ == "__main__":
    main()
