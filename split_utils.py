"""
split_utils.py — Dataset splitting for benchmark v4.

Primary: Bemis–Murcko scaffold split (MoleculeNet / DeepChem convention).
Secondary: random split (deferred; implemented for future comparison runs).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

SPLIT_MODES = ("scaffold", "random")


def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Return Bemis–Murcko scaffold SMILES; fall back to input SMILES on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality
        )
    except Exception:
        return smiles


def scaffold_groups(smiles_list: List[str]) -> Dict[str, List[int]]:
    """Map scaffold SMILES → list of molecule indices."""
    groups: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles_list):
        scaffold = murcko_scaffold(smi)
        groups.setdefault(scaffold, []).append(idx)
    return groups


def scaffold_split_indices(
    smiles_list: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign whole scaffolds to train / val / test (DeepChem / Chemprop convention).

    Algorithm (v5 — size-controlled):
        1. Shuffle scaffold groups with ``seed`` so equal-size groups land in
           different splits from run to run (this is the ONLY source of
           seed-to-seed variation, and it is genuine held-out variance).
        2. Stable-sort the shuffled groups largest-first, so the few very large
           scaffolds (e.g. the single acyclic/empty-scaffold group that dominates
           ESOL and FreeSolv) are always placed *first* and always go to TRAIN.
        3. Fill train up to its target, then val up to its target, then the rest
           to test. Because the largest groups can never overflow into val/test,
           the split proportions stay close to 80/10/10 for every seed.

    This replaces the v4 "deficit balancing without caps" scheme, under which the
    giant empty-scaffold group randomly landed in val or test and produced wildly
    different (near-random) split sizes across seeds — e.g. an ESOL test set of
    32 molecules on one seed and a FreeSolv test set of a *single* molecule on
    another. See ``check_split_proportions`` for the guard that now enforces this.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    n = len(smiles_list)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    groups = scaffold_groups(smiles_list)
    scaffold_sets = list(groups.values())

    # 1) seed-dependent shuffle (variety among equal-size scaffolds)
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_sets)
    # 2) stable sort: biggest scaffolds first (ties keep the shuffled order)
    scaffold_sets.sort(key=len, reverse=True)

    train_cutoff = train_frac * n
    val_cutoff = (train_frac + val_frac) * n

    buckets: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for scaffold_set in scaffold_sets:
        # 3) fill train, then val, then test — largest groups pinned to train
        if len(buckets["train"]) + len(scaffold_set) <= train_cutoff:
            buckets["train"].extend(scaffold_set)
        elif len(buckets["train"]) + len(buckets["val"]) + len(scaffold_set) <= val_cutoff:
            buckets["val"].extend(scaffold_set)
        else:
            buckets["test"].extend(scaffold_set)

    return (
        np.array(buckets["train"], dtype=int),
        np.array(buckets["val"], dtype=int),
        np.array(buckets["test"], dtype=int),
    )


def check_split_proportions(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    tol: float = 0.05,
    min_test: int = 10,
    raise_on_fail: bool = True,
) -> None:
    """Guard against malformed splits.

    Raises (or warns) if any split is more than ``tol`` off its target fraction,
    or if the test set has fewer than ``min_test`` molecules. This makes the v4
    failure mode (a scaffold group blowing up val/test) impossible to miss.
    """
    n = len(train_idx) + len(val_idx) + len(test_idx)
    if n == 0:
        return
    actual = {
        "train": len(train_idx) / n,
        "val": len(val_idx) / n,
        "test": len(test_idx) / n,
    }
    target = {"train": train_frac, "val": val_frac, "test": test_frac}
    problems = [
        f"{name}={actual[name]*100:.1f}% (target {target[name]*100:.0f}%)"
        for name in ("train", "val", "test")
        if abs(actual[name] - target[name]) > tol
    ]
    if len(test_idx) < min_test:
        problems.append(f"test has only {len(test_idx)} molecules (min {min_test})")
    if problems:
        msg = "Malformed split — " + "; ".join(problems)
        if raise_on_fail:
            raise ValueError(msg)
        logger.warning(msg)


def random_split_indices(
    n: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random permutation split (v3 protocol; for later comparison runs)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def make_split_indices(
    smiles_list: List[str],
    *,
    mode: str = "scaffold",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to scaffold or random split."""
    mode = mode.lower()
    if mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split mode {mode!r}. Choose from {SPLIT_MODES}")
    if mode == "scaffold":
        return scaffold_split_indices(
            smiles_list,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
        )
    return random_split_indices(len(smiles_list), train_frac=train_frac, val_frac=val_frac, seed=seed)


def split_summary(
    smiles_list: List[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict:
    """Scaffold statistics per split (for logging / persistence)."""
    groups = scaffold_groups(smiles_list)

    def _scaffold_count(indices: np.ndarray) -> int:
        scaffolds = {murcko_scaffold(smiles_list[i]) for i in indices}
        return len(scaffolds)

    return {
        "n_total": len(smiles_list),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_scaffolds_total": len(groups),
        "n_scaffolds_train": _scaffold_count(train_idx),
        "n_scaffolds_val": _scaffold_count(val_idx),
        "n_scaffolds_test": _scaffold_count(test_idx),
    }


def splits_path(
    results_root: Union[str, Path],
    dataset_slug: str,
    mode: str = "scaffold",
) -> Path:
    """Shared splits location: ``results/splits/{slug}_{mode}_splits.pkl``."""
    return Path(results_root) / "splits" / f"{dataset_slug}_{mode}_splits.pkl"


def save_splits(
    path: Union[str, Path],
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    smiles_list: List[str],
    mode: str = "scaffold",
    seed: int = 42,
) -> Dict:
    """Persist split indices and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_mode": mode,
        "seed": seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "summary": split_summary(smiles_list, train_idx, val_idx, test_idx),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    logger.info(
        "Saved %s split → %s (train=%d, val=%d, test=%d, scaffolds=%d)",
        mode,
        path,
        len(train_idx),
        len(val_idx),
        len(test_idx),
        payload["summary"]["n_scaffolds_total"],
    )
    return payload


def load_splits(path: Union[str, Path]) -> Dict:
    """Load split pickle written by ``save_splits``."""
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _split_matches_smiles(data: Dict, smiles_list: List[str]) -> bool:
    """True if persisted indices are in-range for the current molecule list."""
    n = len(smiles_list)
    if n == 0:
        return False
    summary = data.get("summary") or {}
    if summary.get("n_total") is not None and int(summary["n_total"]) != n:
        return False
    for key in ("train_idx", "val_idx", "test_idx"):
        idx = data.get(key)
        if idx is None:
            return False
        arr = np.asarray(idx)
        if arr.size == 0:
            continue
        if int(arr.min()) < 0 or int(arr.max()) >= n:
            return False
    return True


def get_or_create_splits(
    smiles_list: List[str],
    results_root: Union[str, Path],
    dataset_slug: str,
    *,
    mode: str = "scaffold",
    seed: int = 42,
    force: bool = False,
) -> Dict:
    """Load existing splits or create and save new ones."""
    path = splits_path(results_root, dataset_slug, mode=mode)
    if path.exists() and not force:
        data = load_splits(path)
        stored_seed = data.get("seed")
        if stored_seed is not None and int(stored_seed) != int(seed):
            raise ValueError(
                f"Split file {path} was created with seed={stored_seed}, "
                f"but seed={seed} was requested. Use force=True or a seed-specific "
                f"results root (results/multiseed/seed_{seed}/)."
            )
        if not _split_matches_smiles(data, smiles_list):
            summary = data.get("summary") or {}
            logger.warning(
                "Split file %s does not match current dataset size "
                "(stored n_total=%s, current=%d); regenerating.",
                path,
                summary.get("n_total"),
                len(smiles_list),
            )
        else:
            logger.info("Loaded splits from %s", path)
            return data

    train_idx, val_idx, test_idx = make_split_indices(
        smiles_list, mode=mode, seed=seed
    )
    # Guard: fail loudly if the split is badly proportioned (v4 failure mode).
    check_split_proportions(train_idx, val_idx, test_idx, raise_on_fail=True)
    return save_splits(
        path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        smiles_list=smiles_list,
        mode=mode,
        seed=seed,
    )


def subset_array(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Index a 1D or 2D numpy array."""
    return arr[indices]


def subset_list(items: List, indices: np.ndarray) -> List:
    """Index a Python list."""
    return [items[i] for i in indices]


def subset_graphs_by_valid_idx(
    graphs: List,
    valid_idx: List[int],
    split_indices: np.ndarray,
) -> List:
    """
    Map scaffold-split indices onto a compact conformer graph list.

    ``load_conformer_graphs`` returns only successful 3D graphs plus ``valid_idx``
    (positions in the full smiles list). Split indices refer to the full list,
    not positions in ``graphs``.
    """
    lookup = {i: g for i, g in zip(valid_idx, graphs)}
    missing = [int(i) for i in split_indices if int(i) not in lookup]
    if missing:
        logger.warning(
            "%d molecules in split have no 3D conformer (e.g. idx %s); "
            "excluded from 3D train/val/test subsets",
            len(missing),
            missing[:5],
        )
    return [lookup[int(i)] for i in split_indices if int(i) in lookup]
