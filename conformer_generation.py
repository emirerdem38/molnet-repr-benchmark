"""
conformer_generation.py
=======================
Multi-conformer generation pipeline for the thesis:
  "Comparative Study of Molecular Representations for
   Machine Learning-Based Prediction of Pharmaceutical Properties"

Workflow
--------
1. Generate N conformers per molecule using ETKDG
2. MMFF-optimise each conformer and record its energy
3. Select the lowest-energy conformer as the representative structure
4. Build a 3D PyG graph from that conformer
5. Save/load conformer libraries to/from disk (so you never regenerate)

Why multiple conformers?
------------------------
Flexible molecules have many possible 3D shapes. ETKDG with a single
random seed may not find the global energy minimum. Generating multiple
conformers and selecting the lowest-energy one gives a more reliable
representative geometry, especially for drug-like molecules with many
rotatable bonds.

Usage
-----
# Generate and save conformers for a dataset
python conformer_generation.py --dataset esol --n_confs 10 --data_dir ./data

# Then in the notebook, load them:
from conformer_generation import load_conformer_graphs
graphs, valid_idx = load_conformer_graphs("esol", data_dir="./data")

Author: Emir Erdem  |  RWTH Aachen – Computational Biotechnology
"""

from __future__ import annotations

import os
import pickle
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import torch
from torch_geometric.data import Data

# reuse feature helpers from mol_repr_utils
from mol_repr_utils import (
    _atom_features,
    _bond_features,
    _make_target_tensor,
    load_dataset,
    preprocess_dataset,
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _pickle_safe_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise stats column dtypes after load (legacy DataFrame payloads)."""
    if stats_df is None or len(stats_df) == 0:
        return stats_df
    out = stats_df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype(str)
    return out


def _stats_to_portable(stats) -> Optional[dict]:
    """Store stats as plain dict-of-lists (avoids pandas StringDtype pickle issues)."""
    if stats is None or (isinstance(stats, pd.DataFrame) and len(stats) == 0):
        return None
    if isinstance(stats, dict):
        return stats
    return {col: stats[col].tolist() for col in stats.columns}


def _stats_from_portable(stats) -> pd.DataFrame:
    """Rebuild stats DataFrame from portable dict (or pass through legacy DataFrame)."""
    if stats is None:
        return pd.DataFrame()
    if isinstance(stats, pd.DataFrame):
        return _pickle_safe_stats(stats)
    return pd.DataFrame(stats)


def _pickle_safe_payload(payload: dict) -> dict:
    out = dict(payload)
    if "stats" in out:
        out["stats"] = _stats_to_portable(out["stats"])
    return out

# =============================================================================
# 1. MULTI-CONFORMER GENERATION
# =============================================================================

def generate_conformers(
    mol: Chem.Mol,
    n_confs: int = 10,
    seed: int = 42,
    optimise: bool = True,
    max_iters: int = 2000,
) -> Tuple[Optional[Chem.Mol], List[float]]:
    """
    Generate multiple 3D conformers for a molecule using ETKDG,
    optionally optimise each with MMFF, and return energies.

    Parameters
    ----------
    mol       : RDKit molecule (from SMILES, no existing conformers needed)
    n_confs   : number of conformers to generate
    seed      : random seed for reproducibility
    optimise  : if True, MMFF-optimise each conformer
    max_iters : max MMFF iterations per conformer

    Returns
    -------
    mol_3d   : molecule with all conformers embedded (or None if failed)
    energies : list of MMFF energies per conformer (kcal/mol)
               inf if optimisation failed for that conformer
    """
    mol = Chem.AddHs(mol)

    # ETKDG parameters
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0          # use all available cores
    params.enforceChirality = True
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True

    try:
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    except Exception:
        return None, []

    if len(conf_ids) == 0:
        return None, []

    energies = []

    if optimise:
        try:
            ff_props = AllChem.MMFFGetMoleculeProperties(mol)
        except Exception:
            ff_props = None
        if ff_props is None:
            for _ in conf_ids:
                energies.append(float("inf"))
        else:
            for cid in conf_ids:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=cid)
                except Exception:
                    energies.append(float("inf"))
                    continue
                if ff is None:
                    energies.append(float("inf"))
                    continue
                try:
                    ff.Minimize(maxIts=max_iters)
                    energies.append(ff.CalcEnergy())
                except Exception:
                    energies.append(float("inf"))
    else:
        energies = [0.0] * len(conf_ids)

    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None, []

    return mol, energies


def select_min_energy_conformer(
    mol: Chem.Mol,
    energies: List[float],
) -> Tuple[Optional[Chem.Mol], int, float]:
    """
    Select the conformer with the lowest MMFF energy.

    Returns
    -------
    mol_single : molecule with only the best conformer
    best_idx   : index of the selected conformer
    best_energy: its energy in kcal/mol
    """
    if not energies or mol is None:
        return None, -1, float("inf")

    best_idx    = int(np.argmin(energies))
    best_energy = energies[best_idx]
    best_conf   = mol.GetConformer(best_idx)

    # create a new molecule with only the best conformer
    mol_single = Chem.RWMol(mol)
    mol_single.RemoveAllConformers()
    mol_single.AddConformer(best_conf, assignId=True)

    return mol_single.GetMol(), best_idx, best_energy


# =============================================================================
# 2. CONFORMER STATISTICS
# =============================================================================

def conformer_diversity(mol: Chem.Mol, energies: List[float]) -> Dict:
    """
    Compute statistics across all generated conformers.

    Returns a dict with:
    - n_confs        : number of successfully generated conformers
    - energy_min     : lowest MMFF energy (kcal/mol)
    - energy_max     : highest MMFF energy (kcal/mol)
    - energy_range   : max - min  (proxy for conformational flexibility)
    - energy_std     : std of energies
    - n_rotatable    : number of rotatable bonds (flexibility indicator)
    """
    if mol is None or not energies:
        return {}

    finite_energies = [e for e in energies if np.isfinite(e)]
    n_rot = Descriptors.NumRotatableBonds(mol)

    return {
        "n_confs":      len(finite_energies),
        "energy_min":   float(np.min(finite_energies)) if finite_energies else float("nan"),
        "energy_max":   float(np.max(finite_energies)) if finite_energies else float("nan"),
        "energy_range": float(np.ptp(finite_energies)) if finite_energies else float("nan"),
        "energy_std":   float(np.std(finite_energies)) if finite_energies else float("nan"),
        "n_rotatable":  n_rot,
    }


# =============================================================================
# 3. 3D GRAPH FROM BEST CONFORMER
# =============================================================================

def mol_to_graph_3d(
    mol: Chem.Mol,
    y=None,
    conf_id: int = 0,
) -> Optional[Data]:
    """
    Build a PyG Data object from a molecule that already has a 3D conformer.

    Node features : same 163-dim atom features as the 2D pipeline
    Edge features : 2D bond features + interatomic distance (Å) + mean bond angle (rad)

    This function is used after conformer selection — it takes the
    pre-selected best conformer and builds the graph, rather than
    generating the conformer itself.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None

    conf      = mol.GetConformer(conf_id)
    positions = torch.tensor(conf.GetPositions(), dtype=torch.float)  # (N, 3)
    x         = torch.tensor(
        [_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float
    )

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf   = _bond_features(bond)
        dist = float(torch.norm(positions[j] - positions[i]).item())

        def mean_angle_at(center: int, other: int) -> float:
            c, o = positions[center], positions[other]
            v1   = o - c
            angles = []
            for nb in mol.GetAtomWithIdx(center).GetBonds():
                nb_idx = nb.GetOtherAtomIdx(center)
                if nb_idx == other:
                    continue
                v2    = positions[nb_idx] - c
                denom = torch.norm(v1) * torch.norm(v2) + 1e-8
                cos_a = float(torch.clamp(torch.dot(v1, v2) / denom, -1.0, 1.0))
                angles.append(np.arccos(cos_a))
            return float(np.mean(angles)) if angles else 0.0

        avg_angle = (mean_angle_at(i, j) + mean_angle_at(j, i)) / 2.0
        ef        = bf + [dist, avg_angle]
        edge_indices += [[i, j], [j, i]]
        edge_attrs   += [ef, ef]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attrs,   dtype=torch.float)

    z         = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=positions,
        z=z,
        y=_make_target_tensor(y),
    )


# =============================================================================
# 4. FULL PIPELINE: SMILES LIST → GRAPHS (WITH CACHING)
# =============================================================================

def generate_and_save_conformers(
    smiles_list: List[str],
    targets: np.ndarray,
    save_path: str,
    n_confs: int = 10,
    seed: int = 42,
    show_progress: bool = True,
) -> Tuple[List[Data], List[int], pd.DataFrame]:
    """
    Generate multi-conformer 3D graphs for a list of SMILES,
    select the lowest-energy conformer for each, and save everything to disk.

    Parameters
    ----------
    smiles_list   : list of SMILES strings
    targets       : np.ndarray of target values
    save_path     : path to save the .pkl file (graphs + metadata)
    n_confs       : number of conformers to generate per molecule
    seed          : random seed

    Returns
    -------
    graphs      : List[Data]  — one graph per successfully processed molecule
    valid_idx   : List[int]   — original indices of successful molecules
    stats_df    : pd.DataFrame — conformer statistics per molecule
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    graphs, valid_idx, stats_rows = [], [], []

    iterator = tqdm(
        enumerate(zip(smiles_list, targets)),
        total=len(smiles_list),
        desc=f"Generating {n_confs} conformers/mol",
        disable=not show_progress,
    )

    for i, (smi, y) in iterator:
        try:
            mol_2d = Chem.MolFromSmiles(smi)
            if mol_2d is None:
                continue

            mol_3d, energies = generate_conformers(mol_2d, n_confs=n_confs, seed=seed)
            if mol_3d is None or len(energies) == 0:
                logger.warning(f"Conformer generation failed: {smi[:50]}…")
                continue

            mol_best, best_idx, best_energy = select_min_energy_conformer(mol_3d, energies)
            if mol_best is None:
                continue

            graph = mol_to_graph_3d(mol_best, y=y)
            if graph is None:
                continue

            stats = conformer_diversity(mol_3d, energies)
            stats["smiles"]        = smi
            stats["best_conf_idx"] = best_idx
            stats["best_energy"]   = best_energy
            stats["mol_idx"]       = i
            stats_rows.append(stats)

            graphs.append(graph)
            valid_idx.append(i)
        except Exception as exc:
            logger.warning(f"Skipped molecule idx={i}: {smi[:50]}… ({type(exc).__name__})")
            continue

    stats_df = _pickle_safe_stats(pd.DataFrame(stats_rows))

    # save everything to disk
    payload = _pickle_safe_payload({
        "graphs":    graphs,
        "valid_idx": valid_idx,
        "stats":     stats_df,
        "n_confs":   n_confs,
        "seed":      seed,
        "n_total":   len(smiles_list),
    })
    with open(save_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    logger.info(f"Saved {len(graphs)} / {len(smiles_list)} graphs → {save_path}")
    return graphs, valid_idx, stats_df


def load_conformer_graphs(
    dataset_name: str,
    data_dir: str = "./data",
    n_confs: int = 10,
) -> Tuple[List[Data], List[int], pd.DataFrame]:
    """
    Load pre-generated conformer graphs from disk.
    If the file doesn't exist, raises FileNotFoundError with instructions.

    Parameters
    ----------
    dataset_name : "esol", "bbbp", or "tox21"
    data_dir     : directory where .pkl files are stored
    n_confs      : number of conformers used during generation
                   (used to find the correct file)

    Returns
    -------
    graphs    : List[Data]
    valid_idx : List[int]
    stats_df  : pd.DataFrame with conformer statistics
    """
    path = Path(data_dir) / f"{dataset_name}_conformers_n{n_confs}.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"Conformer file not found: {path}\n"
            f"Run the generation script first:\n"
            f"  python conformer_generation.py --dataset {dataset_name} "
            f"--n_confs {n_confs} --data_dir {data_dir}"
        )

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"Could not unpickle {path} — pandas version mismatch (often StringDtype in stats).\n"
            f"Use the project venv kernel, upgrade pandas if needed, then run:\n"
            f"  python repack_conformer_pkl.py --path {path}\n"
            f"Original error: {exc}"
        ) from exc

    stats = _stats_from_portable(payload.get("stats"))

    logger.info(
        f"Loaded {len(payload['graphs'])} / {payload['n_total']} graphs "
        f"from {path}  (n_confs={payload['n_confs']})"
    )
    return payload["graphs"], payload["valid_idx"], stats


# =============================================================================
# 5. CONFORMER STATISTICS VISUALISATION
# =============================================================================

def plot_conformer_stats(stats_df: pd.DataFrame, dataset_name: str, save_dir: str = "./results"):
    """
    Plot conformer statistics across the dataset:
    - Energy range distribution (proxy for molecular flexibility)
    - Energy of best conformer distribution
    - Rotatable bonds vs energy range (scatter)
    """
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"{dataset_name.upper()} – Conformer Statistics (n_confs={stats_df['n_confs'].iloc[0] if 'n_confs' in stats_df else '?'})",
                 fontsize=13, fontweight="bold")

    # Energy range distribution
    axes[0].hist(stats_df["energy_range"].dropna(), bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Conformational Energy Range")
    axes[0].set_xlabel("Energy range (kcal/mol)\n[proxy for flexibility]")
    axes[0].set_ylabel("Count")

    # Best conformer energy
    axes[1].hist(stats_df["best_energy"].dropna(), bins=40, color="darkorange", edgecolor="white")
    axes[1].set_title("Best Conformer Energy")
    axes[1].set_xlabel("MMFF energy (kcal/mol)")

    # Rotatable bonds vs energy range
    axes[2].scatter(
        stats_df["n_rotatable"], stats_df["energy_range"],
        alpha=0.4, s=15, color="seagreen"
    )
    axes[2].set_title("Flexibility vs Energy Range")
    axes[2].set_xlabel("# Rotatable bonds")
    axes[2].set_ylabel("Energy range (kcal/mol)")

    plt.tight_layout()
    out = save_dir / f"{dataset_name}_conformer_stats.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Saved conformer stats plot → {out}")


# =============================================================================
# 6. COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and save multi-conformer 3D graphs for MoleculeNet datasets"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["esol", "bbbp", "tox21", "freesolv", "lipophilicity", "hiv", "bace"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--n_confs", type=int, default=10,
        help="Number of conformers to generate per molecule (default: 10)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory for dataset CSVs and output .pkl files"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for conformer generation"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results",
        help="Directory to save conformer statistics plots"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index (inclusive) for chunked cluster runs (default: 0)"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None,
        help="End index (exclusive) for chunked cluster runs (default: all molecules)"
    )
    parser.add_argument(
        "--chunk_tag", type=str, default=None,
        help="If set, save to {dataset}_conformers_n{n}_chunk_{tag}.pkl instead of the final file"
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Skip conformer statistics plot (recommended on headless clusters)"
    )
    args = parser.parse_args()

    # load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(Path(args.data_dir) / f"{args.dataset}.csv")
    smiles_list, targets = preprocess_dataset(df, args.dataset)

    n_total = len(smiles_list)
    start = max(0, args.start_idx)
    end = n_total if args.end_idx is None else min(args.end_idx, n_total)
    if start >= end:
        logger.warning(
            f"Empty slice [{start}:{end}) for n_total={n_total} — "
            "nothing to do (array task past end of preprocessed dataset)."
        )
        import sys
        sys.exit(0)

    smiles_chunk = smiles_list[start:end]
    targets_chunk = targets[start:end]
    logger.info(f"Processing molecules [{start}:{end}) of {n_total}")

    # generate conformers and save
    if args.chunk_tag:
        fname = f"{args.dataset}_conformers_n{args.n_confs}_chunk_{args.chunk_tag}.pkl"
    else:
        fname = f"{args.dataset}_conformers_n{args.n_confs}.pkl"
    save_path = Path(args.data_dir) / fname

    graphs, valid_idx, stats_df = generate_and_save_conformers(
        smiles_list=smiles_chunk,
        targets=targets_chunk,
        save_path=save_path,
        n_confs=args.n_confs,
        seed=args.seed,
    )

    # remap valid_idx to global dataset indices
    valid_idx_global = [start + i for i in valid_idx]
    payload = _pickle_safe_payload({
        "graphs": graphs,
        "valid_idx": valid_idx_global,
        "stats": stats_df,
        "n_confs": args.n_confs,
        "seed": args.seed,
        "n_total": n_total,
        "chunk_start": start,
        "chunk_end": end,
    })
    with open(save_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    logger.info(f"Re-saved with global indices → {save_path}")

    # print summary
    print(f"\n{'='*50}")
    print(f"Dataset       : {args.dataset.upper()}")
    print(f"Chunk         : [{start}:{end}) of {n_total}")
    print(f"Chunk size    : {len(smiles_chunk)}")
    print(f"Successful    : {len(graphs)} ({len(graphs)/len(smiles_chunk)*100:.1f}%)")
    print(f"Conformers/mol: {args.n_confs}")
    print(f"Saved to      : {save_path}")
    if len(stats_df) > 0:
        print(f"\nConformer statistics:")
        print(stats_df[["energy_min", "energy_max", "energy_range", "energy_std", "n_rotatable"]].describe().round(3).to_string())

    if not args.no_plot:
        plot_conformer_stats(stats_df, args.dataset, save_dir=args.results_dir)