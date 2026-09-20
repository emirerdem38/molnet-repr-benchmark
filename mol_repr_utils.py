"""
mol_repr_utils.py
=================
Utility functions for the thesis:
  "Comparative Study of Molecular Representations for
   Machine Learning-Based Prediction of Pharmaceutical Properties"

Molecular representations implemented
--------------------------------------
1. Morgan Fingerprints (ECFP)       – fixed-length bit-vector
2. RDKit Descriptors                 – physicochemical descriptor vector
3. 2D Graph                          – atom/bond feature matrices via PyG
4. 3D Graph                          – 2D graph + interatomic distances & angles
5. SMILES strings                    – raw character sequence → LSTM model

Datasets (MoleculeNet)
-----------------------
- ESOL   : aqueous solubility (regression)
- BBBP   : blood-brain barrier permeability (binary classification)
- Tox21  : 12-task toxicity panel (multi-label classification)

Models
-------
- Random Forest (sklearn)           – tabular representations (1, 2)
- Support Vector Machine (sklearn)  – tabular representations (1, 2)
- XGBoost                           – tabular representations (1, 2)
- GCN / GIN (PyTorch Geometric)     – graph representations (3, 4)
- LSTM (PyTorch)                    – SMILES sequence representation (5)

Author: Emir Erdem  |  RWTH Aachen – Computational Biotechnology
"""

from __future__ import annotations
import json

import warnings
warnings.filterwarnings("ignore")

import os
# Must run before sklearn/torch/xgboost load OpenMP (macOS Jupyter OMP Error #179).
for _omp_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_omp_var] = "1"

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ── RDKit ─────────────────────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

# ── sklearn ────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.impute import SimpleImputer

# ── PyTorch / PyG ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from torch_geometric.nn import GCNConv, GINConv, GINEConv, global_mean_pool, global_add_pool
from torch_geometric.nn.models import SchNet as PyGSchNet
from torch.utils.data import Dataset, DataLoader as TorchLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# 1. DATASET LOADING
# =============================================================================

DATASET_URLS = {
    "esol":          "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "bbbp":          "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "tox21":         "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv",
    "freesolv":      "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
    "lipophilicity": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "hiv":           "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
    "bace":          "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
}

DATASET_CONFIG = {
    "esol": {
        "smiles_col":  "smiles",
        "target_cols": ["measured log solubility in mols per litre"],
        "task_type":   "regression",
    },
    "bbbp": {
        "smiles_col":  "smiles",
        "target_cols": ["p_np"],
        "task_type":   "classification",
    },
    "tox21": {
        "smiles_col":  "smiles",
        "target_cols": [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
            "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
            "SR-HSE", "SR-MMP", "SR-p53",
        ],
        "task_type": "multilabel_classification",
    },
    "freesolv": {
        "smiles_col":  "smiles",
        "target_cols": ["expt"],
        "task_type":   "regression",
    },
    "lipophilicity": {
        "smiles_col":  "smiles",
        "target_cols": ["exp"],
        "task_type":   "regression",
    },
    "hiv": {
        "smiles_col":  "smiles",
        "target_cols": ["HIV_active"],
        "task_type":   "classification",
    },
    "bace": {
        "smiles_col":  "mol",
        "target_cols": ["Class"],
        "task_type":   "classification",
    },
}


def load_dataset(name: str, cache_dir: str = "./data") -> pd.DataFrame:
    """Download (or load from cache) a MoleculeNet dataset."""
    name = name.lower()
    assert name in DATASET_URLS, f"Unknown dataset '{name}'. Choose from {list(DATASET_URLS)}"
    cache_path = Path(cache_dir) / f"{name}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        logger.info(f"Loading '{name}' from cache: {cache_path}")
        return pd.read_csv(cache_path)
    logger.info(f"Downloading '{name}' …")
    df = pd.read_csv(DATASET_URLS[name])
    df.to_csv(cache_path, index=False)
    logger.info(f"Saved to {cache_path}")
    return df


def canonical_smiles(smi: str) -> Optional[str]:
    """Canonical isomeric SMILES (Wigh et al. 2022 round-trip convention)."""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_smiles_list(smiles_list: List[str]) -> List[str]:
    """Drop invalid SMILES and return canonical isomeric strings."""
    out: List[str] = []
    for smi in smiles_list:
        can = canonical_smiles(smi)
        if can is not None:
            out.append(can)
    return out


def preprocess_dataset(df: pd.DataFrame, name: str) -> Tuple[List[str], np.ndarray]:
    """Validate SMILES, canonicalize, and return aligned (smiles_list, targets)."""
    cfg        = DATASET_CONFIG[name]
    smiles_col = cfg["smiles_col"]
    target_cols = cfg["target_cols"]
    valid_mask = df[smiles_col].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    df = df[valid_mask].reset_index(drop=True)
    logger.info(f"[{name}] {valid_mask.sum()} / {len(valid_mask)} valid molecules")

    smiles_out: List[str] = []
    target_rows: List[np.ndarray] = []
    for _, row in df.iterrows():
        can = canonical_smiles(str(row[smiles_col]))
        if can is not None:
            smiles_out.append(can)
            target_rows.append(row[target_cols].values.astype(float))

    targets = np.array(target_rows, dtype=float)
    if len(target_cols) == 1:
        targets = targets.ravel()
    return smiles_out, targets


def filter_bonded_molecules(
    smiles_list: List[str], targets: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """Keep only molecules with at least one bond (required for graph / D-MPNN models)."""
    kept_smiles: List[str] = []
    kept_targets: List[np.ndarray] = []
    for smi, yi in zip(smiles_list, targets):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.GetNumBonds() > 0:
            kept_smiles.append(smi)
            kept_targets.append(np.atleast_1d(yi))
    if not kept_targets:
        return [], np.array([])
    stacked = np.vstack([np.atleast_1d(t) for t in kept_targets])
    if stacked.shape[1] == 1:
        return kept_smiles, stacked.ravel()
    return kept_smiles, stacked


# =============================================================================
# 2. MOLECULAR REPRESENTATIONS
# =============================================================================

# ── 2a. Morgan Fingerprints ───────────────────────────────────────────────────

def smiles_to_morgan(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
) -> np.ndarray:
    """
    Compute Morgan (ECFP) fingerprints.
    radius=2 → ECFP4,  radius=3 → ECFP6
    Returns shape (N, n_bits).
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = [generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi)) for smi in smiles_list]
    return np.array(fps, dtype=np.float32)


# ── 2b. RDKit Physicochemical Descriptors ─────────────────────────────────────

_DESCRIPTOR_NAMES = [d[0] for d in Descriptors.descList]
_DESCRIPTOR_CALC  = MoleculeDescriptors.MolecularDescriptorCalculator(_DESCRIPTOR_NAMES)


def smiles_to_descriptors(smiles_list: List[str]) -> np.ndarray:
    """
    Compute the full set of RDKit 2D descriptors (~200 features).
    Returns shape (N, n_descriptors).
    """
    rows = [list(_DESCRIPTOR_CALC.CalcDescriptors(Chem.MolFromSmiles(smi)))
            for smi in smiles_list]
    X = np.array(rows, dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    return X


def smiles_to_combined(smiles_list: List[str]) -> np.ndarray:
    """Concatenate Morgan fingerprints and RDKit descriptors (Wang et al. 2025)."""
    return np.hstack([smiles_to_morgan(smiles_list), smiles_to_descriptors(smiles_list)])


# ── 2c. Shared atom/bond feature helpers ─────────────────────────────────────

ATOM_FEATURES = {
    "atomic_num":    list(range(1, 119)),
    "degree":        list(range(0, 11)),
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "num_hs":        list(range(0, 9)),
    "hybridization": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.OTHER,
    ],
}

BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "stereo": [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ],
}


def _one_hot(val, choices: list) -> List[int]:
    enc = [0] * (len(choices) + 1)
    try:
        enc[choices.index(val)] = 1
    except ValueError:
        enc[-1] = 1
    return enc


def _atom_features(atom) -> List[float]:
    feats  = _one_hot(atom.GetAtomicNum(),     ATOM_FEATURES["atomic_num"])
    feats += _one_hot(atom.GetDegree(),         ATOM_FEATURES["degree"])
    feats += _one_hot(atom.GetFormalCharge(),   ATOM_FEATURES["formal_charge"])
    feats += _one_hot(atom.GetTotalNumHs(),     ATOM_FEATURES["num_hs"])
    feats += _one_hot(atom.GetHybridization(),  ATOM_FEATURES["hybridization"])
    feats += [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    return feats


def _bond_features(bond) -> List[float]:
    feats  = _one_hot(bond.GetBondType(), BOND_FEATURES["bond_type"])
    feats += _one_hot(bond.GetStereo(),   BOND_FEATURES["stereo"])
    feats += [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    return feats


# Precompute dimensions
_ATOM_DIM    = len(_atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
_BOND_DIM    = len(_bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))
_BOND_DIM_3D = _BOND_DIM + 2   # + interatomic distance + bond angle


def _make_target_tensor(y):
    if y is None:
        return None
    return torch.tensor([y] if np.isscalar(y) else y, dtype=torch.float).unsqueeze(0)


# ── 2d. 2D Graph ──────────────────────────────────────────────────────────────

def smiles_to_graph(smi: str, y=None) -> Optional[Data]:
    """Convert SMILES to a 2D PyG Data object (topology + atom/bond features)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumBonds() == 0:
        return None
    x   = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_indices, edge_attrs = [], []
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf   = _bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs   += [bf, bf]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attrs,   dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=_make_target_tensor(y))


def smiles_to_graphs(smiles_list: List[str], targets: np.ndarray) -> List[Data]:
    """Convert a list of SMILES to 2D PyG Data objects."""
    graphs = [smiles_to_graph(smi, y) for smi, y in zip(smiles_list, targets)]
    return [g for g in graphs if g is not None]


# ── 2d-b. Directed bond graph (D-MPNN, Yang et al. 2019) ─────────────────────

class DMPNNData(Data):
    """PyG Data with bond-index offsets for batched D-MPNN message passing."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("b2src", "b2dst"):
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def smiles_to_dmpnn_graph(smi: str, y=None) -> Optional[DMPNNData]:
    """
    Build a directed-bond graph for D-MPNN.

    Each undirected bond yields two directed bonds (i→j and j→i). Initial bond
    features concatenate source atom, bond, and target atom vectors (Chemprop).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumBonds() == 0:
        return None

    atom_x = torch.tensor(
        [_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float
    )
    bond_in, b2src, b2dst = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        af_i = _atom_features(mol.GetAtomWithIdx(i))
        af_j = _atom_features(mol.GetAtomWithIdx(j))
        bond_in.append(af_i + bf + af_j)
        b2src.append(i)
        b2dst.append(j)
        bond_in.append(af_j + bf + af_i)
        b2src.append(j)
        b2dst.append(i)

    return DMPNNData(
        x=atom_x,
        bond_init=torch.tensor(bond_in, dtype=torch.float),
        b2src=torch.tensor(b2src, dtype=torch.long),
        b2dst=torch.tensor(b2dst, dtype=torch.long),
        y=_make_target_tensor(y),
    )


def smiles_to_dmpnn_graphs(smiles_list: List[str], targets: np.ndarray) -> List[DMPNNData]:
    graphs = [smiles_to_dmpnn_graph(smi, y) for smi, y in zip(smiles_list, targets)]
    return [g for g in graphs if g is not None]


# ── 2e. 3D Graph ──────────────────────────────────────────────────────────────

def _generate_conformer(mol) -> Optional[Chem.Mol]:
    """
    Generate a 3D conformer using ETKDG + MMFF optimisation.

    ETKDG (Experimental Torsion Knowledge Distance Geometry) uses torsion
    angle preferences from the Cambridge Structural Database to produce
    realistic drug-like geometries. MMFF then energy-minimises the result.
    Returns None if embedding fails.
    """
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    return mol if mol.GetNumConformers() > 0 else None


def smiles_to_graph_3d(smi: str, y=None) -> Optional[Data]:
    """
    Convert SMILES to a 3D-aware PyG Data object.

    Node features : same 163-dim atom features as 2D graph
    Edge features : 2D bond features (11-dim)
                    + interatomic distance in Å  (1-dim)
                    + mean bond angle in radians  (1-dim)
                  = 13-dim total

    The distance and angle features encode the actual 3D geometry of the
    molecule, allowing GNNs to learn geometry-dependent properties.
    Returns None if conformer generation fails.
    """
    mol_3d = _generate_conformer(Chem.MolFromSmiles(smi))
    if mol_3d is None or mol_3d.GetNumBonds() == 0:
        return None

    conf      = mol_3d.GetConformer()
    positions = torch.tensor(conf.GetPositions(), dtype=torch.float)  # (N, 3)
    x         = torch.tensor([_atom_features(a) for a in mol_3d.GetAtoms()], dtype=torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol_3d.GetBonds():
        i, j  = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf    = _bond_features(bond)
        dist  = float(torch.norm(positions[j] - positions[i]).item())

        # mean bond angle at both endpoints
        def mean_angle_at(center: int, other: int) -> float:
            c, o = positions[center], positions[other]
            v1   = o - c
            angles = []
            for nb in mol_3d.GetAtomWithIdx(center).GetBonds():
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
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                pos=positions, y=_make_target_tensor(y))


def smiles_to_graphs_3d(
    smiles_list: List[str],
    targets: np.ndarray,
    show_progress: bool = True,
) -> Tuple[List[Data], List[int]]:
    """
    Convert a list of SMILES to 3D PyG Data objects.

    Returns (graphs, valid_idx) — valid_idx lets you align targets
    if any conformer generations failed.
    """
    graphs, valid_idx = [], []
    for i, (smi, y) in enumerate(tqdm(
        zip(smiles_list, targets), total=len(smiles_list),
        desc="3D conformers", disable=not show_progress
    )):
        g = smiles_to_graph_3d(smi, y)
        if g is not None and g.edge_index.numel() > 0:
            graphs.append(g)
            valid_idx.append(i)

    logger.info(f"3D graphs: {len(graphs)} / {len(smiles_list)} succeeded")
    return graphs, valid_idx


# ── 2f. SMILES Tokenisation ───────────────────────────────────────────────────

def build_vocab(smiles_list: List[str]) -> Dict[str, int]:
    """
    Build a character-level vocabulary.
    Token 0 = <PAD>,  Token 1 = <UNK>,  all other chars start from 2.
    """
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, c in enumerate(sorted(set("".join(smiles_list)))):
        vocab[c] = i + 2
    return vocab


def tokenize_smiles(
    smiles_list: List[str],
    vocab: Optional[Dict[str, int]] = None,
    max_len: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Character-level tokenisation of SMILES strings.

    Each SMILES character (e.g. 'C', '(', '=', 'O', '#') is mapped to an
    integer. Sequences are zero-padded to max_len. This is the direct input
    to the LSTM — no hand-crafted chemistry, just the raw string.

    Returns
    -------
    X     : np.ndarray  shape (N, max_len)
    vocab : dict  char → int
    """
    if vocab is None:
        vocab = build_vocab(smiles_list)
    if max_len is None:
        max_len = max(len(s) for s in smiles_list)

    X = np.zeros((len(smiles_list), max_len), dtype=np.int32)
    for i, smi in enumerate(smiles_list):
        for j, c in enumerate(smi[:max_len]):
            X[i, j] = vocab.get(c, vocab["<UNK>"])
    return X, vocab


class SMILESDataset(Dataset):
    """PyTorch Dataset wrapping tokenised SMILES + targets for the LSTM."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float)
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# 3. SKLEARN PIPELINES
# =============================================================================

def build_rf_pipeline(task_type: str, **rf_kwargs) -> Pipeline:
    defaults = dict(n_estimators=200, n_jobs=-1, random_state=42)
    defaults.update(rf_kwargs)
    model = (RandomForestRegressor if task_type == "regression"
             else RandomForestClassifier)(**defaults)
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler",  StandardScaler()),
                     ("model",   model)])


def build_svm_pipeline(task_type: str, **svm_kwargs) -> Pipeline:
    defaults = dict(kernel="rbf", C=1.0, random_state=42)
    defaults.update(svm_kwargs)
    if task_type == "regression":
        model = SVR(**{k: v for k, v in defaults.items() if k != "random_state"})
    else:
        model = SVC(probability=True, **defaults)
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler",  StandardScaler()),
                     ("model",   model)])


def build_xgb_pipeline(task_type: str, **xgb_kwargs) -> Pipeline:
    import os
    for _omp_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_omp_var] = "1"
    from xgboost import XGBClassifier, XGBRegressor
    from hpo_config import XGB_N_JOBS

    defaults = dict(
        n_estimators=200,
        n_jobs=XGB_N_JOBS,
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )
    defaults.update(xgb_kwargs)
    if task_type == "regression":
        defaults.setdefault("objective", "reg:squarederror")
        model = XGBRegressor(**defaults)
    else:
        defaults.setdefault("objective", "binary:logistic")
        defaults.setdefault("eval_metric", "logloss")
        model = XGBClassifier(**defaults)
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler",  StandardScaler()),
                     ("model",   model)])


# =============================================================================
# 4. GNN MODELS  (configurable depth — default 4 layers × 256 hidden)
# =============================================================================

GNN_HIDDEN_DEFAULT = 256
GNN_LAYERS_DEFAULT = 4
GNN_DROPOUT_DEFAULT = 0.2


class GCN(nn.Module):
    """Graph Convolutional Network (2D topology only)."""
    uses_edge_attr = False

    def __init__(
        self,
        in_channels,
        hidden,
        out_channels,
        num_layers: int = GNN_LAYERS_DEFAULT,
        dropout: float = GNN_DROPOUT_DEFAULT,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin     = nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_attr=None):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GIN(nn.Module):
    """Graph Isomorphism Network (2D topology only)."""
    uses_edge_attr = False

    def __init__(
        self,
        in_channels,
        hidden,
        out_channels,
        num_layers: int = GNN_LAYERS_DEFAULT,
        dropout: float = GNN_DROPOUT_DEFAULT,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        def mlp(a, b):
            return nn.Sequential(
                nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Linear(b, b),
            )

        self.convs = nn.ModuleList([GINConv(mlp(in_channels, hidden))])
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(mlp(hidden, hidden)))
        self.lin     = nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_attr=None):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        return self.lin(x)


class DMPNN(nn.Module):
    """
    Directed Message Passing Neural Network (Yang et al. 2019 / Chemprop).

    Messages pass on directed bonds; readout sums atom-level vectors built from
    incoming bond messages and atom features.
    """
    is_dmpnn = True
    uses_edge_attr = False

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden: int,
        out_channels: int,
        depth: int = 3,
        dropout: float = GNN_DROPOUT_DEFAULT,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.depth = depth
        self.dropout = dropout
        bond_input_dim = 2 * atom_dim + bond_dim
        self.W_i = nn.Linear(bond_input_dim, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(atom_dim + hidden, hidden)
        self.W_f = nn.Linear(hidden, out_channels)

    def _forward_one(
        self,
        atom_x: torch.Tensor,
        bond_init: torch.Tensor,
        b2src: torch.Tensor,
        b2dst: torch.Tensor,
    ) -> torch.Tensor:
        messages = F.relu(self.W_i(bond_init))
        n_bonds = messages.size(0)

        for _ in range(self.depth - 1):
            agg = torch.zeros_like(messages)
            for b in range(n_bonds):
                src = b2src[b]
                dst = b2dst[b]
                mask = (b2dst == src) & (b2src != dst)
                if mask.any():
                    agg[b] = messages[mask].sum(dim=0)
            messages = F.relu(messages + self.W_h(agg))
            messages = F.dropout(messages, p=self.dropout, training=self.training)

        n_atoms = atom_x.size(0)
        atom_vecs = []
        for a in range(n_atoms):
            mask = b2dst == a
            if mask.any():
                m = messages[mask].sum(dim=0)
            else:
                m = torch.zeros(messages.size(1), device=messages.device)
            atom_vecs.append(F.relu(self.W_o(torch.cat([atom_x[a], m]))))
        mol_vec = torch.stack(atom_vecs, dim=0).sum(dim=0)
        return self.W_f(mol_vec)

    def forward(self, batch) -> torch.Tensor:
        if hasattr(batch, "to_data_list"):
            graphs = batch.to_data_list()
        else:
            graphs = [batch]
        outs = [
            self._forward_one(g.x, g.bond_init, g.b2src, g.b2dst) for g in graphs
        ]
        return torch.stack(outs, dim=0)


class GCN3D(nn.Module):
    """GINE layers + mean pool; uses bond + distance + angle edge features."""
    uses_edge_attr = True

    def __init__(
        self,
        in_channels,
        edge_dim,
        hidden,
        out_channels,
        num_layers: int = GNN_LAYERS_DEFAULT,
        dropout: float = GNN_DROPOUT_DEFAULT,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        def mlp(a, b):
            return nn.Sequential(
                nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Linear(b, b),
            )

        self.convs = nn.ModuleList(
            [GINEConv(mlp(in_channels, hidden), edge_dim=edge_dim)]
        )
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(mlp(hidden, hidden), edge_dim=edge_dim))
        self.lin     = nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_attr):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GIN3D(nn.Module):
    """GINE layers + sum pool; uses bond + distance + angle edge features."""
    uses_edge_attr = True

    def __init__(
        self,
        in_channels,
        edge_dim,
        hidden,
        out_channels,
        num_layers: int = GNN_LAYERS_DEFAULT,
        dropout: float = GNN_DROPOUT_DEFAULT,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        def mlp(a, b):
            return nn.Sequential(
                nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Linear(b, b),
            )

        self.convs = nn.ModuleList(
            [GINEConv(mlp(in_channels, hidden), edge_dim=edge_dim)]
        )
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(mlp(hidden, hidden), edge_dim=edge_dim))
        self.lin     = nn.Linear(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_attr):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, batch)
        return self.lin(x)


def build_gnn(
    arch: str,
    dim: str,
    out_channels: int,
    hidden: int = GNN_HIDDEN_DEFAULT,
    num_layers: int = GNN_LAYERS_DEFAULT,
    dropout: float = GNN_DROPOUT_DEFAULT,
):
    """Factory: 2D -> GIN/DMPNN; 3D -> GIN3D (GINEConv). GCN removed in v4."""
    arch_u = arch.upper()
    if arch_u == "DMPNN":
        if dim.upper() != "2D":
            raise ValueError("D-MPNN is only defined for 2D graphs")
        return DMPNN(
            atom_dim=ATOM_FEATURE_DIM,
            bond_dim=BOND_FEATURE_DIM,
            hidden=hidden,
            out_channels=out_channels,
            depth=num_layers,
            dropout=dropout,
        )
    if dim.upper() == "3D":
        if arch_u == "GCN":
            raise ValueError("GCN (3D) removed in v4; use GIN (3D)")
        return GIN3D(
            in_channels=ATOM_FEATURE_DIM,
            edge_dim=BOND_FEATURE_DIM_3D,
            hidden=hidden,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    if arch_u == "GCN":
        raise ValueError("GCN (2D) removed in v4; use GIN (2D) or D-MPNN (2D)")
    return GIN(
        in_channels=ATOM_FEATURE_DIM,
        hidden=hidden,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
    )


# =============================================================================
# 4b. SCHNET (3D coordinates — min-energy conformer pos + atomic numbers)
# =============================================================================

def _schnet_install_hint() -> str:
    return (
        "SchNet requires pyg-lib>=0.6.0 (PyG 2.6+) or torch-cluster for radius_graph. "
        "On the cluster run: bash cluster/install_pyg_extensions.sh "
        "(after install_cuda_torch.sh). On Mac: bash cluster/setup_mac_env.sh "
        "or pip install pyg-lib torch-cluster -f "
        "https://data.pyg.org/whl/torch-<version>+cpu.html"
    )


class _RadiusInteractionGraphFallback(nn.Module):
    """SchNet radius graph when pyg-lib is missing (uses torch-cluster instead)."""

    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos, batch):
        from torch_cluster import radius_graph

        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


def _patch_schnet_radius_backend(
    net: PyGSchNet,
    cutoff: float,
    max_num_neighbors: int,
) -> None:
    """PyG 2.6+ SchNet calls radius_graph via pyg-lib; fall back to torch-cluster."""
    import torch_geometric.typing as pyg_typing

    if getattr(pyg_typing, "WITH_RADIUS", False):
        return
    try:
        from torch_cluster import radius_graph  # noqa: F401
    except ImportError as exc:
        raise ImportError(_schnet_install_hint()) from exc
    net.interaction_graph = _RadiusInteractionGraphFallback(cutoff, max_num_neighbors)


class SchNet3D(nn.Module):
    """SchNet on atomic numbers + xyz coordinates (radius graph from pos)."""

    def __init__(
        self,
        out_channels: int = 1,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 4,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        readout: str = "add",
    ):
        super().__init__()
        self.net = PyGSchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
        )
        _patch_schnet_radius_backend(self.net, cutoff=cutoff, max_num_neighbors=max_num_neighbors)
        self.net.lin2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, z, pos, batch):
        return self.net(z, pos, batch)


def build_schnet(
    out_channels: int,
    hidden_channels: int = 128,
    num_filters: int = 128,
    num_interactions: int = 4,
    num_gaussians: int = 50,
    cutoff: float = 10.0,
) -> SchNet3D:
    return SchNet3D(
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        num_filters=num_filters,
        num_interactions=num_interactions,
        num_gaussians=num_gaussians,
        cutoff=cutoff,
    )


def ensure_atomic_numbers(graph: Data) -> Data:
    """Attach data.z (atomic numbers) for SchNet; decode from x if missing."""
    if getattr(graph, "z", None) is not None and graph.z.numel() > 0:
        return graph
    n = len(ATOM_FEATURES["atomic_num"]) + 1
    idx = graph.x[:, :n].argmax(dim=1)
    nums = ATOM_FEATURES["atomic_num"]
    z = []
    for i in idx.tolist():
        z.append(nums[i] if i < len(nums) else 0)
    graph.z = torch.tensor(z, dtype=torch.long, device=graph.x.device)
    return graph


def ensure_atomic_numbers_list(graphs: List[Data]) -> List[Data]:
    return [ensure_atomic_numbers(g) for g in graphs]


def rehydrate_pyg_graph(graph: Data) -> Data:
    """Rebuild a PyG Data object (stable batching after PyG upgrades / kernel reloads)."""
    return Data.from_dict(graph.to_dict())


def rehydrate_pyg_graphs(graphs: List[Data]) -> List[Data]:
    return [rehydrate_pyg_graph(g) for g in graphs]


def prepare_schnet_graphs(graphs: List[Data]) -> List[Data]:
    """Attach atomic numbers and rebuild graphs for SchNet DataLoader batching."""
    return rehydrate_pyg_graphs(ensure_atomic_numbers_list(graphs))


class SmilesLSTM(nn.Module):
    """
    Bidirectional LSTM that takes raw tokenised SMILES as input.

    Pipeline: Embedding → BiLSTM → masked mean pool → Linear head

    Why this architecture?
    ----------------------
    - Embedding layer: maps each character token to a dense vector, letting
      the model learn that e.g. 'C' and 'c' are chemically related.
    - Bidirectional LSTM: reads the SMILES both left-to-right and right-to-left,
      so each position has full context of the whole string.
    - Masked mean pooling: averages LSTM outputs over non-padding positions,
      giving a fixed-size molecule embedding regardless of SMILES length.
    - Linear head: maps the molecule embedding to the target property.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                 n_layers=2, out_channels=1, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                                 batch_first=True, bidirectional=bidirectional,
                                 dropout=dropout if n_layers > 1 else 0.0)
        lstm_out = hidden_dim * (2 if bidirectional else 1)
        self.dropout   = nn.Dropout(dropout)
        self.lin       = nn.Linear(lstm_out, out_channels)

    def forward(self, x):
        emb        = self.embedding(x)                        # (B, L, E)
        out, _     = self.lstm(emb)                           # (B, L, H)
        mask       = (x != 0).unsqueeze(-1).float()           # (B, L, 1)
        pooled     = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.lin(self.dropout(pooled))                 # (B, out_channels)


# =============================================================================
# 6. LSTM TRAINING UTILITIES
# =============================================================================

def train_epoch_lstm(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(X_b)
        
        # mask NaN labels for multi-label datasets like Tox21
        mask = ~torch.isnan(y_b)
        if mask.sum() == 0:
            continue
        loss = loss_fn(out[mask], y_b[mask])
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(X_b)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_lstm(model, loader, device, task_type: str) -> Dict:
    model.eval()
    preds, labels = [], []
    for X_b, y_b in loader:
        preds.append(model(X_b.to(device)).cpu())
        labels.append(y_b)
    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return _compute_metrics(preds, labels, task_type)


def train_lstm(
    model: SmilesLSTM,
    train_dataset: SMILESDataset,
    val_dataset:   SMILESDataset,
    task_type: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    patience: Optional[int] = None,
    weight_decay: float = 1e-4,
) -> Dict:
    """Full training loop for the SMILES LSTM."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = TorchLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = TorchLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn      = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_metrics": []}
    best_state, best_metric = None, float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss  = train_epoch_lstm(model, train_loader, optimizer, loss_fn, device)
        val_metrics = eval_lstm(model, val_loader, device, task_type)
        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)

        monitor = val_metrics.get("RMSE", -val_metrics.get("ROC-AUC", val_metrics.get("Mean ROC-AUC", 0)))
        if monitor < best_metric:
            best_metric = monitor
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step(monitor)

        if verbose and epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val={val_metrics}")
        elif patience is not None and (epoch == 1 or epoch % 5 == 0):
            logger.info(f"HPO epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | val={val_metrics}")

        if patience is not None and stale_epochs >= patience:
            logger.info(f"Early stop at epoch {epoch} (patience={patience})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return history


# =============================================================================
# 7. GNN TRAINING UTILITIES
# =============================================================================

def _compute_metrics(preds, labels, task_type: str) -> Dict:
    """Shared metric computation for GNN / SchNet / LSTM eval.

    Always converts to NumPy first. Torch tensors + Tox21 NaN labels break
    sklearn ROC-AUC if passed through without conversion (masking fails).
    """
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    if task_type == "regression":
        return {
            "RMSE": float(np.sqrt(mean_squared_error(labels, preds))),
            "MAE":  float(mean_absolute_error(labels, preds)),
            "R2":   float(r2_score(labels, preds)),
        }
    else:
        probs = 1 / (1 + np.exp(-np.clip(preds, -50, 50)))
        aucs, prcs = [], []
        n_tasks = labels.shape[1] if labels.ndim > 1 else 1
        for t in range(n_tasks):
            y_t = labels[:, t] if labels.ndim > 1 else labels.ravel()
            p_t = probs[:, t]  if probs.ndim  > 1 else probs.ravel()
            mask = np.isfinite(y_t) & np.isfinite(p_t)
            if mask.sum() == 0:
                continue
            y_m, p_m = y_t[mask], p_t[mask]
            if len(np.unique(y_m)) > 1:
                aucs.append(roc_auc_score(y_m, p_m))
                prcs.append(average_precision_score(y_m, p_m))
        return {
            "ROC-AUC": float(np.mean(aucs)) if aucs else float("nan"),
            "PRC-AUC": float(np.mean(prcs)) if prcs else float("nan"),
        }


def _gnn_forward(model, batch):
    if getattr(model, "is_dmpnn", False):
        return model(batch)
    if getattr(model, "uses_edge_attr", False):
        return model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    return model(batch.x, batch.edge_index, batch.batch)


def train_epoch_gnn(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = _gnn_forward(model, batch)
        
        # mask NaN labels (important for multi-label datasets like Tox21)
        mask = ~torch.isnan(batch.y)
        if mask.sum() == 0:
            continue
        loss = loss_fn(out[mask], batch.y[mask])
        
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def eval_gnn(model, loader, device, task_type: str) -> Dict:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(_gnn_forward(model, batch).cpu())
        labels.append(batch.y.cpu())
    return _compute_metrics(torch.cat(preds).numpy(), torch.cat(labels).numpy(), task_type)


def train_gnn(
    model,
    train_graphs: List[Data],
    val_graphs:   List[Data],
    task_type: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    patience: Optional[int] = None,
    weight_decay: float = 1e-4,
) -> Dict:
    """Full training loop for GCN / GIN / D-MPNN.

    patience: if set, stop when validation metric fails to improve for this many
    epochs (used during Optuna HPO). Final refit should use patience=None.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = PyGLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = PyGLoader(val_graphs,   batch_size=batch_size, shuffle=False)
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn      = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_metrics": []}
    best_state, best_metric = None, float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss  = train_epoch_gnn(model, train_loader, optimizer, loss_fn, device)
        val_metrics = eval_gnn(model, val_loader, device, task_type)
        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)

        monitor = val_metrics.get("RMSE", -val_metrics.get("ROC-AUC", val_metrics.get("Mean ROC-AUC", 0)))
        if monitor < best_metric:
            best_metric = monitor
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step(monitor)

        if verbose and epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val={val_metrics}")
        elif patience is not None and (epoch == 1 or epoch % 5 == 0):
            logger.info(f"HPO epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | val={val_metrics}")

        if patience is not None and stale_epochs >= patience:
            logger.info(f"Early stop at epoch {epoch} (patience={patience})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return history


def train_epoch_schnet(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.z, batch.pos, batch.batch)
        y = batch.y
        if y.dim() == 1:
            y = y.view(-1, 1)
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        loss = loss_fn(out[mask], y[mask])
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * mask.sum().item()
        n += mask.sum().item()
    return total / max(n, 1)


def eval_schnet(model, loader, device, task_type: str) -> Dict:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if not hasattr(batch, "z") or batch.z is None:
                batch = ensure_atomic_numbers(batch)
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch)
            preds.append(out.cpu())
            y = batch.y.cpu()
            # Tox21 stores y as (1, n_tasks) per graph → batch (B, n_tasks)
            if y.dim() == 1 and out.size(-1) > 1:
                y = y.view(-1, out.size(-1))
            labels.append(y)
    return _compute_metrics(torch.cat(preds), torch.cat(labels), task_type)


def train_schnet(
    model,
    train_graphs: List[Data],
    val_graphs: List[Data],
    task_type: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    patience: Optional[int] = None,
    weight_decay: float = 1e-4,
) -> Dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_graphs = prepare_schnet_graphs(train_graphs)
    val_graphs = prepare_schnet_graphs(val_graphs)
    model = model.to(device)
    train_loader = PyGLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = PyGLoader(val_graphs, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_metrics": []}
    best_state, best_metric = None, float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_schnet(model, train_loader, optimizer, loss_fn, device)
        val_metrics = eval_schnet(model, val_loader, device, task_type)
        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)
        monitor = val_metrics.get("RMSE", -val_metrics.get("ROC-AUC", val_metrics.get("Mean ROC-AUC", 0)))
        if monitor < best_metric:
            best_metric = monitor
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step(monitor)
        if verbose and epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val={val_metrics}")
        elif patience is not None and (epoch == 1 or epoch % 5 == 0):
            logger.info(f"HPO epoch {epoch:3d}/{epochs} | loss={train_loss:.4f} | val={val_metrics}")
        if patience is not None and stale_epochs >= patience:
            logger.info(f"Early stop at epoch {epoch} (patience={patience})")
            break
    if best_state:
        model.load_state_dict(best_state)
    return history


# =============================================================================
# 8. CROSS-VALIDATION HELPERS
# =============================================================================

def cv_sklearn(pipeline, X, y, task_type, n_splits=5, random_state=42) -> Dict:
    if task_type == "regression":
        cv      = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = {"RMSE": "neg_root_mean_squared_error",
                   "MAE":  "neg_mean_absolute_error", "R2": "r2"}
        res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {"RMSE": (-res["test_RMSE"].mean(), res["test_RMSE"].std()),
                "MAE":  (-res["test_MAE"].mean(),  res["test_MAE"].std()),
                "R2":   ( res["test_R2"].mean(),   res["test_R2"].std())}
    else:
        cv      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = {"ROC-AUC": "roc_auc", "PRC-AUC": "average_precision"}
        res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {
            "ROC-AUC": (res["test_ROC-AUC"].mean(), res["test_ROC-AUC"].std()),
            "PRC-AUC": (res["test_PRC-AUC"].mean(), res["test_PRC-AUC"].std()),
        }


def evaluate_sklearn_test(
    pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
) -> Dict[str, Tuple[float, float]]:
    """Fit on train, report point estimates on scaffold-held-out test set."""
    import warnings
    from sklearn.pipeline import Pipeline

    if task_type == "multilabel_classification" and np.isnan(y_train).any():
        builder = pipeline.named_steps.get("model")
        if builder is None:
            raise ValueError("multilabel NaN targets require a Pipeline with a 'model' step")
        aucs, prcs = [], []
        pre_steps = [(n, s) for n, s in pipeline.steps if n != "model"]
        for t in range(y_train.shape[1]):
            tr_mask = ~np.isnan(y_train[:, t])
            te_mask = ~np.isnan(y_test[:, t])
            if tr_mask.sum() < 10 or te_mask.sum() < 5:
                continue
            if len(np.unique(y_train[tr_mask, t])) < 2 or len(np.unique(y_test[te_mask, t])) < 2:
                continue
            task_pipe = Pipeline(pre_steps + [("model", type(builder)(**builder.get_params()))])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                task_pipe.fit(X_train[tr_mask], y_train[tr_mask, t])
            probs = task_pipe.predict_proba(X_test[te_mask])[:, 1]
            aucs.append(roc_auc_score(y_test[te_mask, t], probs))
            prcs.append(average_precision_score(y_test[te_mask, t], probs))
        return {
            "ROC-AUC": (float(np.mean(aucs)) if aucs else float("nan"), 0.0),
            "PRC-AUC": (float(np.mean(prcs)) if prcs else float("nan"), 0.0),
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    if task_type == "regression":
        preds = pipeline.predict(X_test)
        return {
            "RMSE": (float(np.sqrt(mean_squared_error(y_test, preds))), 0.0),
            "MAE":  (float(mean_absolute_error(y_test, preds)), 0.0),
            "R2":   (float(r2_score(y_test, preds)), 0.0),
        }

    if task_type == "multilabel_classification":
        probs = pipeline.predict_proba(X_test)
        if isinstance(probs, list):
            probs = np.column_stack([p[:, 1] for p in probs])
        aucs, prcs = [], []
        for t in range(y_test.shape[1]):
            mask = ~np.isnan(y_test[:, t])
            if mask.sum() == 0 or len(np.unique(y_test[mask, t])) < 2:
                continue
            aucs.append(roc_auc_score(y_test[mask, t], probs[mask, t]))
            prcs.append(average_precision_score(y_test[mask, t], probs[mask, t]))
        return {
            "ROC-AUC": (float(np.mean(aucs)) if aucs else float("nan"), 0.0),
            "PRC-AUC": (float(np.mean(prcs)) if prcs else float("nan"), 0.0),
        }

    probs = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else pipeline.decision_function(X_test)
    )
    return {
        "ROC-AUC": (float(roc_auc_score(y_test, probs)), 0.0),
        "PRC-AUC": (float(average_precision_score(y_test, probs)), 0.0),
    }


# =============================================================================
# 9. RESULTS & UTILITIES
# =============================================================================

def format_results_table(results: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        row = {"Model": name}
        for metric, (mean, std) in metrics.items():
            row[metric] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"Using device: cuda:0 ({name})")
        return torch.device("cuda")
    logger.info("Using device: cpu (request a GPU node on the cluster for large GNN/LSTM jobs)")
    return torch.device("cpu")


def reset_dataset_run(
    slug: str,
    results_dir: Union[str, Path] = "./results",
    *,
    results_root: Optional[Union[str, Path]] = None,
    split_mode: str = "scaffold",
) -> None:
    """Delete partial metrics, histories, plots, and shared splits for a fresh run."""
    results_dir = Path(results_dir)
    results_root = Path(results_root or results_dir.parent)
    history_dir = results_dir / "histories"
    hpo_dir = results_dir / "hpo" / slug
    targets = [
        results_dir / f"{slug}_partial.json",
        results_dir / f"{slug}_results.csv",
        results_root / "splits" / f"{slug}_{split_mode}_splits.pkl",
    ]
    for path in targets:
        if path.exists():
            path.unlink()
            logger.info(f"Removed {path}")
    if history_dir.exists():
        for path in history_dir.glob(f"{slug}_*.json"):
            path.unlink()
            logger.info(f"Removed {path}")
    for path in results_dir.glob(f"{slug}_*.png"):
        path.unlink()
        logger.info(f"Removed {path}")
    if hpo_dir.exists():
        for path in hpo_dir.glob("*_hpo.json"):
            path.unlink()
            logger.info(f"Removed {path}")
        if not any(hpo_dir.iterdir()):
            hpo_dir.rmdir()

def save_history(history: Dict, path: str):
    """Save training history to JSON."""
    # convert numpy floats to Python floats for JSON serialisation
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        "train_loss": [float(x) for x in history["train_loss"]],
        "val_metrics": [{k: float(v) for k, v in m.items()} 
                        for m in history["val_metrics"]],
    }
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)

def load_history(path: str) -> Dict:
    """Load training history from JSON."""
    with open(path, "r") as f:
        return json.load(f)


MODEL_ORDER = [
    "Morgan + RF", "Morgan + SVM", "Morgan + XGBoost",
    "Descriptors + RF", "Descriptors + SVM", "Descriptors + XGBoost",
    "Combined + XGBoost",
    "GIN (2D)", "D-MPNN (2D)",
    "GIN (3D)",
    "SchNet (3D coords)",
    "SMILES LSTM",
]

CPU_MODELS = MODEL_ORDER[:7]
GPU_MODELS = MODEL_ORDER[7:]


def _metrics_to_json(metrics: Dict) -> Dict:
    return {k: [float(v[0]), float(v[1])] for k, v in metrics.items()}


def _metrics_from_json(data: Dict) -> Dict:
    return {k: (float(v[0]), float(v[1])) for k, v in data.items()}


def load_partial_results(path: Union[str, Path]) -> Dict[str, Dict]:
    """Load incrementally saved model metrics (resume after interrupted runs)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    return {model: _metrics_from_json(metrics) for model, metrics in raw.items()}


def save_partial_results(results: Dict[str, Dict], path: Union[str, Path]) -> None:
    """Persist metrics after each model so training can be resumed cell-by-cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {model: _metrics_to_json(metrics) for model, metrics in results.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def save_model_result(
    results: Dict[str, Dict],
    model_name: str,
    metrics: Dict,
    partial_path: Union[str, Path],
    history: Optional[Dict] = None,
    history_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict]:
    """Update partial results and optionally write a training history JSON."""
    results[model_name] = metrics
    save_partial_results(results, partial_path)
    if history is not None and history_path is not None:
        save_history(history, history_path)
    logger.info(f"Saved {model_name} → {partial_path}")
    return results


def order_results(results: Dict[str, Dict]) -> Dict[str, Dict]:
    return {m: results[m] for m in MODEL_ORDER if m in results}


def finalize_dataset_results(
    slug: str,
    partial_path: Union[str, Path],
    results_dir: Union[str, Path],
    task_type: str,
    plot_title: str,
    ylabel: str,
    primary_metric: Optional[str] = None,
    minimize: bool = True,
) -> pd.DataFrame:
    """
    Load all incrementally saved metrics, write the final CSV, and plot comparison.
    Run this once after all model cells have finished (or partially finished).
    """
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    results = order_results(load_partial_results(partial_path))
    if not results:
        logger.warning(f"No results found in {partial_path}")
        return pd.DataFrame()

    primary_metric = primary_metric or ("RMSE" if task_type == "regression" else "ROC-AUC")
    df = format_results_table(results)
    csv_path = results_dir / f"{slug}_results.csv"
    df.to_csv(csv_path)

    models = list(results.keys())
    vals = [results[m][primary_metric][0] for m in models]
    stds = [results[m][primary_metric][1] for m in models]
    best_idx = int(np.argmin(vals) if minimize else np.argmax(vals))

    fig, ax = plt.subplots(figsize=(13, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, vals, yerr=stds, color=colors, capsize=5, edgecolor="white")
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(2.5)
    if task_type == "classification":
        ax.axhline(0.5, color="grey", ls="--", lw=1, label="Random baseline")
        ax.legend()
        ax.set_ylim(0.4, 1.05)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if minimize else 0.005),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(plot_title, fontsize=13)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plot_suffix = "rmse" if minimize else "auc"
    plot_path = results_dir / f"{slug}_{plot_suffix}_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"=== {slug.upper()} Results ({len(results)}/{len(MODEL_ORDER)} models) ===")
    print(df.to_string())
    print(f"\nFinal CSV : {csv_path}")
    print(f"Plot      : {plot_path}")
    print(f"Partial   : {partial_path}")
    return df


def merge_combined_results(
    slug: str,
    cpu_partial: Union[str, Path],
    gpu_partial: Union[str, Path],
    combined_dir: Union[str, Path],
    task_type: str,
    plot_title: str,
    ylabel: str,
    primary_metric: Optional[str] = None,
    minimize: bool = True,
) -> pd.DataFrame:
    """Merge CPU + GPU partial JSON files and write combined CSV/plot."""
    combined_dir = Path(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)
    merged = {}
    for path in (cpu_partial, gpu_partial):
        merged.update(load_partial_results(path))
    combined_path = combined_dir / f"{slug}_partial.json"
    save_partial_results(merged, combined_path)
    return finalize_dataset_results(
        slug, combined_path, combined_dir, task_type,
        plot_title, ylabel, primary_metric, minimize,
    )


# =============================================================================
# 10. FEATURE DIMENSION CONSTANTS
# =============================================================================

ATOM_FEATURE_DIM    = _ATOM_DIM       # 163 — node features (2D and 3D graphs)
BOND_FEATURE_DIM    = _BOND_DIM       # 11  — edge features (2D graph)
BOND_FEATURE_DIM_3D = _BOND_DIM_3D   # 13  — edge features (3D graph: +dist +angle)