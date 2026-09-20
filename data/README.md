# Data

MoleculeNet CSVs and precomputed conformers are not shipped in this repo
(they are large and already public elsewhere).

Expected layout after setup:

```
data/
  raw/                 # MoleculeNet CSVs (esol, freesolv, ...)
  conformers/          # optional ETKDG/MMFF pickles used by GIN-3D / SchNet
```

The notebooks and `conformer_generation.py` will create what they need if the
raw CSVs are present. For MoleculeNet downloads see the DeepChem / MoleculeNet
documentation, or point `DATA_ROOT` at an existing local cache.
