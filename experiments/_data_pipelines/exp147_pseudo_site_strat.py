"""exp147 — Site-stratified pseudo build.

v3 pseudo (351k entries) → exp136b → LB -0.018. Failure mode: contains
single-site sonotype (16/25 sonos) and single-site bird patterns; SED
fine-tune memorizes those site fingerprints.

New pseudo filter:
  - For each (file_id, window, class) candidate from v33 ensemble (Perch+exp50+exp59 agreement),
    only keep if the SAME class is also predicted high in another DIFFERENT site
  - Effectively requires "this class generalizes across sites" before pseudo

Output: pseudo_v8_site_strat.csv with multi-site-only entries.

This addresses the multi-site-failure dead-zone (24321/22967/22973 etc) by
giving SED extra labeled-style positives across sites WITHOUT introducing
single-site fingerprint amplification.
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import re
import warnings; warnings.filterwarnings("ignore")

from experiments._data_pipelines._shared.data import build_primaries
from experiments._data_pipelines._shared.constants import DATA, N_CLS

PRIMARY_LABELS, l2i = build_primaries()

# Load v33 unlabeled scores (precomputed in exp126)
print("[1/4] Loading v33 unlabeled scores")
v33_path = ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz"
if not v33_path.exists():
    print(f"  v33 unlabeled scores not found at {v33_path}")
    print(f"  Available files in exp126_outputs:")
    for p in (ROOT / "experiments/_data_pipelines/exp126_outputs").iterdir():
        print(f"    {p.name}")
    sys.exit(1)

z = np.load(v33_path, allow_pickle=True)
print(f"  keys: {list(z.keys())}")
for k in z.keys():
    obj = z[k]
    print(f"    {k}: shape={obj.shape if hasattr(obj, 'shape') else 'N/A'}, dtype={obj.dtype if hasattr(obj, 'dtype') else type(obj).__name__}")
