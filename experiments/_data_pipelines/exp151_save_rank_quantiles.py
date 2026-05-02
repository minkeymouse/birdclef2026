"""exp151 — Save per-class score quantiles for rank-blend + Kaggle integration.

For each (model, class), compute 100 quantile values from labeled SS train
distribution. Save as small npz. At inference, rank transform = searchsorted.
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import N_CLS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
import pandas as pd
ss_all_lbl = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
mask_tr = (ss_all_lbl.split == "train").values

perch = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/perch_prob_labeled.npz")["prob"]
exp50 = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/exp50_scores_labeled.npz")["scores"]

P_train = perch[mask_tr].astype(np.float32)  # (617, 234)
E_train = exp50[mask_tr].astype(np.float32)

# Save sorted train scores per class (for searchsorted-based rank transform)
P_sorted = np.sort(P_train, axis=0)  # (617, 234) sorted ascending per class
E_sorted = np.sort(E_train, axis=0)

# Also save mean/std for z-score variant
P_mean = P_train.mean(0); P_std = P_train.std(0)
E_mean = E_train.mean(0); E_std = E_train.std(0)

OUT = ROOT / "model-weights" / "rank_blend_quantiles.npz"
np.savez(OUT,
         P_sorted=P_sorted, E_sorted=E_sorted,
         P_mean=P_mean, P_std=P_std,
         E_mean=E_mean, E_std=E_std)
print(f"Saved {OUT} ({OUT.stat().st_size/1e3:.1f} KB)")
