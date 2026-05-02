"""exp148 — Per-class normalisation + rank blend (hengck23-suggested layer).

Test variants:
  1. Per-class z-score normalize each model's scores BEFORE blend
  2. Per-class rank normalize (uniform [0,1]) BEFORE blend
  3. Cross-model agreement: (Perch + exp50 + AudioMAE)/3 with rank uniform
  4. Mixed: z-score Perch+exp50 then add MLP probe blend
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import DATA, N_CLS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all_lbl = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
mask_tr = (ss_all_lbl.split == "train").values
mask_ev = (ss_all_lbl.split == "eval").values

Y_ev = np.zeros((mask_ev.sum(), N_CLS), dtype=np.float32)
for i, idx in enumerate(np.where(mask_ev)[0]):
    for lbl in ss_all_lbl.iloc[idx].lbls:
        if lbl in l2i: Y_ev[i, l2i[lbl]] = 1.0

# Load teachers
perch = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/perch_prob_labeled.npz")["prob"]
exp50 = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/exp50_scores_labeled.npz")["scores"]
all_probes = np.load(ROOT/"experiments/_data_pipelines/exp146_outputs/all_probes.npz")
M5 = np.zeros_like(perch); M5[mask_ev] = all_probes["M5"]
M3 = np.zeros_like(perch); M3[mask_ev] = all_probes["M3"]

tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
def eval_macro(preds_ev, name=""):
    aucs = []
    by_tax = {t: [] for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
    for c in range(N_CLS):
        yc = Y_ev[:, c]
        if 0 < yc.sum() < len(yc):
            pc = preds_ev[:, c]
            if pc.std() > 1e-9:
                a = roc_auc_score(yc, pc)
                aucs.append(a)
                t = tax.get(PRIMARY_LABELS[c], "Aves")
                if t in by_tax: by_tax[t].append(a)
    print(f"  [{name:55s}] {len(aucs)} cls macro={np.mean(aucs):.4f}", end="")
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        if by_tax[t]: print(f" {t[:3]}={np.mean(by_tax[t]):.3f}", end="")
    print()
    return np.mean(aucs)


def per_class_zscore(preds, mean_train, std_train):
    """Subtract per-class train mean, divide std. Return z-scores."""
    return (preds - mean_train) / (std_train + 1e-6)

def per_class_rank(preds_train, preds_eval):
    """Compute per-class rank in train, then map eval to rank percentile."""
    n_tr, K = preds_train.shape
    out = np.zeros_like(preds_eval)
    for c in range(K):
        # CDF estimated from train
        sorted_tr = np.sort(preds_train[:, c])
        for i, v in enumerate(preds_eval[:, c]):
            # Find position in sorted train
            pos = np.searchsorted(sorted_tr, v) / n_tr
            out[i, c] = pos
    return out


# v33 base
v33_base = 0.7 * perch + 0.3 * exp50

# Per-class train statistics
P_tr_mean = perch[mask_tr].mean(0); P_tr_std = perch[mask_tr].std(0)
E_tr_mean = exp50[mask_tr].mean(0); E_tr_std = exp50[mask_tr].std(0)

# Variant 1: z-score blend
P_z_ev = per_class_zscore(perch[mask_ev], P_tr_mean, P_tr_std)
E_z_ev = per_class_zscore(exp50[mask_ev], E_tr_mean, E_tr_std)
v33_zblend = 0.7 * P_z_ev + 0.3 * E_z_ev

# Variant 2: rank blend
print("Computing rank-percentile per-class (slow)...")
P_rank_ev = per_class_rank(perch[mask_tr], perch[mask_ev])
E_rank_ev = per_class_rank(exp50[mask_tr], exp50[mask_ev])
v33_rblend = 0.7 * P_rank_ev + 0.3 * E_rank_ev

# Baseline
print("\n=== Baseline ===")
eval_macro(v33_base[mask_ev], "v33 raw blend (current)")
print()

print("=== Per-class normalisation variants ===")
eval_macro(v33_zblend, "v33 = 0.7 P_z + 0.3 E_z (per-class z-score)")
eval_macro(v33_rblend, "v33 = 0.7 P_rank + 0.3 E_rank (per-class rank)")
print()

# Try also weighted variants (might Perch dominance be wrong)
for w_p in [0.5, 0.6, 0.7, 0.8]:
    bw = w_p * P_z_ev + (1-w_p) * E_z_ev
    eval_macro(bw, f"z-blend W_P={w_p}")
print()
for w_p in [0.5, 0.6, 0.7, 0.8]:
    bw = w_p * P_rank_ev + (1-w_p) * E_rank_ev
    eval_macro(bw, f"rank-blend W_P={w_p}")

# Add MLP probe
print("\n=== With MLP M5 probe added ===")
M5_tr_proxy = np.zeros((mask_tr.sum(), N_CLS))  # don't have M5 on train
# Approximate: assume M5 train would be similar to M5 eval; use eval stats
M5_z_ev = per_class_zscore(M5[mask_ev], M5[mask_ev].mean(0), M5[mask_ev].std(0))
for am_w in [0.05, 0.10, 0.15, 0.20]:
    ble = (1 - am_w) * v33_zblend + am_w * M5_z_ev
    eval_macro(ble, f"v33-z + {am_w}·M5_z")

# Try per-row max blend (envelope)
print("\n=== Per-row max strategies ===")
v33_max = np.maximum(0.7*perch[mask_ev], 0.3*exp50[mask_ev])
eval_macro(v33_max, "max(0.7P, 0.3E)")

v33_envelope = np.maximum(perch[mask_ev], exp50[mask_ev])
eval_macro(v33_envelope, "max(P, E)")

# Add AudioMAE rank to v33 base
print("\n=== AudioMAE M5 dose sweep on v33 raw blend ===")
for am_w in [0.025, 0.05, 0.075, 0.10]:
    ble = (1-am_w)*v33_base[mask_ev] + am_w*M5[mask_ev]
    eval_macro(ble, f"v33 + {am_w}·M5 raw")
