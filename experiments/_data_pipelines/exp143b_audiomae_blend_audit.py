"""exp143b — Local audit of v33 + α·AudioMAE blend on labeled SS eval (122 rows).

Uses precomputed Perch probs and exp50 SED scores from exp80_outputs to
simulate v33 = 0.7 P + 0.3 exp50 (without V9/Gauss/filemax post-processing,
so it's actually slightly below true v33 performance — but the *delta*
from blending in AudioMAE should still indicate transferability).
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import DATA, N_CLS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
print(f"ss_all: {len(ss_all)}  train={len(ss_train_g)}  eval={len(ss_eval_g)}")

# Load precomputed
perch = np.load("/data/birdclef2026/experiments/_audits_post_v26/exp80_outputs/perch_prob_labeled.npz")["prob"]
exp50 = np.load("/data/birdclef2026/experiments/_audits_post_v26/exp80_outputs/exp50_scores_labeled.npz")["scores"]
print(f"perch: {perch.shape}, exp50: {exp50.shape}")
assert perch.shape == (len(ss_all), N_CLS)
assert exp50.shape == (len(ss_all), N_CLS)

# Load AudioMAE embeddings
audiomae = np.load("/data/birdclef2026/experiments/_data_pipelines/exp143_outputs/audiomae_embs_labeled_ss.npz")
am_embs = audiomae["embs"]
am_split = audiomae["splits"]
am_row_ids = audiomae["row_ids"]
print(f"audiomae embs: {am_embs.shape}")

# Verify row_ids correspond
expected_ids = ss_all["row_id"].values
assert (am_row_ids == expected_ids).all(), "row_id mismatch — different ordering!"
print("row_ids match")

# Build label matrix Y
Y = np.zeros((len(ss_all), N_CLS), dtype=np.float32)
for i, lbls in enumerate(ss_all.lbls):
    for lbl in lbls:
        if lbl in l2i:
            Y[i, l2i[lbl]] = 1.0
print(f"Y total positives: {Y.sum():.0f}")

mask_tr = (am_split == "train")
mask_ev = (am_split == "eval")
print(f"train: {mask_tr.sum()}  eval: {mask_ev.sum()}")

# Train per-class LR probe on AudioMAE (matching exp143 — PCA32 + LR C=0.25)
mu = am_embs[mask_tr].mean(0); sd = am_embs[mask_tr].std(0) + 1e-6
Xn_tr = (am_embs[mask_tr] - mu) / sd
Xn_ev = (am_embs[mask_ev] - mu) / sd
pca = PCA(n_components=32, random_state=0).fit(Xn_tr)
Z_tr = pca.transform(Xn_tr); Z_ev = pca.transform(Xn_ev)
am_preds_ev = np.zeros((mask_ev.sum(), N_CLS), dtype=np.float32)
for c in range(N_CLS):
    yc = Y[mask_tr, c]
    if yc.sum() < 1 or yc.sum() == len(yc):
        am_preds_ev[:, c] = float(yc.mean())
        continue
    try:
        clf = LogisticRegression(C=0.25, max_iter=200, solver="liblinear").fit(Z_tr, yc)
        am_preds_ev[:, c] = clf.predict_proba(Z_ev)[:, 1]
    except Exception:
        am_preds_ev[:, c] = 0.0

# Slice Perch / exp50 to eval
perch_ev = perch[mask_ev]
exp50_ev = exp50[mask_ev]
Y_ev = Y[mask_ev]

# Compute v33 base: 0.7 * perch + 0.3 * exp50 (skipping V9/Gauss/filemax)
v33_simple = 0.7 * perch_ev + 0.3 * exp50_ev


def eval_macro(scores, name=""):
    aucs, taxa = [], {t: [] for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
    tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
    for c in range(N_CLS):
        yc = Y_ev[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc): continue
        sc = scores[:, c]
        if sc.std() < 1e-9: continue
        a = roc_auc_score(yc, sc)
        aucs.append(a)
        t = tax.get(PRIMARY_LABELS[c], "Aves")
        if t in taxa: taxa[t].append(a)
    print(f"\n[{name}] {len(aucs)} cls macro AUC = {np.mean(aucs):.4f}")
    for tn, lst in taxa.items():
        if lst: print(f"   {tn:10s} {len(lst):2d} cls = {np.mean(lst):.4f}")
    return np.mean(aucs), aucs, taxa


print("\n=== Baselines ===")
m_perch, _, _ = eval_macro(perch_ev, "Perch alone")
m_exp50, _, _ = eval_macro(exp50_ev, "exp50 alone")
m_audiomae, _, _ = eval_macro(am_preds_ev, "AudioMAE-probe alone")
m_v33, v33_aucs, v33_taxa = eval_macro(v33_simple, "v33 simple (0.7P+0.3 exp50)")

# Blend AudioMAE: v33_simple + α·AudioMAE
print("\n=== AudioMAE blend sweep ===")
for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    blend = (1 - alpha) * v33_simple + alpha * am_preds_ev
    m, aucs, taxa = eval_macro(blend, f"v33 + {alpha}·AudioMAE uniform")
    sp = np.mean([spearmanr(v33_simple[:, c], blend[:, c]).correlation for c in range(N_CLS) if v33_simple[:, c].std() > 1e-9 and blend[:, c].std() > 1e-9])
    print(f"   sp_row(v33→blend): {sp:.4f}, Δmacro: {m-m_v33:+.4f}")

# Apply Path 1 mask (non-Aves only)
print("\n=== AudioMAE NON-Aves freeze (Aves columns untouched) ===")
tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
is_aves = np.array([tax.get(p, 'Aves') == 'Aves' for p in PRIMARY_LABELS])
print(f"   Aves: {is_aves.sum()}, non-Aves: {(~is_aves).sum()}")
for alpha in [0.10, 0.20, 0.30]:
    w_per_class = np.where(is_aves, 0.0, alpha).astype(np.float32)
    blend = (1 - w_per_class[None, :]) * v33_simple + w_per_class[None, :] * am_preds_ev
    m, _, _ = eval_macro(blend, f"v33 + {alpha}·AudioMAE non-Aves only")
    print(f"   Δmacro: {m-m_v33:+.4f}")
