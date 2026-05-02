"""exp143e — AudioMAE probe architecture sweep + external down-weighting.

Tests:
  C1. PCA32 + LR C=0.25 (original, baseline)
  C2. PCA64 + LR (more variance retained)
  C3. No PCA + LR (full 768-d, may overfit on small train set)
  C4. PCA32 + LR + external 0.3× weight (downweight focal mode)
  C5. PCA32 + LR + external 0.1× weight
  C6. PCA32 + 2-layer MLP probe (768→128→234)

Goal: find probe that beats SS-only 0.8062 macro on labeled SS eval (40 cls).
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import DATA, N_CLS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)

OUT = ROOT / "experiments" / "_data_pipelines" / "exp143_outputs"
ss = np.load(OUT / "audiomae_embs_labeled_ss.npz")
ss_embs = ss["embs"]
ss_split = ss["splits"]

ext = np.load(OUT / "audiomae_external_embs.npz")
ext_embs = ext["embs"]
ext_species = ext["species"]
print(f"SS train: {(ss_split=='train').sum()}, ext: {len(ext_embs)}")

mask_tr = ss_split == "train"
mask_ev = ss_split == "eval"

# Y matrices
def build_Y(rows, lbls_col):
    Y = np.zeros((len(rows), N_CLS), dtype=np.float32)
    for i, lbls in enumerate(rows[lbls_col]):
        for lbl in lbls:
            if lbl in l2i:
                Y[i, l2i[lbl]] = 1.0
    return Y

Y_ss_tr = build_Y(ss_all[mask_tr], "lbls")
Y_ev = build_Y(ss_all[mask_ev], "lbls")

Y_ext = np.zeros((len(ext_embs), N_CLS), dtype=np.float32)
for i, sp in enumerate(ext_species):
    Y_ext[i, l2i[str(sp)]] = 1.0

X_ss = ss_embs[mask_tr]
X_ev = ss_embs[mask_ev]


def eval_macro(preds, name=""):
    aucs = []
    taxa_aucs = {t: [] for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
    tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
    for c in range(N_CLS):
        yc = Y_ev[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc): continue
        p = preds[:, c]
        if p.std() < 1e-9: continue
        a = roc_auc_score(yc, p)
        aucs.append(a)
        t = tax.get(PRIMARY_LABELS[c], "Aves")
        if t in taxa_aucs: taxa_aucs[t].append(a)
    print(f"  [{name}] {len(aucs)} cls macro = {np.mean(aucs):.4f}")
    for tn, lst in taxa_aucs.items():
        if lst: print(f"    {tn:10s} {len(lst):2d} cls: {np.mean(lst):.4f}")
    return np.mean(aucs), aucs


def fit_probe(X_tr, Y_tr, X_ev, n_pca=32, C=0.25, sample_weights=None, balanced=True):
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-6
    Xn_tr = (X_tr - mu) / sd
    Xn_ev = (X_ev - mu) / sd
    if n_pca and n_pca < min(Xn_tr.shape):
        pca = PCA(n_components=n_pca, random_state=0).fit(Xn_tr)
        Z_tr = pca.transform(Xn_tr); Z_ev = pca.transform(Xn_ev)
    else:
        Z_tr, Z_ev = Xn_tr, Xn_ev
    preds = np.zeros((len(X_ev), N_CLS), dtype=np.float32)
    cw = "balanced" if balanced else None
    for c in range(N_CLS):
        yc = Y_tr[:, c]
        if yc.sum() < 2 or yc.sum() == len(yc):
            preds[:, c] = float(yc.mean())
            continue
        try:
            clf = LogisticRegression(C=C, max_iter=300, solver="liblinear", class_weight=cw)
            if sample_weights is not None:
                clf.fit(Z_tr, yc, sample_weight=sample_weights)
            else:
                clf.fit(Z_tr, yc)
            preds[:, c] = clf.predict_proba(Z_ev)[:, 1]
        except Exception:
            preds[:, c] = 0.0
    return preds


# C1: SS-only PCA32 baseline (no class_weight)
print("\n--- C1: SS-only PCA32 LR C=0.25 (no balanced)")
preds = fit_probe(X_ss, Y_ss_tr, X_ev, n_pca=32, balanced=False)
m1, _ = eval_macro(preds, "C1")

# C2: SS+ext PCA32 LR (matches exp143d which got 0.8076)
print("\n--- C2: SS+ext PCA32 LR C=0.25 balanced")
X_full = np.concatenate([X_ss, ext_embs], axis=0)
Y_full = np.concatenate([Y_ss_tr, Y_ext], axis=0)
preds = fit_probe(X_full, Y_full, X_ev, n_pca=32, balanced=True)
m2, _ = eval_macro(preds, "C2 (full)")

# C3-C5: SS+ext with sample_weights downweight on external
print("\n--- C3-C5: external downweight sweep (PCA32, balanced)")
for w_ext in [0.5, 0.3, 0.1]:
    sw = np.concatenate([np.ones(len(X_ss)), np.full(len(ext_embs), w_ext)])
    preds = fit_probe(X_full, Y_full, X_ev, n_pca=32, sample_weights=sw, balanced=True)
    m, _ = eval_macro(preds, f"C3-5 w_ext={w_ext}")

# C6: PCA64 SS-only
print("\n--- C6: SS-only PCA64 LR C=0.25 balanced")
preds = fit_probe(X_ss, Y_ss_tr, X_ev, n_pca=64, balanced=True)
m6, _ = eval_macro(preds, "C6")

# C7: No PCA, full 768-d SS-only
print("\n--- C7: SS-only NO PCA LR C=0.25 balanced")
preds = fit_probe(X_ss, Y_ss_tr, X_ev, n_pca=0, balanced=True, C=0.05)  # smaller C for full-dim
m7, _ = eval_macro(preds, "C7")

# C8: SS-only PCA32 LR with C sweep
print("\n--- C8: SS-only PCA32 C sweep balanced")
for C in [0.1, 0.5, 1.0, 5.0]:
    preds = fit_probe(X_ss, Y_ss_tr, X_ev, n_pca=32, C=C, balanced=True)
    m, _ = eval_macro(preds, f"C8 C={C}")

# C9: SS-only PCA32 LR no balanced (orig recipe)
print("\n--- C9: SS-only PCA32 LR C=0.25 no balanced (original exp143)")
preds = fit_probe(X_ss, Y_ss_tr, X_ev, n_pca=32, balanced=False)
m9, _ = eval_macro(preds, "C9 (=exp143)")

# Summary
print("\n=== Summary ===")
print(f"  exp143 baseline (SS, PCA32 LR no-balance): val_SS = 0.8062")
print(f"  best in this sweep: TBD — pick best for v2 probe replacement")
