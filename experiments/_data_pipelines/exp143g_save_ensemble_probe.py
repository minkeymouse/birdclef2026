"""exp143g — Save AudioMAE ensemble probe (C5 + C7) for v50 candidate.

C5: SS+ext PCA32 LR balanced, w_ext=0.1
C7: SS-only NO PCA LR balanced

Production saves both as separate probes — at inference time, predict from
each and average. To keep notebook simple, we materialize both with their
own (mu, sd, [pca], lr_w, lr_b, lr_valid).
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
from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import N_CLS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)

OUT = ROOT / "experiments" / "_data_pipelines" / "exp143_outputs"
ss = np.load(OUT / "audiomae_embs_labeled_ss.npz")
ss_embs, ss_split = ss["embs"], ss["splits"]
ext = np.load(OUT / "audiomae_external_embs.npz")
ext_embs, ext_species = ext["embs"], ext["species"]
mask_tr = ss_split == "train"

# Y_ss
Y_ss_tr = np.zeros((mask_tr.sum(), N_CLS), dtype=np.float32)
for i, lbls in enumerate(ss_all[mask_tr].lbls):
    for lbl in lbls:
        if lbl in l2i: Y_ss_tr[i, l2i[lbl]] = 1.0
# Y_ext
Y_ext = np.zeros((len(ext_embs), N_CLS), dtype=np.float32)
for i, sp in enumerate(ext_species):
    Y_ext[i, l2i[str(sp)]] = 1.0

X_ss = ss_embs[mask_tr]
X_full = np.concatenate([X_ss, ext_embs], axis=0)
Y_full = np.concatenate([Y_ss_tr, Y_ext], axis=0)


def fit_full(X_tr, Y_tr, n_pca=32, C=0.25, sample_weights=None, balanced=True):
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-6
    Xn = (X_tr - mu) / sd
    if n_pca and n_pca < min(Xn.shape):
        pca = PCA(n_components=n_pca, random_state=0).fit(Xn)
        Z = pca.transform(Xn)
        components = pca.components_.astype(np.float32)
        n_in = n_pca
    else:
        Z = Xn
        components = np.eye(Xn.shape[1], dtype=np.float32)  # identity
        n_in = Xn.shape[1]

    cw = "balanced" if balanced else None
    lr_w = np.zeros((N_CLS, n_in), dtype=np.float32)
    lr_b = np.zeros(N_CLS, dtype=np.float32)
    lr_valid = np.zeros(N_CLS, dtype=bool)
    n_fit = 0
    for c in range(N_CLS):
        yc = Y_tr[:, c]
        if yc.sum() < 2 or yc.sum() == len(yc):
            continue
        try:
            clf = LogisticRegression(C=C, max_iter=300, solver="liblinear", class_weight=cw)
            if sample_weights is not None:
                clf.fit(Z, yc, sample_weight=sample_weights)
            else:
                clf.fit(Z, yc)
            lr_w[c] = clf.coef_[0].astype(np.float32)
            lr_b[c] = float(clf.intercept_[0])
            lr_valid[c] = True
            n_fit += 1
        except Exception:
            pass
    return mu.astype(np.float32), sd.astype(np.float32), components, lr_w, lr_b, lr_valid, n_fit


print("Fitting C5 (SS+ext PCA32 balanced w_ext=0.1)")
sw = np.concatenate([np.ones(len(X_ss)), np.full(len(ext_embs), 0.1)])
c5 = fit_full(X_full, Y_full, n_pca=32, balanced=True, sample_weights=sw)
print(f"  fit {c5[6]} classes")

print("Fitting C7 (SS-only no-PCA balanced)")
c7 = fit_full(X_ss, Y_ss_tr, n_pca=0, balanced=True)
print(f"  fit {c7[6]} classes")

OUTW = ROOT / "model-weights"
np.savez(OUTW / "audiomae_probe_v3_c5.npz",
         mu=c5[0], sd=c5[1], pca_components=c5[2], lr_w=c5[3], lr_b=c5[4], lr_valid=c5[5])
np.savez(OUTW / "audiomae_probe_v3_c7.npz",
         mu=c7[0], sd=c7[1], pca_components=c7[2], lr_w=c7[3], lr_b=c7[4], lr_valid=c7[5])
print(f"\nSaved C5: {(OUTW / 'audiomae_probe_v3_c5.npz').stat().st_size/1e3:.1f} KB")
print(f"Saved C7: {(OUTW / 'audiomae_probe_v3_c7.npz').stat().st_size/1e3:.1f} KB")
