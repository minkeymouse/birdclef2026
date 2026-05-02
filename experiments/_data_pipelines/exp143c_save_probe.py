"""exp143c — Save AudioMAE probe weights for Kaggle integration.

Outputs:
  /data/birdclef2026/model-weights/audiomae_probe.npz
    - mu, sd: feature standardization (768,)
    - pca_components: PCA matrix (32, 768)
    - lr_w: per-class LR weights (234, 32)
    - lr_b: per-class LR biases (234,)
    - lr_valid: bool mask (234,) — true if class was fit
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import N_CLS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)

z = np.load(ROOT / "experiments/_data_pipelines/exp143_outputs/audiomae_embs_labeled_ss.npz")
embs = z["embs"]  # (739, 768)
splits = z["splits"]
row_ids = z["row_ids"]

assert (row_ids == ss_all.row_id.values).all()
mask_tr = splits == "train"

# Build Y on train portion only
Y_tr = np.zeros((mask_tr.sum(), N_CLS), dtype=np.float32)
for i, lbls in enumerate(ss_all[mask_tr].lbls):
    for lbl in lbls:
        if lbl in l2i:
            Y_tr[i, l2i[lbl]] = 1.0

# Use BOTH train+eval to fit probe (we now want production probe, eval is over)
# Actually no — keep eval-leak-free probe as we did in exp143b. Use train only.
X_tr = embs[mask_tr]
mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-6
Xn = (X_tr - mu) / sd
pca = PCA(n_components=32, random_state=0).fit(Xn)
Z = pca.transform(Xn)
print(f"PCA32 var explained: {pca.explained_variance_ratio_.sum():.3f}")

# Per-class LR
lr_w = np.zeros((N_CLS, 32), dtype=np.float32)
lr_b = np.zeros(N_CLS, dtype=np.float32)
lr_valid = np.zeros(N_CLS, dtype=bool)
n_fit = 0
for c in range(N_CLS):
    yc = Y_tr[:, c]
    if yc.sum() < 1 or yc.sum() == len(yc):
        continue
    try:
        clf = LogisticRegression(C=0.25, max_iter=200, solver="liblinear").fit(Z, yc)
        lr_w[c] = clf.coef_[0].astype(np.float32)
        lr_b[c] = float(clf.intercept_[0])
        lr_valid[c] = True
        n_fit += 1
    except Exception:
        pass
print(f"Fit {n_fit}/{N_CLS} classes")

OUT = ROOT / "model-weights" / "audiomae_probe.npz"
np.savez(OUT, mu=mu.astype(np.float32), sd=sd.astype(np.float32),
         pca_components=pca.components_.astype(np.float32),
         lr_w=lr_w, lr_b=lr_b, lr_valid=lr_valid)
print(f"Saved: {OUT} ({OUT.stat().st_size/1e3:.1f} KB)")

# Also copy ONNX into model-weights
import shutil
src = '/tmp/audiomae/audiomae_legacy.onnx'
dst = ROOT / "model-weights" / "audiomae_base_ft_as20k.onnx"
shutil.copy(src, dst)
print(f"Saved: {dst} ({dst.stat().st_size/1e6:.1f} MB)")
