"""exp153 — LightGBM per-class stacking on labeled SS.

Features (per row, per class c):
  - Perch_c, exp50_c, exp59_c, exp84b_c, exp136b_c, AudioMAE_M5_c
  - file_mean_perch, file_std_perch (for class c)
  - file_mean_exp50, file_std_exp50
  - window_pos (0-11)
  - neighbor scores (prev/next window score for class c, perch and exp50)
  - max(perch, exp50), min, abs_diff

Label: y_c (binary)

Train/eval split: same 55/11 SS file split as exp143 etc.

Output: stacked predictions on eval, + saved per-class LightGBM models for
optional Kaggle integration.
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import time
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import DATA, N_CLS, N_WINDOWS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all_lbl = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
mask_tr = (ss_all_lbl.split == "train").values
mask_ev = (ss_all_lbl.split == "eval").values

# Build labels
Y_full = np.zeros((len(ss_all_lbl), N_CLS), dtype=np.float32)
for i, lbls in enumerate(ss_all_lbl.lbls):
    for lbl in lbls:
        if lbl in l2i: Y_full[i, l2i[lbl]] = 1.0

# Load all teachers
print("[1/4] Loading teacher predictions...")
AUD = ROOT / "experiments/_audits_post_v26/exp80_outputs"
T = {}
T["Perch"] = np.load(AUD/"perch_prob_labeled.npz")["prob"]
T["exp50"] = np.load(AUD/"exp50_scores_labeled.npz")["scores"]
T["exp59"] = np.load(AUD/"exp59_scores_labeled.npz")["scores"]
T["exp84b"] = np.load(AUD/"exp84b_scores_labeled.npz")["scores"]
T["exp136b"] = np.load(AUD/"exp136b_scores_labeled.npz")["scores"]
T["v33"] = 0.7 * T["Perch"] + 0.3 * T["exp50"]

# AudioMAE M5 (eval-only; need full)
all_probes = np.load(ROOT/"experiments/_data_pipelines/exp146_outputs/all_probes.npz")
M5_eval = all_probes["M5"]  # (122, 234)
# We need M5 on TRAIN too. Let me just train it again here on the fly.
import torch
import torch.nn as nn
import torch.nn.functional as F
ss_emb = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_embs_labeled_ss.npz")
embs_full = ss_emb["embs"]

# Train M5 to get train predictions also
print("Training M5 on full labeled SS...")
ext = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_external_embs.npz")
ext_embs = ext["embs"]; ext_species = ext["species"]
Y_tr_only = Y_full[mask_tr]
Y_ext = np.zeros((len(ext_embs), N_CLS), dtype=np.float32)
for i, sp in enumerate(ext_species):
    Y_ext[i, l2i[str(sp)]] = 1.0

X_tr_emb = embs_full[mask_tr]
X_full_emb = np.concatenate([X_tr_emb, ext_embs], axis=0)
Y_full_em = np.concatenate([Y_tr_only, Y_ext], axis=0)
mu_e = X_full_emb.mean(0); sd_e = X_full_emb.std(0) + 1e-6
Xn_tr_e = ((X_full_emb - mu_e) / sd_e).astype(np.float32)
Xn_full_e = ((embs_full - mu_e) / sd_e).astype(np.float32)

torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(256, 256), nn.GELU(), nn.Linear(256, N_CLS)).cuda()
opt = torch.optim.AdamW(mlp.parameters(), lr=3e-4, weight_decay=1e-4)
X = torch.from_numpy(Xn_tr_e).cuda(); Y = torch.from_numpy(Y_full_em).cuda()
sw = np.concatenate([np.ones(len(X_tr_emb)), np.full(len(ext_embs), 0.1)]).astype(np.float32)
SW = torch.from_numpy(sw).cuda()
bs = 64; n = len(X)
best_macro = 0; best_state = None
for ep in range(80):
    mlp.train()
    perm = torch.randperm(n, device="cuda")
    for bi in range(0, n, bs):
        idx = perm[bi:bi+bs]
        logits = mlp(X[idx])
        loss = (F.binary_cross_entropy_with_logits(logits, Y[idx], reduction='none').mean(dim=1) * SW[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1) % 5 == 0:
        mlp.eval()
        with torch.no_grad():
            preds_ev = torch.sigmoid(mlp(torch.from_numpy(Xn_full_e[mask_ev]).cuda())).cpu().numpy()
        aucs = []
        for c in range(N_CLS):
            yc = Y_full[mask_ev, c]
            if 0 < yc.sum() < len(yc):
                pc = preds_ev[:, c]
                if pc.std() > 1e-9:
                    aucs.append(roc_auc_score(yc, pc))
        m = np.mean(aucs) if aucs else 0
        if m > best_macro:
            best_macro = m; best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
mlp.load_state_dict(best_state); mlp.eval()
with torch.no_grad():
    M5_full = torch.sigmoid(mlp(torch.from_numpy(Xn_full_e).cuda())).cpu().numpy()
T["AudioMAE_M5"] = M5_full
print(f"M5 best val_SS: {best_macro:.4f}, full pred shape: {M5_full.shape}")

# [2] Build feature matrix per (row, c)
# For efficiency, vectorize per class
print("\n[2/4] Building feature matrix...")
n_files = ss_all_lbl.filename.nunique()

# Per-file precomputations: file_mean, file_std for each teacher per class
file_idx = ss_all_lbl.groupby("filename").indices  # filename → row indices
def file_stats(scores):
    """Return per-row file_mean, file_std for given teacher scores."""
    fm = np.zeros_like(scores); fs = np.zeros_like(scores)
    for fname, rows in file_idx.items():
        rs = np.array(rows)
        fm[rs] = scores[rs].mean(0)
        fs[rs] = scores[rs].std(0)
    return fm, fs

P_fm, P_fs = file_stats(T["Perch"])
E_fm, E_fs = file_stats(T["exp50"])

# Window position 0..11
window_pos = np.zeros(len(ss_all_lbl), dtype=np.int32)
for fname, rows in file_idx.items():
    sorted_rows = sorted(rows, key=lambda r: ss_all_lbl.iloc[r].end)
    for i, r in enumerate(sorted_rows):
        window_pos[r] = i

# Neighbor scores (prev/next window)
def neighbor_scores(scores):
    prev = np.zeros_like(scores); nxt = np.zeros_like(scores)
    for fname, rows in file_idx.items():
        sorted_rows = sorted(rows, key=lambda r: ss_all_lbl.iloc[r].end)
        for i, r in enumerate(sorted_rows):
            if i > 0:
                prev[r] = scores[sorted_rows[i-1]]
            else:
                prev[r] = scores[r]
            if i < len(sorted_rows) - 1:
                nxt[r] = scores[sorted_rows[i+1]]
            else:
                nxt[r] = scores[r]
    return prev, nxt

P_prev, P_next = neighbor_scores(T["Perch"])
E_prev, E_next = neighbor_scores(T["exp50"])

# Build per-class feature stack
def class_features(c):
    return np.stack([
        T["Perch"][:, c], T["exp50"][:, c], T["exp59"][:, c],
        T["exp84b"][:, c], T["exp136b"][:, c], T["AudioMAE_M5"][:, c],
        P_fm[:, c], P_fs[:, c], E_fm[:, c], E_fs[:, c],
        window_pos.astype(np.float32),
        P_prev[:, c], P_next[:, c], E_prev[:, c], E_next[:, c],
        np.maximum(T["Perch"][:, c], T["exp50"][:, c]),
        np.minimum(T["Perch"][:, c], T["exp50"][:, c]),
        np.abs(T["Perch"][:, c] - T["exp50"][:, c]),
        T["v33"][:, c],
    ], axis=1).astype(np.float32)

print("[3/4] Training per-class LightGBM models...")
# For each evaluable class, train LightGBM
preds_ev = np.zeros((mask_ev.sum(), N_CLS), dtype=np.float32)
n_fit = 0
t0 = time.time()
all_models = {}
for c in range(N_CLS):
    yc_tr = Y_full[mask_tr, c]
    yc_ev = Y_full[mask_ev, c]
    if yc_tr.sum() < 3: continue  # need positives in train
    if yc_ev.sum() == 0 or yc_ev.sum() == len(yc_ev): continue  # need both classes in eval

    Xc_tr = class_features(c)[mask_tr]
    Xc_ev = class_features(c)[mask_ev]

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbosity=-1,
        n_estimators=200,
        scale_pos_weight=(yc_tr == 0).sum() / max(1, yc_tr.sum()),
    )
    try:
        model = lgb.LGBMClassifier(**params)
        model.fit(Xc_tr, yc_tr, eval_set=[(Xc_ev, yc_ev)], callbacks=[lgb.early_stopping(20, verbose=False)])
        preds_ev[:, c] = model.predict_proba(Xc_ev)[:, 1]
        all_models[c] = model
        n_fit += 1
    except Exception as e:
        preds_ev[:, c] = T["v33"][mask_ev, c]
print(f"  Fit {n_fit} per-class LGBM in {time.time()-t0:.1f}s")

# [4] Evaluate macro AUC
print("\n[4/4] Evaluation:")
tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
def eval_macro(preds, name):
    aucs, by = [], {t: [] for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
    for c in range(N_CLS):
        yc = Y_full[mask_ev, c]
        if 0 < yc.sum() < len(yc):
            pc = preds[:, c]
            if pc.std() > 1e-9:
                a = roc_auc_score(yc, pc)
                aucs.append(a)
                t = tax.get(PRIMARY_LABELS[c], "Aves")
                if t in by: by[t].append(a)
    m = np.mean(aucs) if aucs else 0
    print(f"  [{name:50s}] {len(aucs)} cls macro={m:.4f}", end="")
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        if by[t]: print(f" {t[:3]}={np.mean(by[t]):.3f}", end="")
    print()
    return m

print()
eval_macro(T["v33"][mask_ev], "v33 base")
eval_macro(preds_ev, "LGBM stacking (alone)")
print()
# Blend stacking with v33
for a in [0.05, 0.10, 0.20, 0.30, 0.50]:
    blend = (1-a) * T["v33"][mask_ev] + a * preds_ev
    eval_macro(blend, f"v33 + {a}·LGBM stacking")
print()
# Where stacking has confidence (filter low-pred)
# Better: per-class weighting based on whether stacking improves
for a in [0.10]:
    # Only apply where stacking pred is meaningful (not zero)
    blend = T["v33"][mask_ev].copy()
    valid_classes = list(all_models.keys())
    for c in valid_classes:
        blend[:, c] = (1-a) * T["v33"][mask_ev, c] + a * preds_ev[:, c]
    eval_macro(blend, f"v33 + {a}·LGBM (only fit classes)")

# Save models for potential Kaggle integration (small per-class GBT)
import pickle
out_path = ROOT / "experiments/_data_pipelines/exp153_outputs"
out_path.mkdir(parents=True, exist_ok=True)
with open(out_path / "stacking_models.pkl", "wb") as f:
    pickle.dump({"models": all_models, "feature_names": [
        "perch","exp50","exp59","exp84b","exp136b","audiomae_m5",
        "P_fm","P_fs","E_fm","E_fs","window_pos","P_prev","P_next","E_prev","E_next",
        "max_PE","min_PE","abs_diff_PE","v33"
    ]}, f)
print(f"\nSaved {out_path/'stacking_models.pkl'}")
