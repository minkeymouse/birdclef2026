"""exp146b — Test v33 + MLP-probe AudioMAE blend (clean MLP, no DANN).

DANN was negative but MLP trunk alone (α=0) gave val_SS 0.858 vs PCA32+LR 0.806.
Test if this transfers to v33 blend audit.
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import DATA, N_CLS

DEVICE = "cuda"
PRIMARY_LABELS, l2i = build_primaries()

# Load AudioMAE features + labels
ss = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_embs_labeled_ss.npz")
ss_embs = ss["embs"]; ss_split = ss["splits"]; ss_row_ids = ss["row_ids"]
ext = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_external_embs.npz")
ext_embs, ext_species = ext["embs"], ext["species"]

ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all_lbl = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
mask_tr = ss_split == "train"; mask_ev = ss_split == "eval"
Y_tr = np.zeros((mask_tr.sum(), N_CLS), dtype=np.float32)
for i, idx in enumerate(np.where(mask_tr)[0]):
    for lbl in ss_all_lbl.iloc[idx].lbls:
        if lbl in l2i: Y_tr[i, l2i[lbl]] = 1.0
Y_ev = np.zeros((mask_ev.sum(), N_CLS), dtype=np.float32)
for i, idx in enumerate(np.where(mask_ev)[0]):
    for lbl in ss_all_lbl.iloc[idx].lbls:
        if lbl in l2i: Y_ev[i, l2i[lbl]] = 1.0

Y_ext = np.zeros((len(ext_embs), N_CLS), dtype=np.float32)
for i, sp in enumerate(ext_species):
    Y_ext[i, l2i[str(sp)]] = 1.0

X_tr = ss_embs[mask_tr]
X_ev = ss_embs[mask_ev]


def train_mlp_probe(X_tr, Y_tr, X_ev, hidden=128, dropout=0.2, lr=3e-4, epochs=80, weight_external=None):
    """Train MLP probe with optional external positives (sample-weighted)."""
    torch.manual_seed(0)
    in_dim = X_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, N_CLS),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-6
    Xn_tr = ((X_tr - mu) / sd).astype(np.float32)
    Xn_ev = ((X_ev - mu) / sd).astype(np.float32)
    X = torch.from_numpy(Xn_tr).to(DEVICE)
    Y = torch.from_numpy(Y_tr).to(DEVICE)
    SW = torch.from_numpy(weight_external if weight_external is not None else np.ones(len(X), dtype=np.float32)).to(DEVICE)

    bs = 64
    n = len(X)
    best_macro = 0
    best_preds = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_loss = 0
        for bi in range(0, n, bs):
            idx = perm[bi:bi+bs]
            x = X[idx]; y = Y[idx]; w = SW[idx]
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none').mean(dim=1) * w
            loss = loss.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model(torch.from_numpy(Xn_ev).to(DEVICE))).cpu().numpy()
            aucs = []
            for c in range(N_CLS):
                yc = Y_ev[:, c]
                if 0 < yc.sum() < len(yc):
                    pc = preds[:, c]
                    if pc.std() > 1e-9:
                        aucs.append(roc_auc_score(yc, pc))
            m = np.mean(aucs) if aucs else 0
            if m > best_macro:
                best_macro = m
                best_preds = preds.copy()
    return best_macro, best_preds


# Baseline references (from exp143/144)
print("Reference: PCA32+LR (C1) val_SS = 0.8062")
print("Reference: DANN α=0 (MLP) val_SS = 0.8578\n")

# Variant 1: clean MLP (no DANN, no external)
print("--- M1: SS-only MLP probe")
m1, p1 = train_mlp_probe(X_tr, Y_tr, X_ev, hidden=128)
print(f"  M1 best val_SS = {m1:.4f}")

# Variant 2: SS + ext (downweight 0.1)
print("\n--- M2: SS + ext × 0.1 weight, MLP")
X_full = np.concatenate([X_tr, ext_embs], axis=0)
Y_full = np.concatenate([Y_tr, Y_ext], axis=0)
sw = np.concatenate([np.ones(len(X_tr)), np.full(len(ext_embs), 0.1)]).astype(np.float32)
m2, p2 = train_mlp_probe(X_full, Y_full, X_ev, hidden=128, weight_external=sw)
print(f"  M2 best val_SS = {m2:.4f}")

# Variant 3: bigger MLP
print("\n--- M3: SS-only MLP (hidden=256)")
m3, p3 = train_mlp_probe(X_tr, Y_tr, X_ev, hidden=256)
print(f"  M3 best val_SS = {m3:.4f}")

# Variant 4: deeper + dropout
print("\n--- M4: SS + ext × 0.2, MLP hidden=256, dropout=0.3")
sw = np.concatenate([np.ones(len(X_tr)), np.full(len(ext_embs), 0.2)]).astype(np.float32)
m4, p4 = train_mlp_probe(X_full, Y_full, X_ev, hidden=256, dropout=0.3, weight_external=sw)
print(f"  M4 best val_SS = {m4:.4f}")


# v33 blend audit using best MLP probe predictions
print("\n=== v33 blend audit ===")
perch = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/perch_prob_labeled.npz")["prob"][mask_ev]
exp50 = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/exp50_scores_labeled.npz")["scores"][mask_ev]
v33_simple = 0.7 * perch + 0.3 * exp50

tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]

def eval_macro(preds, name=""):
    aucs, taxa = [], {t: [] for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
    for c in range(N_CLS):
        yc = Y_ev[:, c]
        if 0 < yc.sum() < len(yc):
            pc = preds[:, c]
            if pc.std() > 1e-9:
                a = roc_auc_score(yc, pc)
                aucs.append(a)
                t = tax.get(PRIMARY_LABELS[c], "Aves")
                if t in taxa: taxa[t].append(a)
    print(f"  [{name:50s}] {len(aucs)} cls macro={np.mean(aucs):.4f}", end="")
    for tn in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        if taxa[tn]: print(f" {tn[:3]}={np.mean(taxa[tn]):.3f}", end="")
    print()
    return np.mean(aucs)

m_base = eval_macro(v33_simple, "v33 simple base")

best_probe_name, best_probe = max(
    [("M1", p1), ("M2", p2), ("M3", p3), ("M4", p4)], key=lambda x: x[1].max()
)
print(f"\nBest probe: {best_probe_name}")

print("\n--- v33 + α × best probe (uniform)")
for a in [0.05, 0.10, 0.15, 0.20, 0.30]:
    blend = (1-a)*v33_simple + a*best_probe
    eval_macro(blend, f"v33 + {a}·{best_probe_name} uniform")

# Also test: PCA32+LR baseline blend (the actual v49 submission)
print("\n--- v33 + α × PCA32+LR (= v49 submitted)")
am_npz = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_probe_preds.npz")
p_orig = am_npz["preds"]
for a in [0.05, 0.10, 0.20]:
    blend = (1-a)*v33_simple + a*p_orig
    eval_macro(blend, f"v33 + {a}·PCA32+LR uniform")

# Save best
import os
np.savez(ROOT/"experiments/_data_pipelines/exp146_outputs/best_mlp_probe_preds.npz",
         M1=p1, M2=p2, M3=p3, M4=p4, val_SS=[m1,m2,m3,m4])
print(f"\nSaved MLP probe predictions to exp146_outputs/")
