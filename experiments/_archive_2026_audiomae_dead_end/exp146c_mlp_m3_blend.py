"""exp146c — Test M3 (SS-only h=256 MLP) in v33 blend properly + per-site spread."""
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
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all_lbl = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
ss_all_lbl["site"] = ss_all_lbl["filename"].str.extract(r"(S\d+)")[0].apply(lambda x: f"S{int(x[1:]):02d}")

ss = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_embs_labeled_ss.npz")
ss_embs = ss["embs"]; ss_split = ss["splits"]; ss_row_ids = ss["row_ids"]
ext = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_external_embs.npz")
ext_embs, ext_species = ext["embs"], ext["species"]
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


def train_mlp_save_best(X_tr, Y_tr, X_ev, hidden=128, dropout=0.2, lr=3e-4, epochs=80, sw=None):
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
    SW = torch.from_numpy(sw if sw is not None else np.ones(len(X), dtype=np.float32)).to(DEVICE)
    bs = 64
    n = len(X)
    best_macro = 0; best_preds = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for bi in range(0, n, bs):
            idx = perm[bi:bi+bs]
            x = X[idx]; y = Y[idx]; w = SW[idx]
            logits = model(x)
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none').mean(dim=1) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 5 == 0:
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
                best_macro = m; best_preds = preds.copy()
    return best_macro, best_preds


print("Training MLP variants and computing per-site spread...")

# Train M1, M2, M3
print("\n--- M1: SS-only h=128")
m1, p1 = train_mlp_save_best(X_tr, Y_tr, X_ev, hidden=128)
print(f"  M1 val_SS = {m1:.4f}")

print("\n--- M3: SS-only h=256")
m3, p3 = train_mlp_save_best(X_tr, Y_tr, X_ev, hidden=256)
print(f"  M3 val_SS = {m3:.4f}")

X_full = np.concatenate([X_tr, ext_embs], axis=0)
Y_full = np.concatenate([Y_tr, Y_ext], axis=0)
print("\n--- M2: SS+ext×0.1 h=128")
sw = np.concatenate([np.ones(len(X_tr)), np.full(len(ext_embs), 0.1)]).astype(np.float32)
m2, p2 = train_mlp_save_best(X_full, Y_full, X_ev, hidden=128, sw=sw)
print(f"  M2 val_SS = {m2:.4f}")

print("\n--- M5: SS+ext×0.1 h=256")
m5, p5 = train_mlp_save_best(X_full, Y_full, X_ev, hidden=256, sw=sw)
print(f"  M5 val_SS = {m5:.4f}")


# Original PCA32+LR baseline
am_npz = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_probe_preds.npz")
p_orig = am_npz["preds"]

# v33 base
perch = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/perch_prob_labeled.npz")["prob"][mask_ev]
exp50 = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/exp50_scores_labeled.npz")["scores"][mask_ev]
v33_simple = 0.7 * perch + 0.3 * exp50

ss_eval_df = ss_all_lbl[mask_ev].reset_index(drop=True)
def per_site_spread(preds, name):
    site_aucs = {}
    for site, g in ss_eval_df.groupby("site"):
        rows = g.index.values
        if len(rows) < 4: continue
        Y_site = Y_ev[rows]
        p_site = preds[rows]
        cls_aucs = []
        for c in range(N_CLS):
            yc = Y_site[:, c]
            if 0 < yc.sum() < len(yc):
                pc = p_site[:, c]
                if pc.std() > 1e-9:
                    cls_aucs.append(roc_auc_score(yc, pc))
        site_aucs[site] = (np.mean(cls_aucs) if cls_aucs else 0, len(cls_aucs))
    if not site_aucs: return None
    aucs_only = [v[0] for v in site_aucs.values()]
    spread = max(aucs_only) - min(aucs_only)
    print(f"  [{name:30s}] sites:")
    for s, (a, n) in sorted(site_aucs.items()):
        print(f"    {s}: AUC {a:.3f} ({n} cls)")
    print(f"    spread (max-min): {spread:.3f}")
    return spread


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


# Per-site spread per probe (key diagnostic for site fragility)
print("\n=== Per-site spread (LOWER = more LB-friendly) ===")
print("\nPCA32+LR (v49 baseline):")
sp_pca = per_site_spread(p_orig, "PCA32+LR")
print("\nMLP M1 (SS-only h=128):")
sp_m1 = per_site_spread(p1, "M1")
print("\nMLP M3 (SS-only h=256):")
sp_m3 = per_site_spread(p3, "M3")
print("\nMLP M2 (SS+ext×0.1 h=128):")
sp_m2 = per_site_spread(p2, "M2")
print("\nMLP M5 (SS+ext×0.1 h=256):")
sp_m5 = per_site_spread(p5, "M5")


# v33 blend audit for each probe
print("\n=== v33 blend audit ===")
m_base = eval_macro(v33_simple, "v33 base")
print()
for probe_name, probe in [("PCA32+LR", p_orig), ("M1", p1), ("M2", p2), ("M3", p3), ("M5", p5)]:
    for a in [0.10, 0.20]:
        blend = (1-a)*v33_simple + a*probe
        eval_macro(blend, f"v33 + {a}·{probe_name}")
    print()

# Save
np.savez(ROOT/"experiments/_data_pipelines/exp146_outputs/all_probes.npz",
         M1=p1, M2=p2, M3=p3, M5=p5, PCA32=p_orig)
print("Saved")
