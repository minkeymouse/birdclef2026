#!/usr/bin/env python3
"""exp43g — Composite pseudo-confidence filter: test on labeled SS with synthetic flips.

Three orthogonal signals for mislabel detection:
  S1 — iVAE z-kNN disagreement (identifiability-motivated, structural)
  S2 — Teacher posterior disagreement: 1 - mean(teacher_prob[labeled_as_1])
       (direct observation: does teacher think the labels are right?)
  S3 — Mahalanobis distance to labeled Perch distribution (density/OOD signal)

Hypothesis: three signals are empirically near-independent (exp43d showed Mahal
vs teacher spearman ~0). Composite filter should beat any single signal on
flip-detection AUC.

Also test: combining composite signal with low-entropy-neighbor label recovery
(exp43c mechanism 2) → does it recover the TRUE (pre-flip) label?
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
CACHE = ROOT / "experiments/exp21_outputs/perch_cache"
OUT = ROOT / "experiments/exp43g_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
Z_DIM = 32
HIDDEN = 256
EPOCHS = 200
BATCH = 64
WARMUP = 50
FREE_BITS = 0.5
K_NEIGH = 10
FLIP_RATE = 0.15


# ─── Data (same as exp43c) ────────────────────────────────────────────────────
def load_data():
    meta = pd.read_parquet(CACHE / "full_perch_meta.parquet")
    emb = np.load(CACHE / "full_perch_arrays.npz")["emb"].astype(np.float32)
    sites = meta["site"].astype("category")
    hours = meta["hour_utc"].astype(int).astype("category")
    u = np.concatenate([
        pd.get_dummies(sites).values.astype(np.float32),
        pd.get_dummies(hours).values.astype(np.float32),
    ], axis=1)
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename", "start", "end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_g["end_sec"].astype(str)
    Y_sc = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g["lbls"]):
        for l in labs:
            if l in l2i: Y_sc[i, l2i[l]] = 1
    idx = sc_g.set_index("row_id")
    Y = np.stack([Y_sc[idx.index.get_loc(rid)] for rid in meta["row_id"]])
    return emb, u, Y, meta


# ─── iVAE (same as exp43b) ────────────────────────────────────────────────────
class iVAE(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim, hidden=HIDDEN):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(x_dim + u_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.enc_mu = nn.Linear(hidden, z_dim); self.enc_lv = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, x_dim))
        self.prior_mu = nn.Linear(u_dim, z_dim); self.prior_lv = nn.Linear(u_dim, z_dim)
    def encode(self, x, u):
        h = self.enc(torch.cat([x, u], -1))
        return self.enc_mu(h), self.enc_lv(h)
    def forward(self, x, u):
        mu, lv = self.encode(x, u)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
        return self.dec(z), mu, lv, self.prior_mu(u), self.prior_lv(u), z


def kl_pd(mu_q, lv_q, mu_p, lv_p):
    return 0.5 * (lv_p - lv_q + (torch.exp(lv_q) + (mu_q - mu_p) ** 2) / torch.exp(lv_p) - 1)


def train_ivae(x, u, epochs=EPOCHS):
    model = iVAE(x.shape[1], u.shape[1], Z_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(u)),
                    batch_size=BATCH, shuffle=True)
    model.train()
    for ep in range(1, epochs + 1):
        beta = min(1.0, ep / WARMUP)
        for xb, ub in dl:
            xb = xb.to(DEVICE); ub = ub.to(DEVICE)
            x_hat, mu_q, lv_q, mu_p, lv_p, z = model(xb, ub)
            rec = ((x_hat - xb) ** 2).sum(-1).mean()
            kl = torch.clamp(kl_pd(mu_q, lv_q, mu_p, lv_p), min=FREE_BITS).sum(-1).mean()
            (rec + beta * kl).backward(); opt.step(); opt.zero_grad()
    model.eval()
    with torch.inference_mode():
        mu, _ = model.encode(torch.from_numpy(x).to(DEVICE), torch.from_numpy(u).to(DEVICE))
    return mu.cpu().numpy().astype(np.float32)


# ─── Three signals for mislabel detection ────────────────────────────────────
def signal_ivae_disagreement(Z, Y, files, sites, k=K_NEIGH, cross_site=True):
    n = Z.shape[0]
    nn_ = NearestNeighbors(n_neighbors=min(n, k + 400), metric="cosine").fit(Z)
    _, I = nn_.kneighbors(Z)
    score = np.zeros(n)
    for i in range(n):
        neigh = []
        for j in I[i]:
            if j == i or files[j] == files[i]: continue
            if cross_site and sites[j] == sites[i]: continue
            neigh.append(j)
            if len(neigh) >= k: break
        if not neigh:
            score[i] = 0.5; continue
        yi = Y[i]
        inter = (Y[neigh] & yi).sum(1)
        union = (Y[neigh] | yi).sum(1)
        jacc = np.where(union > 0, inter / union, 0.0)
        score[i] = 1.0 - jacc.mean()
    return score


def signal_teacher_disagreement(teacher_prob, Y):
    """For each row, for each labeled class, compute 1 - teacher_prob. Avg over labeled."""
    n = Y.shape[0]
    score = np.zeros(n)
    for i in range(n):
        pos = np.where(Y[i] == 1)[0]
        if len(pos) == 0:
            score[i] = 0.5; continue
        score[i] = (1.0 - teacher_prob[i, pos]).mean()
    return score


def signal_mahal(X_ref_perch):
    """Mahalanobis distance in Perch feature space (PCA-32 for stable cov)."""
    pca = PCA(n_components=32, random_state=42)
    X = pca.fit_transform(X_ref_perch)
    mu = X.mean(0)
    cov = np.cov(X, rowvar=False) + 1e-4 * np.eye(X.shape[1])
    cov_inv = np.linalg.inv(cov)
    d = X - mu
    mah = np.einsum("ni,ij,nj->n", d, cov_inv, d)
    return mah  # higher = farther = more OOD = lower confidence


# ─── Synthetic flip injection + AUC ───────────────────────────────────────────
def inject_flips(Y, rate=FLIP_RATE, seed=0):
    rng = np.random.default_rng(seed)
    n, C = Y.shape
    n_flip = int(n * rate)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    Y_dirty = Y.copy()
    for i in flip_idx:
        pres = np.where(Y_dirty[i] == 1)[0]
        if len(pres) > 0:
            Y_dirty[i, rng.choice(pres)] = 0
        Y_dirty[i, rng.integers(0, C)] = 1
    is_flip = np.zeros(n, dtype=np.uint8); is_flip[flip_idx] = 1
    return Y_dirty, is_flip


def rank_norm(s):
    """Rank-normalize signal to [0,1] for composite fairness."""
    return np.argsort(np.argsort(s)).astype(np.float32) / max(len(s) - 1, 1)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    emb, u, Y, meta = load_data()
    files = meta["filename"].values; sites = meta["site"].values
    mu_x = emb.mean(0, keepdims=True); sd_x = emb.std(0, keepdims=True) + 1e-6
    emb_z = ((emb - mu_x) / sd_x).astype(np.float32)

    # Train iVAE once
    print("Training iVAE on 708 labeled...")
    Z = train_ivae(emb_z, u, epochs=EPOCHS)

    # Load teachers
    print("Loading teachers (exp28 Perch probe OOF, exp29 SED29 preds)...")
    teacher_perch = np.load(ROOT / "experiments/exp28_outputs/best_oof.npz")["val_a_probe"].astype(np.float32)
    teacher_sed29 = np.load(ROOT / "experiments/exp29_outputs/val_scores.npz")["preds"].astype(np.float32)
    # z-score teacher columns so ensemble is comparable
    def zn(x):
        m = x.mean(0, keepdims=True); s = x.std(0, keepdims=True) + 1e-8
        return (x - m) / s
    teacher_ens = 0.8 * zn(teacher_perch) + 0.2 * zn(teacher_sed29)
    # convert to pseudo prob via sigmoid (not strictly calibrated, but monotonic)
    teacher_prob = 1 / (1 + np.exp(-teacher_ens))

    # Mahalanobis on raw Perch (constant across flip trials)
    mah = signal_mahal(emb)

    # Flip trial
    print(f"\nInjecting {FLIP_RATE*100:.0f}% flips...")
    Y_dirty, is_flip = inject_flips(Y, rate=FLIP_RATE, seed=42)

    # Compute three signals with DIRTY labels
    print("Computing signals...")
    s_ivae = signal_ivae_disagreement(Z, Y_dirty, files, sites, k=K_NEIGH, cross_site=True)
    s_perch = signal_ivae_disagreement(emb_z, Y_dirty, files, sites, k=K_NEIGH, cross_site=True)
    s_teach_ens = signal_teacher_disagreement(teacher_prob, Y_dirty)
    s_teach_perch = signal_teacher_disagreement(teacher_perch, Y_dirty)
    s_mah = mah  # doesn't depend on labels

    def auc(s): return roc_auc_score(is_flip, s)

    print("\n=== Individual signals (higher = more mislabel-ish) ===")
    print(f"  iVAE z-kNN disagreement (k=10, cross-site):  AUC {auc(s_ivae):.4f}")
    print(f"  Perch kNN disagreement  (k=10, cross-site):  AUC {auc(s_perch):.4f}")
    print(f"  Teacher-ensemble disagreement (1-P(y|x)):    AUC {auc(s_teach_ens):.4f}")
    print(f"  Teacher-perch disagreement:                  AUC {auc(s_teach_perch):.4f}")
    print(f"  Mahalanobis (Perch PCA-32):                  AUC {auc(s_mah):.4f}")

    # Rank-normalize and combine
    r_ivae = rank_norm(s_ivae)
    r_perch = rank_norm(s_perch)
    r_teach = rank_norm(s_teach_ens)
    r_mah = rank_norm(s_mah)

    print("\n=== Composite signals ===")
    print(f"  iVAE + Teacher:                   AUC {auc(0.5*r_ivae + 0.5*r_teach):.4f}")
    print(f"  Perch + Teacher:                  AUC {auc(0.5*r_perch + 0.5*r_teach):.4f}")
    print(f"  iVAE + Teacher + Mahal:           AUC {auc((r_ivae + r_teach + r_mah)/3):.4f}")
    print(f"  Perch + Teacher + Mahal:          AUC {auc((r_perch + r_teach + r_mah)/3):.4f}")
    print(f"  ALL 4 (iVAE+Perch+Teacher+Mahal): AUC {auc((r_ivae + r_perch + r_teach + r_mah)/4):.4f}")

    # Spearman correlations
    from scipy.stats import spearmanr
    print("\n=== Spearman correlations (orthogonality of signals) ===")
    signals = {"iVAE": s_ivae, "Perch_kNN": s_perch, "Teacher": s_teach_ens, "Mahal": s_mah}
    names = list(signals.keys())
    print("       " + "  ".join(f"{n:>10}" for n in names))
    for ni in names:
        row = f"{ni:<10}"
        for nj in names:
            r, _ = spearmanr(signals[ni], signals[nj])
            row += f"  {r:+.3f}    "
        print(row)

    # Save
    results = {
        "individual": {
            "iVAE": float(auc(s_ivae)),
            "Perch_kNN": float(auc(s_perch)),
            "Teacher_ens": float(auc(s_teach_ens)),
            "Teacher_perch": float(auc(s_teach_perch)),
            "Mahal": float(auc(s_mah)),
        },
        "composite": {
            "iVAE+Teacher": float(auc(0.5*r_ivae + 0.5*r_teach)),
            "Perch+Teacher": float(auc(0.5*r_perch + 0.5*r_teach)),
            "iVAE+Teacher+Mahal": float(auc((r_ivae + r_teach + r_mah)/3)),
            "Perch+Teacher+Mahal": float(auc((r_perch + r_teach + r_mah)/3)),
            "ALL4": float(auc((r_ivae + r_perch + r_teach + r_mah)/4)),
        },
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT}/results.json")


if __name__ == "__main__":
    main()
