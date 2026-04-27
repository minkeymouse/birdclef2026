#!/usr/bin/env python3
"""exp43c — Mislabel detection and entropy-guided disambiguation via iVAE z-space.

Paper thesis (from user): latent space structure lets us:
  (1) detect mislabeled clips by finding clips whose k-NN neighbors in z have
      systematically DIFFERENT labels (neighbor-agreement-weighted).
  (2) resolve AMBIGUOUS (high teacher-entropy) clips by cross-referencing with
      low-entropy neighbors in z-space.

This harness on 708 labeled SS windows validates both mechanisms BEFORE
investing in full-scale 128k iVAE training. Compares three latent spaces:
  A. Raw Perch 1536-d (baseline)
  B. PCA-32 (cheap linear)
  C. iVAE z-32 (from exp43b)

For each: measure
  - mislabel detection signal: for each window w, compute *disagreement score* =
    1 - (fraction of k-NN that share ≥1 label with w, cross-site diff-file).
    Then inject 10-20% synthetic label flips and check if disagreement score
    ranks flipped windows higher than clean ones (ROC-AUC on flip-detection).
  - entropy-guided resolution: synthesize "ambiguous" queries by removing one
    of the multi-label entries per clip; check if the removed label is
    recoverable by majority-vote over k-NN neighbors.
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
CACHE = ROOT / "experiments/exp21_outputs/perch_cache"
OUT = ROOT / "experiments/exp43c_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
Z_DIM = 32
HIDDEN = 256
EPOCHS = 200
BATCH = 64
WARMUP = 50
FREE_BITS = 0.5
K_NEIGHBORS = [5, 10, 20]
FLIP_RATE = 0.15


# ─── Data loading (shared with exp43b) ────────────────────────────────────────
def load_data():
    meta = pd.read_parquet(CACHE / "full_perch_meta.parquet")
    emb = np.load(CACHE / "full_perch_arrays.npz")["emb"].astype(np.float32)
    sites = meta["site"].astype("category")
    hours = meta["hour_utc"].astype(int).astype("category")
    site_oh = pd.get_dummies(sites).values.astype(np.float32)
    hour_oh = pd.get_dummies(hours).values.astype(np.float32)
    u = np.concatenate([site_oh, hour_oh], axis=1)

    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    Y_sc = np.zeros((len(sc), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc["lbls"]):
        for l in labs:
            if l in l2i: Y_sc[i, l2i[l]] = 1
    idx = sc.set_index("row_id")
    Y = np.stack([Y_sc[idx.index.get_loc(rid)] for rid in meta["row_id"]])
    return emb, u, Y, meta, primary


# ─── iVAE (same as exp43b) ────────────────────────────────────────────────────
class iVAE(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim, hidden=HIDDEN):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim + u_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )
        self.prior_mu = nn.Linear(u_dim, z_dim)
        self.prior_logvar = nn.Linear(u_dim, z_dim)

    def encode(self, x, u):
        h = self.enc(torch.cat([x, u], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    def forward(self, x, u):
        mu_q, lv_q = self.encode(x, u)
        z = mu_q + torch.exp(0.5 * lv_q) * torch.randn_like(lv_q)
        x_hat = self.dec(z)
        mu_p, lv_p = self.prior_mu(u), self.prior_logvar(u)
        return x_hat, mu_q, lv_q, mu_p, lv_p, z


def kl_per_dim(mu_q, lv_q, mu_p, lv_p):
    return 0.5 * (lv_p - lv_q + (torch.exp(lv_q) + (mu_q - mu_p) ** 2) / torch.exp(lv_p) - 1)


def train_ivae(model, x, u, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(u))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model.train()
    for ep in range(1, epochs + 1):
        beta = min(1.0, ep / WARMUP)
        for xb, ub in dl:
            xb = xb.to(DEVICE); ub = ub.to(DEVICE)
            x_hat, mu_q, lv_q, mu_p, lv_p, z = model(xb, ub)
            rec = ((x_hat - xb) ** 2).sum(-1).mean()
            kl = torch.clamp(kl_per_dim(mu_q, lv_q, mu_p, lv_p), min=FREE_BITS).sum(-1).mean()
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def extract_z(model, x, u):
    model.eval()
    with torch.inference_mode():
        mu, _ = model.encode(torch.from_numpy(x).to(DEVICE), torch.from_numpy(u).to(DEVICE))
    return mu.cpu().numpy()


# ─── Mislabel detection ───────────────────────────────────────────────────────
def neighbor_labels(X, files, sites, cross_site=False, k=10, oversample=400):
    """Return per-row list of neighbor indices (excluding same file, optionally same site)."""
    n = X.shape[0]
    nn_ = NearestNeighbors(n_neighbors=min(n, k + oversample), metric="cosine").fit(X)
    _, I = nn_.kneighbors(X)
    out = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        taken = 0
        for j in I[i]:
            if j == i or files[j] == files[i]:
                continue
            if cross_site and sites[j] == sites[i]:
                continue
            out[i, taken] = j
            taken += 1
            if taken >= k:
                break
    return out


def disagreement_score(Y, neigh_idx):
    """For each row, score = 1 - (avg Jaccard of label with neighbors). Higher → more likely mislabel."""
    n, k = neigh_idx.shape
    score = np.zeros(n)
    for i in range(n):
        valid = neigh_idx[i][neigh_idx[i] >= 0]
        if len(valid) == 0:
            score[i] = 0.5
            continue
        yi = Y[i]
        inter = (Y[valid] & yi).sum(1)
        union = (Y[valid] | yi).sum(1)
        jacc = np.where(union > 0, inter / union, 0.0)
        score[i] = 1.0 - jacc.mean()
    return score


def flip_and_score(X, Y, files, sites, primary_labels, flip_rate=FLIP_RATE, k=10, seed=0, cross_site=True):
    """Inject random label flips, measure flip-detection ROC-AUC."""
    rng = np.random.default_rng(seed)
    n, C = Y.shape
    n_flip = int(n * flip_rate)
    flip_idx = rng.choice(n, size=n_flip, replace=False)

    Y_dirty = Y.copy()
    for i in flip_idx:
        # remove one existing label (if any) and add one random other
        present = np.where(Y_dirty[i] == 1)[0]
        if len(present) > 0:
            drop = rng.choice(present)
            Y_dirty[i, drop] = 0
        add = rng.integers(0, C)
        Y_dirty[i, add] = 1

    is_flipped = np.zeros(n, dtype=np.uint8)
    is_flipped[flip_idx] = 1

    neigh = neighbor_labels(X, files, sites, cross_site=cross_site, k=k)
    score = disagreement_score(Y_dirty, neigh)
    # skip rows that found no valid neighbors
    mask = (neigh >= 0).any(1)
    if mask.sum() < 10 or is_flipped[mask].sum() == 0:
        return None
    auc = roc_auc_score(is_flipped[mask], score[mask])
    return auc, score, is_flipped


# ─── Entropy-guided label recovery ────────────────────────────────────────────
def label_recovery(X, Y, files, sites, k=10, cross_site=True, seed=0):
    """For each multi-label row, remove ONE label, try to recover by neighbor vote.
       Recovery accuracy = frac of 'removed label' ranked in top-3 of neighbor vote."""
    rng = np.random.default_rng(seed)
    n, C = Y.shape
    neigh = neighbor_labels(X, files, sites, cross_site=cross_site, k=k)

    top1_hits = 0
    top3_hits = 0
    evaluated = 0
    for i in range(n):
        present = np.where(Y[i] == 1)[0]
        if len(present) < 2:
            continue  # need ≥2 labels to remove one and have ground truth
        removed = rng.choice(present)
        Y_query = Y[i].copy()
        Y_query[removed] = 0

        valid = neigh[i][neigh[i] >= 0]
        if len(valid) == 0:
            continue

        # Neighbor vote: sum of neighbor label vectors (exclude already-known labels)
        vote = Y[valid].astype(np.float32).sum(0)
        vote[Y_query == 1] = -1  # exclude known-present
        # top-k candidate classes
        top = np.argsort(-vote)[:3]
        if top[0] == removed:
            top1_hits += 1
        if removed in top:
            top3_hits += 1
        evaluated += 1
    if evaluated == 0:
        return None
    return top1_hits / evaluated, top3_hits / evaluated, evaluated


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    emb, u, Y, meta, primary = load_data()
    files = meta["filename"].values
    sites = meta["site"].values
    print(f"Data: emb {emb.shape}  Y {Y.shape}  positives {Y.sum()}  "
          f"multi-label rows {(Y.sum(1) >= 2).sum()}")

    mu_x = emb.mean(0, keepdims=True); sd_x = emb.std(0, keepdims=True) + 1e-6
    emb_z = ((emb - mu_x) / sd_x).astype(np.float32)

    pca = PCA(n_components=Z_DIM, random_state=42)
    emb_pca32 = pca.fit_transform(emb_z).astype(np.float32)

    print(f"\n[iVAE] training on 708 labeled windows")
    model = iVAE(emb_z.shape[1], u.shape[1], Z_DIM).to(DEVICE)
    train_ivae(model, emb_z, u, epochs=EPOCHS)
    Z_ivae = extract_z(model, emb_z, u).astype(np.float32)
    print(f"  z_std {Z_ivae.std(0).mean():.3f}")

    spaces = {
        "Perch-1536 std": emb_z,
        "PCA-32": emb_pca32,
        "iVAE-32": Z_ivae,
        "iVAE-32 + PCA-32": np.concatenate([Z_ivae, emb_pca32], axis=1),
    }

    results = {"mislabel_auc": {}, "recovery": {}}

    # ── (1) Mislabel detection: inject 15% flips, predict with disagreement score
    print("\n=== Mislabel detection (15% label flips, cross-site kNN) ===")
    print(f"{'space':<22} | AUC@k=5   AUC@k=10  AUC@k=20")
    for name, X in spaces.items():
        row = {}
        line = f"{name:<22} | "
        for k in K_NEIGHBORS:
            out = flip_and_score(X, Y, files, sites, primary, flip_rate=FLIP_RATE, k=k, seed=42, cross_site=True)
            auc = out[0] if out else float("nan")
            row[f"k{k}"] = auc
            line += f"{auc:.4f}   "
        print(line)
        results["mislabel_auc"][name] = row

    # ── (2) Label recovery: remove 1 label from multi-label clips, recover by neighbor vote
    print("\n=== Label recovery (remove 1 label from multi-label, cross-site vote) ===")
    print(f"{'space':<22} | top1@k=5   top3@k=5   top1@k=10  top3@k=10")
    for name, X in spaces.items():
        row = {}
        line = f"{name:<22} | "
        for k in [5, 10]:
            out = label_recovery(X, Y, files, sites, k=k, cross_site=True, seed=42)
            if out is None:
                row[f"top1@k{k}"] = row[f"top3@k{k}"] = float("nan")
                line += "     nan        nan    "
                continue
            t1, t3, n = out
            row[f"top1@k{k}"] = t1; row[f"top3@k{k}"] = t3; row[f"n@k{k}"] = n
            line += f"{t1:.3f}      {t3:.3f}     "
        print(line + f"  (n={row['n@k5']})")
        results["recovery"][name] = row

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved → {OUT}/results.json")


if __name__ == "__main__":
    main()
