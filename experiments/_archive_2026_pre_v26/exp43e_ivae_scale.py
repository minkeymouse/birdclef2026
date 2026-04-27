#!/usr/bin/env python3
"""exp43e — iVAE on full 128k Perch embeddings (labeled + unlabeled SS).

Paper's decisive test: at n=708, iVAE-z vs raw-Perch-kNN Spearman was 0.84
(near-redundant). Does scale (n=128k) unlock iVAE's identifiability
advantage? Gate: cross-site flip-detection AUC on labeled subset must
exceed raw-Perch 0.606 by ≥0.02 at k=10.

Training:
  x = Perch 1536-d (standardized)
  u = site one-hot (23) + hour one-hot (18) = 41-d
  z = 32-d identifiable latent
  Encoder: 2-layer MLP 512 hidden
  Decoder: 2-layer MLP 512 hidden
  Prior p(z|u): linear mu + logvar from u (Khemakhem 2020)
  Loss: rec (sum over x_dim) + β * KL (free-bits 0.5, warmup 0→1 over 10 ep)
  Optim: AdamW 3e-4, cosine, 30 epochs
  Batch: 512. Est wall: ~2-3 min/epoch × 30 = 60-90 min on RTX 5090

Outputs:
  exp43e_outputs/ivae_ckpt.pt
  exp43e_outputs/z_all.npz       — z for all 127,896 windows
  exp43e_outputs/flip_auc.json   — validation vs raw Perch on labeled subset
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
OUT = ROOT / "experiments/exp43e_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
Z_DIM = 32
HIDDEN = 512
EPOCHS = 30
BATCH = 512
WARMUP = 10
FREE_BITS = 0.5
LR = 3e-4
K_FLIP = [5, 10, 20]
FLIP_RATE = 0.15


# ─── Data load ────────────────────────────────────────────────────────────
def load_all():
    emb = np.load(EXP43A / "perch_ss_all.npz")["emb"].astype(np.float32)
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    sites = meta["site"].astype("category")
    hours = meta["hour_utc"].astype(int).astype("category")
    u = np.concatenate([
        pd.get_dummies(sites).values.astype(np.float32),
        pd.get_dummies(hours).values.astype(np.float32),
    ], axis=1)
    return emb, u, meta


def load_labeled_subset(meta):
    """Identify rows in meta that match labeled SS. Returns (mask, Y)."""
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

    rid_to_labels = {}
    for _, r in sc_g.iterrows():
        Y = np.zeros(len(primary), dtype=np.uint8)
        for l in r.lbls:
            if l in l2i: Y[l2i[l]] = 1
        rid_to_labels[r.row_id] = Y

    mask = np.zeros(len(meta), dtype=bool)
    Y = np.zeros((len(meta), len(primary)), dtype=np.uint8)
    for i, rid in enumerate(meta["row_id"].values):
        if rid in rid_to_labels:
            mask[i] = True
            Y[i] = rid_to_labels[rid]
    return mask, Y


# ─── Model ────────────────────────────────────────────────────────────────
class iVAE(nn.Module):
    def __init__(self, x_dim, u_dim, z_dim, hidden=HIDDEN):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim + u_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_lv = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, x_dim),
        )
        self.prior_mu = nn.Linear(u_dim, z_dim)
        self.prior_lv = nn.Linear(u_dim, z_dim)

    def encode(self, x, u):
        h = self.enc(torch.cat([x, u], -1))
        return self.enc_mu(h), self.enc_lv(h)

    def forward(self, x, u):
        mu_q, lv_q = self.encode(x, u)
        z = mu_q + torch.exp(0.5 * lv_q) * torch.randn_like(lv_q)
        return self.dec(z), mu_q, lv_q, self.prior_mu(u), self.prior_lv(u)


def kl_pd(mu_q, lv_q, mu_p, lv_p):
    return 0.5 * (lv_p - lv_q + (torch.exp(lv_q) + (mu_q - mu_p) ** 2) / torch.exp(lv_p) - 1)


# ─── Train ────────────────────────────────────────────────────────────────
def train(model, x, u):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    dl = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(u)),
        batch_size=BATCH, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True,
    )
    history = []
    model.train()
    for ep in range(1, EPOCHS + 1):
        beta = min(1.0, ep / WARMUP)
        rec_sum = kl_sum = zstd_sum = n_seen = 0
        t0 = time.time()
        for xb, ub in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            ub = ub.to(DEVICE, non_blocking=True)
            x_hat, mu_q, lv_q, mu_p, lv_p = model(xb, ub)
            rec = ((x_hat - xb) ** 2).sum(-1).mean()
            kl_d = kl_pd(mu_q, lv_q, mu_p, lv_p)
            kl_fb = torch.clamp(kl_d, min=FREE_BITS).sum(-1).mean()
            loss = rec + beta * kl_fb
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            rec_sum += rec.item() * xb.size(0)
            kl_sum += kl_d.sum(-1).mean().item() * xb.size(0)
            zstd_sum += mu_q.std(0).mean().item() * xb.size(0)
            n_seen += xb.size(0)
        sched.step()
        dt = time.time() - t0
        row = {"ep": ep, "beta": beta, "rec": rec_sum/n_seen, "kl": kl_sum/n_seen,
               "zstd": zstd_sum/n_seen, "time_s": dt}
        history.append(row)
        print(f"  ep {ep:02d}  β={beta:.2f}  rec {row['rec']:.1f}  kl {row['kl']:.2f}  "
              f"z_std {row['zstd']:.3f}  ({dt:.0f}s)")
    return history


def extract_z(model, x, u, batch=2048):
    model.eval()
    n = x.shape[0]
    Z = np.zeros((n, Z_DIM), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, n, batch):
            xb = torch.from_numpy(x[i:i+batch]).to(DEVICE)
            ub = torch.from_numpy(u[i:i+batch]).to(DEVICE)
            mu, _ = model.encode(xb, ub)
            Z[i:i+batch] = mu.cpu().numpy()
    return Z


# ─── Eval: flip-detection AUC on labeled subset ──────────────────────────
def knn_disagree(X, Y, files, sites, k, cross_site=True, sample=None, seed=0):
    n = X.shape[0]
    if sample and n > sample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=sample, replace=False)
        X, Y, files, sites = X[sel], Y[sel], files[sel], sites[sel]
    nn_ = NearestNeighbors(n_neighbors=min(len(X), k + 400), metric="cosine").fit(X)
    _, I = nn_.kneighbors(X)
    score = np.zeros(len(X))
    for i in range(len(X)):
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


def inject_flips(Y, rate=FLIP_RATE, seed=0):
    rng = np.random.default_rng(seed)
    n, C = Y.shape
    n_flip = int(n * rate)
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    Y_d = Y.copy()
    for i in flip_idx:
        pres = np.where(Y_d[i] == 1)[0]
        if len(pres) > 0:
            Y_d[i, rng.choice(pres)] = 0
        Y_d[i, rng.integers(0, C)] = 1
    is_flip = np.zeros(n, dtype=np.uint8); is_flip[flip_idx] = 1
    return Y_d, is_flip


def main():
    print("Loading 128k Perch embeddings + metadata...")
    emb, u, meta = load_all()
    mask, Y_lab = load_labeled_subset(meta)
    print(f"  total {emb.shape}  labeled subset {mask.sum()}/{mask.size}")

    mu_x = emb.mean(0, keepdims=True); sd_x = emb.std(0, keepdims=True) + 1e-6
    x = ((emb - mu_x) / sd_x).astype(np.float32)

    # Train
    print(f"\nTraining iVAE  z_dim={Z_DIM}  batch={BATCH}  epochs={EPOCHS}")
    model = iVAE(x.shape[1], u.shape[1], Z_DIM).to(DEVICE)
    history = train(model, x, u)
    ckpt = {"state_dict": model.state_dict(), "z_dim": Z_DIM,
            "x_mean": mu_x, "x_std": sd_x, "history": history}
    torch.save(ckpt, OUT / "ivae_ckpt.pt")

    # Extract z for all 128k windows
    print("\nExtracting z for all windows...")
    Z = extract_z(model, x, u)
    np.savez_compressed(OUT / "z_all.npz", z=Z)
    print(f"  z {Z.shape}  std/dim mean {Z.std(0).mean():.3f}")

    # Flip detection on labeled subset
    print("\n=== Flip-detection on labeled subset (cross-site kNN) ===")
    Y_lab_d, is_flip = inject_flips(Y_lab[mask], rate=FLIP_RATE, seed=42)
    files_lab = meta["filename"].values[mask]
    sites_lab = meta["site"].values[mask]

    results = {}
    for name, X_lab in [("raw_perch", x[mask]), ("ivae_z", Z[mask])]:
        row = {}
        for k in K_FLIP:
            s = knn_disagree(X_lab, Y_lab_d, files_lab, sites_lab, k=k, cross_site=True)
            auc = roc_auc_score(is_flip, s)
            row[f"k{k}"] = auc
            print(f"  {name:<12} k={k:2d}  AUC {auc:.4f}")
        results[name] = row

    # also with neighbors drawn from FULL 128k (not just labeled)
    print("\n=== Flip-detection with full-128k neighbor pool ===")
    for name, X_all, X_lab in [("raw_perch_full", x, x[mask]),
                               ("ivae_z_full",   Z, Z[mask])]:
        row = {}
        for k in K_FLIP:
            # need custom: query is labeled mask, search pool is all
            nn_ = NearestNeighbors(n_neighbors=k + 400, metric="cosine").fit(X_all)
            _, I = nn_.kneighbors(X_lab)
            score = np.zeros(len(X_lab))
            files_all = meta["filename"].values
            sites_all = meta["site"].values
            for i_l in range(len(X_lab)):
                # find original idx in full
                orig_idx = np.where(mask)[0][i_l]
                neigh = []
                for j in I[i_l]:
                    if j == orig_idx or files_all[j] == files_all[orig_idx]: continue
                    if sites_all[j] == sites_all[orig_idx]: continue
                    neigh.append(j)
                    if len(neigh) >= k: break
                if not neigh:
                    score[i_l] = 0.5; continue
                # Y for neighbors is ONLY defined for labeled ones; for unlabeled, use "no information" → skip
                neigh_lab = [j for j in neigh if mask[j]]
                if not neigh_lab:
                    score[i_l] = 0.5; continue
                yi = Y_lab_d[i_l]
                Y_n = Y_lab[neigh_lab]   # ground truth for labeled neighbors
                inter = (Y_n & yi).sum(1); union = (Y_n | yi).sum(1)
                jacc = np.where(union > 0, inter / union, 0.0)
                score[i_l] = 1.0 - jacc.mean()
            auc = roc_auc_score(is_flip, score)
            row[f"k{k}"] = auc
            print(f"  {name:<15} k={k:2d}  AUC {auc:.4f}")
        results[name] = row

    # Spearman between ivae-z and raw-perch kNN disagreement (redundancy check)
    s_raw = knn_disagree(x[mask], Y_lab_d, files_lab, sites_lab, k=10, cross_site=True)
    s_iv  = knn_disagree(Z[mask], Y_lab_d, files_lab, sites_lab, k=10, cross_site=True)
    from scipy.stats import spearmanr
    r, _ = spearmanr(s_raw, s_iv)
    print(f"\nSpearman(raw_perch_kNN, ivae_z_kNN) disagreement scores: {r:+.3f}")
    print(f"  (at n=708 was 0.84 — redundant; target at n=128k: ≤ 0.5 for distinct signals)")
    results["spearman_raw_ivae"] = float(r)

    with open(OUT / "flip_auc.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}/flip_auc.json")


if __name__ == "__main__":
    main()
