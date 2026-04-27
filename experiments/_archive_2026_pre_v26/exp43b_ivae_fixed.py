#!/usr/bin/env python3
"""exp43b — iVAE on labeled SS Perch, with proper ELBO and anti-collapse tricks.

Fixes vs exp43:
  1. Rec loss reduction="sum"/B (not mean over x_dim) → rec and kl comparable scale
  2. Standardize Perch embeddings (zero-mean unit-var per dim)
  3. β annealing 0 → 1 over first 50 epochs (KL warmup)
  4. Free-bits: max(kl_per_dim, FB) prevents per-dim collapse
  5. Report per-epoch rec / kl / z_std to detect collapse
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
CACHE = ROOT / "experiments/exp21_outputs/perch_cache"
OUT = ROOT / "experiments/exp43b_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
Z_DIM = 32
HIDDEN = 256
EPOCHS = 200
BATCH = 64
WARMUP = 50
FREE_BITS = 0.5  # nats per dim floor


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

    return emb, u, Y, meta


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

    def reparam(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def forward(self, x, u):
        mu_q, lv_q = self.encode(x, u)
        z = self.reparam(mu_q, lv_q)
        x_hat = self.dec(z)
        mu_p, lv_p = self.prior_mu(u), self.prior_logvar(u)
        return x_hat, mu_q, lv_q, mu_p, lv_p, z


def kl_per_dim(mu_q, lv_q, mu_p, lv_p):
    """Per-dim KL(q||p) for diagonal Gaussians. Shape (B, z_dim)."""
    var_q = torch.exp(lv_q); var_p = torch.exp(lv_p)
    return 0.5 * (lv_p - lv_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)


def train_ivae(model, x, u, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(u))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    model.train()
    x_dim = x.shape[1]
    for ep in range(1, epochs + 1):
        beta = min(1.0, ep / WARMUP)
        rec_sum = kl_sum = zstd_sum = n = 0
        for xb, ub in dl:
            xb = xb.to(DEVICE); ub = ub.to(DEVICE)
            x_hat, mu_q, lv_q, mu_p, lv_p, z = model(xb, ub)
            # rec: sum over x_dim, mean over batch — elbo scale
            rec = ((x_hat - xb) ** 2).sum(-1).mean()
            kl_d = kl_per_dim(mu_q, lv_q, mu_p, lv_p)           # (B, z_dim)
            # free bits: floor each dim at FREE_BITS nats
            kl_fb = torch.clamp(kl_d, min=FREE_BITS).sum(-1).mean()
            loss = rec + beta * kl_fb
            opt.zero_grad(); loss.backward(); opt.step()
            rec_sum += rec.item() * xb.size(0)
            kl_sum += kl_d.sum(-1).mean().item() * xb.size(0)
            zstd_sum += z.std(0).mean().item() * xb.size(0)
            n += xb.size(0)
        if ep % 20 == 0 or ep <= 3:
            print(f"  ep {ep:03d}  β={beta:.2f}  rec {rec_sum/n:.2f}  "
                  f"kl {kl_sum/n:.2f}  z_std {zstd_sum/n:.3f}")


def extract_z(model, x, u):
    model.eval()
    with torch.inference_mode():
        mu, _ = model.encode(torch.from_numpy(x).to(DEVICE),
                             torch.from_numpy(u).to(DEVICE))
    return mu.cpu().numpy()


def probe_auc(Z, Y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    strat = Y.sum(1).clip(0, 5)
    oof = np.zeros_like(Y, dtype=np.float32)
    for tr, va in skf.split(Z, strat):
        for c in range(Y.shape[1]):
            if Y[tr, c].sum() < 1: continue
            try:
                clf = LogisticRegression(C=0.25, max_iter=200)
                clf.fit(Z[tr], Y[tr, c])
                oof[va, c] = clf.predict_proba(Z[va])[:, 1]
            except Exception:
                pass
    keep = Y.sum(0) > 0
    return float(roc_auc_score(Y[:, keep], oof[:, keep], average="macro"))


def main():
    emb, u, Y, meta = load_data()
    print(f"emb {emb.shape}  u {u.shape}  Y positives {Y.sum():d}")

    # Standardize
    mu_x = emb.mean(0, keepdims=True)
    sd_x = emb.std(0, keepdims=True) + 1e-6
    emb_z = ((emb - mu_x) / sd_x).astype(np.float32)

    # Baselines on standardized emb
    pca = PCA(n_components=32, random_state=42)
    emb_pca32 = pca.fit_transform(emb_z)
    auc_pca = probe_auc(emb_pca32, Y)
    auc_raw = probe_auc(emb_z, Y)
    print(f"\nPCA-32 probe AUC: {auc_pca:.4f}")
    print(f"Raw 1536 probe AUC: {auc_raw:.4f}")

    print(f"\n=== Training iVAE (z={Z_DIM}, warmup={WARMUP}, free_bits={FREE_BITS}) ===")
    model = iVAE(emb_z.shape[1], u.shape[1], Z_DIM).to(DEVICE)
    t0 = time.time()
    train_ivae(model, emb_z, u, epochs=EPOCHS)
    print(f"iVAE trained in {time.time()-t0:.0f}s")

    Z = extract_z(model, emb_z, u)
    print(f"\nZ std per dim mean: {Z.std(0).mean():.3f}")

    auc_z = probe_auc(Z, Y)
    print(f"\niVAE Z-32 probe AUC: {auc_z:.4f}")
    print(f"  Δ vs PCA-32: {auc_z - auc_pca:+.4f}")
    print(f"  Δ vs Raw 1536: {auc_z - auc_raw:+.4f}")

    # Also: concat iVAE-z + PCA-32 → does combining help?
    Z_cat = np.concatenate([Z, emb_pca32], axis=1)
    auc_cat = probe_auc(Z_cat, Y)
    print(f"iVAE-Z + PCA-32 (64d) probe AUC: {auc_cat:.4f}  "
          f"(Δ vs PCA: {auc_cat - auc_pca:+.4f})")

    np.savez_compressed(OUT / "results.npz",
                        Z=Z, emb_pca32=emb_pca32,
                        auc_pca=auc_pca, auc_raw=auc_raw,
                        auc_ivae=auc_z, auc_cat=auc_cat)


if __name__ == "__main__":
    main()
