#!/usr/bin/env python3
"""exp43 — iVAE on Perch embeddings for site-invariant species representation.

Setup:
  x = Perch 1536-dim embedding (from jaejohn cache, labeled SS)
  u = auxiliary (site one-hot + hour one-hot) → identifies conditional factorization
  z = 32-dim latent (identifiable up to permutation/scaling by iVAE theorem)

Training:
  Encoder q(z|x,u) = N(mu_phi(x,u), sigma_phi(x,u))
  Decoder p(x|z) = N(mu_theta(z), sigma_theta)
  Prior p(z|u) = N(mu_lambda(u), sigma_lambda(u))   ← KEY for identifiability
  ELBO = E_q[log p(x|z)] - KL(q(z|x,u) || p(z|u))

After training:
  - Check site/hour clustering in z (should still separate if u-conditional)
  - Extract z for each of 708 labeled SS chunks
  - Re-fit LogReg probe on z (instead of PCA-32 Perch) → test if probe AUC improves
  - Also: compute OOD score = -log p(z|u) for unlabeled SS (needs Perch on unlabeled, NA)

Practical scope: just labeled SS (708 samples). iVAE with only 708 samples is
underpowered but we can still see if latent separates semantically.
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
CACHE = ROOT / "experiments/exp21_outputs/perch_cache"
OUT = ROOT / "experiments/exp43_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
Z_DIM = 32
HIDDEN = 256
EPOCHS = 300
LR = 1e-3
BATCH = 64


def load_data():
    """Load Perch cache + metadata. Returns (X, u, Y, filename_meta)."""
    meta = pd.read_parquet(CACHE / "full_perch_meta.parquet")
    emb = np.load(CACHE / "full_perch_arrays.npz")["emb"].astype(np.float32)  # (708, 1536)
    print(f"Perch emb: {emb.shape}  meta: {meta.shape}")

    # Build u: site one-hot + hour one-hot
    sites = meta["site"].astype("category")
    hours = meta["hour_utc"].astype(int).astype("category")
    site_oh = pd.get_dummies(sites).values.astype(np.float32)
    hour_oh = pd.get_dummies(hours).values.astype(np.float32)
    u = np.concatenate([site_oh, hour_oh], axis=1)
    print(f"u: {u.shape}  (sites={site_oh.shape[1]}, hours={hour_oh.shape[1]})")

    # Load labels
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
        # Encoder q(z | x, u)
        self.enc = nn.Sequential(
            nn.Linear(x_dim + u_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        # Decoder p(x | z)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

        # Prior p(z | u) — identifiability driver
        self.prior_mu = nn.Linear(u_dim, z_dim)
        self.prior_logvar = nn.Linear(u_dim, z_dim)

    def encode(self, x, u):
        h = self.enc(torch.cat([x, u], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z): return self.dec(z)

    def prior(self, u):
        return self.prior_mu(u), self.prior_logvar(u)

    def forward(self, x, u):
        mu_q, logvar_q = self.encode(x, u)
        z = self.reparam(mu_q, logvar_q)
        x_hat = self.decode(z)
        mu_p, logvar_p = self.prior(u)
        return x_hat, mu_q, logvar_q, mu_p, logvar_p, z


def kl_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)


def train_ivae(model, x, u, epochs=EPOCHS, lr=LR, batch=BATCH):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(u))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    model.train()
    beta = 1.0
    for ep in range(1, epochs + 1):
        total_rec = total_kl = 0.0
        for xb, ub in dl:
            xb = xb.to(DEVICE); ub = ub.to(DEVICE)
            x_hat, mu_q, lv_q, mu_p, lv_p, z = model(xb, ub)
            rec = F.mse_loss(x_hat, xb, reduction="mean")
            kl = kl_gaussian(mu_q, lv_q, mu_p, lv_p).sum(-1).mean()
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            total_rec += rec.item() * xb.size(0)
            total_kl += kl.item() * xb.size(0)
        if ep % 30 == 0 or ep == 1:
            print(f"  ep {ep:03d}  rec {total_rec/len(ds):.4f}  kl {total_kl/len(ds):.2f}")


def extract_z(model, x, u):
    model.eval()
    xb = torch.from_numpy(x).to(DEVICE)
    ub = torch.from_numpy(u).to(DEVICE)
    with torch.inference_mode():
        mu, _ = model.encode(xb, ub)
    return mu.cpu().numpy()


def probe_auc(Z, Y, n_splits=5):
    """5-fold CV LogReg probe on Z, per-class macro AUC averaged."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Need to stratify by something; use sum of labels (non-ideal but workable)
    strat = Y.sum(1).clip(0, 5)
    oof = np.zeros_like(Y, dtype=np.float32)
    for tr, va in skf.split(Z, strat):
        for c in range(Y.shape[1]):
            if Y[tr, c].sum() < 1: continue
            clf = LogisticRegression(C=0.25, max_iter=200, n_jobs=-1)
            try:
                clf.fit(Z[tr], Y[tr, c])
                oof[va, c] = clf.predict_proba(Z[va])[:, 1]
            except Exception:
                pass
    keep = Y.sum(0) > 0
    return float(roc_auc_score(Y[:, keep], oof[:, keep], average="macro"))


def main():
    emb, u, Y, meta = load_data()

    # Check baselines
    print("\n=== Baselines ===")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=32, random_state=42)
    emb_pca32 = pca.fit_transform(emb.astype(np.float32))
    auc_pca = probe_auc(emb_pca32, Y)
    print(f"PCA-32 probe OOF macro-AUC: {auc_pca:.4f}")

    auc_raw = probe_auc(emb, Y)
    print(f"Raw 1536 probe OOF macro-AUC: {auc_raw:.4f}")

    # Train iVAE
    print("\n=== Training iVAE ===")
    x_dim = emb.shape[1]; u_dim = u.shape[1]
    model = iVAE(x_dim, u_dim, Z_DIM).to(DEVICE)
    t0 = time.time()
    train_ivae(model, emb, u, epochs=EPOCHS)
    print(f"iVAE trained in {time.time()-t0:.0f}s")

    # Extract Z
    Z = extract_z(model, emb, u)
    print(f"\nZ shape: {Z.shape}  mean std: {Z.std(0).mean():.3f}")

    # Probe on Z
    auc_z = probe_auc(Z, Y)
    print(f"iVAE Z-32 probe OOF macro-AUC: {auc_z:.4f}")

    # Compare: does iVAE z separate species better than PCA?
    print(f"\n=== Comparison ===")
    print(f"  PCA-32: {auc_pca:.4f}")
    print(f"  Raw:    {auc_raw:.4f}")
    print(f"  iVAE-32: {auc_z:.4f}  (Δ vs PCA: {auc_z - auc_pca:+.4f})")

    np.savez_compressed(OUT / "results.npz",
                        Z=Z, emb_pca32=emb_pca32,
                        auc_pca=auc_pca, auc_raw=auc_raw, auc_ivae=auc_z)
    print(f"Saved: {OUT}/results.npz")


if __name__ == "__main__":
    main()
