#!/usr/bin/env python3
"""exp43k — iVDFM (Chang 2026, ucrl-iclr2026-ivdfm) applied to BirdCLEF.

Key differences from exp43e (which failed with Spearman 0.90 redundancy):
  1. TIME-SERIES input: y_t = Perch spatial_embedding[t] ∈ R^1536, T=16 per window
  2. INNOVATION-level conditioning (not latent state):
         η_t ~ p(η_t | u_t, e_t)  — Laplace, component-wise
  3. REGIME embedding:
         e_t = Σ_k π_{t,k} e_k,  π_t = softmax(RegimeNet(u_t)),  K=4
  4. LINEAR DIAGONAL DYNAMICS:
         f_{t+1} = diag(A_bar) f_t + diag(B_bar) η_t
      (optionally with regime mix)
  5. NON-GAUSSIAN innovation (Laplace); Gaussian is degenerate (paper App).

Architecture:
  Encoder  q_φ(η_t | y_{1:T}, u_t, e_t):
      BiGRU over (y_t ⊕ u_t ⊕ e_t) → per-time-step μ, logσ for η_t.
      Reparameterize with Laplace (location/scale) — NOT Gaussian.
  Prior    p_θ(η_t | u_t, e_t) = Laplace(μ_λ(u,e), σ_λ(u,e)) component-wise
      μ_λ, σ_λ are linear from concat(u_t, e_t).
  Dynamics f_0 = 0; f_{t+1} = a ⊙ f_t + b ⊙ η_t,  a, b ∈ R^r learned (diagonal)
  Decoder  g(f_t) → y_hat_t, MLP 2-layer, 1536-d output.
  ELBO     Σ_t [log p(y_t | f_t) − KL(q_φ(η_t) || p_θ(η_t | u_t, e_t))]

BirdCLEF auxiliary:
  u_t = concat(site_one_hot, hour_one_hot, position_in_window_one_hot(t))
       = (23 + 18 + 16) = 57 dim

Gates (for paper/LB viability):
  A1. flip-AUC @k=10 on labeled SS > raw-Perch 0.606 + 0.02 = 0.626
  A2. Spearman(iVDFM-f-kNN, raw-Perch-kNN) ≤ 0.50 (distinct signal)
  A3. Regime posterior π_t clusters meaningfully by taxa (qualitative check)
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43J = ROOT / "experiments/exp43j_outputs"
OUT = ROOT / "experiments/exp43k_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"

# Model hyperparameters
T_LEN = 16       # time steps per 5-sec window
X_DIM = 1536     # per-time-step feature
R_DIM = 16       # innovation/factor dim
K_REG = 4        # regimes
E_DIM = 16       # regime embedding dim
HIDDEN = 256

# Training
EPOCHS = 25
BATCH = 128      # 128 windows = 2048 time steps per batch
WARMUP = 8
FREE_BITS = 0.3  # nats per innovation dim floor
LR = 3e-4
FLIP_RATE = 0.15
K_FLIP = [5, 10, 20]


# ─── Model ────────────────────────────────────────────────────────────────
class RegimeNet(nn.Module):
    """π_t = softmax(MLP(u_t)) ∈ Δ^{K-1}. Deterministic given u_t."""
    def __init__(self, u_dim, K=K_REG, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(u_dim, hidden), nn.GELU(),
            nn.Linear(hidden, K),
        )
    def forward(self, u):
        return F.softmax(self.net(u), dim=-1)   # (..., K)


class iVDFM(nn.Module):
    def __init__(self, x_dim=X_DIM, r_dim=R_DIM, u_dim=None,
                 K=K_REG, e_dim=E_DIM, hidden=HIDDEN):
        super().__init__()
        assert u_dim is not None
        self.r_dim, self.x_dim, self.K = r_dim, x_dim, K
        # Regime
        self.regime_net = RegimeNet(u_dim, K)
        self.regime_codes = nn.Parameter(torch.randn(K, e_dim) * 0.1)  # e_k
        # Prior on innovation: Laplace(μ, b) component-wise; params from (u,e)
        ue_dim = u_dim + e_dim
        self.prior_mu = nn.Linear(ue_dim, r_dim)
        self.prior_log_scale = nn.Linear(ue_dim, r_dim)  # log b
        # Encoder: BiGRU over (y, u, e) → per-step posterior q(η | ·)
        enc_in_dim = x_dim + ue_dim
        self.enc_rnn = nn.GRU(enc_in_dim, hidden, num_layers=2, bidirectional=True,
                              batch_first=True, dropout=0.1)
        self.enc_mu = nn.Linear(hidden * 2, r_dim)
        self.enc_log_scale = nn.Linear(hidden * 2, r_dim)
        # Dynamics: diagonal A, B (optionally regime-mixed); here K regimes × diag
        self.A_diag = nn.Parameter(torch.ones(K, r_dim) * 0.9)   # init near stable
        self.B_diag = nn.Parameter(torch.ones(K, r_dim))
        # Decoder: MLP f_t → y_t
        self.dec = nn.Sequential(
            nn.Linear(r_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, x_dim),
        )

    def regime(self, u):
        """Returns (π (..., K), e (..., E))."""
        pi = self.regime_net(u)
        e = pi @ self.regime_codes                 # deterministic expected embedding
        return pi, e

    def prior_params(self, u, e):
        ue = torch.cat([u, e], dim=-1)
        return self.prior_mu(ue), self.prior_log_scale(ue).clamp(min=-3.0, max=3.0)

    def encode(self, y, u, e):
        """
        y: (B, T, x_dim)   u: (B, T, u_dim)   e: (B, T, e_dim)
        returns mu, log_scale: (B, T, r_dim)
        log_scale clamped to [-3, 3] so b ∈ [0.05, 20] — prevents Laplace explosion.
        """
        h = torch.cat([y, u, e], dim=-1)
        h, _ = self.enc_rnn(h)
        return self.enc_mu(h), self.enc_log_scale(h).clamp(min=-3.0, max=3.0)

    @staticmethod
    def sample_laplace(mu, log_scale):
        """Reparameterized Laplace: mu + b * (-sign(u) * log(1 - 2|u|)).
           Bounded: clip noise term to prevent extreme tail samples."""
        b = torch.exp(log_scale).clamp(min=1e-4, max=20.0)
        u = torch.rand_like(mu) - 0.5              # Uniform(-0.5, 0.5)
        # Use 1e-4 floor for log1p arg → max |noise|/b ≤ log(1e4) ≈ 9.2
        noise = -torch.sign(u) * torch.log1p(-2 * u.abs() + 1e-4)
        return mu + b * noise

    def dynamics(self, pi, eta):
        """
        Apply regime-mixed diagonal dynamics: f_{t+1} = a_eff ⊙ f_t + b_eff ⊙ η_t
        with a_eff = Σ π_{t,k} a_k (element-wise).
        pi: (B, T, K)    eta: (B, T, r)
        Returns f: (B, T, r).
        """
        a_eff = pi @ self.A_diag                   # (B, T, r)
        b_eff = pi @ self.B_diag
        B, T, r = eta.shape
        f = torch.zeros(B, r, device=eta.device)
        outs = []
        for t in range(T):
            f = a_eff[:, t] * f + b_eff[:, t] * eta[:, t]
            outs.append(f)
        return torch.stack(outs, dim=1)

    def forward(self, y, u):
        pi, e = self.regime(u)                     # (B, T, K), (B, T, e_dim)
        mu_q, ls_q = self.encode(y, u, e)
        eta = self.sample_laplace(mu_q, ls_q)
        mu_p, ls_p = self.prior_params(u, e)
        f = self.dynamics(pi, eta)
        y_hat = self.dec(f)
        return {"y_hat": y_hat, "mu_q": mu_q, "ls_q": ls_q,
                "mu_p": mu_p, "ls_p": ls_p, "eta": eta, "f": f, "pi": pi}


def kl_laplace(mu_q, ls_q, mu_p, ls_p):
    """KL(Laplace(μ_q, b_q) || Laplace(μ_p, b_p)) component-wise.
       See e.g. Meyer (2021): KL = log(b_p/b_q) + (|μ_q−μ_p| + b_q exp(−|μ_q−μ_p|/b_q)) / b_p − 1
    """
    b_q = torch.exp(ls_q).clamp(min=1e-6)
    b_p = torch.exp(ls_p).clamp(min=1e-6)
    d = (mu_q - mu_p).abs()
    return ls_p - ls_q + (d + b_q * torch.exp(-d / b_q)) / b_p - 1.0


# ─── Data ──────────────────────────────────────────────────────────────────
def load_data(labeled_only=False):
    spat = np.load(EXP43J / "spatial_ss_all.npz")["spatial"]   # (N, 16, 1536) fp16
    meta = pd.read_parquet(EXP43J / "spatial_ss_all_meta.parquet")
    print(f"spatial: {spat.shape} {spat.dtype}  meta: {len(meta)}")
    return spat, meta


def build_u(meta, T=T_LEN):
    """u_t has shape (N, T, u_dim) = site_oh + hour_oh + position_in_window_oh(t)."""
    site_oh = pd.get_dummies(meta["site"]).values.astype(np.float32)
    hour_oh = pd.get_dummies(meta["hour_utc"].astype(int)).values.astype(np.float32)
    pos_oh = np.eye(T, dtype=np.float32)
    # broadcast: (N, S) → (N, T, S), (N, H) → (N, T, H), (T, T) → (N, T, T)
    N = len(meta)
    site_rep = np.tile(site_oh[:, None, :], (1, T, 1))
    hour_rep = np.tile(hour_oh[:, None, :], (1, T, 1))
    pos_rep = np.tile(pos_oh[None, :, :], (N, 1, 1))
    return np.concatenate([site_rep, hour_rep, pos_rep], axis=-1)


def load_labels(meta):
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
    rid2y = {}
    for _, r in sc_g.iterrows():
        Y = np.zeros(len(primary), dtype=np.uint8)
        for l in r.lbls:
            if l in l2i: Y[l2i[l]] = 1
        rid2y[r.row_id] = Y
    mask = np.zeros(len(meta), dtype=bool)
    Y = np.zeros((len(meta), len(primary)), dtype=np.uint8)
    for i, rid in enumerate(meta["row_id"].values):
        if rid in rid2y:
            mask[i] = True; Y[i] = rid2y[rid]
    return mask, Y


# ─── Training ──────────────────────────────────────────────────────────────
def train(model, spat, u, epochs=EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Standardize spatial (per channel across the whole corpus)
    # Cast fp16→fp32 FIRST to avoid overflow in sum-reduce across 2M rows.
    spat_f32 = spat.astype(np.float32)
    flat = spat_f32.reshape(-1, spat_f32.shape[-1])
    mu_x = flat.mean(0)
    sd_x = flat.std(0) + 1e-6
    del flat
    x_std_t = (spat_f32 - mu_x) / sd_x  # (N, T, X)
    del spat_f32
    ds = TensorDataset(torch.from_numpy(x_std_t).float(), torch.from_numpy(u).float())
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True)
    history = []
    model.train()
    for ep in range(1, epochs + 1):
        beta = min(1.0, ep / WARMUP)
        rec_s = kl_s = fstd_s = n = 0
        t0 = time.time()
        for yb, ub in dl:
            yb = yb.to(DEVICE, non_blocking=True)
            ub = ub.to(DEVICE, non_blocking=True)
            out = model(yb, ub)
            rec = ((out["y_hat"] - yb) ** 2).sum(-1).mean()   # sum over x_dim, mean over B,T
            kl = kl_laplace(out["mu_q"], out["ls_q"], out["mu_p"], out["ls_p"])  # (B, T, r)
            kl = torch.clamp(kl, min=FREE_BITS).sum(-1).mean()
            loss = rec + beta * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rec_s += rec.item() * yb.size(0)
            kl_s += kl.item() * yb.size(0)
            fstd_s += out["f"].std(1).mean().item() * yb.size(0)
            n += yb.size(0)
        sched.step()
        row = {"ep": ep, "beta": beta, "rec": rec_s/n, "kl": kl_s/n,
               "f_std_time": fstd_s/n, "time_s": time.time()-t0}
        history.append(row)
        print(f"  ep {ep:02d}  β={beta:.2f}  rec {row['rec']:.1f}  kl {row['kl']:.2f}  "
              f"f_std_time {row['f_std_time']:.3f}  ({row['time_s']:.0f}s)")
    return history, mu_x, sd_x


def extract_f(model, spat, u, mu_x, sd_x, batch=256):
    model.eval()
    x_std = (spat.astype(np.float32) - mu_x) / sd_x
    N = x_std.shape[0]
    f_win = np.zeros((N, R_DIM), dtype=np.float32)        # pooled per window
    pi_win = np.zeros((N, K_REG), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, N, batch):
            yb = torch.from_numpy(x_std[i:i+batch]).float().to(DEVICE)
            ub = torch.from_numpy(u[i:i+batch]).float().to(DEVICE)
            out = model(yb, ub)
            # Pool f over time steps for window-level retrieval key
            f_win[i:i+batch] = out["f"].mean(1).cpu().numpy()
            pi_win[i:i+batch] = out["pi"].mean(1).cpu().numpy()
    return f_win, pi_win


# ─── Evaluation: flip detection ────────────────────────────────────────────
def knn_disagree(X, Y, files, sites, k, cross_site=True):
    n = X.shape[0]
    nn_ = NearestNeighbors(n_neighbors=min(n, k + 400), metric="cosine").fit(X)
    _, I = nn_.kneighbors(X)
    score = np.zeros(n)
    for i in range(n):
        neigh = []
        for j in I[i]:
            if j == i or files[j] == files[i]: continue
            if cross_site and sites[j] == sites[i]: continue
            neigh.append(j)
            if len(neigh) >= k: break
        if not neigh: score[i] = 0.5; continue
        yi = Y[i]
        inter = (Y[neigh] & yi).sum(1); union = (Y[neigh] | yi).sum(1)
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
    print("Loading spatial embeddings...")
    spat, meta = load_data()
    u = build_u(meta)
    print(f"u: {u.shape}  u_dim={u.shape[-1]}")
    mask, Y = load_labels(meta)
    print(f"Labeled subset: {mask.sum()}/{mask.size}")

    model = iVDFM(x_dim=X_DIM, r_dim=R_DIM, u_dim=u.shape[-1],
                  K=K_REG, e_dim=E_DIM, hidden=HIDDEN).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"iVDFM params: {n_params/1e6:.2f} M")

    print(f"\nTraining iVDFM for {EPOCHS} epochs on {spat.shape[0]} windows...")
    history, mu_x, sd_x = train(model, spat, u, epochs=EPOCHS)
    torch.save({"state_dict": model.state_dict(), "mu_x": mu_x, "sd_x": sd_x,
                "history": history,
                "config": dict(r_dim=R_DIM, K=K_REG, e_dim=E_DIM, hidden=HIDDEN,
                               x_dim=X_DIM, u_dim=u.shape[-1])},
               OUT / "ivdfm_ckpt.pt")

    print("\nExtracting window-level f and pi...")
    f_win, pi_win = extract_f(model, spat, u, mu_x, sd_x)
    np.savez_compressed(OUT / "f_pi.npz", f=f_win, pi=pi_win)
    print(f"  f_win: {f_win.shape}  pi_win: {pi_win.shape}")
    print(f"  regime avg: {pi_win.mean(0)}")

    # Also pool Perch spatial over time for raw-Perch baseline comparison
    x_pooled = spat.astype(np.float32).mean(1)  # (N, 1536)

    # Flip detection on labeled subset
    print("\n=== Flip-detection (cross-site kNN, labeled subset) ===")
    Y_lab, is_flip = inject_flips(Y[mask], rate=FLIP_RATE, seed=42)
    files = meta["filename"].values[mask]
    sites = meta["site"].values[mask]

    results = {}
    for name, X in [("raw_perch_pooled", x_pooled[mask]),
                    ("iVDFM_f_window",   f_win[mask]),
                    ("iVDFM_pi_window",  pi_win[mask])]:
        row = {}
        for k in K_FLIP:
            s = knn_disagree(X, Y_lab, files, sites, k=k, cross_site=True)
            auc = roc_auc_score(is_flip, s)
            row[f"k{k}"] = auc
            print(f"  {name:<22} k={k:2d}  AUC {auc:.4f}")
        results[name] = row

    # Spearman between iVDFM f-kNN and raw-Perch kNN
    from scipy.stats import spearmanr
    s_raw = knn_disagree(x_pooled[mask], Y_lab, files, sites, k=10, cross_site=True)
    s_f   = knn_disagree(f_win[mask],   Y_lab, files, sites, k=10, cross_site=True)
    r, _ = spearmanr(s_raw, s_f)
    results["spearman_raw_iVDFM"] = float(r)
    print(f"\nSpearman(raw_perch_kNN, iVDFM_f_kNN): {r:+.3f}  (target ≤ 0.50 for distinct signal)")
    if r > 0.5:
        print("  ⚠️  Still redundant with raw Perch — iVDFM did not extract distinct signal.")
    else:
        print("  ✓ Distinct signal — iVDFM identifies structure raw Perch misses.")

    with open(OUT / "flip_auc.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT}/flip_auc.json")


if __name__ == "__main__":
    main()
