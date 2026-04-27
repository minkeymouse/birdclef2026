#!/usr/bin/env python3
"""exp43l — exercise iVDFM structure for species ID, not just kNN-AUC.

User insight: "η_t is the process by which a new species 'emerges' at time t.
Visualize how that emergence is represented, don't just correlate kNN scores."

This script does what exp43k SHOULD have done:
  1. Re-encode labeled windows with trained iVDFM → (η_t, f_t, π_t) per time step
  2. Innovation-magnitude vs vocalization: silence vs labeled windows, onset t*
  3. Species-conditional η direction and clustering (vs raw Perch baseline)
  4. Linear probe on η_{1:16} flattened vs raw Perch → species classification
  5. Factor trajectory UMAP colored by species (top-k classes)
  6. Regime posterior inspection (collapsed but may have subtle structure)
  7. Sparse event count as pseudo vocalization detector
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43J = ROOT / "experiments/exp43j_outputs"
EXP43K = ROOT / "experiments/exp43k_outputs"
OUT = ROOT / "experiments/exp43l_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
T_LEN = 16


# ─── Same model as exp43k ─────────────────────────────────────────────────
class RegimeNet(nn.Module):
    def __init__(self, u_dim, K=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(u_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, K))
    def forward(self, u): return F.softmax(self.net(u), dim=-1)


class iVDFM(nn.Module):
    def __init__(self, x_dim, r_dim, u_dim, K, e_dim, hidden):
        super().__init__()
        self.r_dim, self.x_dim, self.K = r_dim, x_dim, K
        self.regime_net = RegimeNet(u_dim, K)
        self.regime_codes = nn.Parameter(torch.randn(K, e_dim) * 0.1)
        ue_dim = u_dim + e_dim
        self.prior_mu = nn.Linear(ue_dim, r_dim)
        self.prior_log_scale = nn.Linear(ue_dim, r_dim)
        self.enc_rnn = nn.GRU(x_dim + ue_dim, hidden, num_layers=2, bidirectional=True,
                              batch_first=True, dropout=0.1)
        self.enc_mu = nn.Linear(hidden * 2, r_dim)
        self.enc_log_scale = nn.Linear(hidden * 2, r_dim)
        self.A_diag = nn.Parameter(torch.ones(K, r_dim) * 0.9)
        self.B_diag = nn.Parameter(torch.ones(K, r_dim))
        self.dec = nn.Sequential(nn.Linear(r_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, x_dim))

    def regime(self, u):
        pi = self.regime_net(u)
        return pi, pi @ self.regime_codes

    def encode(self, y, u, e):
        h = self.enc_rnn(torch.cat([y, u, e], -1))[0]
        return self.enc_mu(h), self.enc_log_scale(h).clamp(min=-3.0, max=3.0)

    def dynamics(self, pi, eta):
        a_eff = pi @ self.A_diag; b_eff = pi @ self.B_diag
        B, T, r = eta.shape
        f = torch.zeros(B, r, device=eta.device)
        outs = []
        for t in range(T):
            f = a_eff[:, t] * f + b_eff[:, t] * eta[:, t]
            outs.append(f)
        return torch.stack(outs, dim=1)


# ─── Load ──────────────────────────────────────────────────────────────────
def load_everything():
    ckpt = torch.load(EXP43K / "ivdfm_ckpt.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]; mu_x = ckpt["mu_x"]; sd_x = ckpt["sd_x"]
    model = iVDFM(**cfg).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    spat = np.load(EXP43J / "spatial_ss_all.npz")["spatial"]
    meta = pd.read_parquet(EXP43J / "spatial_ss_all_meta.parquet")
    sites = meta["site"].astype("category")
    hours = meta["hour_utc"].astype(int).astype("category")
    T = T_LEN
    site_rep = np.tile(pd.get_dummies(sites).values.astype(np.float32)[:, None, :], (1, T, 1))
    hour_rep = np.tile(pd.get_dummies(hours).values.astype(np.float32)[:, None, :], (1, T, 1))
    pos_rep = np.tile(np.eye(T, dtype=np.float32)[None, :, :], (len(meta), 1, 1))
    u = np.concatenate([site_rep, hour_rep, pos_rep], axis=-1)
    print(f"spatial {spat.shape}  u {u.shape}  meta {len(meta)}")
    return model, spat, u, meta, mu_x, sd_x


def load_labels(meta):
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
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
    return mask, Y, primary


def encode_all(model, spat, u, mu_x, sd_x, batch=256):
    """Return per-time-step (eta, f, pi) for ALL windows."""
    n = spat.shape[0]
    eta = np.zeros((n, T_LEN, model.r_dim), dtype=np.float32)
    f   = np.zeros((n, T_LEN, model.r_dim), dtype=np.float32)
    pi  = np.zeros((n, T_LEN, model.K), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, n, batch):
            y = (spat[i:i+batch].astype(np.float32) - mu_x) / sd_x
            yb = torch.from_numpy(y).to(DEVICE)
            ub = torch.from_numpy(u[i:i+batch]).to(DEVICE)
            pi_b, e_b = model.regime(ub)
            # Encoder returns q(η | y, u, e); use posterior mean as eta estimate
            mu_q, _ = model.encode(yb, ub, e_b)
            eta_b = mu_q                                    # deterministic (posterior mean)
            f_b = model.dynamics(pi_b, eta_b)
            eta[i:i+batch] = eta_b.cpu().numpy()
            f[i:i+batch] = f_b.cpu().numpy()
            pi[i:i+batch] = pi_b.cpu().numpy()
    return eta, f, pi


# ─── Analyses ─────────────────────────────────────────────────────────────
def linear_probe_auc(X, Y, n_splits=5):
    """5-fold linear LogReg probe → per-class AUC averaged. Returns macro AUC and top-100 mean AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    strat = Y.sum(1).clip(0, 5)
    oof = np.zeros_like(Y, dtype=np.float32)
    for tr, va in skf.split(X, strat):
        for c in range(Y.shape[1]):
            if Y[tr, c].sum() < 2: continue
            try:
                clf = LogisticRegression(C=0.25, max_iter=300)
                clf.fit(X[tr], Y[tr, c])
                oof[va, c] = clf.predict_proba(X[va])[:, 1]
            except Exception:
                pass
    keep = Y.sum(0) > 0
    aucs = []
    for c in np.where(keep)[0]:
        try:
            aucs.append(roc_auc_score(Y[:, c], oof[:, c]))
        except Exception:
            pass
    aucs = np.array(aucs)
    return float(aucs.mean()), float(np.sort(aucs)[-100:].mean()), int(keep.sum())


def innovation_magnitude_analysis(eta_lab, Y_lab):
    """Is ||η_t|| larger in windows with species than silent windows?"""
    has_species = (Y_lab.sum(1) > 0).astype(np.int64)
    if has_species.sum() == 0 or (has_species == 0).sum() == 0:
        return None
    # per-window: mean magnitude of η over time
    mag = np.linalg.norm(eta_lab, axis=-1)     # (N, T)
    mean_mag = mag.mean(-1)                    # (N,)
    max_mag = mag.max(-1)                      # (N,)
    auc_mean = roc_auc_score(has_species, mean_mag)
    auc_max  = roc_auc_score(has_species, max_mag)
    return {
        "labeled_windows": int(len(Y_lab)),
        "silent_windows": int((has_species == 0).sum()),
        "voiced_windows": int(has_species.sum()),
        "AUC_meanmag_voiced_vs_silent": float(auc_mean),
        "AUC_maxmag_voiced_vs_silent":  float(auc_max),
        "mag_median_voiced": float(np.median(mean_mag[has_species == 1])),
        "mag_median_silent": float(np.median(mean_mag[has_species == 0])),
    }


def event_sparsity_auc(eta_lab, Y_lab, tau_sigma=2.0):
    """Sparse event = ||η_t|| > mean + tau*std. Count events per window → predict voiced."""
    mag = np.linalg.norm(eta_lab, axis=-1)
    mu = mag.mean(); sd = mag.std()
    thresh = mu + tau_sigma * sd
    events = (mag > thresh).sum(-1)            # (N,)
    has_species = (Y_lab.sum(1) > 0).astype(np.int64)
    if has_species.sum() == 0 or (has_species == 0).sum() == 0:
        return None
    return {
        "threshold": float(thresh),
        "mean_events_voiced": float(events[has_species == 1].mean()),
        "mean_events_silent": float(events[has_species == 0].mean()),
        "AUC_event_count_voiced_vs_silent": float(roc_auc_score(has_species, events)),
    }


def regime_class_alignment(pi_lab, Y_lab):
    """Does regime posterior align with any species? Use AMI between argmax regime and species presence."""
    regime_id = pi_lab.mean(1).argmax(-1)           # (N,) dominant regime per window
    # For classes with ≥10 positives, AMI between "class present" and regime id
    Y_ml = Y_lab.sum(1).clip(0, 1)                   # any species
    ami_any = adjusted_mutual_info_score(Y_ml, regime_id)
    per_class = []
    for c in range(Y_lab.shape[1]):
        if Y_lab[:, c].sum() >= 10:
            ami = adjusted_mutual_info_score(Y_lab[:, c], regime_id)
            per_class.append((c, int(Y_lab[:, c].sum()), ami))
    per_class.sort(key=lambda x: -x[2])
    return {
        "regime_distribution": {int(k): int(v) for k, v in zip(*np.unique(regime_id, return_counts=True))},
        "AMI_regime_vs_any_species": float(ami_any),
        "top5_AMI_class_vs_regime": [{"class_idx": int(c), "n_pos": int(n), "AMI": float(a)} for c, n, a in per_class[:5]],
    }


def kmeans_alignment(X, Y, k_list=(8, 16, 32, 64)):
    """K-means on representation, compare cluster assignment to species-presence vector."""
    results = []
    Y_ml = Y.sum(1).clip(0, 1)
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X)
        ami = adjusted_mutual_info_score(Y_ml, km.labels_)
        # per-species AMI for abundant classes
        per_class_ami = []
        for c in range(Y.shape[1]):
            if Y[:, c].sum() >= 10:
                per_class_ami.append(adjusted_mutual_info_score(Y[:, c], km.labels_))
        top5 = sorted(per_class_ami, reverse=True)[:5]
        results.append({"k": k, "AMI_any_species": float(ami),
                        "top5_class_AMI_mean": float(np.mean(top5)) if top5 else 0.0})
    return results


def species_cluster_purity(eta_lab, Y_lab, n_top_species=10):
    """Do windows of same species cluster tightly in η-sequence-flattened space?
       For each top species: within-class cosine similarity vs between-class."""
    flat = eta_lab.reshape(len(eta_lab), -1)
    # Select top-N species by positive count
    counts = Y_lab.sum(0)
    top_idx = np.argsort(-counts)[:n_top_species]
    top_idx = [int(c) for c in top_idx if counts[c] >= 10]
    metrics = []
    # Normalize rows
    flat_n = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
    for c in top_idx:
        in_mask = Y_lab[:, c] == 1
        out_mask = Y_lab[:, c] == 0
        if in_mask.sum() < 3 or out_mask.sum() < 10:
            continue
        # within: mean pairwise cos of in-class
        X_in = flat_n[in_mask]
        X_out = flat_n[out_mask]
        within = (X_in @ X_in.T).mean()
        between = (X_in @ X_out.T).mean()
        metrics.append({"class_idx": int(c), "n_pos": int(in_mask.sum()),
                        "within_cos": float(within), "between_cos": float(between),
                        "gap": float(within - between)})
    return metrics


def main():
    model, spat, u, meta, mu_x, sd_x = load_everything()
    mask, Y, primary = load_labels(meta)
    print(f"Labeled: {mask.sum()}")

    print("\nEncoding all windows with trained iVDFM...")
    t0 = time.time()
    eta, f, pi = encode_all(model, spat, u, mu_x, sd_x)
    print(f"  eta {eta.shape}  f {f.shape}  pi {pi.shape}  wall {time.time()-t0:.1f}s")

    eta_lab, f_lab, pi_lab = eta[mask], f[mask], pi[mask]
    Y_lab = Y[mask]
    perch_pooled = spat[mask].astype(np.float32).mean(1)    # (N, 1536) time-pooled

    results = {}

    # ─── 1. Innovation magnitude: voiced vs silent ─────────────────────────
    print("\n[1] Innovation magnitude — voiced vs silent windows")
    r1 = innovation_magnitude_analysis(eta_lab, Y_lab)
    print(f"  {json.dumps(r1, indent=2) if r1 else 'N/A'}")
    results["innovation_magnitude"] = r1

    # ─── 2. Sparse event detector ──────────────────────────────────────────
    print("\n[2] Sparse event count (||η_t|| > μ+2σ)")
    r2 = event_sparsity_auc(eta_lab, Y_lab, tau_sigma=2.0)
    print(f"  {json.dumps(r2, indent=2) if r2 else 'N/A'}")
    results["event_sparsity"] = r2

    # ─── 3. Linear probe: η flattened vs raw Perch vs f ────────────────────
    print("\n[3] Linear probe AUC on labeled subset")
    spaces = {
        "raw_perch_pooled_1536": perch_pooled,
        "eta_flat_256":          eta_lab.reshape(len(eta_lab), -1),  # 16 × 16
        "f_flat_256":            f_lab.reshape(len(f_lab), -1),
        "f_pooled_16":           f_lab.mean(1),
        "eta_magvec_16":         np.linalg.norm(eta_lab, axis=-1),
    }
    probe_results = {}
    for name, X in spaces.items():
        mean_auc, top100, n = linear_probe_auc(X, Y_lab)
        print(f"  {name:<28} macro_AUC={mean_auc:.4f}  top100_mean={top100:.4f}  (n_classes={n})")
        probe_results[name] = {"macro_AUC": mean_auc, "top100_mean": top100, "n_classes": n}
    results["linear_probe"] = probe_results

    # ─── 4. Regime collapse: is there any structure? ───────────────────────
    print("\n[4] Regime posterior alignment with species")
    r4 = regime_class_alignment(pi_lab, Y_lab)
    print(f"  {json.dumps(r4, indent=2)}")
    results["regime_alignment"] = r4

    # ─── 5. K-means on representation vs species ───────────────────────────
    print("\n[5] K-means clustering AMI vs species presence")
    km_eta = kmeans_alignment(eta_lab.reshape(len(eta_lab), -1), Y_lab)
    km_f   = kmeans_alignment(f_lab.reshape(len(f_lab), -1), Y_lab)
    km_raw = kmeans_alignment(perch_pooled, Y_lab)
    print(f"  eta_flat:   {km_eta}")
    print(f"  f_flat:     {km_f}")
    print(f"  raw_perch:  {km_raw}")
    results["kmeans"] = {"eta": km_eta, "f": km_f, "raw_perch": km_raw}

    # ─── 6. Species cluster purity: within vs between class cosine ─────────
    print("\n[6] Species cluster purity (within- vs between-class cosine, top-10 species)")
    r6_eta = species_cluster_purity(eta_lab, Y_lab)
    r6_raw = species_cluster_purity(perch_pooled[:, None, :], Y_lab)  # 1536 → (N, 1, 1536)
    print(f"  eta:  mean_gap={np.mean([x['gap'] for x in r6_eta]):+.4f}  per-class={[round(x['gap'],4) for x in r6_eta]}")
    print(f"  raw:  mean_gap={np.mean([x['gap'] for x in r6_raw]):+.4f}  per-class={[round(x['gap'],4) for x in r6_raw]}")
    results["cluster_purity"] = {"eta": r6_eta, "raw_perch": r6_raw}

    # ─── 7. Innovation onset pattern (qualitative sample) ──────────────────
    print("\n[7] Innovation onset pattern — per-t magnitude (voiced vs silent sample)")
    has_species = (Y_lab.sum(1) > 0).astype(np.int64)
    mag_t = np.linalg.norm(eta_lab, axis=-1)  # (N_lab, T)
    mag_t_voiced = mag_t[has_species == 1].mean(0) if has_species.sum() > 0 else np.zeros(T_LEN)
    mag_t_silent = mag_t[has_species == 0].mean(0) if (1-has_species).sum() > 0 else np.zeros(T_LEN)
    print(f"  voiced mean per t: {[round(v,2) for v in mag_t_voiced]}")
    print(f"  silent mean per t: {[round(v,2) for v in mag_t_silent]}")
    results["magnitude_per_timestep"] = {
        "voiced": mag_t_voiced.tolist(), "silent": mag_t_silent.tolist()
    }

    # Save
    with open(OUT / "species_analysis.json", "w") as fp:
        json.dump(results, fp, indent=2, default=float)
    np.savez_compressed(OUT / "encoded_labeled.npz",
                        eta=eta_lab, f=f_lab, pi=pi_lab, Y=Y_lab)
    print(f"\nSaved → {OUT}/species_analysis.json + encoded_labeled.npz")


if __name__ == "__main__":
    main()
