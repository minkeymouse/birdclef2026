#!/usr/bin/env python3
"""exp43o — Does iVDFM latent catch impostors within teacher-assigned pseudo groups?

User hypothesis (precise):
  Teacher pseudo says species c is present in windows {w1, w2, ..., wN}.
  Some of those assignments are correct, some are teacher mistakes.
  In iVDFM latent space, if the teacher-mistake window is an outlier from the
  centroid of the true-positive windows, we have a mislabel-correction signal
  that is NOT just "raw Perch kNN".

  Test: does iVDFM latent give LARGER impostor vs true-positive separation
  than raw Perch?

Setup (on labeled SS only, where we have ground truth):
  For each species c with ≥ 5 labeled positives:
    1. teacher_c_high = {windows | Perch_score(c) > τ}  ← teacher assigns c
    2. within teacher_c_high:
         true_pos  = windows with Y[c] = 1  (actually contains c)
         impostor  = windows with Y[c] = 0  (teacher mistake)
    3. centroid_c = mean latent of true_pos (leave-one-out per query)
    4. distance(w, centroid_c) in each representation
    5. AUC(is_impostor ⊥ distance) = how well does distance separate mistakes?

  Compare AUC across:
    - raw_perch_pooled
    - iVDFM_eta_win, iVDFM_f_win
    - iVDFM_eta_flat, iVDFM_f_flat
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP43J = ROOT / "experiments/exp43j_outputs"
EXP43K = ROOT / "experiments/exp43k_outputs"
OUT = ROOT / "experiments/exp43o_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
T_LEN = 16
TAU_SCORE = 0.3   # teacher-c-high threshold on Perch score


# ─── iVDFM re-encode (same as exp43n) ─────────────────────────────────────
class RegimeNet(nn.Module):
    def __init__(self, u_dim, K=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(u_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, K))
    def forward(self, u): return F.softmax(self.net(u), dim=-1)


class iVDFM(nn.Module):
    def __init__(self, x_dim, r_dim, u_dim, K, e_dim, hidden):
        super().__init__()
        self.r_dim = r_dim; self.K = K
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
        pi = self.regime_net(u); return pi, pi @ self.regime_codes
    def encode(self, y, u, e):
        h = self.enc_rnn(torch.cat([y, u, e], -1))[0]
        return self.enc_mu(h), self.enc_log_scale(h).clamp(min=-3.0, max=3.0)
    def dynamics(self, pi, eta):
        a_eff = pi @ self.A_diag; b_eff = pi @ self.B_diag
        f = torch.zeros(eta.size(0), eta.size(-1), device=eta.device)
        outs = []
        for t in range(eta.size(1)):
            f = a_eff[:, t] * f + b_eff[:, t] * eta[:, t]
            outs.append(f)
        return torch.stack(outs, 1)


def load_all():
    ckpt = torch.load(EXP43K / "ivdfm_ckpt.pt", map_location=DEVICE, weights_only=False)
    model = iVDFM(**ckpt["config"]).to(DEVICE).eval()
    model.load_state_dict(ckpt["state_dict"])
    spat = np.load(EXP43J / "spatial_ss_all.npz")["spatial"]
    meta = pd.read_parquet(EXP43J / "spatial_ss_all_meta.parquet")
    scores = np.load(EXP43A / "perch_ss_all.npz")["scores"]  # (N, 234) teacher proxy
    T = T_LEN
    site_rep = np.tile(pd.get_dummies(meta["site"].astype("category"))
                       .values.astype(np.float32)[:, None, :], (1, T, 1))
    hour_rep = np.tile(pd.get_dummies(meta["hour_utc"].astype(int).astype("category"))
                       .values.astype(np.float32)[:, None, :], (1, T, 1))
    pos_rep = np.tile(np.eye(T, dtype=np.float32)[None, :, :], (len(meta), 1, 1))
    u = np.concatenate([site_rep, hour_rep, pos_rep], axis=-1)
    return model, spat, u, meta, scores, ckpt["mu_x"], ckpt["sd_x"]


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
    rid2y = {r.row_id: r.lbls for _, r in sc_g.iterrows()}
    mask = np.zeros(len(meta), dtype=bool)
    Y = np.zeros((len(meta), len(primary)), dtype=np.uint8)
    for i, rid in enumerate(meta["row_id"].values):
        if rid in rid2y:
            mask[i] = True
            for l in rid2y[rid]:
                if l in l2i: Y[i, l2i[l]] = 1
    return mask, Y, primary


def encode_iVDFM(model, spat, u, mu_x, sd_x, batch=256):
    n = spat.shape[0]
    eta = np.zeros((n, T_LEN, model.r_dim), dtype=np.float32)
    f = np.zeros_like(eta)
    with torch.inference_mode():
        for i in range(0, n, batch):
            y = (spat[i:i+batch].astype(np.float32) - mu_x) / sd_x
            yb = torch.from_numpy(y).to(DEVICE); ub = torch.from_numpy(u[i:i+batch]).to(DEVICE)
            pi_b, e_b = model.regime(ub)
            mu_q, _ = model.encode(yb, ub, e_b)
            eta[i:i+batch] = mu_q.cpu().numpy()
            f[i:i+batch] = model.dynamics(pi_b, mu_q).cpu().numpy()
    return eta, f


def cosine_dist(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return 1.0 - a @ b.T


def impostor_auc_per_class(X_lab, Y_lab, scores_lab, files_lab, min_pos=5, min_imp=3, tau=TAU_SCORE):
    """For each class c:
         teacher_hi = scores_lab[:, c] > tau
         true_pos   = teacher_hi & Y_lab[:, c] == 1
         impostor   = teacher_hi & Y_lab[:, c] == 0
       For each window w ∈ teacher_hi: distance from w to leave-one-out centroid of true_pos.
       Compute AUC(is_impostor | distance).
    """
    results = {}
    for c in range(Y_lab.shape[1]):
        teacher_hi = scores_lab[:, c] > tau
        is_pos = (Y_lab[:, c] == 1)
        tp_idx = np.where(teacher_hi & is_pos)[0]
        imp_idx = np.where(teacher_hi & ~is_pos)[0]
        if len(tp_idx) < min_pos or len(imp_idx) < min_imp:
            continue
        # Leave-one-out distance for true positives
        all_idx = np.concatenate([tp_idx, imp_idx])
        labels = np.concatenate([np.zeros(len(tp_idx)), np.ones(len(imp_idx))])  # 1 = impostor
        # distances: for true pos, LOO centroid; for impostor, full tp centroid
        dists = np.zeros(len(all_idx))
        tp_mat = X_lab[tp_idx]
        for i, g_idx in enumerate(all_idx):
            if i < len(tp_idx):
                # leave-one-out: exclude this tp
                mask = np.ones(len(tp_idx), dtype=bool); mask[i] = False
                centroid = tp_mat[mask].mean(0, keepdims=True)
            else:
                centroid = tp_mat.mean(0, keepdims=True)
            dists[i] = cosine_dist(X_lab[g_idx:g_idx+1], centroid)[0, 0]
        try:
            auc = roc_auc_score(labels, dists)
        except Exception:
            continue
        # Also enforce cross-file condition: impostors should be from different files than tp
        n_cross_file = 0
        for j in imp_idx:
            if files_lab[j] not in set(files_lab[tp_idx]):
                n_cross_file += 1
        results[int(c)] = {
            "n_true_pos": int(len(tp_idx)),
            "n_impostor": int(len(imp_idx)),
            "n_impostor_cross_file": int(n_cross_file),
            "auc_impostor_detection": float(auc),
        }
    return results


def aggregate(results_per_class):
    if not results_per_class: return {}
    aucs = [r["auc_impostor_detection"] for r in results_per_class.values()]
    weights = [r["n_impostor"] for r in results_per_class.values()]
    return {
        "n_classes_eval": len(aucs),
        "mean_auc": float(np.mean(aucs)),
        "median_auc": float(np.median(aucs)),
        "weighted_auc": float(np.average(aucs, weights=weights)),
        "frac_auc_above_0.5": float(np.mean([a > 0.5 for a in aucs])),
        "frac_auc_above_0.7": float(np.mean([a > 0.7 for a in aucs])),
    }


def main():
    print("Loading model + spatial + scores...")
    model, spat, u, meta, scores, mu_x, sd_x = load_all()
    mask, Y, primary = load_labels(meta)
    print(f"  labeled {mask.sum()}/{mask.size}")

    print("Encoding iVDFM (eta, f)...")
    t0 = time.time()
    eta, f = encode_iVDFM(model, spat, u, mu_x, sd_x)
    print(f"  wall {time.time()-t0:.1f}s")

    # Representations
    perch_pooled = spat.astype(np.float32).mean(1)
    reprs = {
        "raw_perch_pooled":  perch_pooled[mask],
        "iVDFM_eta_win":     eta.mean(1)[mask],
        "iVDFM_f_win":       f.mean(1)[mask],
        "iVDFM_eta_flat":    eta.reshape(len(eta), -1)[mask],
        "iVDFM_f_flat":      f.reshape(len(f), -1)[mask],
    }
    Y_lab = Y[mask]
    scores_lab = scores[mask]
    files_lab = meta["filename"].values[mask]

    print(f"\nImpostor detection AUC (τ={TAU_SCORE}, ≥5 tp, ≥3 imp)")
    print(f"  {'rep':<22}  {'n_cls':>5}  {'mean':>6}  {'median':>6}  {'wAUC':>6}  {'%>0.5':>6}  {'%>0.7':>6}")
    all_res = {}
    for name, X in reprs.items():
        per = impostor_auc_per_class(X, Y_lab, scores_lab, files_lab)
        agg = aggregate(per)
        all_res[name] = {"per_class": per, "aggregate": agg}
        if agg:
            print(f"  {name:<22}  {agg['n_classes_eval']:>5}  "
                  f"{agg['mean_auc']:>6.3f}  {agg['median_auc']:>6.3f}  "
                  f"{agg['weighted_auc']:>6.3f}  {agg['frac_auc_above_0.5']:>6.3f}  "
                  f"{agg['frac_auc_above_0.7']:>6.3f}")
        else:
            print(f"  {name:<22}  no classes qualified")

    # Per-class head-to-head: for classes where BOTH have AUC, does iVDFM beat raw?
    print("\nPaired per-class AUC: classes where iVDFM_f_win > raw_perch_pooled:")
    base = all_res["raw_perch_pooled"]["per_class"]
    common_classes = set(base.keys())
    for name in ["iVDFM_eta_win", "iVDFM_f_win", "iVDFM_eta_flat", "iVDFM_f_flat"]:
        ivf = all_res[name]["per_class"]
        common = common_classes & set(ivf.keys())
        wins = 0; ties = 0
        deltas = []
        for c in common:
            d = ivf[c]["auc_impostor_detection"] - base[c]["auc_impostor_detection"]
            deltas.append(d)
            if d > 0.01: wins += 1
            elif abs(d) <= 0.01: ties += 1
        print(f"  {name:<22}  n_common={len(common):3d}  "
              f"wins={wins:3d}  ties={ties:2d}  losses={len(common)-wins-ties:3d}  "
              f"mean_Δ={np.mean(deltas):+.4f}  median_Δ={np.median(deltas):+.4f}")
        all_res[name]["paired_vs_raw"] = {
            "n_common": len(common), "wins": wins, "ties": ties, "losses": len(common)-wins-ties,
            "mean_delta": float(np.mean(deltas)), "median_delta": float(np.median(deltas)),
        }

    with open(OUT / "impostor_detection.json", "w") as fp:
        json.dump(all_res, fp, indent=2, default=float)
    print(f"\nSaved → {OUT}/impostor_detection.json")


if __name__ == "__main__":
    main()
