#!/usr/bin/env python3
"""exp43n — iVDFM conditional advantage for hard species / ambiguous windows.

Paper hypothesis (user): iVDFM's identifiable factors help on cases where
raw Perch + teacher posterior cannot decide — rare species, ambiguous Zone C
windows, taxonomic structure.

Three tests against the hypothesis (we want iVDFM > raw Perch HERE, even if
iVDFM loses on the easy majority):

  T1 — Rare-species retrieval (leave-one-out kNN recall@k per class,
        stratified by class abundance).  Gate: Δ(recall | iVDFM − raw Perch)
        must be MORE POSITIVE for rare than for abundant classes.

  T2 — Ambiguous Zone C resolution (teacher top1-top2 gap < 0.2).
        For labeled windows in Zone C, assign class by kNN majority vote in
        iVDFM-f space and raw Perch space; compare to ground truth.  Gate:
        iVDFM top1 accuracy > raw Perch top1 accuracy in Zone C.

  T3 — Species-conditional innovation direction vs taxonomy.  Compute
        mean η̄_c per species; hierarchical cluster; correlate cluster
        tree with taxonomic class_name labels (Aves/Amphibia/...).  Gate:
        cophenetic correlation of η cluster > raw Perch cluster.

Uses existing exp43k ckpt + exp41a teacher pseudo (for Zone C window set).
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43J = ROOT / "experiments/exp43j_outputs"
EXP43K = ROOT / "experiments/exp43k_outputs"
EXP41 = ROOT / "experiments/exp41_outputs"
OUT = ROOT / "experiments/exp43n_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
T_LEN = 16


# ─── Reconstruct iVDFM to re-encode ────────────────────────────────────────
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


# ─── Data ──────────────────────────────────────────────────────────────────
def load_all():
    ckpt = torch.load(EXP43K / "ivdfm_ckpt.pt", map_location=DEVICE, weights_only=False)
    model = iVDFM(**ckpt["config"]).to(DEVICE).eval()
    model.load_state_dict(ckpt["state_dict"])
    mu_x = ckpt["mu_x"]; sd_x = ckpt["sd_x"]

    spat = np.load(EXP43J / "spatial_ss_all.npz")["spatial"]
    meta = pd.read_parquet(EXP43J / "spatial_ss_all_meta.parquet")
    T = T_LEN
    site_rep = np.tile(pd.get_dummies(meta["site"].astype("category"))
                       .values.astype(np.float32)[:, None, :], (1, T, 1))
    hour_rep = np.tile(pd.get_dummies(meta["hour_utc"].astype(int).astype("category"))
                       .values.astype(np.float32)[:, None, :], (1, T, 1))
    pos_rep = np.tile(np.eye(T, dtype=np.float32)[None, :, :], (len(meta), 1, 1))
    u = np.concatenate([site_rep, hour_rep, pos_rep], axis=-1)

    return model, spat, u, meta, mu_x, sd_x


def load_labels_with_taxa(meta):
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    label_to_class = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
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
    return mask, Y, primary, label_to_class


def encode_all(model, spat, u, mu_x, sd_x, batch=256):
    n = spat.shape[0]
    eta = np.zeros((n, T_LEN, model.r_dim), dtype=np.float32)
    f = np.zeros_like(eta)
    with torch.inference_mode():
        for i in range(0, n, batch):
            y = (spat[i:i+batch].astype(np.float32) - mu_x) / sd_x
            yb = torch.from_numpy(y).to(DEVICE)
            ub = torch.from_numpy(u[i:i+batch]).to(DEVICE)
            pi_b, e_b = model.regime(ub)
            mu_q, _ = model.encode(yb, ub, e_b)
            f_b = model.dynamics(pi_b, mu_q)
            eta[i:i+batch] = mu_q.cpu().numpy()
            f[i:i+batch] = f_b.cpu().numpy()
    return eta, f


# ─── Test 1: Rare-species retrieval (LOO kNN recall) ──────────────────────
def loo_recall(X, Y, files, k_list=(5, 10), min_per_class=2):
    """Leave-one-out per positive window: retrieve k NN (diff-file), fraction sharing class."""
    files = np.asarray(files)
    class_recall = {}  # class_idx → list of recall values
    for c in range(Y.shape[1]):
        idx = np.where(Y[:, c] == 1)[0]
        if len(idx) < min_per_class:
            continue
        nn_ = NearestNeighbors(n_neighbors=min(len(X), max(k_list) + 200), metric="cosine").fit(X)
        _, I = nn_.kneighbors(X[idx])
        recs = {k: [] for k in k_list}
        for q_ordinal, q_global in enumerate(idx):
            neigh = [j for j in I[q_ordinal] if j != q_global and files[j] != files[q_global]]
            for k in k_list:
                top_k = neigh[:k]
                if len(top_k) == 0:
                    recs[k].append(0.0); continue
                recs[k].append(float(Y[top_k, c].mean()))
        class_recall[c] = {f"recall@{k}": float(np.mean(recs[k])) for k in k_list}
        class_recall[c]["n_pos"] = int(len(idx))
    return class_recall


def stratify_recall(class_recall, abundant_thresh=20, rare_thresh=10):
    """Group per-class recall by abundance."""
    groups = {"abundant": [], "medium": [], "rare": []}
    for c, r in class_recall.items():
        n = r["n_pos"]
        g = "abundant" if n >= abundant_thresh else ("rare" if n <= rare_thresh else "medium")
        groups[g].append(r)
    out = {}
    for g, rs in groups.items():
        out[g] = {"n_classes": len(rs)}
        if rs:
            for k in rs[0]:
                if k.startswith("recall"):
                    out[g][f"mean_{k}"] = float(np.mean([x[k] for x in rs]))
    return out


# ─── Test 2: Ambiguous Zone C resolution ──────────────────────────────────
def zone_c_accuracy(X_all, X_lab_idx_in_all, Y_lab, files_all, meta_all, k=15):
    """For labeled windows also ambiguous per Perch scores (Zone C), assign top-1 class via
       kNN majority vote in X-space. Compare top1 accuracy across representations."""
    # Use Perch-scores from exp43a as teacher proxy for labeled windows
    scores = np.load(EXP43J.with_name("exp43a_outputs") / "perch_ss_all.npz")["scores"]
    # scores: (N, 234) — use as teacher proxy
    scores_lab = scores[X_lab_idx_in_all]  # (739, 234)
    s_max = scores_lab.max(1)
    s_sorted = np.sort(scores_lab, axis=-1)[:, ::-1]
    s_gap = s_sorted[:, 0] - s_sorted[:, 1]

    zone_c = (s_max > 0.3) & (s_gap < 0.2)
    easy   = (s_max > 0.5) & (s_gap > 0.3)

    # For each labeled window in zone_c: top1 kNN vote (k neighbors, cross-file, diff-site)
    files_lab = files_all[X_lab_idx_in_all]
    sites_lab = meta_all["site"].values[X_lab_idx_in_all]
    files_all_arr = files_all
    sites_all_arr = meta_all["site"].values

    results = {}
    for name, X_representation in X_all.items():
        X_lab_rep = X_representation[X_lab_idx_in_all]
        nn_ = NearestNeighbors(n_neighbors=min(len(X_representation), k + 400),
                               metric="cosine").fit(X_representation)
        _, I = nn_.kneighbors(X_lab_rep)
        for zone_name, zone_mask in [("zone_c", zone_c), ("easy", easy)]:
            hits, tot = 0, 0
            for i_l in np.where(zone_mask)[0]:
                orig = X_lab_idx_in_all[i_l]
                neigh = []
                for j in I[i_l]:
                    if j == orig or files_all_arr[j] == files_lab[i_l]: continue
                    if sites_all_arr[j] == sites_lab[i_l]: continue
                    neigh.append(j)
                    if len(neigh) >= k: break
                # Vote: for each class in neighbors, use scores (soft) to tally
                if not neigh: continue
                # Only labeled neighbors have ground truth — use teacher scores as "votes"
                vote = scores[neigh].mean(0)  # (234,)
                pred = int(np.argmax(vote))
                if Y_lab[i_l, pred] == 1:
                    hits += 1
                tot += 1
            results.setdefault(name, {})[f"{zone_name}_top1_acc"] = (hits / tot if tot else 0.0)
            results[name][f"{zone_name}_n"] = tot
    return results


# ─── Test 3: η̄_c vs taxonomy ──────────────────────────────────────────────
def taxonomy_structure_alignment(X_lab, Y_lab, primary, label_to_class):
    """Per-class mean vec; hierarchical cluster; cophenetic correlation with taxon labels."""
    # Collect mean representation per abundant class
    classes = []; means = []; taxa = []
    for c in range(Y_lab.shape[1]):
        idx = np.where(Y_lab[:, c] == 1)[0]
        if len(idx) < 5: continue
        mean_c = X_lab[idx].mean(0)
        sp = primary[c]
        tax = label_to_class.get(sp, "Unknown")
        classes.append(sp); means.append(mean_c); taxa.append(tax)
    if len(means) < 5:
        return {"n_classes_with_data": len(means)}
    M = np.stack(means)
    # cosine dist matrix
    M_n = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    D = 1.0 - M_n @ M_n.T
    D = (D + D.T) / 2
    D = np.clip(D, 0.0, None)                # float-precision fix for linkage
    np.fill_diagonal(D, 0.0)
    from scipy.spatial.distance import squareform
    try:
        condensed = squareform(D, checks=False)
        link = linkage(condensed, method="average")
    except Exception as e:
        return {"error": str(e)}
    # Ground truth: same-taxon pair has distance 0, diff-taxon 1
    tax_arr = np.array(taxa)
    gt = np.zeros_like(D)
    for i in range(len(taxa)):
        for j in range(len(taxa)):
            if i != j and tax_arr[i] != tax_arr[j]:
                gt[i, j] = 1.0
    gt_cond = squareform(gt, checks=False)
    coph = cophenet(link)
    corr = float(np.corrcoef(coph, gt_cond)[0, 1])
    return {
        "n_classes": len(classes),
        "tax_counts": {t: int((tax_arr == t).sum()) for t in np.unique(tax_arr)},
        "cophenetic_corr_with_taxonomy": corr,
    }


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print("Loading iVDFM ckpt + spatial + metadata...")
    model, spat, u, meta, mu_x, sd_x = load_all()
    mask, Y, primary, label_to_class = load_labels_with_taxa(meta)
    print(f"  labeled {mask.sum()}/{mask.size}  n_classes={Y.shape[1]}")

    print("Encoding all 128k windows (eta, f)...")
    t0 = time.time()
    eta, f = encode_all(model, spat, u, mu_x, sd_x)
    print(f"  eta {eta.shape}  f {f.shape}  wall {time.time()-t0:.1f}s")

    # Representations to compare
    eta_win = eta.mean(1)                                   # (N, r)
    f_win = f.mean(1)                                       # (N, r)
    eta_flat = eta.reshape(len(eta), -1)                    # (N, T*r)
    f_flat = f.reshape(len(f), -1)
    perch_pooled = spat.astype(np.float32).mean(1)         # (N, 1536)

    X_all = {
        "raw_perch_pooled":  perch_pooled,
        "iVDFM_eta_win":     eta_win,
        "iVDFM_f_win":       f_win,
        "iVDFM_eta_flat":    eta_flat,
        "iVDFM_f_flat":      f_flat,
    }
    lab_idx = np.where(mask)[0]
    Y_lab = Y[mask]
    files_all = meta["filename"].values

    results = {}

    # ── T1: Rare-species LOO recall ─────────────────────────────────────
    print("\n[T1] Per-class LOO recall@k (need n_pos ≥ 2)")
    t1 = {}
    for name, X in X_all.items():
        X_lab = X[mask]
        cr = loo_recall(X_lab, Y_lab, files_all[mask], k_list=(5, 10))
        strat = stratify_recall(cr)
        t1[name] = strat
        print(f"  {name:<20} {strat}")
    results["T1_LOO_recall"] = t1

    # Compute gate: Δ(iVDFM − raw) for rare vs abundant
    print("\n  Gate: Δ recall@10 (iVDFM − raw Perch) by abundance")
    base = t1["raw_perch_pooled"]
    for name in ["iVDFM_eta_win", "iVDFM_f_win", "iVDFM_eta_flat", "iVDFM_f_flat"]:
        for g in ["abundant", "medium", "rare"]:
            if base.get(g, {}).get("mean_recall@10") is None: continue
            if t1[name].get(g, {}).get("mean_recall@10") is None: continue
            delta = t1[name][g]["mean_recall@10"] - base[g]["mean_recall@10"]
            print(f"    {name:<20} {g:<9} Δ = {delta:+.4f}  ({t1[name][g]['n_classes']} classes)")

    # ── T2: Zone C resolution ──────────────────────────────────────────
    print("\n[T2] Zone C ambiguous window resolution")
    try:
        zone_res = zone_c_accuracy(X_all, lab_idx, Y_lab, files_all, meta, k=15)
        for name, v in zone_res.items():
            print(f"  {name:<20} {v}")
        results["T2_zone_c"] = zone_res
    except Exception as e:
        print(f"  skipped: {e}")
        results["T2_zone_c"] = {"error": str(e)}

    # ── T3: Taxonomy alignment ─────────────────────────────────────────
    print("\n[T3] Species-conditional mean vector → taxonomy cophenetic correlation")
    t3 = {}
    for name, X in X_all.items():
        res = taxonomy_structure_alignment(X[mask], Y_lab, primary, label_to_class)
        t3[name] = res
        print(f"  {name:<20} {res}")
    results["T3_taxonomy_cophenetic"] = t3

    with open(OUT / "conditional_advantage.json", "w") as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f"\nSaved → {OUT}/conditional_advantage.json")


if __name__ == "__main__":
    main()
