#!/usr/bin/env python3
"""exp46 — Quantify v12 vs v20 vs v21 actual prediction differences.

Hypothesis: LB identically 0.929 across 3 pipelines because:
  (a) gate changes concentrate on rare species NOT in public LB evaluable set
  (b) gate per-row effect is small relative to Perch signal dominance
  (c) public LB has only ~3-decimal precision — sub-rounding changes invisible

Investigate:
  1. For each of 234 species columns, compute % of row pairs reordered by
     gate on the 11-file eval (proxy for test)
  2. Per-class AUC change: how many classes have |ΔAUC| > 0.01?
  3. Which species are actually affected — are they plausibly in public LB?
  4. Simulate "if public LB had 5× more precision, would scores differ?"
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP41F = ROOT / "experiments/exp41f_outputs"
EXP45A = ROOT / "experiments/exp45a_outputs"
OUT = ROOT / "experiments/exp46_outputs"
OUT.mkdir(exist_ok=True)
SEED = 42; EVAL_N = 11
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]


def build_eval():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N])
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["lbls"]):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_eval, Y, primary


def align_43a(sc_eval):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs, embs = d["scores"], d["emb"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(sc_eval), scs.shape[1]), np.float32)
    E = np.zeros((len(sc_eval), embs.shape[1]), np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]; E[i] = embs[j]
    return S, E


def align_old(sc_eval, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(sc_eval), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s
def gauss_pf(scores, sc_eval, sigma=0.5):
    out = np.zeros_like(scores)
    for fname in sc_eval["filename"].unique():
        m = (sc_eval["filename"] == fname).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out


class TaxonHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=256, n_taxa=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, n_taxa))
    def forward(self, x): return self.net(x)


def main():
    sc_eval, Y, primary = build_eval()
    S_perch, E_perch = align_43a(sc_eval)
    S_sed29 = align_old(sc_eval, EXP29 / "val_scores.npz")
    S_sed41f = align_old(sc_eval, EXP41F / "val_scores.npz" if (EXP41F / "val_scores.npz").exists() else EXP41F / "val_scores_full.npz")

    perch_prob = sigmoid(S_perch)
    z29 = zs(np.nan_to_num(S_sed29, nan=0.0)) if S_sed29 is not None else None
    z41 = zs(np.nan_to_num(S_sed41f, nan=0.0)) if S_sed41f is not None else None
    zP = zs(perch_prob)

    # Taxon head
    ckpt = torch.load(EXP45A / "taxon_head.pt", map_location="cuda", weights_only=False)
    tm = TaxonHead().cuda(); tm.load_state_dict(ckpt["state_dict"]); tm.eval()
    species_to_taxon = np.asarray(ckpt["species_to_taxon"], dtype=np.int64)
    with torch.no_grad():
        tprob = torch.sigmoid(tm(torch.from_numpy(E_perch).cuda())).cpu().numpy()
    gate = np.clip(tprob[:, species_to_taxon] + 0.1, 0.0, 1.0)   # (N, 234)

    # Build the three pipelines (reproducing cell 50 logic)
    # v12 base: 0.8*zP + 0.2*z29, Gauss 0.5, sigmoid, temperature
    # (we skip some Kaggle-specific post-processing like prior, thresholds)

    def pipeline(zP, z29=None, z41=None, gate=None):
        blend = 0.8 * zP
        if z29 is not None: blend += 0.2 * z29
        if z41 is not None: blend += 0.2 * z41
        smoothed = gauss_pf(blend, sc_eval, sigma=0.5)
        probs = sigmoid(smoothed)
        if gate is not None:
            probs = probs * gate
        return probs

    p_v12 = pipeline(zP, z29=z29)                   # v12
    p_v17 = pipeline(zP, z41=z41)                   # v17
    p_v20 = pipeline(zP, z41=z41, gate=gate)         # v20
    p_v21 = pipeline(zP, z29=z29, gate=gate)         # v21

    # Total pairwise pipeline differences
    print("[1] Mean absolute prob diff between pipelines (over all rows×cols):")
    for name, a, b in [("v12 vs v20", p_v12, p_v20),
                       ("v12 vs v21", p_v12, p_v21),
                       ("v20 vs v21", p_v20, p_v21),
                       ("v12 vs v17", p_v12, p_v17)]:
        mad = np.abs(a - b).mean()
        nonzero = (np.abs(a - b) > 1e-6).mean()
        print(f"  {name:<15}  MAD={mad:.5f}  {nonzero*100:.1f}% cells changed")

    # Per-class AUC diff
    def per_class_auc(P):
        out = {}
        for c in range(Y.shape[1]):
            y = Y[:, c].astype(int)
            if y.sum() == 0 or y.sum() == len(y): continue
            if not np.isfinite(P[:, c]).all(): continue
            try: out[c] = float(roc_auc_score(y, P[:, c]))
            except Exception: pass
        return out

    auc_v12 = per_class_auc(p_v12)
    auc_v20 = per_class_auc(p_v20)
    auc_v21 = per_class_auc(p_v21)

    classes = set(auc_v12) & set(auc_v20) & set(auc_v21)
    print(f"\n[2] Per-class AUC changes ({len(classes)} classes evaluable):")
    big_change_v12_v20 = [(c, auc_v20[c] - auc_v12[c]) for c in classes if abs(auc_v20[c] - auc_v12[c]) > 0.01]
    big_change_v12_v21 = [(c, auc_v21[c] - auc_v12[c]) for c in classes if abs(auc_v21[c] - auc_v12[c]) > 0.01]
    print(f"  v12 → v20: {len(big_change_v12_v20)} classes with |ΔAUC| > 0.01")
    print(f"  v12 → v21: {len(big_change_v12_v21)} classes with |ΔAUC| > 0.01")

    print(f"\n[3] Top-10 classes with biggest |ΔAUC| (v12 → v21):")
    sorted_chg = sorted(classes, key=lambda c: abs(auc_v21[c] - auc_v12[c]), reverse=True)[:10]
    for c in sorted_chg:
        t = TAXA[species_to_taxon[c]]
        print(f"  {str(primary[c]):<12} ({t:<8}) v12={auc_v12[c]:.3f}  v21={auc_v21[c]:.3f}  Δ={auc_v21[c]-auc_v12[c]:+.3f}")

    # If local macros differ, but Kaggle shows 0.929 for both, then:
    # EITHER Kaggle uses a different class subset, OR precision is masked
    print(f"\n[4] Macro AUC on eval 40 classes (rounded to 4 decimals):")
    print(f"  v12 macro: {np.mean([auc_v12[c] for c in classes]):.4f}")
    print(f"  v17 macro: {np.mean([per_class_auc(p_v17).get(c, float('nan')) for c in classes]):.4f}")
    print(f"  v20 macro: {np.mean([auc_v20[c] for c in classes]):.4f}")
    print(f"  v21 macro: {np.mean([auc_v21[c] for c in classes]):.4f}")

    # Which species had AUC >0.5 in v12 but dropped in v21 (gate HURT these)
    hurt_by_gate = sorted(
        [(c, auc_v12[c] - auc_v21[c]) for c in classes if auc_v12[c] - auc_v21[c] > 0.02],
        key=lambda x: -x[1])[:15]
    print(f"\n[5] Species HURT by gate (top-15 v12 → v21 drops > 0.02):")
    for c, d in hurt_by_gate:
        t = TAXA[species_to_taxon[c]]
        print(f"  {str(primary[c]):<12} ({t:<8}) {auc_v12[c]:.3f} → {auc_v21[c]:.3f}  Δ={-d:+.3f}")

    helped_by_gate = sorted(
        [(c, auc_v21[c] - auc_v12[c]) for c in classes if auc_v21[c] - auc_v12[c] > 0.02],
        key=lambda x: -x[1])[:15]
    print(f"\n[6] Species HELPED by gate (top-15 v12 → v21 gains > 0.02):")
    for c, d in helped_by_gate:
        t = TAXA[species_to_taxon[c]]
        print(f"  {str(primary[c]):<12} ({t:<8}) {auc_v12[c]:.3f} → {auc_v21[c]:.3f}  Δ={+d:+.3f}")

    # How many rows have gate value near 1 (gate is no-op) vs far from 1
    print(f"\n[7] Gate value distribution per taxon (proxy for 'is gate doing something'):")
    for tidx, tname in enumerate(TAXA):
        vals = tprob[:, tidx]
        print(f"  {tname:<10}  mean={vals.mean():.3f}  median={np.median(vals):.3f}  q10={np.quantile(vals,.1):.3f}  q90={np.quantile(vals,.9):.3f}")


if __name__ == "__main__":
    main()
