#!/usr/bin/env python3
"""exp93 — FFT on Perch's spatial_embedding (its 'thinking through time')
+ TP/FP/TN/FN signal hunt.

Spatial_embedding: (B, 16, 1536) — Perch outputs 16 temporal patches per
5-sec window, each with 1536-d feature vector. FFT along time axis (16 →
9 magnitude bins) reveals temporal modulation patterns of Perch's
internal state.

Hypothesis: Perch's "thought trajectory" (how its features evolve over
the 5-sec window) differs systematically between
  - rows where it's right (TP, TN) vs wrong (FP, FN)
  - rows where the species is in vocab vs not

Pipeline:
  A. Align spatial_emb to 739 labeled rows.
  B. FFT along time axis. Compute per-row spectrum stats (total energy,
     low/high band ratio, spectral entropy, dominant frequency).
  C. Classify each (row, class) pair by TP/FP/TN/FN at threshold 0.5
     (using v33 predictions). Restrict to Aves classes with both
     positive and negative examples in 739.
  D. Compare FFT features + Perch embedding stats across 4 categories.
  E. Test if any feature is a clean separator.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP43A, EXP80, ROOT, N_CLS, primary_labels)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend


def get_cached(name): return np.load(EXP80 / name)["scores"]


def get_spatial_emb_labeled(sc_g):
    """Align spatial_embedding cache to 739 labeled rows."""
    cache = EXP80 / "spatial_emb_labeled.npz"
    if cache.exists():
        return np.load(cache)["spatial"]
    print("  building spatial_emb cache (one-time)...", flush=True)
    src = ROOT / "experiments/_archive_2026_pre_v26/exp43j_outputs/spatial_ss_all.npz"
    meta = pd.read_parquet(ROOT / "experiments/_archive_2026_pre_v26/exp43j_outputs/spatial_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    full = np.load(src)["spatial"]   # (127k, 16, 1536)
    out = np.zeros((len(sc_g), 16, 1536), dtype=np.float32)
    miss = 0
    for i, rid in enumerate(sc_g.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = full[j]
        else: miss += 1
    print(f"  aligned: {len(sc_g)-miss}/{len(sc_g)}, missing {miss}", flush=True)
    cache.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache, spatial=out)
    return out


def compute_spectral_features(spatial_emb):
    """spatial_emb: (n, 16, 1536). Returns dict of per-row features."""
    fft = np.fft.rfft(spatial_emb, axis=1)        # (n, 9, 1536)
    mag = np.abs(fft)
    # Total energy across time-FFT and feature dims
    total = mag.sum(axis=(1, 2))
    # DC component (mag[:, 0, :]) = mean over time
    dc_energy = mag[:, 0].sum(axis=-1)
    # AC component (mag[:, 1:, :]) = modulation
    ac_energy = mag[:, 1:].sum(axis=(1, 2))
    # Low (bins 1-3, ~0.6-1.9 Hz) vs high (5-8, ~3-5 Hz)
    low_band = mag[:, 1:4].sum(axis=(1, 2))
    high_band = mag[:, 5:].sum(axis=(1, 2))
    low_high_ratio = low_band / (high_band + 1e-6)
    # Spectral entropy along FFT axis (averaged across feat dim)
    eps = 1e-12
    pp = mag[:, 1:] / (mag[:, 1:].sum(axis=1, keepdims=True) + eps)   # exclude DC
    spec_ent = -(pp * np.log(pp + eps)).sum(axis=1) / np.log(mag.shape[1] - 1)  # (n, 1536)
    spec_ent_mean = spec_ent.mean(axis=-1)
    # Peak frequency: argmax along FFT axis (avg across feat dims)
    peak_freq = mag[:, 1:].argmax(axis=1).mean(axis=-1)   # (n,)
    # Embedding-vs-time variance: how much does Perch state change over the 5 sec?
    time_var = spatial_emb.var(axis=1).mean(axis=-1)      # (n,)
    return {
        "total_energy": total,
        "dc_energy": dc_energy,
        "ac_energy": ac_energy,
        "ac_dc_ratio": ac_energy / (dc_energy + 1e-6),
        "low_high_ratio": low_high_ratio,
        "spec_ent_mean": spec_ent_mean,
        "peak_freq_mean": peak_freq,
        "time_var": time_var,
    }


def main():
    print("=== exp93: spatial-embedding FFT + TP/FP/TN/FN signal hunt ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    print(f"data loaded", flush=True)

    print("Loading spatial_embedding (Perch's temporal feature trajectory)...", flush=True)
    spatial_emb = get_spatial_emb_labeled(sc_g)
    print(f"spatial_emb: {spatial_emb.shape}", flush=True)

    # === Compute per-row spectral features ===
    print("Computing FFT spectral features along time axis...", flush=True)
    feats = compute_spectral_features(spatial_emb)
    print("  features:", list(feats.keys()))

    # Build v33 predictions for TP/FP labelling
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # === A. UNK vs KNOWN comparison (sanity check that spatial FFT shows what mel did) ===
    col_var = perch_prob.var(axis=0)
    unmapped_idx = np.where(col_var < 1e-6)[0]
    mapped_idx = np.where(col_var >= 1e-6)[0]
    classes = []
    for i in range(len(sc_g)):
        gt = np.where(Y[i] == 1)[0]
        gt_unmapped = np.intersect1d(gt, unmapped_idx)
        gt_mapped = np.intersect1d(gt, mapped_idx)
        if len(gt) == 0:
            classes.append("BG")
        elif len(gt_unmapped) > 0 and len(gt_mapped) == 0:
            classes.append("UNK_ONLY")
        elif len(gt_unmapped) > 0 and len(gt_mapped) > 0:
            classes.append("UNK_MIXED")
        else:
            classes.append("KNOWN_ONLY")
    classes = np.array(classes)

    print("\n=== A. Spatial-FFT features per row class ===")
    print(f"  {'class':<14} {'n':>5}  " + " ".join(f"{k:>14}" for k in feats))
    for c in ["KNOWN_ONLY", "UNK_ONLY", "UNK_MIXED", "BG"]:
        m = (classes == c)
        if m.sum() == 0: continue
        vals = " ".join(f"{feats[k][m].mean():>10.4f}±{feats[k][m].std():.4f}" for k in feats)
        print(f"  {c:<14} {m.sum():>5}  {vals}")

    # Within-site z-scored separability: UNK_ONLY (1) vs KNOWN_ONLY (0)
    print("\n=== A2. Within-site z-scored UNK_ONLY-vs-KNOWN_ONLY AUC for spatial FFT ===")
    sites = sc_g.site.values
    sites_with_both = []
    for s in sorted(set(sites)):
        n_unk = ((classes == "UNK_ONLY") & (sites == s)).sum()
        n_kno = ((classes == "KNOWN_ONLY") & (sites == s)).sum()
        if n_unk >= 3 and n_kno >= 3:
            sites_with_both.append(s)
    pool_mask = ((classes == "UNK_ONLY") | (classes == "KNOWN_ONLY")) & np.isin(sites, sites_with_both)
    y_pool = (classes[pool_mask] == "UNK_ONLY").astype(int)

    def site_zscore(values, sites):
        out = np.zeros_like(values, dtype=np.float64)
        for s in np.unique(sites):
            mm = (sites == s)
            if mm.sum() < 2: continue
            mu = values[mm].mean(); sd = values[mm].std() + 1e-6
            out[mm] = (values[mm] - mu) / sd
        return out

    print(f"  pooled n_pos={y_pool.sum()}, n_neg={len(y_pool)-y_pool.sum()}")
    for k, v in feats.items():
        zv = site_zscore(v, sites)
        try:
            auc = roc_auc_score(y_pool, zv[pool_mask])
            cross_auc = roc_auc_score((classes == "UNK_ONLY")[(classes == "UNK_ONLY") | (classes == "KNOWN_ONLY")].astype(int),
                                       v[(classes == "UNK_ONLY") | (classes == "KNOWN_ONLY")])
            print(f"  {k:<18} within-site AUC = {auc:.3f}  (cross-site = {cross_auc:.3f})")
        except: pass

    # === B. TP/FP/TN/FN per (row, class) ===
    # Restrict to mapped Aves classes with at least 5 positives in 739
    aves_mask = sp_taxon == "Aves"
    candidate_classes = []
    for c in range(N_CLS):
        if not aves_mask[c]: continue
        if c in unmapped_idx: continue
        n_pos = Y[:, c].sum()
        n_neg = (Y[:, c] == 0).sum()
        if n_pos >= 5 and n_neg >= 50:
            candidate_classes.append(c)
    print(f"\n=== B. TP/FP/TN/FN classification ===")
    print(f"  Mapped Aves classes with ≥5 pos and ≥50 neg: {len(candidate_classes)}")

    # For each candidate class, threshold v33 at 0.5 (or class-specific median?)
    THRESH = 0.5
    quadrant_features = {q: {f: [] for f in feats} for q in ("TP", "FP", "TN", "FN")}
    quadrant_perch_top1 = {q: [] for q in ("TP", "FP", "TN", "FN")}
    quadrant_emb_L2 = {q: [] for q in ("TP", "FP", "TN", "FN")}

    for c in candidate_classes:
        pred = v33[:, c] > THRESH
        for i in range(len(sc_g)):
            y = Y[i, c]; p = pred[i]
            if y == 1 and p:    q = "TP"
            elif y == 1 and not p: q = "FN"
            elif y == 0 and p:  q = "FP"
            else:                q = "TN"
            for f in feats:
                quadrant_features[q][f].append(feats[f][i])
            quadrant_perch_top1[q].append(perch_prob[i].max())
            quadrant_emb_L2[q].append(np.linalg.norm(perch_emb[i]))

    print(f"\n  Quadrant counts (across all candidate classes):")
    for q in ("TP", "FN", "FP", "TN"):
        print(f"    {q}: {len(quadrant_features[q]['total_energy'])}")

    print(f"\n=== Per-quadrant feature means ===")
    print(f"  {'feat':<18} {'TP':>14} {'FN':>14} {'FP':>14} {'TN':>14}")
    all_feats = list(feats) + ["perch_top1", "emb_L2"]
    for f in all_feats:
        cells = []
        for q in ("TP", "FN", "FP", "TN"):
            if f == "perch_top1":
                arr = np.array(quadrant_perch_top1[q])
            elif f == "emb_L2":
                arr = np.array(quadrant_emb_L2[q])
            else:
                arr = np.array(quadrant_features[q][f])
            if len(arr) == 0:
                cells.append(f"{'--':>14}")
                continue
            cells.append(f"{arr.mean():>10.3f}±{arr.std()/np.sqrt(len(arr)):.3f}")
        print(f"  {f:<18} " + " ".join(cells))

    # === C. Pairwise AUC: TP vs FP, TN vs FN ===
    print(f"\n=== C. Pairwise separability AUC ===")
    print(f"  {'feat':<18} {'TP_vs_FP':>10} {'TN_vs_FN':>10} {'CORR_vs_WRONG':>14}")
    for f in all_feats:
        # TP vs FP
        if f == "perch_top1":
            tp_arr = np.array(quadrant_perch_top1["TP"]); fp_arr = np.array(quadrant_perch_top1["FP"])
            tn_arr = np.array(quadrant_perch_top1["TN"]); fn_arr = np.array(quadrant_perch_top1["FN"])
        elif f == "emb_L2":
            tp_arr = np.array(quadrant_emb_L2["TP"]); fp_arr = np.array(quadrant_emb_L2["FP"])
            tn_arr = np.array(quadrant_emb_L2["TN"]); fn_arr = np.array(quadrant_emb_L2["FN"])
        else:
            tp_arr = np.array(quadrant_features["TP"][f]); fp_arr = np.array(quadrant_features["FP"][f])
            tn_arr = np.array(quadrant_features["TN"][f]); fn_arr = np.array(quadrant_features["FN"][f])
        cells = [f"  {f:<18}"]
        try:
            auc = roc_auc_score(np.concatenate([np.zeros(len(tp_arr)), np.ones(len(fp_arr))]),
                                 np.concatenate([tp_arr, fp_arr]))
            cells.append(f"{auc:>10.3f}")
        except: cells.append(f"{'--':>10}")
        try:
            auc = roc_auc_score(np.concatenate([np.zeros(len(tn_arr)), np.ones(len(fn_arr))]),
                                 np.concatenate([tn_arr, fn_arr]))
            cells.append(f"{auc:>10.3f}")
        except: cells.append(f"{'--':>10}")
        # CORRECT (TP+TN) vs WRONG (FP+FN)
        try:
            corr = np.concatenate([tp_arr, tn_arr])
            wrong = np.concatenate([fp_arr, fn_arr])
            auc = roc_auc_score(np.concatenate([np.zeros(len(corr)), np.ones(len(wrong))]),
                                 np.concatenate([corr, wrong]))
            cells.append(f"{auc:>14.3f}")
        except: cells.append(f"{'--':>14}")
        print(" ".join(cells))


if __name__ == "__main__":
    main()
