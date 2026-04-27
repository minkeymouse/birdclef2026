#!/usr/bin/env python3
"""exp92 — Scale sweep on C-suppress + FFT spectral analysis on mel.

Two parts:
A. Wider scale/threshold sweep on C-suppress (best lever from exp91).
   See if larger scale gives measurably bigger macro_d while keeping
   sp_row > 0.99.

B. FFT on mel-time-axis of the (16, 128) pooled mel cache. Hypothesis:
   Insecta sonotypes have sustained narrow-band tones, bird calls have
   transient broadband. The temporal modulation FFT spectrum should
   differ. If FFT feature has separability, add it to U_combined.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, load_labeled_mel,
                        EXP80, N_CLS, TAXA)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name): return np.load(EXP80 / name)["scores"]


def entropy_norm(p, axis=-1):
    eps = 1e-12
    pp = p / (p.sum(axis=axis, keepdims=True) + eps)
    return -(pp * np.log(pp + eps)).sum(axis=axis) / np.log(p.shape[axis])


def site_zscore(values, sites):
    out = np.zeros_like(values, dtype=np.float64)
    for s in np.unique(sites):
        mask = (sites == s)
        if mask.sum() < 2: continue
        m = values[mask].mean()
        sd = values[mask].std() + 1e-6
        out[mask] = (values[mask] - m) / sd
    return out.astype(np.float32)


def main():
    print("=== exp92: scale sweep + FFT spectral analysis ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")

    col_var = perch_prob.var(axis=0)
    unmapped_idx = np.where(col_var < 1e-6)[0]
    mapped_idx = np.where(col_var >= 1e-6)[0]

    classes = []
    for i in range(len(sc_g)):
        gt = np.where(Y[i] == 1)[0]
        if len(gt) == 0:
            classes.append("BG"); continue
        gt_unmapped = np.intersect1d(gt, unmapped_idx)
        gt_mapped = np.intersect1d(gt, mapped_idx)
        if len(gt_unmapped) > 0 and len(gt_mapped) == 0:
            classes.append("UNK_ONLY")
        elif len(gt_unmapped) > 0 and len(gt_mapped) > 0:
            classes.append("UNK_MIXED")
        elif len(gt_mapped) > 0:
            classes.append("KNOWN_ONLY")
        else:
            classes.append("BG")
    classes = np.array(classes)
    sites_arr = sc_g.site.values

    # ===== Part A: wider C-suppress sweep =====
    print("\n=== A. Wider C-suppress sweep ===")
    H_234 = entropy_norm(perch_prob, axis=-1)
    top1 = perch_prob.max(axis=-1)
    sorted_p = np.sort(perch_prob, axis=-1)[:, ::-1]
    top5_mass = sorted_p[:, :5].sum(axis=-1) / (sorted_p.sum(axis=-1) + 1e-12)
    emb_L2 = np.linalg.norm(perch_emb, axis=-1)
    emb_max = np.abs(perch_emb).max(axis=-1)
    emb_var = perch_emb.var(axis=-1)

    z_H = site_zscore(H_234, sites_arr)
    z_top1 = site_zscore(top1, sites_arr)
    z_top5 = site_zscore(top5_mass, sites_arr)
    z_L2 = site_zscore(emb_L2, sites_arr)
    z_max = site_zscore(emb_max, sites_arr)
    z_var = site_zscore(emb_var, sites_arr)

    U_z = (-z_H + z_top1 + z_top5 - z_L2 - z_max - z_var) / 6.0

    # Build v33 ref
    base_cert = 0.7 * perch_prob + 0.3 * exp50
    gated_ref = apply_v9_gate(base_cert, perch_emb, sp_taxon, offset=0.1)
    v33_ref = file_max_blend(gated_ref, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33_ref, v33_ref, ev_mask, Y, sp_taxon, "v33 ref")]

    print(f"  threshold × scale grid:")
    for thresh in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for scale in [0.10, 0.20, 0.30, 0.40, 0.50]:
            mod = v33_ref.copy()
            high_unc = U_z > thresh
            n_unc = high_unc.sum()
            if n_unc == 0: continue
            mod[high_unc][:, mapped_idx] = mod[high_unc][:, mapped_idx] * (1 - scale)
            # Properly apply: 2D indexing
            mod = v33_ref.copy()
            for i in np.where(high_unc)[0]:
                mod[i, mapped_idx] = mod[i, mapped_idx] * (1 - scale)
            rows.append(evaluate(mod, v33_ref, ev_mask, Y, sp_taxon,
                                  f"C-suppress U>{thresh} scale={scale} (n={n_unc})"))

    # ===== Part B: FFT on mel time axis =====
    print("\n=== B. FFT on pooled mel time axis (T=16, F=128) ===")
    mel = load_labeled_mel()  # (739, 16, 128)
    print(f"  mel shape: {mel.shape}")

    # FFT along time axis (axis=1, length 16). Magnitude on first 9 bins.
    fft_mel = np.abs(np.fft.rfft(mel, axis=1))  # (739, 9, 128)
    # Per-row spectral features
    fft_total = fft_mel.sum(axis=(1, 2))                     # total spectral magnitude
    fft_low = fft_mel[:, :3].sum(axis=(1, 2))                # low-freq (modulations 0-1 Hz)
    fft_high = fft_mel[:, 5:].sum(axis=(1, 2))               # high-freq (>1.6 Hz modulations)
    fft_low_high_ratio = fft_low / (fft_high + 1e-6)         # high ratio = sustained tones
    # spectral peak frequency (per row, per mel bin, take peak then average)
    fft_peak_idx = fft_mel.argmax(axis=1)                    # (739, 128)
    fft_peak_mean = fft_peak_idx.mean(axis=1)                # average peak bin across mel bins
    # spectral entropy along time-FFT axis (per mel bin, then average)
    eps = 1e-12
    pp = fft_mel / (fft_mel.sum(axis=1, keepdims=True) + eps)
    spec_ent = -(pp * np.log(pp + eps)).sum(axis=1) / np.log(fft_mel.shape[1])
    spec_ent_mean = spec_ent.mean(axis=1)                    # (739,)

    print(f"\n  Per-class FFT feature means:")
    print(f"  {'class':<14} {'fft_total':>12} {'low/high':>12} {'peak_mean':>12} {'spec_ent':>12}")
    for c in ["KNOWN_ONLY", "UNK_ONLY", "UNK_MIXED"]:
        m = (classes == c)
        if m.sum() == 0: continue
        print(f"  {c:<14} {fft_total[m].mean():>12.2f} {fft_low_high_ratio[m].mean():>12.3f} "
              f"{fft_peak_mean[m].mean():>12.3f} {spec_ent_mean[m].mean():>12.3f}")

    # Z-score per site for fairness
    z_fft_total = site_zscore(fft_total, sites_arr)
    z_fft_lh = site_zscore(fft_low_high_ratio, sites_arr)
    z_fft_peak = site_zscore(fft_peak_mean, sites_arr)
    z_spec_ent = site_zscore(spec_ent_mean, sites_arr)

    # Pooled within-site UNK_ONLY vs KNOWN_ONLY AUC
    print("\n  Pooled within-site z-scored AUC (UNK_ONLY=1 vs KNOWN_ONLY=0, qualifying sites only):")
    sites_with_both = []
    for s in sorted(set(sites_arr)):
        n_unk = ((classes == "UNK_ONLY") & (sites_arr == s)).sum()
        n_kno = ((classes == "KNOWN_ONLY") & (sites_arr == s)).sum()
        if n_unk >= 3 and n_kno >= 3:
            sites_with_both.append(s)
    pool_mask = ((classes == "UNK_ONLY") | (classes == "KNOWN_ONLY")) & np.isin(sites_arr, sites_with_both)
    y_pool = (classes[pool_mask] == "UNK_ONLY").astype(int)
    print(f"    n_pos={y_pool.sum()}, n_neg={len(y_pool)-y_pool.sum()}")
    for nm, vv in [("z_fft_total", z_fft_total), ("z_fft_lh_ratio", z_fft_lh),
                   ("z_fft_peak_mean", z_fft_peak), ("z_spec_ent", z_spec_ent)]:
        try:
            auc = roc_auc_score(y_pool, vv[pool_mask])
            strength = abs(auc - 0.5) * 2
            print(f"    {nm:<22} AUC = {auc:.3f}  (strength = {strength:.3f})")
        except: pass

    # Cross-site separation (with the caveat it's site-confounded)
    print("\n  Cross-site separation AUC (UNK_ONLY=1 vs KNOWN_ONLY=0):")
    cross_mask = (classes == "UNK_ONLY") | (classes == "KNOWN_ONLY")
    y_cross = (classes[cross_mask] == "UNK_ONLY").astype(int)
    for nm, vv in [("fft_total", fft_total), ("fft_lh_ratio", fft_low_high_ratio),
                   ("fft_peak_mean", fft_peak_mean), ("spec_ent", spec_ent_mean)]:
        try:
            auc = roc_auc_score(y_cross, vv[cross_mask])
            strength = abs(auc - 0.5) * 2
            print(f"    {nm:<22} AUC = {auc:.3f}  (strength = {strength:.3f})")
        except: pass

    # ===== Part C: Build enhanced U_z with FFT =====
    print("\n=== C. Enhanced U_z (Perch features + FFT spec_ent) — routing test ===")
    # Idea: sustained Insecta calls have lower spec_ent (concentrated FFT). UNK has lower spec_ent.
    # Add FFT spec_ent to U_z
    U_z_fft = (-z_H + z_top1 + z_top5 - z_L2 - z_max - z_var - z_spec_ent) / 7.0

    print(f"  U_z_fft UNK_ONLY mean: {U_z_fft[classes == 'UNK_ONLY'].mean():+.3f}")
    print(f"  U_z_fft UNK_MIXED mean: {U_z_fft[classes == 'UNK_MIXED'].mean():+.3f}")
    print(f"  U_z_fft KNOWN_ONLY mean: {U_z_fft[classes == 'KNOWN_ONLY'].mean():+.3f}")

    # Pooled AUC
    if pool_mask.sum() > 0:
        auc_orig = roc_auc_score(y_pool, U_z[pool_mask])
        auc_fft = roc_auc_score(y_pool, U_z_fft[pool_mask])
        print(f"  Pooled within-site AUC: U_z = {auc_orig:.3f}, U_z_fft = {auc_fft:.3f}")

    # Routing with U_z_fft
    print("\n  C-suppress with U_z_fft, scale × thresh sweep:")
    for thresh in [0.5, 1.0, 1.5]:
        for scale in [0.20, 0.30, 0.40]:
            mod = v33_ref.copy()
            high_unc = U_z_fft > thresh
            n_unc = high_unc.sum()
            if n_unc == 0: continue
            for i in np.where(high_unc)[0]:
                mod[i, mapped_idx] = mod[i, mapped_idx] * (1 - scale)
            rows.append(evaluate(mod, v33_ref, ev_mask, Y, sp_taxon,
                                  f"C-FFT U_fft>{thresh} scale={scale} (n={n_unc})"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== TOP results sorted by macro_d desc ===")
    print(res.sort_values("macro_d", ascending=False)[cols].head(15).to_string(index=False))

    res.to_csv(EXP80 / "exp92_scale_fft.csv", index=False)


if __name__ == "__main__":
    main()
