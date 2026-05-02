#!/usr/bin/env python3
"""exp91 — Within-site z-scored confidence: is overconfidence a consistent
UNK signal across sites?

User hypothesis: even if absolute confidence direction reverses (S09
showed reverse), perhaps the DEVIATION from per-site mean confidence is
systematically POSITIVE on UNK rows. I.e., UNK rows may always be in the
upper tail of within-site confidence distribution.

Test:
  1. Compute per-site mean of each metric.
  2. For each row, compute z-score = (value - site_mean) / site_std.
  3. Check UNK_ONLY vs KNOWN_ONLY AUC of z-scored values.
  4. Test on richer pool: include UNK_MIXED + KNOWN_ONLY too.

If z-scored AUC > 0.65 consistently, the signal is real but expressed
through within-site distribution shape, not absolute values.

Then test routing again with z-scored uncertainty as the variable.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name): return np.load(EXP80 / name)["scores"]


def entropy_norm(p, axis=-1):
    eps = 1e-12
    pp = p / (p.sum(axis=axis, keepdims=True) + eps)
    return -(pp * np.log(pp + eps)).sum(axis=axis) / np.log(p.shape[axis])


def site_zscore(values: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Z-score per site."""
    out = np.zeros_like(values, dtype=np.float64)
    for s in np.unique(sites):
        mask = (sites == s)
        if mask.sum() < 2: continue
        m = values[mask].mean()
        sd = values[mask].std() + 1e-6
        out[mask] = (values[mask] - m) / sd
    return out.astype(np.float32)


def main():
    print("=== exp91: within-site z-scored confidence test ===\n", flush=True)
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

    # Per-row metrics
    H_234 = entropy_norm(perch_prob, axis=-1)
    top1 = perch_prob.max(axis=-1)
    sorted_p = np.sort(perch_prob, axis=-1)[:, ::-1]
    top5_mass = sorted_p[:, :5].sum(axis=-1) / (sorted_p.sum(axis=-1) + 1e-12)
    top1_minus_top2 = sorted_p[:, 0] - sorted_p[:, 1]   # margin
    emb_L2 = np.linalg.norm(perch_emb, axis=-1)
    emb_max = np.abs(perch_emb).max(axis=-1)
    emb_var = perch_emb.var(axis=-1)
    emb_pr = (emb_L2 ** 4) / ((perch_emb ** 4).sum(axis=-1) + 1e-12)  # participation ratio

    # Z-score each per site
    metric_dict = {
        "H_234": H_234, "top1": top1, "top5_mass": top5_mass, "top1_top2_margin": top1_minus_top2,
        "emb_L2": emb_L2, "emb_max": emb_max, "emb_var": emb_var, "emb_pr": emb_pr,
    }
    z_metrics = {n: site_zscore(v, sites_arr) for n, v in metric_dict.items()}

    # === Test 1: per-site z-scored AUC for UNK_ONLY vs KNOWN_ONLY ===
    print("=== Per-site z-scored AUC: UNK_ONLY vs KNOWN_ONLY (≥3 each) ===")
    sites_with_both = []
    for s in sorted(set(sites_arr)):
        n_unk = ((classes == "UNK_ONLY") & (sites_arr == s)).sum()
        n_kno = ((classes == "KNOWN_ONLY") & (sites_arr == s)).sum()
        if n_unk >= 3 and n_kno >= 3:
            sites_with_both.append(s)

    print(f"  Qualifying sites: {sites_with_both}\n")

    for m, zv in z_metrics.items():
        aucs_per_site = []
        for s in sites_with_both:
            mask = (sites_arr == s) & ((classes == "UNK_ONLY") | (classes == "KNOWN_ONLY"))
            y = (classes[mask] == "UNK_ONLY").astype(int)
            scores = zv[mask]
            try:
                auc = roc_auc_score(y, scores)
                aucs_per_site.append(auc)
            except: pass
        if aucs_per_site:
            mean_auc = np.mean(aucs_per_site)
            strength = abs(mean_auc - 0.5) * 2
            print(f"  {m:<20} per-site AUCs = {[f'{a:.3f}' for a in aucs_per_site]}")
            print(f"    {'':<20} mean = {mean_auc:.3f}  (strength = {strength:.3f})")

    # === Test 2: pooled z-scored AUC across all qualifying sites ===
    print("\n=== Pooled across all qualifying sites (UNK_ONLY vs KNOWN_ONLY) ===")
    pool_mask = ((classes == "UNK_ONLY") | (classes == "KNOWN_ONLY")) & np.isin(sites_arr, sites_with_both)
    y_pool = (classes[pool_mask] == "UNK_ONLY").astype(int)
    print(f"  pooled n_UNK={y_pool.sum()}, n_KNOWN={len(y_pool)-y_pool.sum()}")
    for m, zv in z_metrics.items():
        try:
            auc = roc_auc_score(y_pool, zv[pool_mask])
            strength = abs(auc - 0.5) * 2
            print(f"  {m:<20} AUC = {auc:.3f}  (strength = {strength:.3f})")
        except Exception as e:
            print(f"  {m:<20} ERROR: {e}")

    # === Test 3: UNK_MIXED + UNK_ONLY (all UNK present) vs KNOWN_ONLY ===
    print("\n=== Pooled: ANY_UNK_PRESENT vs KNOWN_ONLY (richer sample) ===")
    any_unk = (classes == "UNK_ONLY") | (classes == "UNK_MIXED")
    pool_mask2 = (any_unk | (classes == "KNOWN_ONLY")) & np.isin(sites_arr, sites_with_both)
    y_pool2 = any_unk[pool_mask2].astype(int)
    print(f"  pooled n_UNK_present={y_pool2.sum()}, n_KNOWN={len(y_pool2)-y_pool2.sum()}")
    for m, zv in z_metrics.items():
        try:
            auc = roc_auc_score(y_pool2, zv[pool_mask2])
            strength = abs(auc - 0.5) * 2
            print(f"  {m:<20} AUC = {auc:.3f}  (strength = {strength:.3f})")
        except Exception as e:
            print(f"  {m:<20} ERROR: {e}")

    # === Test 4: routing with z-scored uncertainty ===
    print("\n=== Routing test with z-scored U ===", flush=True)
    # Build U_z = combined z-scored uncertainty (consistent direction:
    #   higher U_z = MORE UNK-like)
    # Based on findings: UNK has lower H, higher top1, higher top5_mass, lower emb_L2, lower emb_max, lower emb_var
    U_z = (
        -z_metrics["H_234"] +     # UNK has lower H → -z higher on UNK
        z_metrics["top1"] +
        z_metrics["top5_mass"] +
        -z_metrics["emb_L2"] +
        -z_metrics["emb_max"] +
        -z_metrics["emb_var"]
    ) / 6.0

    # Verify on labeled subset
    print(f"  U_z UNK_ONLY mean: {U_z[classes == 'UNK_ONLY'].mean():+.3f}")
    print(f"  U_z UNK_MIXED mean: {U_z[classes == 'UNK_MIXED'].mean():+.3f}")
    print(f"  U_z KNOWN_ONLY mean: {U_z[classes == 'KNOWN_ONLY'].mean():+.3f}")

    # Build v33 baseline + routed variants
    base_cert = 0.7 * perch_prob + 0.3 * exp50

    # v33 ref
    gated = apply_v9_gate(base_cert, perch_emb, sp_taxon, offset=0.1)
    v33_ref = file_max_blend(gated, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33_ref, v33_ref, ev_mask, Y, sp_taxon, "v33 ref")]

    # Variant A: when U_z high, REDUCE confidence (multiply final by 1 - σ * U_z)
    # Idea: don't trust the prediction when Perch is OOD
    print("\n  Variant A: confidence dampening on uncertain rows")
    for scale in [0.05, 0.10, 0.20]:
        damping = 1.0 - scale * np.clip(U_z, 0, 3)  # damp by up to scale*3
        damping = np.clip(damping, 0.5, 1.0)[:, None]
        damped = v33_ref * damping
        rows.append(evaluate(damped.astype(np.float32), v33_ref, ev_mask, Y, sp_taxon,
                              f"A-damp scale={scale}"))

    # Variant B: route uncertain rows to exp50-only (no Perch)
    print("  Variant B: route uncertain rows to exp50-only")
    for thresh in [0.5, 1.0, 1.5]:
        is_unc = U_z > thresh
        bp = base_cert.copy()
        bp[is_unc] = exp50[is_unc]   # full route to exp50
        gated = apply_v9_gate(bp, perch_emb, sp_taxon, offset=0.1)
        P = file_max_blend(gated, sc_g, alpha=0.10)
        rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon,
                              f"B-route(exp50) U_z>{thresh} (n_unc={is_unc.sum()})"))

    # Variant C: amplify NO-NOTHING signal: when uncertain, also reduce all UNK_idx columns
    # since Perch's overconfident-wrong is on mapped Aves columns, suppress those when uncertain
    print("  Variant C: suppress confident-Aves predictions on uncertain rows")
    for scale in [0.10, 0.20, 0.30]:
        # On uncertain rows, soft-suppress mapped columns
        mod = v33_ref.copy()
        for i in np.where(U_z > 1.0)[0]:
            mod[i, mapped_idx] = mod[i, mapped_idx] * (1 - scale)
        rows.append(evaluate(mod, v33_ref, ev_mask, Y, sp_taxon,
                              f"C-suppress(map) U_z>1.0 scale={scale}"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== ROUTING RESULTS ===")
    print(res[cols].sort_values("macro_d", ascending=False).to_string(index=False))
    res.to_csv(EXP80 / "exp91_zscored_routing.csv", index=False)


if __name__ == "__main__":
    main()
