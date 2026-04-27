#!/usr/bin/env python3
"""exp97 — Robustness checks on exp96's best lever.

Tests whether the +0.0068 macro_d on 122 eval rows is genuine signal or
site-fingerprint amplification, via:

  R1. Train-vs-eval consistency: apply the SAME lever to 617 train rows.
      If train gain >> eval gain (e.g. 3×), suggests overfitting to
      train-distribution patterns (site shortcut).
  R2. Per-site analysis: which sites benefit? If gain concentrated in
      train-Insecta sites (S08/S15/S19/S23), high site-shortcut risk.
  R3. Per-class-pair attribution: which specific (row, class)
      modifications drive the gain? Is it broad-spectrum or
      concentrated on a few classes?
  R4. Holdout sensitivity: would the lever still help if we remove the
      site with biggest contribution?

  R5. Finer parameter sweep around the best (FN_R th=0.4 α=0.5 δ=0.3).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS, TAXA)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman, per_class_auc
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend


def get_cached(name): return np.load(EXP80 / name)["scores"]


def fn_rescue(v33, perch_prob, exp50, exp59, thresh_low=0.4, alpha_fn=0.5, delta_t=0.3):
    out = v33.copy()
    low_mask = v33 < thresh_low
    teacher_max = np.maximum(exp50, exp59) if exp59 is not None else exp50
    teacher_signal = np.maximum(teacher_max - delta_t, 0.0)
    boost = alpha_fn * teacher_signal * (1.0 - v33)
    out[low_mask] = v33[low_mask] + boost[low_mask]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def macro_on_subset(P, Y, sp_taxon, mask):
    """Macro AUC restricted to subset rows."""
    P_sub = P[mask]; Y_sub = Y[mask]
    aucs = per_class_auc(Y_sub, P_sub)
    return (np.mean(list(aucs.values())) if aucs else np.nan), len(aucs)


def main():
    print("=== exp97: robustness checks on exp96 best lever ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")

    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # Best config from exp96
    BEST = dict(thresh_low=0.4, alpha_fn=0.5, delta_t=0.3)
    P_lev = fn_rescue(v33, perch_prob, exp50, exp59, **BEST)

    ev_mask = sc_g.split.values == "eval"
    tr_mask = sc_g.split.values == "train"
    sites = sc_g.site.values

    # === R1: Train-vs-eval consistency ===
    print("=== R1: train-vs-eval consistency ===")
    eval_v33, n_ev = macro_on_subset(v33, Y, sp_taxon, ev_mask)
    eval_lev, _ = macro_on_subset(P_lev, Y, sp_taxon, ev_mask)
    train_v33, n_tr = macro_on_subset(v33, Y, sp_taxon, tr_mask)
    train_lev, _ = macro_on_subset(P_lev, Y, sp_taxon, tr_mask)
    print(f"  EVAL ({n_ev} cls): v33 macro = {eval_v33:.4f}, lever = {eval_lev:.4f}, Δ = {eval_lev - eval_v33:+.4f}")
    print(f"  TRAIN ({n_tr} cls): v33 macro = {train_v33:.4f}, lever = {train_lev:.4f}, Δ = {train_lev - train_v33:+.4f}")
    ratio = (train_lev - train_v33) / max(eval_lev - eval_v33, 1e-6)
    print(f"  TRAIN/EVAL gain ratio: {ratio:.2f}  (≈1: consistent generalization, >>1: train-distribution overfit)")

    # === R2: Per-site analysis ===
    print("\n=== R2: Per-site analysis ===")
    print(f"  {'site':<6} {'split':<6} {'n_rows':>7} {'v33':>8} {'lever':>8} {'Δ':>8}")
    for site in sorted(set(sites)):
        for split_name, mask in [("eval", ev_mask), ("train", tr_mask)]:
            ms = (sites == site) & mask
            if ms.sum() < 5: continue
            v_auc, n = macro_on_subset(v33, Y, sp_taxon, ms)
            l_auc, _ = macro_on_subset(P_lev, Y, sp_taxon, ms)
            if np.isnan(v_auc): continue
            print(f"  {site:<6} {split_name:<6} {ms.sum():>7} {v_auc:>8.4f} {l_auc:>8.4f} {l_auc-v_auc:>+8.4f}")

    # === R3: Per-class attribution ===
    print("\n=== R3: Per-class attribution (top contributors to eval gain) ===")
    aucs_v33 = per_class_auc(Y[ev_mask], v33[ev_mask])
    aucs_lev = per_class_auc(Y[ev_mask], P_lev[ev_mask])
    diffs = []
    for c in sorted(set(aucs_v33) & set(aucs_lev)):
        diffs.append((primary[c], sp_taxon[c], aucs_v33[c], aucs_lev[c], aucs_lev[c] - aucs_v33[c],
                       int(Y[ev_mask, c].sum())))
    diffs_df = pd.DataFrame(diffs, columns=["class", "taxon", "v33", "lever", "Δ", "n_pos_eval"])
    print("  Classes with biggest |Δ|:")
    print(diffs_df.sort_values("Δ", key=abs, ascending=False).head(15).to_string(index=False))

    # === R4: Holdout-site sensitivity ===
    print("\n=== R4: Holdout-site sensitivity (drop one site, recompute eval gain) ===")
    print(f"  {'holdout':<8} {'n_left':>7} {'v33':>8} {'lever':>8} {'Δ':>8}")
    eval_sites = sorted(set(sites[ev_mask]))
    for ho_site in eval_sites:
        ms = ev_mask & (sites != ho_site)
        if ms.sum() < 5: continue
        v_auc, _ = macro_on_subset(v33, Y, sp_taxon, ms)
        l_auc, _ = macro_on_subset(P_lev, Y, sp_taxon, ms)
        print(f"  {ho_site:<8} {ms.sum():>7} {v_auc:>8.4f} {l_auc:>8.4f} {l_auc-v_auc:>+8.4f}")

    # === R5: Finer parameter sweep around best ===
    print("\n=== R5: Finer parameter sweep around (th=0.4, α=0.5, δ=0.3) ===")
    print(f"  {'config':<30} {'macro_d':>9} {'sp_row':>8} {'Aves':>8}")
    from exp82_q1_4way_on_v33 import evaluate
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]
    for tl in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        for af in [0.30, 0.40, 0.50, 0.60, 0.70]:
            for dt in [0.20, 0.30, 0.40]:
                P = fn_rescue(v33, perch_prob, exp50, exp59, thresh_low=tl, alpha_fn=af, delta_t=dt)
                r = evaluate(P, v33, ev_mask, Y, sp_taxon, f"tl={tl} α={af} δ={dt}")
                rows.append(r)
    res = pd.DataFrame(rows)
    safe = res[res.predicted.str.startswith("A") & (res.label != "v33 ref")]
    print(f"\n  Top-10 class-A by Aves Δ:")
    print(safe.sort_values("Aves", ascending=False).head(10)[
        ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil"]
    ].to_string(index=False))
    print(f"\n  Top-10 class-A by macro_d:")
    print(safe.sort_values("macro_d", ascending=False).head(10)[
        ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil"]
    ].to_string(index=False))

    res.to_csv(EXP80 / "exp97_robustness.csv", index=False)


if __name__ == "__main__":
    main()
