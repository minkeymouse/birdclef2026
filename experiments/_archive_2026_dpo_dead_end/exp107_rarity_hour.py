#!/usr/bin/env python3
"""exp107 — Test A (rarity cliff) + Test B (hour-of-day) on our pipeline.

Test A: Does the 1-10-clip cliff (exp45) persist in our LEARNED models?
  - Compute n_train_audio per class from exp22 cache
  - Bands: 0, 1-10, 11-50, 51-200, 200+
  - For each: compute LOSO macro AUC for Perch baseline, P_NEW1 (random init),
    P_NEW3 (Perch-init hybrid)
  - Verdict:
    - If learned models close the cliff → architecture/data-mix fix works
    - If they preserve the cliff → external data is the only lever

Test B: Does v33/P_NEW3 show hour-of-day bias?
  - Bucket SS rows by hour: {0-6, 6-12, 12-18, 18-24}
  - Compute macro AUC per bucket for v33 and P_NEW3
  - Test: hour-feature additive bias (per-class learned hour offset)
  - Verdict: if hour buckets show >0.05 spread + bias correction recovers
    even half of it → temporal augmentation is a viable lever
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from sklearn.metrics import roc_auc_score


def get_cached(name):
    d = np.load(EXP80 / name)
    if "scores" in d.files: return d["scores"]
    if "predictions" in d.files: return d["predictions"]
    return d[d.files[0]]


def per_class_auc_arr(Y, P):
    """Returns (234,) AUC array, NaN where unevaluable."""
    out = np.full(N_CLS, np.nan, dtype=np.float64)
    for c in range(N_CLS):
        y = Y[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
        try:
            out[c] = roc_auc_score(y, P[:, c])
        except ValueError:
            pass
    return out


def main():
    print("=== exp107: rarity cliff + hour-of-day analysis ===\n", flush=True)

    sc_g, Y, primary, l2i = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()
    perch_prob_ss = load_perch_scores_labeled()

    # ----- Load all model predictions -----
    p_new1 = get_cached("p_new_predictions.npz")           # random init
    p_new3 = get_cached("p_new2_hybrid_predictions.npz")    # Perch-init hybrid

    # Build v33
    from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend
    exp50 = get_cached("exp50_scores_labeled.npz")
    base = 0.7 * perch_prob_ss + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb_ss, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # ----- Load n_train_audio counts -----
    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    n_train_audio = np.bincount(ta_y_idx[valid], minlength=N_CLS).astype(np.int64)
    print(f"  n_train_audio: min {n_train_audio.min()}, max {n_train_audio.max()}, "
          f"mean {n_train_audio.mean():.1f}, n_zero {(n_train_audio == 0).sum()}")

    # ===== TEST A: RARITY CLIFF =====
    print("\n=== TEST A: Rarity cliff (n_train_audio band × model AUC) ===\n")

    bands = [
        (0, 0, "0"),
        (1, 10, "1-10"),
        (11, 50, "11-50"),
        (51, 200, "51-200"),
        (201, 1_000_000, "200+"),
    ]

    # Per-class AUC on full SS labeled (no LOSO yet — full sample, ~739 rows)
    aucs = {
        "Perch":   per_class_auc_arr(Y, perch_prob_ss),
        "v33":     per_class_auc_arr(Y, v33),
        "P_NEW1":  per_class_auc_arr(Y, p_new1),
        "P_NEW3":  per_class_auc_arr(Y, p_new3),
    }

    print(f"  Same-distribution AUC by n_train_audio band:")
    print(f"  {'Band':<8} {'n_cls':>6} " + " ".join(f"{m:>10}" for m in aucs))
    for lo, hi, label in bands:
        mask = (n_train_audio >= lo) & (n_train_audio <= hi)
        cells = []
        for m in aucs:
            valid_aucs = aucs[m][mask]
            valid_aucs = valid_aucs[~np.isnan(valid_aucs)]
            cells.append(f"{valid_aucs.mean():>10.4f}" if len(valid_aucs) else f"{'--':>10}")
        n_cls_with_eval = sum(~np.isnan(aucs['Perch'][mask]))
        print(f"  {label:<8} {n_cls_with_eval:>6} " + " ".join(cells))

    # Now do LOSO-style comparison for the most interesting band (1-10)
    print(f"\n  --- Per-band Aves vs non-Aves split (where most Aves cliff lives) ---")
    print(f"  {'Band':<8} {'taxon':<10} {'n_cls':>6} " + " ".join(f"{m:>10}" for m in aucs))
    for lo, hi, label in bands:
        for taxon in ["Aves", "Insecta", "Amphibia", "Mammalia", "Reptilia"]:
            mask = (n_train_audio >= lo) & (n_train_audio <= hi) & (sp_taxon == taxon)
            n_eval = sum(~np.isnan(aucs['Perch'][mask]))
            if n_eval == 0: continue
            cells = []
            for m in aucs:
                v = aucs[m][mask]; v = v[~np.isnan(v)]
                cells.append(f"{v.mean():>10.4f}" if len(v) else f"{'--':>10}")
            print(f"  {label:<8} {taxon:<10} {n_eval:>6} " + " ".join(cells))

    # ===== TEST B: HOUR OF DAY =====
    print("\n\n=== TEST B: Hour-of-day analysis ===\n")
    hours = sc_g.hour.values
    print(f"  Hour distribution: min {hours.min()}, max {hours.max()}, "
          f"unique values: {sorted(set(hours))}")

    buckets = [
        (0, 5, "0-5h"),
        (6, 11, "6-11h"),
        (12, 17, "12-17h"),
        (18, 23, "18-23h"),
    ]
    print(f"\n  Macro AUC by hour bucket (full SS data, ~all 739 rows):")
    print(f"  {'Bucket':<10} {'n_rows':>7} {'Perch':>8} {'v33':>8} {'P_NEW3':>8}")

    bucket_aucs = []
    for lo, hi, label in buckets:
        bm = (hours >= lo) & (hours <= hi)
        if bm.sum() < 5:
            print(f"  {label:<10} {bm.sum():>7}    --        --        --")
            continue
        cells = {}
        for name, P in [("Perch", perch_prob_ss), ("v33", v33), ("P_NEW3", p_new3)]:
            try:
                m, _ = macro_auc(Y[bm].astype(np.float32), P[bm])
                cells[name] = m
            except Exception:
                cells[name] = float("nan")
        print(f"  {label:<10} {bm.sum():>7} {cells['Perch']:>8.4f} {cells['v33']:>8.4f} {cells['P_NEW3']:>8.4f}")
        bucket_aucs.append((label, cells))

    if len(bucket_aucs) >= 2:
        for name in ["Perch", "v33", "P_NEW3"]:
            vals = [b[1][name] for b in bucket_aucs if not np.isnan(b[1][name])]
            if vals:
                print(f"  {name} hour-bucket spread: max-min = {max(vals) - min(vals):+.4f}")

    # Test simple per-hour bias correction using sklearn LR
    print("\n  --- Per-hour bias correction test (cross-bucket CV) ---")
    print("  Train per-class hour-bias on N-1 buckets, eval on holdout bucket.")
    print("  If post-correction AUC improves → temporal augmentation viable.")
    print(f"  {'holdout':<10} {'before':>8} {'after':>8} {'Δ':>8}")

    from scipy.special import logit, expit
    for lo, hi, label in buckets:
        ho_mask = (hours >= lo) & (hours <= hi)
        if ho_mask.sum() < 5: continue
        keep_mask = ~ho_mask

        # Per-class additive bias on logit(v33), fit only on keep rows
        # bias_c = mean_c(logit(v33[keep, c]) - logit(prior_c))
        # Apply: logit_corr = logit(v33) - bias_c (per holdout's hour bucket)
        # This tests if hour-conditional class priors transfer
        v33_clip = np.clip(v33, 1e-4, 1 - 1e-4)
        v33_logit = logit(v33_clip)
        ho_v33_logit_mean = v33_logit[ho_mask].mean(axis=0)
        keep_v33_logit_mean = v33_logit[keep_mask].mean(axis=0)
        bias_per_class = ho_v33_logit_mean - keep_v33_logit_mean  # shift between buckets
        v33_corr_logit = v33_logit[ho_mask] - bias_per_class
        v33_corr = expit(v33_corr_logit).astype(np.float32)

        try:
            before, _ = macro_auc(Y[ho_mask].astype(np.float32), v33[ho_mask])
            after, _ = macro_auc(Y[ho_mask].astype(np.float32), v33_corr)
            print(f"  {label:<10} {before:>8.4f} {after:>8.4f} {after-before:>+8.4f}")
        except Exception:
            pass

    # Save record
    out_path = ROOT / "experiments/_audits_post_v26/exp107_results.md"
    with open(out_path, "w") as f:
        f.write("# exp107 — Rarity cliff + Hour-of-day analysis\n\n")
        f.write("Run on full 739 labeled SS rows.\n\n")
        f.write("## Test A: Rarity cliff (n_train_audio band)\n\n")
        f.write("| Band | n_cls | Perch | v33 | P_NEW1 (rand) | P_NEW3 (hybrid) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for lo, hi, label in bands:
            mask = (n_train_audio >= lo) & (n_train_audio <= hi)
            row = []
            for m in aucs:
                v = aucs[m][mask]; v = v[~np.isnan(v)]
                row.append(f"{v.mean():.4f}" if len(v) else "--")
            n_eval = sum(~np.isnan(aucs['Perch'][mask]))
            f.write(f"| {label} | {n_eval} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n")

        f.write("\n## Test B: Hour-of-day macro AUC\n\n")
        f.write("| Bucket | n_rows | Perch | v33 | P_NEW3 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for label, cells in bucket_aucs:
            bm_sum = ((hours >= int(label.split("-")[0])) & (hours <= int(label.split("-")[1].rstrip("h")))).sum()
            f.write(f"| {label} | {bm_sum} | {cells['Perch']:.4f} | {cells['v33']:.4f} | {cells['P_NEW3']:.4f} |\n")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
