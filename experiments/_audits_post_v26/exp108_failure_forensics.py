#!/usr/bin/env python3
"""exp108 — Deep failure-mode analysis on 122 SS eval rows.

For each model in {Perch, exp50, v33, P_NEW3 LOSO}:
  1. FN analysis: where labels are positive, what did the model predict?
     - Confident miss (pred < 0.1): structural blind spot
     - Weak miss (0.1-0.5): signal exists but threshold issue
     - What was predicted INSTEAD (top-3 confused species)?
  2. FP analysis: where labels are negative but pred > 0.5
     - Per-row: how many false positives?
     - Per-class: which species over-predicted?
  3. Cross-model agreement on errors:
     - Errors all 4 models share = "structural" (data limitation)
     - Errors only some make = "model-specific"
  4. Site / hour stratification:
     - Which (site, hour) bucket is worst per model?
  5. Score distribution analysis:
     - Are predictions polarized (clear 0/1) or uncertain (0.4-0.6)?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import macro_auc
from sklearn.metrics import roc_auc_score


def get_cached(name):
    d = np.load(EXP80 / name)
    if "scores" in d.files: return d["scores"]
    if "predictions" in d.files: return d["predictions"]
    return d[d.files[0]]


def main():
    print("=== exp108: Failure-mode forensics on 122 SS eval rows ===\n", flush=True)

    sc_g, Y, primary, l2i = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()
    perch_prob_ss = load_perch_scores_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")

    # Build v33
    from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend
    base = 0.7 * perch_prob_ss + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb_ss, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # Restrict to 122 eval rows
    ev_mask = sc_g.split.values == "eval"
    sc_ev = sc_g[ev_mask].reset_index(drop=True)
    Y_ev = Y[ev_mask]
    n_eval = len(Y_ev)

    # We need P_NEW3 predictions on EVAL only. Rebuild quickly.
    print(f"  Loading models, building P_NEW3 LOSO predictions on eval rows...")
    from exp106_pnew_hybrid import build_perch_init, train_hybrid

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0

    # P_NEW3 trained on TA + SS_train (excluding eval rows)
    tr_mask = sc_g.split.values == "train"
    X_train = np.concatenate([ta_emb[valid], perch_emb_ss[tr_mask]], axis=0)
    Y_train = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])
    W_init, b_init, _ = build_perch_init()
    _, p_new3_ev, _, _, _ = train_hybrid(
        X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ev.astype(np.float32),
        W_init, b_init, n_epochs=12, verbose=False
    )

    # Per-eval-row predictions for all models
    models = {
        "Perch":  perch_prob_ss[ev_mask],
        "exp50":  exp50[ev_mask],
        "v33":    v33[ev_mask],
        "P_NEW3": p_new3_ev,
    }

    # =================================================================
    # 1. PER-CLASS PERFORMANCE BREAKDOWN
    # =================================================================
    print("\n=== 1. Per-class performance on 122 eval rows ===\n")
    eval_classes = np.where(Y_ev.sum(axis=0) > 0)[0]
    print(f"  Evaluable classes (≥1 positive in 122 rows): {len(eval_classes)}")

    rows = []
    for c in eval_classes:
        n_pos = int(Y_ev[:, c].sum())
        rec = {"class_idx": c, "label": primary[c], "taxon": sp_taxon[c], "n_pos": n_pos}
        for name, P in models.items():
            try:
                rec[name] = roc_auc_score(Y_ev[:, c], P[:, c])
            except ValueError:
                rec[name] = float("nan")
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["best_model"] = df[list(models.keys())].idxmax(axis=1)

    print("\n  Per-taxon AUC summary:")
    print(df.groupby("taxon")[list(models.keys())].mean().round(4).to_string())

    print("\n  Best-model distribution by taxon:")
    print(pd.crosstab(df.taxon, df.best_model).to_string())

    # Bottom-K per model
    print("\n  Bottom-5 classes per model:")
    for name in models:
        bot = df.nsmallest(5, name)[["label","taxon","n_pos",name,"best_model"]]
        print(f"\n    {name}:")
        print(bot.to_string(index=False))

    # =================================================================
    # 2. FN ANALYSIS — confident vs weak misses
    # =================================================================
    print("\n\n=== 2. False-negative analysis (positive labels, model says 'no') ===\n")

    # For positive (row, class) entries, bucket by prediction value
    fn_buckets = [(0.0, 0.1, "confident_miss"),
                    (0.1, 0.3, "weak_miss"),
                    (0.3, 0.5, "borderline"),
                    (0.5, 1.01, "caught")]

    for name, P in models.items():
        pos_mask = Y_ev > 0  # (rows, classes) where label = 1
        n_pos = int(pos_mask.sum())
        print(f"  {name}: {n_pos} positive (row, class) entries")
        for lo, hi, label in fn_buckets:
            sel = pos_mask & (P >= lo) & (P < hi)
            print(f"    pred ∈ [{lo:.2f}, {hi:.2f}) ({label:<16}): {int(sel.sum()):>4} ({100*sel.sum()/n_pos:>5.1f}%)")

    # =================================================================
    # 3. CROSS-MODEL ERROR CORRELATION
    # =================================================================
    print("\n\n=== 3. Cross-model error correlation ===\n")
    print("  How often do models miss the SAME positives?\n")

    pos_idx = np.where(Y_ev > 0)
    n_pos = len(pos_idx[0])

    # For each model, binary "missed" = pred < 0.5 on positives
    miss_arr = {}
    for name, P in models.items():
        miss_arr[name] = (P[pos_idx] < 0.5).astype(np.int8)

    # Pairwise agreement: P(both miss | one misses)
    print("  Pairwise miss agreement (% of one's misses that other also misses):")
    print(f"  {'':<10} " + " ".join(f"{m:>10}" for m in models))
    for name1 in models:
        cells = []
        for name2 in models:
            both = (miss_arr[name1] & miss_arr[name2]).sum()
            either = miss_arr[name1].sum()
            cells.append(f"{100*both/either:>10.1f}" if either else f"{'--':>10}")
        print(f"  {name1:<10} " + " ".join(cells))

    # Universally missed = all 4 miss
    all_miss = miss_arr["Perch"] & miss_arr["exp50"] & miss_arr["v33"] & miss_arr["P_NEW3"]
    none_miss = (~miss_arr["Perch"].astype(bool)) & (~miss_arr["exp50"].astype(bool)) & \
                 (~miss_arr["v33"].astype(bool)) & (~miss_arr["P_NEW3"].astype(bool))

    print(f"\n  Out of {n_pos} positives:")
    print(f"    All 4 models MISS:      {int(all_miss.sum())} ({100*all_miss.sum()/n_pos:.1f}%) — STRUCTURAL")
    print(f"    All 4 models CATCH:     {int(none_miss.sum())} ({100*none_miss.sum()/n_pos:.1f}%) — easy")
    inconsistent = n_pos - all_miss.sum() - none_miss.sum()
    print(f"    Inconsistent (some-only): {int(inconsistent)} ({100*inconsistent/n_pos:.1f}%) — recoverable via ensemble")

    # Class-level breakdown of structural misses
    print("\n  Classes most often universally-missed (all 4 fail):")
    universal_miss_per_class = np.zeros(N_CLS)
    universal_total_per_class = np.zeros(N_CLS)
    for i in range(n_pos):
        c = pos_idx[1][i]
        universal_total_per_class[c] += 1
        if all_miss[i]:
            universal_miss_per_class[c] += 1

    rec_rows = []
    for c in eval_classes:
        if universal_total_per_class[c] > 0:
            rec_rows.append({
                "label": primary[c],
                "taxon": sp_taxon[c],
                "n_pos": int(universal_total_per_class[c]),
                "n_universal_miss": int(universal_miss_per_class[c]),
                "rate": universal_miss_per_class[c] / universal_total_per_class[c],
            })
    rec_df = pd.DataFrame(rec_rows).sort_values("rate", ascending=False)
    top10 = rec_df[rec_df.n_universal_miss >= 1].head(10)
    print(top10.to_string(index=False))

    # =================================================================
    # 4. CONFUSION TARGETS — what do models predict INSTEAD of truth?
    # =================================================================
    print("\n\n=== 4. Confusion targets: when model misses positive, what does it predict? ===\n")
    print("  For each (row, class) with positive label that ALL models miss,")
    print("  list the top-3 species predicted (per Perch and per v33).\n")

    confusion_examples = []
    for i in range(min(n_pos, 200)):
        if all_miss[i]:
            r = pos_idx[0][i]; c = pos_idx[1][i]
            # Top-3 predicted by Perch
            perch_top = np.argsort(perch_prob_ss[ev_mask][r])[::-1][:3]
            v33_top = np.argsort(v33[ev_mask][r])[::-1][:3]
            confusion_examples.append({
                "row": r,
                "true_class": primary[c],
                "true_taxon": sp_taxon[c],
                "perch_score_on_true": float(perch_prob_ss[ev_mask][r, c]),
                "v33_score_on_true": float(v33[ev_mask][r, c]),
                "perch_top3": [primary[p] for p in perch_top],
                "perch_top3_scores": [round(float(perch_prob_ss[ev_mask][r, p]), 3) for p in perch_top],
                "v33_top3": [primary[p] for p in v33_top],
                "v33_top3_scores": [round(float(v33[ev_mask][r, p]), 3) for p in v33_top],
            })

    if confusion_examples:
        print(f"  {len(confusion_examples)} structural-miss examples (all 4 models fail).")
        print("\n  First 5 examples:")
        for ex in confusion_examples[:5]:
            print(f"\n    Row {ex['row']}: true = {ex['true_class']} ({ex['true_taxon']})")
            print(f"      Perch on truth: {ex['perch_score_on_true']:.3f}, top-3: " +
                   ", ".join(f"{l}({s})" for l, s in zip(ex['perch_top3'], ex['perch_top3_scores'])))
            print(f"      v33 on truth:   {ex['v33_score_on_true']:.3f}, top-3: " +
                   ", ".join(f"{l}({s})" for l, s in zip(ex['v33_top3'], ex['v33_top3_scores'])))

    # =================================================================
    # 5. FALSE POSITIVE ANALYSIS
    # =================================================================
    print("\n\n=== 5. False-positive analysis (negative labels, model says 'yes') ===\n")
    for name, P in models.items():
        neg_mask = Y_ev == 0
        fp_high = neg_mask & (P > 0.7)
        fp_med = neg_mask & (P > 0.5) & (P <= 0.7)
        n_neg = int(neg_mask.sum())
        print(f"  {name}: of {n_neg} negative entries: high FP (>0.7) = {int(fp_high.sum())} ({100*fp_high.sum()/n_neg:.2f}%), med FP (0.5-0.7) = {int(fp_med.sum())} ({100*fp_med.sum()/n_neg:.2f}%)")

    print("\n  Per-class top-3 over-predicted classes (most FP per class):")
    for name, P in models.items():
        neg_mask = Y_ev == 0
        # For each class, count FP > 0.7
        fp_per_class = (neg_mask & (P > 0.7)).sum(axis=0)
        top_fp = np.argsort(fp_per_class)[::-1][:3]
        cls_info = [(primary[c], sp_taxon[c], int(fp_per_class[c])) for c in top_fp]
        print(f"    {name}: " + ", ".join(f"{l}({t}, {n})" for l, t, n in cls_info))

    # =================================================================
    # 6. SITE / HOUR STRATIFICATION
    # =================================================================
    print("\n\n=== 6. Per-site error rates (122 eval rows) ===\n")
    print("  Macro AUC by site (only sites with ≥3 evaluable classes):\n")
    sites_ev = sc_ev.site.values
    print(f"  {'site':<6} {'n_rows':>7} {'n_eval_cls':>10} " + " ".join(f"{m:>10}" for m in models))
    for site in sorted(set(sites_ev)):
        sm = sites_ev == site
        if sm.sum() < 3: continue
        n_eval_cls = (Y_ev[sm].sum(axis=0) > 0).sum()
        if n_eval_cls < 3: continue
        cells = []
        for name, P in models.items():
            try:
                m, _ = macro_auc(Y_ev[sm].astype(np.float32), P[sm])
                cells.append(f"{m:>10.4f}")
            except Exception:
                cells.append(f"{'--':>10}")
        print(f"  {site:<6} {sm.sum():>7} {n_eval_cls:>10} " + " ".join(cells))

    # =================================================================
    # 7. SCORE DISTRIBUTION SHAPE
    # =================================================================
    print("\n\n=== 7. Score distribution analysis ===\n")
    print("  Are predictions polarized (clear 0/1) or uncertain?\n")
    for name, P in models.items():
        # Histogram of all scores
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        hist = np.histogram(P.flatten(), bins=bins)[0]
        total = hist.sum()
        print(f"  {name:<10} " + " ".join(f"[{bins[i]:.1f}-{bins[i+1]:.1f}]:{100*hist[i]/total:>5.1f}%" for i in range(len(bins)-1)))

    print("\n  Polarization metric (% of scores in [0.45, 0.55]):")
    for name, P in models.items():
        uncertain = ((P >= 0.45) & (P < 0.55)).mean()
        print(f"    {name}: {100*uncertain:.2f}%")

    # =================================================================
    # 8. WHERE DOES P_NEW3 BLEND BREAK?
    # =================================================================
    print("\n\n=== 8. v33+P_NEW3 blend: where does it help / hurt? ===\n")
    blend = (0.9 * v33[ev_mask] + 0.1 * p_new3_ev).astype(np.float32)

    # Per-class AUC delta
    deltas = []
    for c in eval_classes:
        try:
            v33_auc = roc_auc_score(Y_ev[:, c], v33[ev_mask][:, c])
            blend_auc = roc_auc_score(Y_ev[:, c], blend[:, c])
            d = blend_auc - v33_auc
            deltas.append({
                "label": primary[c], "taxon": sp_taxon[c],
                "n_pos": int(Y_ev[:, c].sum()),
                "v33": v33_auc, "blend": blend_auc, "delta": d,
            })
        except ValueError:
            pass
    dd = pd.DataFrame(deltas).sort_values("delta")
    print("  Top-5 classes where blend HURTS (negative delta):")
    print(dd.head(5).round(4).to_string(index=False))
    print("\n  Top-5 classes where blend HELPS (positive delta):")
    print(dd.tail(5).round(4).to_string(index=False))
    print(f"\n  Blend deltas: mean {dd.delta.mean():+.4f}, "
          f"std {dd.delta.std():.4f}, "
          f"% positive {100*(dd.delta > 0).mean():.1f}%, "
          f"% negative {100*(dd.delta < 0).mean():.1f}%")

    # Save full forensic report
    out_dir = ROOT / "experiments/_audits_post_v26/exp108_outputs"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "per_class_auc.csv", index=False)
    rec_df.to_csv(out_dir / "universal_misses.csv", index=False)
    dd.to_csv(out_dir / "blend_deltas.csv", index=False)
    pd.DataFrame(confusion_examples).to_csv(out_dir / "confusion_examples.csv", index=False)
    print(f"\nFull tables saved to {out_dir}")


if __name__ == "__main__":
    main()
