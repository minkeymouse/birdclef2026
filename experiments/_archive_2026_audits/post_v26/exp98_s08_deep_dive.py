#!/usr/bin/env python3
"""exp98 — S08 regression diagnostic.

The exp97 lever helps overall but HURTS S08 eval rows by macro_d −0.023.
S08 is Insecta-dominant (43 of 60 SS rows are UNK_ONLY). Hypothesis:
exp50/exp59 give high "Aves" predictions on S08 rows because they were
trained with S08 labels containing mostly Insecta — i.e., they 'memorize'
that S08 audio is non-Aves territory and produce specific biases.

The lever boosts class c when teacher is high. If teacher (exp50/exp59)
produces inflated prediction on certain Aves classes at S08 due to site
fingerprint, the lever amplifies an incorrect signal.

This script:
  1. Compare per-class teacher behavior on S08 vs other sites
  2. Identify specific (S08_row, class) pairs where lever pushes wrong
  3. Test variants that suppress lever effect on S08-like sites
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
from _lib.eval_metrics import per_class_auc
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name): return np.load(EXP80 / name)["scores"]


def fn_rescue(v33, perch_prob, exp50, exp59, thresh_low=0.55, alpha_fn=0.7, delta_t=0.2):
    out = v33.copy()
    low_mask = v33 < thresh_low
    teacher_max = np.maximum(exp50, exp59)
    teacher_signal = np.maximum(teacher_max - delta_t, 0.0)
    boost = alpha_fn * teacher_signal * (1.0 - v33)
    out[low_mask] = v33[low_mask] + boost[low_mask]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main():
    print("=== exp98: S08 deep dive ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")

    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    sites = sc_g.site.values
    ev_mask = sc_g.split.values == "eval"
    s08_eval = ev_mask & (sites == "S08")
    print(f"S08 eval rows: {s08_eval.sum()}")

    # ===== 1. Compare exp50/exp59 per-class predictions: S08 vs other sites =====
    print("\n=== 1. Per-class teacher inflation: S08 vs other sites (mean diff) ===")
    other_eval = ev_mask & (sites != "S08")
    e50_s08_mean = exp50[s08_eval].mean(axis=0)
    e50_other_mean = exp50[other_eval].mean(axis=0)
    e59_s08_mean = exp59[s08_eval].mean(axis=0)
    e59_other_mean = exp59[other_eval].mean(axis=0)
    print(f"  Top-15 mapped Aves classes where exp50 inflates on S08 vs other:")
    aves_mask = sp_taxon == "Aves"
    col_var = perch_prob.var(axis=0)
    mapped_idx = np.where(col_var >= 1e-6)[0]
    candidate = [c for c in range(N_CLS) if aves_mask[c] and c in mapped_idx]
    diffs = []
    for c in candidate:
        diffs.append((primary[c], e50_s08_mean[c], e50_other_mean[c],
                       e50_s08_mean[c] - e50_other_mean[c],
                       e59_s08_mean[c] - e59_other_mean[c]))
    df_diffs = pd.DataFrame(diffs, columns=["class", "exp50_S08", "exp50_other", "exp50_Δ", "exp59_Δ"])
    print(df_diffs.sort_values("exp50_Δ", ascending=False).head(15).to_string(index=False))
    print("\n  Top-10 classes where exp50 DEFLATES on S08:")
    print(df_diffs.sort_values("exp50_Δ", ascending=True).head(10).to_string(index=False))

    # ===== 2. Per-row at S08: which (row, class) pairs does lever modify, and is it correct? =====
    print("\n=== 2. S08 eval rows: lever modifications ===")
    P_lev = fn_rescue(v33, perch_prob, exp50, exp59)
    s08_idx = np.where(s08_eval)[0]
    print(f"  S08 eval rows: indices {s08_idx.tolist()}")
    print(f"  GT species in S08 eval rows:")
    for i in s08_idx:
        gt = [primary[c] for c in np.where(Y[i] == 1)[0]]
        print(f"    row {i} ({sc_g.iloc[i].row_id}): GT={gt}")

    # For each S08 eval row, find (class, v33, lever, GT) where lever changed
    print(f"\n  Per-row top boosts on S08 (class, v33→lever, is_GT?):")
    for i in s08_idx[:5]:
        gt_set = set(np.where(Y[i] == 1)[0])
        diffs = P_lev[i] - v33[i]
        top_boost_idx = diffs.argsort()[::-1][:5]
        print(f"  row {i}:")
        for c in top_boost_idx:
            if diffs[c] < 0.001: continue
            is_gt = "GT" if c in gt_set else "  "
            print(f"    {primary[c]:<14} {sp_taxon[c]:<10} v33={v33[i,c]:.3f} → lever={P_lev[i,c]:.3f} (Δ {diffs[c]:+.3f}) {is_gt}")

    # ===== 3. Variant: suppress lever on rows with high "S08-like" signature =====
    # S08-like = high Insecta-channel sum OR high site-Insecta detector
    # Use: row_insecta_sum = sum of exp50 predictions on Insecta sonotype columns
    insecta_cols = np.where(sp_taxon == "Insecta")[0]
    valid_insecta = np.array([c for c in insecta_cols if str(primary[c]).startswith("47158son") and exp50[:, c].var() > 1e-6])
    print(f"\n=== 3. Variant testing — suppress lever on Insecta-fingerprint rows ===")
    print(f"  Valid Insecta sonotype cols (with non-zero var): {len(valid_insecta)}")

    if len(valid_insecta) == 0:
        # Use all Insecta cols regardless
        valid_insecta = insecta_cols
        print(f"  Using all Insecta cols: {len(valid_insecta)}")

    insecta_score = exp50[:, valid_insecta].mean(axis=1) if len(valid_insecta) > 0 else np.zeros(len(sc_g))
    print(f"  insecta_score on S08 eval rows: mean = {insecta_score[s08_eval].mean():.3f}")
    print(f"  insecta_score on other eval rows: mean = {insecta_score[other_eval].mean():.3f}")

    # Variant: scale lever by (1 - insecta_score)
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]
    base_lever = fn_rescue(v33, perch_prob, exp50, exp59)
    rows.append(evaluate(base_lever, v33, ev_mask, Y, sp_taxon, "FN_R best (no S08 protection)"))

    for ins_thresh in [0.3, 0.5, 0.7]:
        for shrink in [0.3, 0.5, 1.0]:
            # On rows where insecta_score > thresh, shrink the boost
            mod = base_lever.copy()
            high_ins = insecta_score > ins_thresh
            # Original v33 + shrunk boost on those rows
            mod[high_ins] = v33[high_ins] + shrink * (base_lever[high_ins] - v33[high_ins])
            rows.append(evaluate(mod, v33, ev_mask, Y, sp_taxon,
                                  f"FN_R + Insecta-shrink (thresh={ins_thresh}, shrink={shrink})"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== Variants ===")
    print(res[cols].sort_values("macro_d", ascending=False).to_string(index=False))

    # Per-site analysis of best variant
    print("\n=== Best variant per-site analysis ===")
    best = res[res.label.str.startswith("FN_R + Insecta-shrink")].sort_values("macro_d", ascending=False).iloc[0]
    print(f"  best variant: {best['label']}")
    # Re-build best
    parts = best["label"].split("=")
    ins_thresh = float(parts[1].split(",")[0].strip())
    shrink = float(parts[2].split(")")[0].strip())
    mod = base_lever.copy()
    high_ins = insecta_score > ins_thresh
    mod[high_ins] = v33[high_ins] + shrink * (base_lever[high_ins] - v33[high_ins])
    print(f"  Per-site eval Δ:")
    for s in sorted(set(sites[ev_mask])):
        ms = ev_mask & (sites == s)
        if ms.sum() < 5: continue
        v_aucs = per_class_auc(Y[ms], v33[ms])
        l_aucs = per_class_auc(Y[ms], mod[ms])
        common = set(v_aucs) & set(l_aucs)
        if not common: continue
        v_macro = np.mean([v_aucs[c] for c in common])
        l_macro = np.mean([l_aucs[c] for c in common])
        print(f"    {s:<6} n={ms.sum():>3}: v33={v_macro:.4f}, lever={l_macro:.4f}, Δ={l_macro-v_macro:+.4f}")

    res.to_csv(EXP80 / "exp98_s08_deep_dive.csv", index=False)


if __name__ == "__main__":
    main()
