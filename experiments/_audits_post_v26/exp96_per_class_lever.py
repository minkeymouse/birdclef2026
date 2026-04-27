#!/usr/bin/env python3
"""exp96 — Per-(row, class) selective lever using exp95's discoveries.

Two surgical interventions on v33 base:

  FN_RESCUE: where v33[i,c] < THRESH_LOW AND (sed_on_c OR cnxt_on_c) is HIGH,
    boost v33[i,c] proportionally. Uses sed_on_c TN/FN AUC = 0.983 finding.

  FP_SUPPRESS: where v33[i,c] > THRESH_HIGH AND |perch[i,c] - exp50[i,c]|
    is HIGH (teachers disagree), suppress v33[i,c]. Uses TP/FP AUC = 0.890.

Key safety property: each modification is per-(row, class) — does NOT change
W_PERCH globally. Avoids the v36 trap (W_PERCH 0.5 → −0.017 LB).

Sweep:
  - thresh_low ∈ {0.3, 0.4, 0.5}
  - thresh_high ∈ {0.5, 0.6, 0.7}
  - alpha_fn ∈ {0.1, 0.2, 0.3, 0.5}
  - alpha_fp ∈ {0.05, 0.1, 0.2}
  - delta_disagree ∈ {0.3, 0.5}

Eval: 122 held-out rows on 40 evaluable classes. Report
  (macro_d, sp_row, per-taxon Δ, predicted_LB_class).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name): return np.load(EXP80 / name)["scores"]


def fn_rescue(v33, perch_prob, exp50, exp59, thresh_low=0.4, alpha_fn=0.2, delta_t=0.3):
    """Boost v33 where v33<thresh_low AND (sed_on_c OR cnxt_on_c) > delta_t."""
    out = v33.copy()
    low_mask = v33 < thresh_low
    teacher_max = np.maximum(exp50, exp59) if exp59 is not None else exp50
    teacher_signal = np.maximum(teacher_max - delta_t, 0.0)
    # boost = alpha_fn * teacher_signal * (1 - v33)  — proportional to room left
    boost = alpha_fn * teacher_signal * (1.0 - v33)
    out[low_mask] = v33[low_mask] + boost[low_mask]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fp_suppress(probs, perch_prob, exp50, thresh_high=0.6, alpha_fp=0.1, delta_d=0.3):
    """Suppress where probs>thresh_high AND |perch-exp50| > delta_d."""
    out = probs.copy()
    high_mask = probs > thresh_high
    disagree = np.abs(perch_prob - exp50)
    sup_strength = np.clip((disagree - delta_d) / (1.0 - delta_d), 0.0, 1.0)
    suppress = alpha_fp * sup_strength
    out[high_mask] = probs[high_mask] * (1.0 - suppress[high_mask])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main():
    print("=== exp96: per-(row, class) selective lever ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")

    # Build v33 baseline
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    # === Sweep FN_RESCUE alone ===
    print("\n=== A. FN_RESCUE alone (low-prediction boost) ===")
    for thresh_low in [0.3, 0.4, 0.5]:
        for alpha_fn in [0.1, 0.2, 0.3, 0.5]:
            for delta_t in [0.3, 0.5]:
                P = fn_rescue(v33, perch_prob, exp50, exp59,
                              thresh_low=thresh_low, alpha_fn=alpha_fn, delta_t=delta_t)
                rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                                      f"FN_R th={thresh_low} α={alpha_fn} δ={delta_t}"))

    # === Sweep FP_SUPPRESS alone ===
    print("\n=== B. FP_SUPPRESS alone (high-prediction disagree-suppress) ===")
    for thresh_high in [0.5, 0.6, 0.7]:
        for alpha_fp in [0.05, 0.1, 0.2, 0.3]:
            for delta_d in [0.3, 0.5]:
                P = fp_suppress(v33, perch_prob, exp50,
                                 thresh_high=thresh_high, alpha_fp=alpha_fp, delta_d=delta_d)
                rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                                      f"FP_S th={thresh_high} α={alpha_fp} δ={delta_d}"))

    # === Sweep FN_RESCUE + FP_SUPPRESS combined ===
    print("\n=== C. Combined FN_RESCUE + FP_SUPPRESS ===")
    for tl in [0.4, 0.5]:
        for af in [0.2, 0.3]:
            for th in [0.5, 0.6]:
                for ap in [0.1, 0.2]:
                    P = fn_rescue(v33, perch_prob, exp50, exp59, thresh_low=tl, alpha_fn=af, delta_t=0.3)
                    P = fp_suppress(P, perch_prob, exp50, thresh_high=th, alpha_fp=ap, delta_d=0.3)
                    rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                                          f"FN+FP tl={tl} αf={af} th={th} αp={ap}"))

    # === D. Conservative variant: only use single-teacher boost (exp50 only, not exp59) ===
    print("\n=== D. Single-teacher (exp50) FN_RESCUE — even more conservative ===")
    for tl in [0.4, 0.5]:
        for af in [0.1, 0.2, 0.3]:
            P = fn_rescue(v33, perch_prob, exp50, None, thresh_low=tl, alpha_fn=af, delta_t=0.3)
            rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                                  f"FN_R(exp50_only) tl={tl} αf={af}"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]

    print("\n=== ALL RESULTS sorted by macro_d desc ===")
    print(res.sort_values("macro_d", ascending=False)[cols].head(25).to_string(index=False))
    res.to_csv(EXP80 / "exp96_per_class_lever.csv", index=False)
    print(f"\nSaved → {EXP80}/exp96_per_class_lever.csv")

    print("\n=== Top class-A candidates (sp_row≥0.99 AND Aves Δ ≥ 0) by Aves Δ ===")
    safe = res[res.predicted.str.startswith("A") & (res.label != "v33 ref")]
    if len(safe) > 0:
        top = safe.sort_values("Aves", ascending=False).head(10)
        print(top[cols].to_string(index=False))

    print("\n=== Top class-A candidates by macro_d ===")
    if len(safe) > 0:
        top2 = safe.sort_values("macro_d", ascending=False).head(10)
        print(top2[cols].to_string(index=False))


if __name__ == "__main__":
    main()
