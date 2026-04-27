#!/usr/bin/env python3
"""exp101 — Multi-teacher lever variants beyond exp100's flat correction.

User's intuition: when Perch shows FP signature on a (row, class), trust
SED more. When FN signature, boost via SED. Each per-(row, class).

Variants tested:

  V_A (= exp100 baseline): linear correction on final v33
       FP_high  → v33 *= (1 - β * P_FP)
       FN_high  → v33 += α * P_FN * (1 - v33)

  V_B: Perch-replace. When P_FP > thresh, replace v33[i,c] with SED average.
       v33[i,c] := P_FP * mean(exp50, exp59)[i,c] + (1-P_FP) * v33[i,c]

  V_C: SED-boost. When P_FN > thresh, blend in max(SED).
       v33[i,c] += α * P_FN * (max(exp50, exp59)[i,c] - v33[i,c])

  V_D: Combined V_B + V_C.

  V_E: Taxon-aware variant. For Aves (where SED's site shortcut hurts)
       use ConvNeXt only (slightly different recipe). For non-Aves
       use full SED average.

  V_F: Disagreement-aware. Use perch_sed_disagree directly to interpolate
       between Perch-only and SED-only views.

Pick best class-A by joint criteria (Aves Δ + sp_row + per-taxon
balance), then promote to LB-candidate.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS)
from _lib.eval_metrics import per_class_auc
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name): return np.load(EXP80 / name)["scores"]


# ============== Re-build LR detectors (universal features only) ==============
def build_lr_detectors(sc_g, perch_prob, exp50, exp59, v33, sp_taxon, Y, candidate_classes,
                       file_mean, file_std):
    """Return P_FP_grid (n_rows, N_CLS) and P_FN_grid."""
    UNIVERSAL = ["perch_on_c", "exp50_on_c", "exp59_on_c", "perch_sed_disagree",
                  "perch_low_sed_high", "file_mean_e50_c", "file_std_e50_c",
                  "file_uniform_e50_c", "v33_on_c"]

    def make_features(i, c):
        return [
            perch_prob[i, c], exp50[i, c], exp59[i, c],
            abs(perch_prob[i, c] - exp50[i, c]),
            max(0, exp50[i, c] - perch_prob[i, c]),
            file_mean[i, c], file_std[i, c],
            1.0 - (file_std[i, c] / (file_mean[i, c] + 1e-6)),
            v33[i, c],
        ]

    # Build train rows for FP / FN
    fp_rows = []; fn_rows = []
    for i in range(len(sc_g)):
        if sc_g.iloc[i].split != "train": continue
        for c in candidate_classes:
            v = v33[i, c]
            if v > 0.5:
                fp_rows.append((i, c, make_features(i, c), int(Y[i, c] == 0)))
            else:
                fn_rows.append((i, c, make_features(i, c), int(Y[i, c] == 1)))

    X_fp = np.array([r[2] for r in fp_rows]); y_fp = np.array([r[3] for r in fp_rows])
    X_fn = np.array([r[2] for r in fn_rows]); y_fn = np.array([r[3] for r in fn_rows])
    sc_fp = StandardScaler().fit(X_fp); sc_fn = StandardScaler().fit(X_fn)
    clf_fp = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42).fit(sc_fp.transform(X_fp), y_fp)
    clf_fn = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42).fit(sc_fn.transform(X_fn), y_fn)

    # Predict on ALL pairs
    P_FP = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    P_FN = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    for i in range(len(sc_g)):
        for c in candidate_classes:
            v = v33[i, c]
            f = np.array([make_features(i, c)])
            if v > 0.5:
                P_FP[i, c] = clf_fp.predict_proba(sc_fp.transform(f))[0, 1]
            else:
                P_FN[i, c] = clf_fn.predict_proba(sc_fn.transform(f))[0, 1]
    return P_FP, P_FN


def main():
    print("=== exp101: multi-teacher lever variants ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")
    exp73 = get_cached("exp73_scores_labeled.npz")
    exp84b = get_cached("exp84b_scores_labeled.npz")

    # v33 baseline
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)
    ev_mask = sc_g.split.values == "eval"

    # File-level stats
    file_mean = np.zeros_like(exp50); file_std = np.zeros_like(exp50)
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = exp50[idx]
        m = sub.mean(0); sd = sub.std(0)
        for ii in idx: file_mean[ii] = m; file_std[ii] = sd

    col_var = perch_prob.var(axis=0)
    mapped_idx = np.where(col_var >= 1e-6)[0]
    aves_mask = sp_taxon == "Aves"
    candidate_classes = [c for c in range(N_CLS)
                          if aves_mask[c] and c in mapped_idx
                          and Y[:, c].sum() >= 5 and (Y[:, c] == 0).sum() >= 50]
    print(f"Candidate classes: {len(candidate_classes)}", flush=True)

    print("Building LR detectors (this takes ~20s)...", flush=True)
    P_FP, P_FN = build_lr_detectors(sc_g, perch_prob, exp50, exp59, v33,
                                      sp_taxon, Y, candidate_classes, file_mean, file_std)
    print("  P_FP nonzero:", (P_FP > 0).sum(), " P_FN nonzero:", (P_FN > 0).sum(), flush=True)

    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    sed_avg = (exp50 + exp59) / 2.0
    sed_max = np.maximum(exp50, exp59)
    sed_avg_4 = (exp50 + exp59 + exp73 + exp84b) / 4.0

    # ===== V_A baseline (exp100): linear FP suppress + FN boost =====
    print("\n=== V_A: linear correction (exp100 baseline) ===", flush=True)
    for af in [0.10, 0.20, 0.30, 0.50]:
        for bf in [0.00, 0.10, 0.20]:
            P = v33.copy()
            high_fn = (v33 < 0.5)
            P[high_fn] = v33[high_fn] + af * P_FN[high_fn] * (1 - v33[high_fn])
            high_fp = (v33 > 0.5)
            P[high_fp] = v33[high_fp] * (1 - bf * P_FP[high_fp])
            rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                                  f"V_A α={af} β={bf}"))

    # ===== V_B: Perch-replace on high FP (substitute SED average) =====
    print("\n=== V_B: Perch-replace by SED-average on high FP ===", flush=True)
    for thresh_fp in [0.5, 0.7, 0.85]:
        for sed_choice in [("avg2", sed_avg), ("max", sed_max), ("avg4", sed_avg_4)]:
            sed_pred = sed_choice[1]
            P = v33.copy()
            high_fp = (v33 > 0.5) & (P_FP > thresh_fp)
            P[high_fp] = sed_pred[high_fp]
            rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                                  f"V_B thresh={thresh_fp} {sed_choice[0]}"))

    # ===== V_C: SED-boost on high FN =====
    print("\n=== V_C: SED-boost on high FN ===", flush=True)
    for thresh_fn in [0.5, 0.7]:
        for af in [0.30, 0.50, 0.70]:
            for sed_choice in [("max", sed_max), ("avg2", sed_avg)]:
                sed_pred = sed_choice[1]
                P = v33.copy()
                high_fn = (v33 < 0.5) & (P_FN > thresh_fn)
                gap = sed_pred - v33
                P[high_fn] = v33[high_fn] + af * gap[high_fn]
                rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                                      f"V_C fn={thresh_fn} α={af} {sed_choice[0]}"))

    # ===== V_D: Combined V_B + V_C =====
    print("\n=== V_D: V_B + V_C combined ===", flush=True)
    for tfp in [0.7]:
        for tfn in [0.5, 0.7]:
            for af in [0.30, 0.50]:
                for sed_choice in [("max", sed_max), ("avg2", sed_avg)]:
                    sed_pred = sed_choice[1]
                    P = v33.copy()
                    # FN boost
                    high_fn = (v33 < 0.5) & (P_FN > tfn)
                    P[high_fn] = v33[high_fn] + af * (sed_pred - v33)[high_fn]
                    # FP suppress
                    high_fp = (v33 > 0.5) & (P_FP > tfp)
                    P[high_fp] = sed_pred[high_fp]
                    rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                                          f"V_D tfp={tfp} tfn={tfn} α={af} {sed_choice[0]}"))

    # ===== V_F: continuous disagreement-aware interpolation =====
    print("\n=== V_F: continuous P_FP-weighted Perch-toSED interpolation ===", flush=True)
    for amp in [0.50, 1.0]:
        # When P_FP high, lean toward sed_avg; when P_FP low, keep v33
        P = v33 * (1 - amp * P_FP) + sed_avg * (amp * P_FP)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"V_F P_FP amp={amp}"))
    for amp in [0.50, 1.0]:
        # When P_FN high, lean toward sed_max
        P = v33 * (1 - amp * P_FN) + sed_max * (amp * P_FN)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"V_F P_FN amp={amp}"))

    # ===== Best variant from V_A (= exp100) for reference =====
    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]

    print("\n=== ALL RESULTS sorted by macro_d desc ===")
    print(res.sort_values("macro_d", ascending=False)[cols].head(20).to_string(index=False))

    print("\n=== Top class-A (sp_row≥0.99 + Aves Δ ≥ 0) by macro_d ===")
    safe = res[res.predicted.str.startswith("A") & (res.label != "v33 ref")]
    if len(safe) > 0:
        print(safe.sort_values("macro_d", ascending=False).head(10)[cols].to_string(index=False))

    print("\n=== Top class-A by Aves Δ ===")
    if len(safe) > 0:
        print(safe.sort_values("Aves", ascending=False).head(10)[cols].to_string(index=False))

    res.to_csv(EXP80 / "exp101_multi_teacher.csv", index=False)
    print(f"\nSaved → {EXP80}/exp101_multi_teacher.csv")


if __name__ == "__main__":
    main()
