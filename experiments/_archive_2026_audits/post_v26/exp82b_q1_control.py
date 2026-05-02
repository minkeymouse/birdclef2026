#!/usr/bin/env python3
"""exp82b — Q1 CONTROL: distinguish architecture diversity from simple
weight reallocation.

Hypothesis check: exp50 (HGNet) and exp59 (ConvNeXt-tiny) have prediction
Pearson 0.981 — extremely correlated. Yet 4-way blend (0.6P + 0.2exp50 +
0.2exp59) gave macro +0.009, Aves +0.017 vs v33 (0.7P + 0.3exp50).

Is this architecture diversity, or simply increasing SED weight from 0.3
to 0.4? Test:
  A: 0.6P + 0.2 exp50 + 0.2 exp59      (4-way, w_sed_total=0.4)
  B: 0.6P + 0.4 exp50                  (3-way, w_sed_total=0.4, exp50 only)  ← control
  C: 0.6P + 0.4 exp59                  (3-way, w_sed_total=0.4, exp59 only)  ← control

If A ≈ B at higher SED weight: just weight effect, no real diversity benefit.
If A > max(B, C) clearly: real diversity contribution.

Also: lower-weight controls
  D: 0.7P + 0.3 exp59      (single-SED swap)  for direct compare
  E: 0.7P + 0.3 mean(exp50, exp59)  (mathematical avg, same weight)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, MW, TAXA)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman

# Reuse exp82's helpers via import
sys.path.insert(0, str(Path(__file__).parent))
from exp82_q1_4way_on_v33 import (apply_v9_gate, file_max_blend, evaluate)


def get_cached(name):
    return np.load(EXP80 / name)["scores"]


def build_pipeline(perch_prob, sed_prob, perch_emb, sc_g, sp_taxon, wP, w_sed):
    base = wP * perch_prob + w_sed * sed_prob
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    return file_max_blend(gated, sc_g, alpha=0.10)


def build_pipeline_4way(perch_prob, exp50, exp59, perch_emb, sc_g, sp_taxon, wP, w50, w59):
    base = wP * perch_prob + w50 * exp50 + w59 * exp59
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    return file_max_blend(gated, sc_g, alpha=0.10)


def main():
    print("=== exp82b Q1 CONTROL: architecture diversity vs weight reallocation ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")
    print(f"loaded all caches", flush=True)

    ev_mask = sc_g.split.values == "eval"

    # v33 reference
    v33_ref = build_pipeline(perch_prob, exp50, perch_emb, sc_g, sp_taxon, 0.7, 0.3)

    rows = [evaluate(v33_ref, v33_ref, ev_mask, Y, sp_taxon, "v33 ref (0.7P + 0.3 exp50)")]

    print("\n=== Weight-control experiments ===")
    # Same SED total weight 0.4, single SED:
    P_b = build_pipeline(perch_prob, exp50, perch_emb, sc_g, sp_taxon, 0.6, 0.4)
    rows.append(evaluate(P_b, v33_ref, ev_mask, Y, sp_taxon, "B: 0.6P + 0.4 exp50 (HGNet only, w=0.4)"))

    P_c = build_pipeline(perch_prob, exp59, perch_emb, sc_g, sp_taxon, 0.6, 0.4)
    rows.append(evaluate(P_c, v33_ref, ev_mask, Y, sp_taxon, "C: 0.6P + 0.4 exp59 (ConvNeXt only, w=0.4)"))

    # 4-way (the result we want to verify)
    P_a = build_pipeline_4way(perch_prob, exp50, exp59, perch_emb, sc_g, sp_taxon, 0.6, 0.2, 0.2)
    rows.append(evaluate(P_a, v33_ref, ev_mask, Y, sp_taxon, "A: 0.6P + 0.2 exp50 + 0.2 exp59 (4-way)"))

    # Mathematical equivalent: average exp50 & exp59 then weight 0.4
    sed_avg = (exp50 + exp59) / 2.0
    P_e = build_pipeline(perch_prob, sed_avg, perch_emb, sc_g, sp_taxon, 0.6, 0.4)
    rows.append(evaluate(P_e, v33_ref, ev_mask, Y, sp_taxon, "E: 0.6P + 0.4 mean(exp50,exp59) (same as A)"))

    # Single SED swap at w=0.3 (Perch unchanged)
    P_d = build_pipeline(perch_prob, exp59, perch_emb, sc_g, sp_taxon, 0.7, 0.3)
    rows.append(evaluate(P_d, v33_ref, ev_mask, Y, sp_taxon, "D: 0.7P + 0.3 exp59 (single-SED swap)"))

    # Heavier exp50 weight (control for "diversity vs more SED")
    P_f = build_pipeline(perch_prob, exp50, perch_emb, sc_g, sp_taxon, 0.5, 0.5)
    rows.append(evaluate(P_f, v33_ref, ev_mask, Y, sp_taxon, "F: 0.5P + 0.5 exp50 (HGNet w=0.5)"))

    # Mid weight 4-way
    P_g = build_pipeline_4way(perch_prob, exp50, exp59, perch_emb, sc_g, sp_taxon, 0.5, 0.25, 0.25)
    rows.append(evaluate(P_g, v33_ref, ev_mask, Y, sp_taxon, "G: 0.5P + 0.25 + 0.25 (4-way w=0.5)"))

    df = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== Results (sorted by macro_d desc) ===")
    print(df[cols].sort_values("macro_d", ascending=False).to_string(index=False))
    df.to_csv(EXP80 / "exp82b_q1_control.csv", index=False)
    print(f"\nSaved → {EXP80}/exp82b_q1_control.csv")

    # Decisive comparisons
    print("\n=== INTERPRETATION ===")
    A = df[df.label.str.startswith("A:")].iloc[0]
    B = df[df.label.str.startswith("B:")].iloc[0]
    C = df[df.label.str.startswith("C:")].iloc[0]
    E = df[df.label.str.startswith("E:")].iloc[0]
    print(f"\n  A (4-way 0.6P + 0.2 + 0.2):     macro_d={A.macro_d:+.4f}  Aves={A.Aves:+.4f}")
    print(f"  B (3-way 0.6P + 0.4 exp50):     macro_d={B.macro_d:+.4f}  Aves={B.Aves:+.4f}")
    print(f"  C (3-way 0.6P + 0.4 exp59):     macro_d={C.macro_d:+.4f}  Aves={C.Aves:+.4f}")
    print(f"  E (0.6P + 0.4 mean(50,59)):     macro_d={E.macro_d:+.4f}  Aves={E.Aves:+.4f}")
    print(f"\n  Diversity benefit (A - max(B,C)):  macro_d {A.macro_d - max(B.macro_d, C.macro_d):+.4f}")
    print(f"  A vs E (math equiv):                macro_d {A.macro_d - E.macro_d:+.4f}")
    if abs(A.macro_d - max(B.macro_d, C.macro_d)) < 0.002:
        print("\n  → Architectural diversity contribution is NEGLIGIBLE (within ±0.002 noise).")
        print("    The +0.009 vs v33 is a WEIGHT REALLOCATION effect, not diversity.")
    else:
        print(f"\n  → 4-way diversity contributes {A.macro_d - max(B.macro_d, C.macro_d):+.4f} beyond pure weight effect.")


if __name__ == "__main__":
    main()
