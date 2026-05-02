#!/usr/bin/env python3
"""exp85 — Final local audit comparing all SED variants on v33 pipeline.

Tests Q1, Q2, Q3a, Q4 hypotheses comprehensively. For each candidate
config, reports (macro_d, sp_row, per-taxon Δ, predicted_LB_class).

SEDs available:
  exp50 — production HGNet B0 + 2025 BG (BCE)        ← v33 component
  exp59 — ConvNeXt-tiny + 2025 BG (BCE, val_SS 0.86)  ← Q1 candidate
  exp73 — exp50 + 21 external clip fine-tune          ← Q3a candidate
  exp83 — HGNet B0 + 2025 BG + FOCAL loss             ← Q2 candidate
  exp84b — exp50 + 545 external clip fine-tune        ← Q4 candidate

Combinations tested:
  v33 ref (Q0 baseline)
  Q1 best: 4-way with exp59
  Q2 alone: replace exp50 with exp83
  Q2 blend: include exp83 in 4-way
  Q3a: include exp73 in 4-way
  Q4 alone: replace exp50 with exp84b
  Q4 blend: include exp84b in 4-way
  All-in: 5-way with everything (Perch + exp50 + exp59 + exp83 + exp84b)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT)
from exp82_q1_4way_on_v33 import (get_sed_scores, apply_v9_gate, file_max_blend, evaluate)


def get_cached(name):
    return np.load(EXP80 / name)["scores"]


def build_pipeline_n(perch_prob, *sed_probs_with_w, perch_emb, sc_g, sp_taxon, wP):
    base = wP * perch_prob
    for sp, w in sed_probs_with_w:
        base = base + w * sp
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    return file_max_blend(gated, sc_g, alpha=0.10)


def main():
    print("=== exp85: comprehensive final local audit ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")
    exp73 = get_cached("exp73_scores_labeled.npz")

    EXP83_CKPT = ROOT / "experiments/_data_pipelines/exp83_q2_outputs/best_ckpt.pt"
    EXP84B_CKPT = ROOT / "experiments/_data_pipelines/exp84b_q4_outputs/best_ckpt.pt"

    print("Loading exp83 (focal-loss) scores...", flush=True)
    exp83 = get_sed_scores(sc_g, EXP83_CKPT, "hgnetv2_b0.ssld_stage2_ft_in1k", "exp83_scores_labeled.npz")
    print("Loading exp84b (large external) scores...", flush=True)
    exp84b = get_sed_scores(sc_g, EXP84B_CKPT, "hgnetv2_b0.ssld_stage2_ft_in1k", "exp84b_scores_labeled.npz")
    print(f"all loaded.\n", flush=True)

    # Independence diagnostic
    from scipy.stats import pearsonr
    print("=== Pearson correlation (lower = more independent) ===")
    pairs = [("Perch", perch_prob), ("exp50", exp50), ("exp59", exp59),
             ("exp73", exp73), ("exp83 focal", exp83), ("exp84b ext", exp84b)]
    for i, (n1, p1) in enumerate(pairs):
        for n2, p2 in pairs[i+1:]:
            print(f"  {n1:<14} ↔ {n2:<14}: {pearsonr(p1.flatten(), p2.flatten())[0]:+.3f}")

    ev_mask = sc_g.split.values == "eval"

    # === v33 ref ===
    v33_ref = build_pipeline_n(perch_prob, (exp50, 0.3),
                                perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=0.7)
    rows = [evaluate(v33_ref, v33_ref, ev_mask, Y, sp_taxon, "v33 ref (0.7P + 0.3 exp50)")]

    print("\n=== Single-SED swaps ===", flush=True)
    for label, sed in [("(swap to exp59) 0.7P + 0.3 exp59", exp59),
                        ("(swap to exp73) 0.7P + 0.3 exp73", exp73),
                        ("(swap to exp83) 0.7P + 0.3 exp83", exp83),
                        ("(swap to exp84b) 0.7P + 0.3 exp84b", exp84b)]:
        P = build_pipeline_n(perch_prob, (sed, 0.3),
                              perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=0.7)
        rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, label))

    print("=== 4-way blends (each SED 0.10) ===", flush=True)
    # Q1 best already known: 4-way with 0.6P + 0.2 exp50 + 0.2 exp59. Test variants.
    configs_4way = [
        ("Q1 best: 0.6P + 0.2 e50 + 0.2 e59",
         [(exp50, 0.2), (exp59, 0.2)], 0.6),
        ("Q3a best: 0.6P + 0.15 e50 + 0.15 e59 + 0.10 e73",
         [(exp50, 0.15), (exp59, 0.15), (exp73, 0.10)], 0.6),
        ("Q2-blend: 0.6P + 0.15 e50 + 0.15 e83",
         [(exp50, 0.15), (exp83, 0.15)], 0.6),
        ("Q2-blend2: 0.6P + 0.2 e50 + 0.2 e83",
         [(exp50, 0.2), (exp83, 0.2)], 0.6),
        ("Q4-blend: 0.6P + 0.2 e50 + 0.2 e84b",
         [(exp50, 0.2), (exp84b, 0.2)], 0.6),
        ("Q4-blend2: 0.6P + 0.15 e50 + 0.25 e84b",
         [(exp50, 0.15), (exp84b, 0.25)], 0.6),
    ]
    for label, seds, wP in configs_4way:
        P = build_pipeline_n(perch_prob, *seds, perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=wP)
        rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, label))

    print("=== 5-way / 6-way (everything) ===", flush=True)
    configs_big = [
        ("5-way: 0.5P + e50/e59/e73/e84b each 0.125",
         [(exp50, 0.125), (exp59, 0.125), (exp73, 0.125), (exp84b, 0.125)], 0.5),
        ("5-way: 0.5P + e50/e59/e83/e84b each 0.125",
         [(exp50, 0.125), (exp59, 0.125), (exp83, 0.125), (exp84b, 0.125)], 0.5),
        ("6-way: 0.5P + all 5 SEDs each 0.10",
         [(exp50, 0.10), (exp59, 0.10), (exp73, 0.10), (exp83, 0.10), (exp84b, 0.10)], 0.5),
        ("6-way conservative: 0.6P + each SED 0.08",
         [(exp50, 0.08), (exp59, 0.08), (exp73, 0.08), (exp83, 0.08), (exp84b, 0.08)], 0.6),
    ]
    for label, seds, wP in configs_big:
        P = build_pipeline_n(perch_prob, *seds, perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=wP)
        rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, label))

    print("=== Q4 substitution variants (use exp84b instead of exp50 in 4-way) ===", flush=True)
    configs_sub = [
        ("0.6P + 0.2 e84b + 0.2 e59 (replace e50→e84b in Q1)",
         [(exp84b, 0.2), (exp59, 0.2)], 0.6),
        ("0.6P + 0.15 e84b + 0.15 e59 + 0.10 e73",
         [(exp84b, 0.15), (exp59, 0.15), (exp73, 0.10)], 0.6),
        ("0.7P + 0.15 e50 + 0.15 e84b",
         [(exp50, 0.15), (exp84b, 0.15)], 0.7),
        ("0.7P + 0.10 e50 + 0.20 e84b",
         [(exp50, 0.10), (exp84b, 0.20)], 0.7),
    ]
    for label, seds, wP in configs_sub:
        P = build_pipeline_n(perch_prob, *seds, perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=wP)
        rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, label))

    df = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== ALL RESULTS sorted by macro_d desc ===")
    print(df[cols].sort_values("macro_d", ascending=False).to_string(index=False))

    df.to_csv(EXP80 / "exp85_final_local_audit.csv", index=False)
    print(f"\nSaved → {EXP80}/exp85_final_local_audit.csv")

    # Top class-A candidates by Aves Δ (most LB-predictive)
    print("\n=== Top-5 class-A candidates ranked by Aves Δ (LB transfer signal) ===")
    safe = df[df.predicted.str.startswith("A") & (df.label != "v33 ref (0.7P + 0.3 exp50)")]
    if len(safe) > 0:
        top = safe.sort_values("Aves", ascending=False).head(5)
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
