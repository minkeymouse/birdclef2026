#!/usr/bin/env python3
"""exp82c Q3a — Swap exp50 with exp73 (exp50 + 21 external clip fine-tune)
inside the v33 pipeline. Tests whether the val_SS +0.013 from exp73 fine-tune
translates to local macro-AUC delta in the production pipeline.

If yes → external data finetune does add value beyond exp50 alone.
If no → val_SS improvement was overfit / not transferable.

Tests several blends:
  - v33 with exp73 replacing exp50 entirely
  - v33 with average(exp50, exp73) — mix the original + finetuned
  - v33 with exp50, exp59, exp73 4-way blend
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT)
from exp82_q1_4way_on_v33 import (get_sed_scores, apply_v9_gate, file_max_blend, evaluate)

EXP73_CKPT = ROOT / "experiments/_data_pipelines/exp73_outputs/best_ckpt.pt"


def get_cached(name):
    return np.load(EXP80 / name)["scores"]


def build_pipeline_n(perch_prob, *sed_probs_with_w, perch_emb, sc_g, sp_taxon, wP):
    """Generic n-way blend → V9 gate → file-max."""
    base = wP * perch_prob
    for sp, w in sed_probs_with_w:
        base = base + w * sp
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    return file_max_blend(gated, sc_g, alpha=0.10)


def main():
    print("=== exp82c Q3a: exp73 swap test in v33 pipeline ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")
    print(f"loaded perch+exp50+exp59", flush=True)

    print("Loading exp73 (exp50 + external fine-tune) scores (build cache if missing)...", flush=True)
    exp73 = get_sed_scores(sc_g, EXP73_CKPT, "hgnetv2_b0.ssld_stage2_ft_in1k", "exp73_scores_labeled.npz")
    print(f"exp73: {exp73.shape}, range [{exp73.min():.3f}, {exp73.max():.3f}]", flush=True)

    from scipy.stats import pearsonr
    print(f"\nPearson:")
    print(f"  exp50 ↔ exp73: {pearsonr(exp50.flatten(), exp73.flatten())[0]:.3f}")
    print(f"  exp59 ↔ exp73: {pearsonr(exp59.flatten(), exp73.flatten())[0]:.3f}")
    print(f"  Perch ↔ exp73: {pearsonr(perch_prob.flatten(), exp73.flatten())[0]:.3f}")

    ev_mask = sc_g.split.values == "eval"

    v33_ref = build_pipeline_n(perch_prob, (exp50, 0.3),
                                perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=0.7)
    rows = [evaluate(v33_ref, v33_ref, ev_mask, Y, sp_taxon, "v33 ref (0.7P + 0.3 exp50)")]

    print("\n=== Variants ===", flush=True)
    # Direct exp73 swap
    P = build_pipeline_n(perch_prob, (exp73, 0.3),
                          perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=0.7)
    rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, "S1: 0.7P + 0.3 exp73 (full swap)"))

    # exp50+exp73 mix (smooth ensemble)
    P = build_pipeline_n(perch_prob, (exp50, 0.15), (exp73, 0.15),
                          perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=0.7)
    rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, "S2: 0.7P + 0.15 exp50 + 0.15 exp73"))

    P = build_pipeline_n(perch_prob, (exp50, 0.2), (exp73, 0.1),
                          perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=0.7)
    rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, "S3: 0.7P + 0.2 exp50 + 0.1 exp73"))

    # 4-way: original 50, conv-tiny 59, finetune 73
    P = build_pipeline_n(perch_prob, (exp50, 0.15), (exp59, 0.15), (exp73, 0.10),
                          perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=0.6)
    rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, "S4: 0.6P + 0.15 e50 + 0.15 e59 + 0.10 e73"))

    P = build_pipeline_n(perch_prob, (exp50, 0.1), (exp59, 0.15), (exp73, 0.15),
                          perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=0.6)
    rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, "S5: 0.6P + 0.10 e50 + 0.15 e59 + 0.15 e73"))

    # Heavier exp73
    P = build_pipeline_n(perch_prob, (exp73, 0.4),
                          perch_emb=perch_emb, sc_g=sc_g, sp_taxon=sp_taxon, wP=0.6)
    rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, "S6: 0.6P + 0.4 exp73"))

    df = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== Results (sorted by macro_d desc) ===")
    print(df[cols].sort_values("macro_d", ascending=False).to_string(index=False))
    df.to_csv(EXP80 / "exp82c_q3a_results.csv", index=False)


if __name__ == "__main__":
    main()
