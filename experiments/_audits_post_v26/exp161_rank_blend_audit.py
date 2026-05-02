#!/usr/bin/env python3
"""exp161 — Rank-percentile blend audit (v56 prep).

Compare linear blend (current v33) vs rank-percentile blend (Mattia 0.943
recipe) on labeled SS.

Rank-blend recipe (Mattia line 2240):
  xa = rank_pct(streamA)            # ProtoSSM/Perch
  xb = rank_pct(streamB)            # SED ensemble
  pred = xa * (1 - SED_W) + xb * SED_W

Optionally with V9 gate before rank, file_max after.

Prereq: tucker_sed_5fold_labeled.npz (from exp160) + exp50_scores_labeled.npz.
Run AFTER exp160 produces the Tucker scores cache.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def rank_pct(arr):
    """Per-column (per-class) rank percentile, shape (N, C). NaN-safe."""
    return pd.DataFrame(arr).rank(axis=0, pct=True).to_numpy(dtype=np.float32)


def main():
    print("=== exp161: Rank-percentile blend audit (v56 prep) ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]
    tucker_path = EXP80 / "tucker_sed_5fold_labeled.npz"
    if not tucker_path.exists():
        print(f"ERROR: {tucker_path} missing. Run exp160 first.")
        sys.exit(1)
    tucker = np.load(tucker_path)["scores"]

    # === Reference: current linear v33 ===
    base_linear = 0.7 * perch_prob + 0.3 * exp50
    g_linear = apply_v9_gate(base_linear, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(g_linear, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 linear ref")]

    # Variant A: linear with Tucker swap (v55 candidate without rank)
    base_lin_T = 0.7 * perch_prob + 0.3 * tucker
    g = apply_v9_gate(base_lin_T, perch_emb, sp_taxon, offset=0.1)
    v33_T_lin = file_max_blend(g, sc_g, alpha=0.10)
    rows.append(evaluate(v33_T_lin, v33, ev_mask, Y, sp_taxon,
                          "v55 linear: 0.7P + 0.3 Tucker (swap, linear)"))

    # === Variant B: rank-pct blend, exp50 SED stream ===
    # streamA = (P+ProtoSSM proxied by perch_prob alone for now)
    # streamB = exp50
    xa = rank_pct(perch_prob)
    xb = rank_pct(exp50)
    for sed_w in [0.20, 0.30, 0.40]:
        pred_rank = xa * (1.0 - sed_w) + xb * sed_w
        # Apply same V9 gate + file-max in rank space (note: gate operates on probs)
        # For audit, skip V9 gate after rank (rank is not a prob; gate would distort)
        # Just do: pred_rank → file_max coherence on rank space
        v_rank = file_max_blend(pred_rank, sc_g, alpha=0.10)
        rows.append(evaluate(v_rank, v33, ev_mask, Y, sp_taxon,
                              f"rank-blend exp50 SED_W={sed_w}"))

    # === Variant C: rank-pct blend, Tucker SED stream ===
    xb_tucker = rank_pct(tucker)
    for sed_w in [0.20, 0.30, 0.40]:
        pred_rank = xa * (1.0 - sed_w) + xb_tucker * sed_w
        v_rank = file_max_blend(pred_rank, sc_g, alpha=0.10)
        rows.append(evaluate(v_rank, v33, ev_mask, Y, sp_taxon,
                              f"rank-blend Tucker SED_W={sed_w}"))

    # === Variant D: rank with V9 gate applied first (gate prob, then rank) ===
    g_perch = apply_v9_gate(perch_prob, perch_emb, sp_taxon, offset=0.1)
    g_tucker = apply_v9_gate(tucker, perch_emb, sp_taxon, offset=0.1)
    xa_g = rank_pct(g_perch)
    xb_g = rank_pct(g_tucker)
    for sed_w in [0.30]:
        pred_rank = xa_g * (1.0 - sed_w) + xb_g * sed_w
        v_rank = file_max_blend(pred_rank, sc_g, alpha=0.10)
        rows.append(evaluate(v_rank, v33, ev_mask, Y, sp_taxon,
                              f"V9-then-rank-blend Tucker SED_W={sed_w}"))

    # === Variant E: hybrid — linear blend probs to score, then rank only at final ===
    # This is the SAFEST variant: keep current pipeline, just rank-transform final probs
    rank_v33 = rank_pct(v33)
    rows.append(evaluate(rank_v33, v33, ev_mask, Y, sp_taxon,
                          "final-rank-only of v33"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal",
            "Amphib", "Reptil", "predicted"]
    print("\n=== Audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
