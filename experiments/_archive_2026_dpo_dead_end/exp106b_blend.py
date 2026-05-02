#!/usr/bin/env python3
"""exp106b — Blend P_NEW3 hybrid predictions with v33."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name):
    d = np.load(EXP80 / name)
    if "scores" in d.files: return d["scores"]
    if "predictions" in d.files: return d["predictions"]
    return d[d.files[0]]


def main():
    print("=== exp106b: P_NEW3 hybrid + v33 blend test ===\n", flush=True)
    sc_g, Y, primary, l2i = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()
    perch_prob_ss = load_perch_scores_labeled()

    p_new3 = get_cached("p_new2_hybrid_predictions.npz")
    print(f"  P_NEW3 hybrid: {p_new3.shape}")

    p_new1 = get_cached("p_new_predictions.npz")
    print(f"  P_NEW (random): {p_new1.shape}")

    exp50 = get_cached("exp50_scores_labeled.npz")
    base = 0.7 * perch_prob_ss + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb_ss, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    # Both alones
    rows.append(evaluate(p_new3, v33, ev_mask, Y, sp_taxon, "P_NEW3 alone"))
    rows.append(evaluate(p_new1, v33, ev_mask, Y, sp_taxon, "P_NEW alone"))

    # Additive blends with v33 (hybrid)
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        P = (1 - w) * v33 + w * p_new3
        P = np.clip(P, 0, 1).astype(np.float32)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"v33 + P_NEW3 w={w}"))

    # Replace exp50 with P_NEW3 in v33 base
    for w_n in [0.20, 0.30]:
        base_new = 0.7 * perch_prob_ss + (0.3 - w_n) * exp50 + w_n * p_new3
        gated_new = apply_v9_gate(base_new, perch_emb_ss, sp_taxon, offset=0.1)
        P = file_max_blend(gated_new, sc_g, alpha=0.10)
        P = np.clip(P, 0, 1).astype(np.float32)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"v33-style 0.7P+{0.3-w_n:.2f}exp50+{w_n}P_NEW3"))

    # 3-way blends (replicate exp104 best config)
    for w_n in [0.15, 0.20, 0.25]:
        P = (1 - 2*w_n) * v33 + w_n * exp50 + w_n * p_new3
        P = np.clip(P, 0, 1).astype(np.float32)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"3-way v33+{w_n}exp50+{w_n}P_NEW3"))

    # Pure base + P_NEW3 (no exp50)
    for w_n in [0.20, 0.30]:
        base_new = (1 - w_n) * perch_prob_ss + w_n * p_new3
        gated_new = apply_v9_gate(base_new, perch_emb_ss, sp_taxon, offset=0.1)
        P = file_max_blend(gated_new, sc_g, alpha=0.10)
        P = np.clip(P, 0, 1).astype(np.float32)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"v33-style {1-w_n}P+{w_n}P_NEW3 (no exp50)"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n  Sorted by macro_d desc:")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
