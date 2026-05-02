#!/usr/bin/env python3
"""exp157 — exp84b non-Aves freeze blend audit.

v47 confirmed exp136b non-Aves freeze yielded +0.002 LB vs v46 uniform.
Apply the same recipe to exp84b (the cleaner teacher: val_SS 0.861 vs
exp136b 0.907 — less site-fingerprint absorption per CLAUDE.md).

Blend:
  Aves columns:     v33 unchanged
  non-Aves columns: v33 + W × exp84b_scores

Then V9 gate + file-max α=0.10 (same as v33).

Sweep W ∈ {0.05, 0.10, 0.15, 0.20}. Compare to v33 reference.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS, TAXA)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def nonaves_blend(base, teacher, sp_taxon, w):
    """Add w × teacher only to non-Aves columns; Aves columns unchanged."""
    aves_mask = (sp_taxon == "Aves")
    out = base.copy()
    out[:, ~aves_mask] = base[:, ~aves_mask] + w * teacher[:, ~aves_mask]
    return out


def main():
    print("=== exp157: exp84b non-Aves freeze blend ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]
    exp84b = np.load(EXP80 / "exp84b_scores_labeled.npz")["scores"]

    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    # Sweep non-Aves freeze
    for w in [0.05, 0.10, 0.15, 0.20]:
        blended = nonaves_blend(base, exp84b, sp_taxon, w)
        gated_b = apply_v9_gate(blended, perch_emb, sp_taxon, offset=0.1)
        v33_b = file_max_blend(gated_b, sc_g, alpha=0.10)
        rows.append(evaluate(v33_b, v33, ev_mask, Y, sp_taxon,
                              f"non-Aves freeze W={w}"))

    # Also test uniform (as reference for what v48 did at w=0.05)
    for w in [0.05, 0.10]:
        blended = base + w * exp84b
        gated_b = apply_v9_gate(blended, perch_emb, sp_taxon, offset=0.1)
        v33_b = file_max_blend(gated_b, sc_g, alpha=0.10)
        rows.append(evaluate(v33_b, v33, ev_mask, Y, sp_taxon,
                              f"uniform W={w} (v48-style)"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal",
            "Amphib", "Reptil", "predicted"]
    print("\n=== Audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
