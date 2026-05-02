#!/usr/bin/env python3
"""exp87 — Cross-site rare-Aves selective routing on v33 base.

Apply exp84b (exp50 + 545 ext clip ft) ONLY on specific weak-Aves columns
where we have cross-site external clips. All other 230+ columns stay at
v33 production output. Preserves W_PERCH = 0.7 invariant globally.

Target columns picked from exp86 audit:
  bafcur1: v33 AUC 0.746, exp84b val_SS per-target 0.79 (+0.04)
  litnig1: v33 AUC 0.835, exp84b val_SS per-target 1.00 (saturated)
  67107  : Amphibia, v33 AUC 0.79 (eval), exp84b 0.87 (+0.08)
  74113  : Mammalia, v33 AUC 0.77 (eval), exp84b 0.77 (~)
  326272 : Amphibia, v33 AUC 0.59 (eval), exp84b 0.48 (regressed!)

Will SKIP 326272 since exp84b regressed on it. Test multiple subsets.

Routing modes:
  M1: replace v33[:, target_c] with exp84b[:, target_c] entirely
  M2: weighted blend final[:, target_c] = (1-w)*v33[:,c] + w*exp84b[:,c]
  M3: log-space (z-score) blend
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name):
    return np.load(EXP80 / name)["scores"]


def build_v33(perch_prob, exp50, perch_emb, sc_g, sp_taxon):
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    return file_max_blend(gated, sc_g, alpha=0.10)


def main():
    print("=== exp87: cross-site rare specialist selective routing ===\n", flush=True)
    sc_g, Y, primary, l2i = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp84b = get_cached("exp84b_scores_labeled.npz")
    print(f"loaded all caches", flush=True)

    v33 = build_v33(perch_prob, exp50, perch_emb, sc_g, sp_taxon)
    ev_mask = sc_g.split.values == "eval"

    # Test routing on different subsets
    subsets = {
        "S0_baseline": [],
        "S1_bafcur+litnig (Aves)": ["bafcur1", "litnig1"],
        "S2_bafcur only (Aves)": ["bafcur1"],
        "S3_67107+74113 (Amphib+Mamm)": ["67107", "74113"],
        "S4_4-class (no 326272)": ["bafcur1", "litnig1", "67107", "74113"],
        "S5_5-class (incl 326272)": ["bafcur1", "litnig1", "67107", "74113", "326272"],
        "S6_litnig+67107 only": ["litnig1", "67107"],
    }

    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    for name, target_labels in subsets.items():
        if not target_labels: continue
        target_idx = [l2i[lbl] for lbl in target_labels if lbl in l2i]
        if not target_idx: continue

        # M1: full replacement
        P = v33.copy()
        P[:, target_idx] = exp84b[:, target_idx]
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"M1 full-swap {name}"))

        # M2 blend variants
        for w in [0.3, 0.5, 0.7]:
            P = v33.copy()
            P[:, target_idx] = (1 - w) * v33[:, target_idx] + w * exp84b[:, target_idx]
            rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"M2 w={w} {name}"))

    df = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== ALL RESULTS sorted by macro_d desc ===")
    print(df[cols].sort_values("macro_d", ascending=False).to_string(index=False))
    df.to_csv(EXP80 / "exp87_selective_routing.csv", index=False)
    print(f"\nSaved → {EXP80}/exp87_selective_routing.csv")

    print("\n=== Top class-A candidates (sp_row ≥ 0.99 AND Aves Δ ≥ 0) ===")
    safe = df[df.predicted.str.startswith("A") & (df.label != "v33 ref")]
    top = safe.sort_values("Aves", ascending=False).head(8)
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
