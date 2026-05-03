#!/usr/bin/env python3
"""exp166 — 3-stream linear blend audit for slot 5 candidate.

Given exp165 finding (linear v33 + Tucker W=0.30 macro_d +0.117 sp_row 0.991),
test if adding a third teacher residually pushes higher.

Candidates (all linear):
  v33 + Tucker W=0.30                            (baseline = exp165 best)
  v33 + Tucker W=0.30 + exp59  W=0.05            (ConvNeXt SED diversity)
  v33 + Tucker W=0.30 + exp84b W=0.05            (external iNat positives)
  v33 + Tucker W=0.30 + exp136b W=0.05           (v3 pseudo)
  v33 + Tucker W=0.30 + AudioMAE M5 W=0.05       (probe path)
  v33 + Tucker W=0.30 + (AVG of exp59 + exp84b) W=0.05

Pick the highest macro_d with sp_row > 0.99 + Aves Δ ≥ 0 + per-taxon
positive for slot 5 LB submission.
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

EPS = 1e-6
W_TUCKER = 0.30


def linear3(streamA, streamB, streamC, w_b=0.30, w_c=0.05):
    """A * (1 - w_b - w_c) + B * w_b + C * w_c"""
    pa = np.clip(streamA, EPS, 1.0 - EPS)
    pb = np.clip(streamB, EPS, 1.0 - EPS)
    pc = np.clip(streamC, EPS, 1.0 - EPS)
    out = pa * (1.0 - w_b - w_c) + pb * w_b + pc * w_c
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def main():
    print("=== exp166: 3-stream linear blend audit ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]
    tucker = np.load(EXP80 / "tucker_sed_5fold_labeled.npz")["scores"]

    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)
    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    # Baseline: v33 + Tucker linear W=0.30
    v58_pred = ((1.0 - W_TUCKER) * v33 + W_TUCKER * tucker).astype(np.float32)
    rows.append(evaluate(v58_pred, v33, ev_mask, Y, sp_taxon,
                          "v58: v33 + Tucker W=0.30 (baseline)"))

    # Optional 3rd streams to try
    candidates = []
    for name in ["exp59_scores_labeled", "exp84b_scores_labeled",
                  "exp136b_scores_labeled", "exp121_scores_labeled",
                  "exp123_scores_labeled"]:
        path = EXP80 / f"{name}.npz"
        if path.exists():
            scores = np.load(path)["scores"]
            candidates.append((name.replace("_scores_labeled", ""), scores))

    print(f"available 3rd streams: {[n for n,_ in candidates]}\n")

    # Sweep each as 3rd stream at W=0.05 and W=0.10
    for name, scores in candidates:
        from scipy.stats import pearsonr
        rho_v33 = pearsonr(v33.flatten(), scores.flatten())[0]
        rho_tucker = pearsonr(tucker.flatten(), scores.flatten())[0]
        for w_c in [0.05, 0.10]:
            pred = linear3(v33, tucker, scores, w_b=W_TUCKER, w_c=w_c)
            rows.append(evaluate(
                pred, v33, ev_mask, Y, sp_taxon,
                f"+ {name} W={w_c}  ρ(v33)={rho_v33:.2f} ρ(T)={rho_tucker:.2f}"
            ))

    # Also AVG of all available residual SEDs
    if len(candidates) >= 2:
        avg = np.mean(np.stack([s for _, s in candidates], axis=0), axis=0).astype(np.float32)
        for w_c in [0.05, 0.10]:
            pred = linear3(v33, tucker, avg, w_b=W_TUCKER, w_c=w_c)
            rows.append(evaluate(pred, v33, ev_mask, Y, sp_taxon,
                                  f"+ AVG-of-{len(candidates)}SEDs W={w_c}"))

    # And try W_TUCKER sweep at 0.35 / 0.40 (no 3rd stream)
    for w in [0.35, 0.40]:
        p = ((1.0 - w) * v33 + w * tucker).astype(np.float32)
        rows.append(evaluate(p, v33, ev_mask, Y, sp_taxon,
                              f"v33 + Tucker W={w} (dose sweep)"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal",
            "Amphib", "Reptil", "predicted"]
    print("\n=== Audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
