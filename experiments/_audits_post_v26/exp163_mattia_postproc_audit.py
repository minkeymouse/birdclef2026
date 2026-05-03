#!/usr/bin/env python3
"""exp163 — Mattia post-processing components audit (no Tucker dependency).

Tucker-line dead (v55=≤0.932 linear, v56=0.931/0.932 rank). Pivot to
OTHER Mattia 0.943 components that don't require Tucker:

  1) Rank-aware file_max^0.4 scaling
       out_window = score * file_max^0.4
     vs our linear file_max α=0.10:
       out = (1-α) * sub + α * fmax

  2) Adaptive delta smoothing
       alpha = base_alpha * (1 - conf)
       new_t = (1-alpha) * old_t + alpha * (old_{t-1} + old_{t+1}) / 2
     vs our fixed Gauss σ=0.5

  3) Isotonic calibration + per-class F1 threshold
     Per-class fit isotonic regression on OOF, optimize threshold.

Apply to v33 baseline. Sweep parameters. Compare macro Δ.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate

N_WINDOWS = 12


def rank_aware_scale(probs, sc_g, power=0.4):
    """out = score * file_max^power per-class. Uses sc_g groupby (labeled SS
    has variable rows per file)."""
    out = probs.copy()
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = probs[idx]
        fmax = sub.max(axis=0, keepdims=True)  # (1, n_cls)
        scale = np.power(np.maximum(fmax, 1e-9), power)
        out[idx] = sub * scale
    return out.astype(np.float32)


def adaptive_delta_smooth(probs, sc_g, base_alpha=0.20):
    """Adaptive delta smoothing per file using groupby."""
    out = probs.copy()
    for fname, idx in sc_g.groupby("filename").indices.items():
        idx_sorted = idx[np.argsort(sc_g.iloc[idx]["end_sec"].values)]
        sub = probs[idx_sorted]
        n = len(sub)
        if n < 2:
            continue
        smoothed = sub.copy()
        for t in range(n):
            prev = sub[max(0, t-1)]
            nxt = sub[min(n-1, t+1)]
            conf = sub[t].max()  # scalar
            alpha = base_alpha * (1.0 - conf)
            neighbor_avg = 0.5 * (prev + nxt)
            smoothed[t] = (1.0 - alpha) * sub[t] + alpha * neighbor_avg
        out[idx_sorted] = smoothed
    return out.astype(np.float32)


def isotonic_calibrate(probs, Y, ev_mask):
    """Per-class isotonic regression on TRAIN split, applied to all rows.
    Returns calibrated probs (same shape as probs)."""
    out = probs.copy()
    train_mask = ~ev_mask
    n_calib = 0
    for c in range(N_CLS):
        if Y[train_mask, c].sum() < 3:
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(probs[train_mask, c], Y[train_mask, c])
        out[:, c] = ir.predict(probs[:, c])
        n_calib += 1
    return out.astype(np.float32), n_calib


def main():
    print("=== exp163: Mattia post-proc components audit ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]

    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    # (1) Rank-aware file_max^p (replace linear file_max)
    # Apply on `gated` instead of file_max_blend, then compare
    for p in [0.2, 0.4, 0.6]:
        v_ra = rank_aware_scale(gated, sc_g, power=p)
        rows.append(evaluate(v_ra, v33, ev_mask, Y, sp_taxon,
                              f"rank-aware file_max^{p:.1f} (replace linear)"))

    # (2) Rank-aware ADDED on top of v33 (file-max linear stays)
    for p in [0.2, 0.4, 0.6]:
        v_ra2 = rank_aware_scale(v33, sc_g, power=p)
        rows.append(evaluate(v_ra2, v33, ev_mask, Y, sp_taxon,
                              f"rank-aware file_max^{p:.1f} on v33"))

    # (3) Adaptive delta smoothing on v33
    for ba in [0.10, 0.20, 0.30]:
        v_ads = adaptive_delta_smooth(v33, sc_g, base_alpha=ba)
        rows.append(evaluate(v_ads, v33, ev_mask, Y, sp_taxon,
                              f"adaptive delta smooth base_alpha={ba}"))

    # (4) Isotonic calibration on v33
    v_iso, n_cal = isotonic_calibrate(v33, Y, ev_mask)
    rows.append(evaluate(v_iso, v33, ev_mask, Y, sp_taxon,
                          f"isotonic calib (n_cal={n_cal})"))

    # (5) Combined: rank-aware + adaptive smoothing
    v_ra_best = rank_aware_scale(gated, sc_g, power=0.4)
    v_combo = adaptive_delta_smooth(v_ra_best, sc_g, base_alpha=0.20)
    rows.append(evaluate(v_combo, v33, ev_mask, Y, sp_taxon,
                          "rank-aware^0.4 + adaptive smooth ba=0.20"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal",
            "Amphib", "Reptil", "predicted"]
    print("\n=== Audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
