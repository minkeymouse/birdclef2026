#!/usr/bin/env python3
"""exp43d — Analyze teacher pseudo-label entropy distribution on unlabeled SS.

Paper-motivated: iVAE confidence filter only helps where teacher is UNCERTAIN.
If teacher's pseudo is sharp (low entropy) for 90% of clips, iVAE filter is
only useful on the 10% tail. If teacher is fuzzy everywhere, iVAE becomes the
primary signal.

Output:
  - entropy histogram per clip (top-k entropy over classes)
  - how many clips have max_prob < 0.5 (currently-discarded in exp41c)
  - how many clips are 'ambiguous' (multi-modal top-3)
  - correlation with Mahalanobis confidence (exp41b)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

ROOT = Path("/data/birdclef2026")
EXP41 = ROOT / "experiments/exp41_outputs"
OUT = ROOT / "experiments/exp43d_outputs"
OUT.mkdir(exist_ok=True)


def main():
    print("Loading pseudo probs (ensemble teacher, per exp41c2)...")
    d = np.load(EXP41 / "pseudo_probs_ensemble.npz", allow_pickle=True)
    probs = d["probs"].astype(np.float32)  # (n_files, 12, 234)
    filenames = d["filenames"]
    print(f"  probs {probs.shape}")

    n_files, n_win, n_cls = probs.shape
    flat = probs.reshape(-1, n_cls)  # (n_windows, 234)

    # Per-window max prob
    max_prob = flat.max(1)
    # Per-window Shannon entropy (normalized to [0,1] by log(n_cls))
    # treat as distribution: softmax-ish? probs here are sigmoid outputs.
    # Normalize to sum=1 per window for entropy calc.
    p_norm = flat / (flat.sum(1, keepdims=True) + 1e-8)
    H = entropy(p_norm.T, base=2) / np.log2(n_cls)  # ∈ [0, 1]

    # Top-k structure: is there a clear winner or a near-tie?
    top3 = -np.sort(-flat, axis=1)[:, :3]
    top1_gap = top3[:, 0] - top3[:, 1]    # confidence gap 1st - 2nd
    top12_sum = top3[:, 0] + top3[:, 1]   # multi-modal score

    print(f"\n=== Max prob distribution ===")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  q{q:.2f}: {np.quantile(max_prob, q):.4f}")
    print(f"  mean: {max_prob.mean():.4f}")
    print(f"  frac < 0.1: {(max_prob < 0.1).mean():.3f}")
    print(f"  frac < 0.3: {(max_prob < 0.3).mean():.3f}")
    print(f"  frac < 0.5: {(max_prob < 0.5).mean():.3f}  ← exp41c discards these")
    print(f"  frac > 0.7: {(max_prob > 0.7).mean():.3f}")
    print(f"  frac > 0.9: {(max_prob > 0.9).mean():.3f}")

    print(f"\n=== Normalized Shannon entropy (top-3 structure) ===")
    for q in [0.1, 0.5, 0.9]:
        print(f"  q{q:.2f}: H={np.quantile(H, q):.3f}  gap1-2={np.quantile(top1_gap, q):.3f}")

    print(f"\n=== Ambiguity zones (for iVAE filter utility) ===")
    # Zone A: confident (max > 0.7, gap > 0.3) — iVAE not needed
    # Zone B: clear top candidate but moderate (max 0.3-0.7, gap > 0.2)
    # Zone C: ambiguous (max > 0.3, gap < 0.2) — iVAE most valuable
    # Zone D: silent / OOD (max < 0.3) — filter out entirely
    zones = {
        "A_confident (max>0.7)": (max_prob > 0.7),
        "B_moderate (0.3≤max≤0.7, gap>0.2)": ((max_prob >= 0.3) & (max_prob <= 0.7) & (top1_gap > 0.2)),
        "C_ambiguous (max>0.3, gap<0.2)": ((max_prob > 0.3) & (top1_gap < 0.2)),
        "D_silent (max<0.3)": (max_prob < 0.3),
    }
    for name, mask in zones.items():
        print(f"  {name}: {mask.mean():.3f}  ({mask.sum():d} windows)")

    # Correlation with Mahalanobis (exp41b)
    print("\n=== Correlation with Mahalanobis weight (exp41b) ===")
    mpath = EXP41 / "mahal_weights.npz"
    if mpath.exists():
        mw = np.load(mpath, allow_pickle=True)["confidence_weights"].astype(np.float32).flatten()
        assert len(mw) == flat.shape[0], f"{len(mw)} vs {flat.shape[0]}"
        from scipy.stats import spearmanr, pearsonr
        r_max, _ = spearmanr(max_prob, mw)
        r_H, _ = spearmanr(H, mw)
        r_gap, _ = spearmanr(top1_gap, mw)
        print(f"  spearman(max_prob, mahal): {r_max:+.3f}")
        print(f"  spearman(entropy,  mahal): {r_H:+.3f}")
        print(f"  spearman(gap12,    mahal): {r_gap:+.3f}")
        print(f"  → if |r|<0.3: mahal and teacher-posterior are NEARLY INDEPENDENT signals,")
        print(f"    iVAE-z adds a THIRD orthogonal signal with strongest theoretical basis")

    np.savez_compressed(OUT / "pseudo_stats.npz",
                        max_prob=max_prob, entropy=H, top1_gap=top1_gap)
    with open(OUT / "zones.json", "w") as f:
        json.dump({k: int(v.sum()) for k, v in zones.items()}, f, indent=2)
    print(f"\nSaved → {OUT}/pseudo_stats.npz")


if __name__ == "__main__":
    main()
