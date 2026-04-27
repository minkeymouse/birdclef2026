#!/usr/bin/env python3
"""exp41c — Prepare pseudo-label training set.

Applies 2025-winner pseudo filtering:
  1. chunk max_prob < 0.5  → discard chunk
  2. class_prob < 0.1 in retained chunks → set to 0
  3. power transform p^0.5 (noise reduction, 2025 1st prize)
  4. Mahalanobis confidence weight per chunk (sample weight = pseudo_sum × mahal_w)

Output:
  - pseudo_ensemble_df.parquet: (filename, start_sec, end_sec, soft_labels, weight)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
EXP41 = ROOT / "experiments/exp41_outputs"

# Hyperparameters
MAX_PROB_THRESH = 0.5  # chunk discard threshold (2nd prize)
CLASS_PROB_FLOOR = 0.1  # class-level zero
POWER = 0.5            # noise reduction


def main():
    d = np.load(EXP41 / "pseudo_probs_ensemble.npz", allow_pickle=True)
    probs = d["probs"].astype(np.float32)  # (10592, 12, 234)
    filenames = d["filenames"]
    n_files, n_win, n_cls = probs.shape
    print(f"Pseudo probs: {probs.shape}")

    # Mahalanobis weights (if available)
    mahal_w = None
    if (EXP41 / "mahal_weights.npz").exists():
        m = np.load(EXP41 / "mahal_weights.npz", allow_pickle=True)
        mahal_w = m["confidence_weights"].astype(np.float32)  # (10592, 12)
        print(f"Mahalanobis weights: {mahal_w.shape}  mean={mahal_w.mean():.3f}")
    else:
        print("⚠️  Mahalanobis not available; using uniform weights")
        mahal_w = np.ones((n_files, n_win), dtype=np.float32)

    # Step 1: discard chunks where max prob < threshold
    max_w = probs.max(-1)  # (n, 12)
    keep = max_w >= MAX_PROB_THRESH
    print(f"\nChunk-level filter (max_prob >= {MAX_PROB_THRESH}):")
    print(f"  kept: {keep.sum()}/{keep.size} ({keep.mean()*100:.1f}%)")

    # Step 2: zero out classes below floor
    p_filt = probs.copy()
    p_filt[probs < CLASS_PROB_FLOOR] = 0
    print(f"Class floor (<{CLASS_PROB_FLOOR} → 0):")
    print(f"  nonzero per chunk median: {(p_filt > 0).sum(-1).mean():.1f}")

    # Step 3: power transform (noise reduction)
    p_pow = p_filt ** POWER
    print(f"Power transform p^{POWER}:")
    print(f"  max value: {p_pow.max():.3f}  median nonzero: {np.median(p_pow[p_pow > 0]):.3f}")

    # Build long-form DF: one row per (file, window) where keep=True
    rows = []
    keep_idx = np.argwhere(keep)  # (N_keep, 2) of (file_idx, win_idx)
    for fi, wi in keep_idx:
        lbl_vec = p_pow[fi, wi]
        w_sum = float(lbl_vec.sum())
        if w_sum < 0.05:
            continue  # after power transform, too weak
        weight = float(mahal_w[fi, wi] * w_sum)
        rows.append({
            "filename": str(filenames[fi]),
            "win_idx": int(wi),
            "end_sec": int((wi + 1) * 5),
            "soft_labels": lbl_vec.astype(np.float16),
            "weight": weight,
            "max_prob": float(max_w[fi, wi]),
            "n_pos": int((lbl_vec > 0).sum()),
        })

    df = pd.DataFrame(rows)
    print(f"\nFinal pseudo DF: {len(df)} rows across {df.filename.nunique()} files")
    print(f"  weight distribution: median={df.weight.median():.2f}  p90={df.weight.quantile(0.9):.2f}")
    print(f"  n_pos per row: median={df.n_pos.median():.0f}  max={df.n_pos.max()}")

    # Save labels as stacked numpy for fast loading
    soft_labels = np.stack(df.soft_labels.values)
    df = df.drop(columns=["soft_labels"])
    df.to_parquet(EXP41 / "pseudo_ensemble_df.parquet")
    np.savez_compressed(EXP41 / "pseudo_ensemble_labels.npz", labels=soft_labels)
    print(f"\nSaved: {EXP41}/pseudo_ensemble_df.parquet ({len(df)} rows)")
    print(f"Saved: {EXP41}/pseudo_ensemble_labels.npz")


if __name__ == "__main__":
    main()
