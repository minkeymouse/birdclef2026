#!/usr/bin/env python3
"""exp43h — Replace Mahalanobis with teacher sharpness in pseudo weights.

exp43g evidence: on labeled SS flip detection,
  Mahalanobis AUC 0.463  (useless, near-random)
  Teacher posterior AUC 0.687 (strongest single signal)

exp41c current weight: w = mahal_w * w_sum
This exp: w = sharpness * w_sum, where
  sharpness(p) = (p_top1 - p_top2) / (p_top1 + eps)  in [0, 1]

Output: experiments/exp41_outputs/pseudo_ensemble_df_teacher.parquet (new weights)
        experiments/exp41_outputs/pseudo_ensemble_labels_teacher.npz (same labels)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
EXP41 = ROOT / "experiments/exp41_outputs"

MAX_PROB_THRESH = 0.5
CLASS_PROB_FLOOR = 0.1
POWER = 0.5
EPS = 1e-6


def sharpness(probs):
    """Per-window sharpness: (top1 - top2) / top1. Higher = more confident teacher."""
    top2 = -np.sort(-probs, axis=-1)[..., :2]
    gap = top2[..., 0] - top2[..., 1]
    return gap / (top2[..., 0] + EPS)


def main():
    d = np.load(EXP41 / "pseudo_probs_ensemble.npz", allow_pickle=True)
    probs = d["probs"].astype(np.float32)  # (n_files, 12, 234)
    filenames = d["filenames"]
    n_files, n_win, n_cls = probs.shape
    print(f"Pseudo probs: {probs.shape}")

    sharp = sharpness(probs)  # (n_files, 12)
    print(f"Sharpness dist: q10={np.quantile(sharp, .1):.3f}  "
          f"q50={np.quantile(sharp, .5):.3f}  q90={np.quantile(sharp, .9):.3f}")

    # Same filter chain as exp41c2
    max_w = probs.max(-1)
    keep = max_w >= MAX_PROB_THRESH
    p_filt = np.where(probs >= CLASS_PROB_FLOOR, probs, 0)
    p_pow = p_filt ** POWER

    rows = []
    keep_idx = np.argwhere(keep)
    for fi, wi in keep_idx:
        lbl_vec = p_pow[fi, wi]
        w_sum = float(lbl_vec.sum())
        if w_sum < 0.05:
            continue
        # NEW weight: sharpness instead of Mahalanobis
        weight = float(sharp[fi, wi] * w_sum)
        rows.append({
            "filename": str(filenames[fi]),
            "win_idx": int(wi),
            "end_sec": int((wi + 1) * 5),
            "soft_labels": lbl_vec.astype(np.float16),
            "weight": weight,
            "sharp": float(sharp[fi, wi]),
            "max_prob": float(max_w[fi, wi]),
            "n_pos": int((lbl_vec > 0).sum()),
        })

    df = pd.DataFrame(rows)
    print(f"\nPseudo DF: {len(df)} rows across {df.filename.nunique()} files")
    print(f"  weight   p50={df.weight.median():.3f}  p90={df.weight.quantile(0.9):.3f}")
    print(f"  sharp    p50={df.sharp.median():.3f}  p90={df.sharp.quantile(0.9):.3f}")

    # Compare to Mahalanobis weights on same rows
    mw = np.load(EXP41 / "mahal_weights.npz", allow_pickle=True)["confidence_weights"].astype(np.float32)
    print(f"\nCompare to mahal_w: mean {mw.mean():.3f}, p50 {np.median(mw):.3f}")
    # Spearman between new and old weighting on retained rows
    from scipy.stats import spearmanr
    mahal_on_rows = np.array([mw[df.filename.values[i] == filenames.astype(str), df.win_idx.values[i]][0]
                              if (df.filename.values[i] == filenames.astype(str)).any() else np.nan
                              for i in range(min(1000, len(df)))])
    mahal_on_rows = np.array([
        mw[int(np.where(filenames.astype(str) == r.filename)[0][0]), int(r.win_idx)]
        for _, r in df.head(1000).iterrows()
    ])
    r, _ = spearmanr(df.head(1000).weight.values, mahal_on_rows)
    print(f"spearman(new_weight, mahal_weight on first 1000 rows): {r:+.3f}")

    soft_labels = np.stack(df.soft_labels.values)
    df_out = df.drop(columns=["soft_labels"])
    df_out.to_parquet(EXP41 / "pseudo_ensemble_df_teacher.parquet")
    np.savez_compressed(EXP41 / "pseudo_ensemble_labels_teacher.npz", labels=soft_labels)
    print(f"\nSaved: {EXP41}/pseudo_ensemble_df_teacher.parquet ({len(df)} rows)")


if __name__ == "__main__":
    main()
