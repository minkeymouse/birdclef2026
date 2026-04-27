#!/usr/bin/env python3
"""exp42 — per-class normalization on v12 pipeline.

Tests multiple variants since ROC-AUC is rank-invariant per-class:
  V1: pure rank-norm (should be no-op for AUC but used as sanity)
  V2: per-class mean-centering + std-scaling (z-score per class) — also no-op
  V3: per-class quantile → uniform (rank → percentile)
  V4: Platt per-class (sigmoid calibration) on OOF folds — exp34 failed, retest
  V5: combined rank norm + temperature per-class
  V6: file-level rank norm (rank each file's predictions across windows)

Since vanilla AUC is invariant to per-class monotonic, any gain must come from:
  - tie-breaking (when many predictions tied)
  - cross-class interactions after aggregation
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP21 = ROOT / "experiments/exp21_outputs/perch_cache"
EXP28 = ROOT / "experiments/exp28_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
OUT = ROOT / "experiments/exp42_outputs"
OUT.mkdir(exist_ok=True)


def macro_auc(Y, S):
    k = Y.sum(0) > 0
    return float(roc_auc_score(Y[:, k], S[:, k], average="macro"))


def zs(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


def gauss(S, files, sigma=0.5):
    out = np.zeros_like(S)
    for f in np.unique(files):
        m = files == f
        out[m] = gaussian_filter1d(S[m], sigma=sigma, axis=0)
    return out


def build():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)
    meta = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    Y_sc = np.zeros((len(sc), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc["lbls"]):
        for l in labs:
            if l in l2i: Y_sc[i, l2i[l]] = 1
    idx = sc.set_index("row_id")
    Y = np.stack([Y_sc[idx.index.get_loc(rid)] for rid in meta["row_id"]])
    return meta, Y


# ─── normalization variants ─────────────────────────────────────────
def v1_rank_global(S):
    """rank each class globally across all rows"""
    out = np.zeros_like(S, dtype=np.float32)
    for c in range(S.shape[1]):
        out[:, c] = rankdata(S[:, c]) / len(S)
    return out


def v3_rank_per_file(S, files):
    """rank within each file (across 12 windows)"""
    out = np.zeros_like(S, dtype=np.float32)
    for f in np.unique(files):
        m = files == f
        for c in range(S.shape[1]):
            out[m, c] = rankdata(S[m, c]) / m.sum()
    return out


def v4_cross_class_rank(S):
    """rank across classes per row (confidence ordering within a window)"""
    out = np.zeros_like(S, dtype=np.float32)
    for i in range(len(S)):
        out[i] = rankdata(S[i]) / S.shape[1]
    return out


def v5_centered_rescale(S):
    """center per class, rescale by class std (z-score per class)"""
    return zs(S)


def v6_tempscale_perclass(S, Y, T_default=1.0):
    """per-class temperature fit on logits (heuristic, not proper calibration)"""
    # logit transform
    eps = 1e-6
    P = np.clip(S, eps, 1 - eps)
    logits = np.log(P / (1 - P))
    # for each class, scale by 1/std of positive vs negative means
    out = logits.copy()
    for c in range(S.shape[1]):
        pos = logits[Y[:, c] == 1, c]
        neg = logits[Y[:, c] == 0, c]
        if len(pos) > 1 and len(neg) > 1:
            T = max(0.1, (pos.std() + neg.std()) / 2)
            out[:, c] = logits[:, c] / T
    return 1.0 / (1.0 + np.exp(-out))


def main():
    meta, Y = build()
    files = meta["filename"].values

    perch = np.load(EXP28 / "best_oof.npz")["val_a_smoothed"]
    sed29 = np.load(EXP29 / "val_scores.npz")["preds"]

    # v12 baseline: z-score blend (P=0.80, SED29=0.20) + Gauss 0.5
    zP, z29 = zs(perch), zs(sed29)
    v12_blend = 0.80 * zP + 0.20 * z29
    v12 = gauss(v12_blend, files, sigma=0.5)
    auc_v12 = macro_auc(Y, v12)
    print(f"v12 baseline (P·0.80 + SED29·0.20 + Gauss): {auc_v12:.6f}")

    results = {"v12_baseline": auc_v12}

    # Perclass norm variants applied ON TOP of v12
    for name, fn in [
        ("rank_global_per_class", lambda S: v1_rank_global(S)),
        ("rank_per_file", lambda S: v3_rank_per_file(S, files)),
        ("cross_class_rank", lambda S: v4_cross_class_rank(S)),
        ("zscore_per_class", lambda S: v5_centered_rescale(S)),
        ("tempscale_perclass", lambda S: v6_tempscale_perclass(S, Y)),
    ]:
        out = fn(v12)
        auc = macro_auc(Y, out)
        print(f"  + {name}: {auc:.6f}  (Δ={auc - auc_v12:+.4f})")
        results[f"v12+{name}"] = auc

    # Combined: per-class rank on Perch + SED separately, then z-blend
    print("\nApply rank norm BEFORE blending:")
    P_rank = v1_rank_global(perch)
    S_rank = v1_rank_global(sed29)
    rb = 0.80 * zs(P_rank) + 0.20 * zs(S_rank)
    rb_g = gauss(rb, files, sigma=0.5)
    auc_pre = macro_auc(Y, rb_g)
    print(f"  rank→zscore→blend→gauss: {auc_pre:.6f}  (Δ={auc_pre - auc_v12:+.4f})")
    results["pre_blend_rank"] = auc_pre

    # Gauss sigma sweep on v12 (cheap alternative lever)
    print("\nGauss sigma sweep on v12 blend:")
    for sigma in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5]:
        s = gauss(v12_blend, files, sigma=sigma)
        auc = macro_auc(Y, s)
        print(f"  σ={sigma}: {auc:.6f}")
        results[f"gauss_s{sigma}"] = auc

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved: {OUT}/results.json")


if __name__ == "__main__":
    main()
