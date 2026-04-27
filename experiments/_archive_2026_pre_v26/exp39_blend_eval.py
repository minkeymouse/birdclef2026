#!/usr/bin/env python3
"""exp39 blend vs Perch + SED29. Uses full 59-file Val-A."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP21 = ROOT / "experiments/exp21_outputs/perch_cache"
EXP28 = ROOT / "experiments/exp28_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP39 = ROOT / "experiments/exp39_outputs"


def macro_auc(Y, S):
    keep = Y.sum(0) > 0
    return float(roc_auc_score(Y[:, keep], S[:, keep], average="macro"))


def zscore(X):
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


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


def main():
    meta, Y = build()
    perch = np.load(EXP28 / "best_oof.npz")["val_a_smoothed"]
    sed29 = np.load(EXP29 / "val_scores.npz")["preds"]
    sed39 = np.load(EXP39 / "val_scores.npz")["preds"]

    print(f"Val-A rows: {len(Y)}, active classes: {int((Y.sum(0)>0).sum())}")
    print(f"Perch alone:  {macro_auc(Y, perch):.4f}")
    print(f"SED29 alone:  {macro_auc(Y, sed29):.4f}")
    print(f"SED39 alone:  {macro_auc(Y, sed39):.4f}")

    # Pearson correlation between SED29 and SED39 on active classes
    keep = Y.sum(0) > 0
    from scipy.stats import pearsonr
    s29f = sed29[:, keep].flatten()
    s39f = sed39[:, keep].flatten()
    r, _ = pearsonr(s29f, s39f)
    print(f"SED29 vs SED39 Pearson (active): {r:.3f}")

    zP, z29, z39 = zscore(perch), zscore(sed29), zscore(sed39)

    print("\n=== Perch + SED39 (2-way) ===")
    for a in np.arange(0.5, 0.96, 0.05):
        auc = macro_auc(Y, a * zP + (1 - a) * z39)
        print(f"  α={a:.2f}: {auc:.4f}")

    print("\n=== 3-way: wP + w29 + w39 (simplex) ===")
    best = (-1, None)
    for wP in np.arange(0.5, 0.96, 0.05):
        for w29 in np.arange(0.0, 1.01 - wP, 0.05):
            w39 = 1.0 - wP - w29
            if w39 < -1e-9: continue
            auc = macro_auc(Y, wP * zP + w29 * z29 + w39 * z39)
            if auc > best[0]:
                best = (auc, (wP, w29, w39))
    print(f"Best 3-way: {best[0]:.4f}  (wP={best[1][0]:.2f}, w29={best[1][1]:.2f}, w39={best[1][2]:.2f})")

    # Gauss smoothing on best
    from scipy.ndimage import gaussian_filter1d
    wP, w29, w39 = best[1]
    s = wP * zP + w29 * z29 + w39 * z39
    files = meta["filename"].values
    s_sm = np.zeros_like(s)
    for f in np.unique(files):
        m = files == f
        s_sm[m] = gaussian_filter1d(s[m], sigma=0.5, axis=0)
    auc_sm = macro_auc(Y, s_sm)
    print(f"Best 3-way + Gauss σ=0.5: {auc_sm:.4f}")

    # v12 reference
    v12 = 0.80 * zP + 0.20 * z29
    files = meta["filename"].values
    v12_sm = np.zeros_like(v12)
    for f in np.unique(files):
        m = files == f
        v12_sm[m] = gaussian_filter1d(v12[m], sigma=0.5, axis=0)
    print(f"\nv12 reproduction (P 0.80 + SED29 0.20 + Gauss σ=0.5): {macro_auc(Y, v12_sm):.4f}")

    out = {"val_a_alone": {"perch": macro_auc(Y, perch), "sed29": macro_auc(Y, sed29), "sed39": macro_auc(Y, sed39)},
           "pearson_29_39": float(r),
           "best_3way": {"wP": best[1][0], "w29": best[1][1], "w39": best[1][2], "val_a": best[0]},
           "best_3way_gauss0.5": auc_sm,
           "v12_reference": macro_auc(Y, v12_sm)}
    (EXP39 / "blend_results.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
