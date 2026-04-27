#!/usr/bin/env python3
"""Fair comparison: SED29 vs SED38 on the SAME 11 held-out files + blend with Perch."""
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
EXP38 = ROOT / "experiments/exp38_outputs"

SEED = 42
EVAL_N_FILES = 11


def macro_auc(Y, S):
    keep = Y.sum(0) > 0
    if keep.sum() < 2:
        return float("nan")
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
          .apply(lambda s: sorted({l for x in s for l in parse(x)}))
          .reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    meta = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    Y_sc = np.zeros((len(sc), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc["lbls"]):
        for l in labs:
            if l in l2i: Y_sc[i, l2i[l]] = 1
    idx = sc.set_index("row_id")
    Y = np.stack([Y_sc[idx.index.get_loc(rid)] for rid in meta["row_id"]])

    # Reproduce exp38 split
    rng = np.random.RandomState(SEED)
    ss_sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    files_all = sorted(ss_sc.filename.unique())
    rng.shuffle(files_all)
    eval_files = set(files_all[:EVAL_N_FILES])
    train_files = set(files_all[EVAL_N_FILES:])
    return meta, Y, eval_files, train_files


def main():
    meta, Y, eval_files, train_files = build()
    print(f"exp38 split — train {len(train_files)} / eval {len(eval_files)} files")

    # Filter meta to only exp29-Val-A files that are in exp38's eval set
    mask_eval = meta["filename"].isin(eval_files).values
    mask_train = meta["filename"].isin(train_files).values
    print(f"Rows: eval={mask_eval.sum()}, train-leaked={mask_train.sum()}, total={len(meta)}")

    perch = np.load(EXP28 / "best_oof.npz")["val_a_smoothed"]
    sed29 = np.load(EXP29 / "val_scores.npz")["preds"]
    sed38 = np.load(EXP38 / "val_scores_full.npz")["preds"]

    # --- On 11 held-out files (fair) ---
    print("\n=== Fair (exp38 eval-holdout 11 files) ===")
    Ye = Y[mask_eval]; Pe = perch[mask_eval]; S29e = sed29[mask_eval]; S38e = sed38[mask_eval]
    print(f"active classes: {int((Ye.sum(0)>0).sum())}")
    print(f"Perch alone:   {macro_auc(Ye, Pe):.4f}")
    print(f"SED29 alone:   {macro_auc(Ye, S29e):.4f}")
    print(f"SED38 alone:   {macro_auc(Ye, S38e):.4f}  (seen 0, unbiased)")

    zP, z29, z38 = zscore(Pe), zscore(S29e), zscore(S38e)
    print("\nPerch + SED29:")
    best29 = (-1, None)
    for a in np.arange(0.0, 1.01, 0.1):
        s = a * zP + (1 - a) * z29
        auc = macro_auc(Ye, s)
        if auc > best29[0]: best29 = (auc, a)
        print(f"  α={a:.1f}: {auc:.4f}")
    print(f"  best: α={best29[1]:.1f} → {best29[0]:.4f}")

    print("\nPerch + SED38:")
    best38 = (-1, None)
    for a in np.arange(0.0, 1.01, 0.1):
        s = a * zP + (1 - a) * z38
        auc = macro_auc(Ye, s)
        if auc > best38[0]: best38 = (auc, a)
        print(f"  α={a:.1f}: {auc:.4f}")
    print(f"  best: α={best38[1]:.1f} → {best38[0]:.4f}")

    print("\n3-way: wP + w29 + w38:")
    best3 = (-1, None)
    for wP in np.arange(0.0, 1.01, 0.05):
        for w29 in np.arange(0.0, 1.01 - wP, 0.05):
            w38 = 1.0 - wP - w29
            if w38 < -1e-9: continue
            auc = macro_auc(Ye, wP * zP + w29 * z29 + w38 * z38)
            if auc > best3[0]: best3 = (auc, (wP, w29, w38))
    print(f"  best: {best3[0]:.4f}  (wP={best3[1][0]:.2f}, w29={best3[1][1]:.2f}, w38={best3[1][2]:.2f})")

    # --- On 48 train-leaked files (upper bound / sanity) ---
    print("\n=== Train-leaked (48 files, UPPER BOUND only) ===")
    Yt = Y[mask_train]; Pt = perch[mask_train]; S29t = sed29[mask_train]; S38t = sed38[mask_train]
    print(f"Perch: {macro_auc(Yt, Pt):.4f}  SED29: {macro_auc(Yt, S29t):.4f}  SED38: {macro_auc(Yt, S38t):.4f}")

    out = {
        "fair_eval_11files": {
            "perch_alone": macro_auc(Ye, Pe),
            "sed29_alone": macro_auc(Ye, S29e),
            "sed38_alone": macro_auc(Ye, S38e),
            "best_P+29": {"alpha": best29[1], "val_a": best29[0]},
            "best_P+38": {"alpha": best38[1], "val_a": best38[0]},
            "best_3way": {"wP": best3[1][0], "w29": best3[1][1], "w38": best3[1][2], "val_a": best3[0]},
        },
        "leaked_train_48files_upper_bound": {
            "perch": macro_auc(Yt, Pt),
            "sed29": macro_auc(Yt, S29t),
            "sed38": macro_auc(Yt, S38t),
        },
    }
    (EXP38 / "fair_blend.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {EXP38}/fair_blend.json")


if __name__ == "__main__":
    main()
