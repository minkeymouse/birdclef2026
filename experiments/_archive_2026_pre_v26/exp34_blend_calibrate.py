"""
exp34 — final blend + per-class calibration under Val-A / Val-B.

Inputs:
  - exp28 best_oof.npz          : Perch probe (val_a_probe, val_a_smoothed, val_b_probe, val_b_smoothed)
  - exp29 val_scores.npz        : HGNet SED basic (Val-A 0.737)
  - exp31 val_scores.npz        : HGNet SED + BG (Val-A 0.604) — probably useless but we'll test

Tests:
  T1  Perch probe smoothed (exp28 baseline)                 : 0.8943 reference
  T2  T1 + per-class Platt calibration (isotonic & sigmoid)  : expected +0.002
  T3  Blend α·Perch + (1-α)·exp29                             : grid α ∈ {0..1}
  T4  Blend α·Perch + (1-α)·exp31                             : grid α
  T5  3-way: w_P·Perch + w_29·exp29 + w_31·exp31              : simplex grid
  T6  T5 + Platt on top of final blend

For Platt per-class: we CANNOT use the same data for both training and eval AUC (leak).
Use file-stratified 5-fold: on each fold, fit Platt on training folds' raw probe → y,
apply to validation fold. Rebuild OOF calibrated score.
"""
from __future__ import annotations
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path("/data/birdclef2026")
CACHE = ROOT / "experiments/exp21_outputs/perch_cache"
DATA = ROOT / "data/birdclef-2026"
EXP28 = ROOT / "experiments/exp28_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP31 = ROOT / "experiments/exp31_outputs"
OUT = ROOT / "experiments/exp34_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_truth():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sample_sub.columns[1:].tolist()
    lab2idx = {c: i for i, c in enumerate(primary)}

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    meta = pd.read_parquet(CACHE / "full_perch_meta.parquet")
    Y = np.zeros((len(meta), len(primary)), dtype=np.uint8)
    by_rowid = sc.set_index("row_id")
    for i, rid in enumerate(meta["row_id"]):
        if rid in by_rowid.index:
            for l in by_rowid.loc[rid, "lbls"]:
                if l in lab2idx:
                    Y[i, lab2idx[l]] = 1
    return meta, Y, primary


def val_a_folds(meta):
    files = meta.drop_duplicates("filename").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f2f = {}
    for fold, (_, vi) in enumerate(skf.split(files["filename"], files["site"])):
        for f in files.iloc[vi]["filename"].values:
            f2f[f] = fold
    return meta["filename"].map(f2f).values.astype(int)


def val_b_folds(meta):
    gkf = GroupKFold(n_splits=min(5, meta["site"].nunique()))
    folds = np.full(len(meta), -1)
    for fold, (_, vi) in enumerate(gkf.split(meta, groups=meta["site"])):
        folds[vi] = fold
    return folds


def macro_auc(y_true, y_score):
    keep = y_true.sum(0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def calibrate_platt(scores, Y, folds):
    """Per-class fold-wise Platt calibration. Returns calibrated OOF scores."""
    out = np.zeros_like(scores, dtype=np.float32)
    for c in range(Y.shape[1]):
        if Y[:, c].sum() < 4:
            out[:, c] = scores[:, c]
            continue
        for f in np.unique(folds):
            ti = folds != f; vi = ~ti
            if Y[ti, c].sum() < 3 or (1 - Y[ti, c]).sum() < 3:
                out[vi, c] = scores[vi, c]
                continue
            try:
                lr = LogisticRegression(max_iter=200, C=1.0)
                lr.fit(scores[ti, c:c+1], Y[ti, c])
                out[vi, c] = lr.decision_function(scores[vi, c:c+1])
            except Exception:
                out[vi, c] = scores[vi, c]
    return out


def calibrate_isotonic(scores, Y, folds):
    out = np.zeros_like(scores, dtype=np.float32)
    for c in range(Y.shape[1]):
        if Y[:, c].sum() < 4:
            out[:, c] = scores[:, c]
            continue
        for f in np.unique(folds):
            ti = folds != f; vi = ~ti
            if Y[ti, c].sum() < 3 or (1 - Y[ti, c]).sum() < 3:
                out[vi, c] = scores[vi, c]
                continue
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(scores[ti, c], Y[ti, c])
                out[vi, c] = iso.transform(scores[vi, c])
            except Exception:
                out[vi, c] = scores[vi, c]
    return out


def zscore(x):
    """Per-class z-score normalization so scales align before blending."""
    m = x.mean(0, keepdims=True)
    s = x.std(0, keepdims=True) + 1e-6
    return (x - m) / s


def main():
    t0 = time.time()
    meta, Y, primary = load_truth()
    folds_a = val_a_folds(meta)
    folds_b = val_b_folds(meta)

    # Load all
    e28 = np.load(EXP28 / "best_oof.npz")
    perch_a = e28["val_a_smoothed"].astype(np.float32)   # 0.8943 expected
    perch_b = e28["val_b_smoothed"].astype(np.float32)
    perch_a_raw = e28["val_a_probe"].astype(np.float32)  # 0.8891
    perch_b_raw = e28["val_b_probe"].astype(np.float32)
    sed29 = np.load(EXP29 / "val_scores.npz")["preds"].astype(np.float32)
    sed31 = np.load(EXP31 / "val_scores.npz")["preds"].astype(np.float32)

    # Reference AUCs
    results = []
    def log(name, val_a_score, val_b_score=None):
        aa = macro_auc(Y, val_a_score)
        ab = macro_auc(Y, val_b_score) if val_b_score is not None else None
        r = {"name": name, "val_a": aa, "val_b": ab}
        s = f"  {name:45s}  Val-A {aa:.4f}"
        if ab is not None: s += f"  Val-B {ab:.4f}"
        print(s)
        results.append(r)
        return aa, ab

    print("\n[T1] References")
    log("perch_probe_raw", perch_a_raw, perch_b_raw)
    log("perch_probe_smoothed", perch_a, perch_b)
    log("sed29_basic", sed29)
    log("sed31_bgmix", sed31)

    print("\n[T2] Per-class Platt / isotonic on Perch smoothed")
    perch_a_platt = calibrate_platt(perch_a, Y, folds_a)
    perch_b_platt = calibrate_platt(perch_b, Y, folds_b)
    log("perch_smoothed+Platt", perch_a_platt, perch_b_platt)
    perch_a_iso = calibrate_isotonic(perch_a, Y, folds_a)
    perch_b_iso = calibrate_isotonic(perch_b, Y, folds_b)
    log("perch_smoothed+Isotonic", perch_a_iso, perch_b_iso)

    # Also on raw probe (before smoothing)
    perch_a_raw_platt = calibrate_platt(perch_a_raw, Y, folds_a)
    perch_b_raw_platt = calibrate_platt(perch_b_raw, Y, folds_b)
    log("perch_probe_raw+Platt", perch_a_raw_platt, perch_b_raw_platt)

    print("\n[T3] Blend Perch + SED29 (z-scored before blend)")
    perch_a_z = zscore(perch_a)
    perch_b_z = zscore(perch_b)
    sed29_z = zscore(sed29)
    sed31_z = zscore(sed31)
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend_a = alpha * perch_a_z + (1 - alpha) * sed29_z
        # For Val-B we don't have site-fold-specific SED, just reuse sed29 (it's the same prediction)
        blend_b = alpha * perch_b_z + (1 - alpha) * sed29_z
        log(f"blend_perch_sed29_a{alpha:.1f}", blend_a, blend_b)

    print("\n[T4] Blend Perch + SED31 (bg mix)")
    for alpha in np.arange(0.0, 1.01, 0.1):
        blend_a = alpha * perch_a_z + (1 - alpha) * sed31_z
        blend_b = alpha * perch_b_z + (1 - alpha) * sed31_z
        log(f"blend_perch_sed31_a{alpha:.1f}", blend_a, blend_b)

    print("\n[T5] 3-way Perch + SED29 + SED31 simplex")
    best_3way = None
    for wp in np.arange(0.0, 1.01, 0.1):
        for w29 in np.arange(0.0, 1.0 - wp + 0.01, 0.1):
            w31 = 1.0 - wp - w29
            if w31 < -1e-6: continue
            ba = wp * perch_a_z + w29 * sed29_z + w31 * sed31_z
            bb = wp * perch_b_z + w29 * sed29_z + w31 * sed31_z
            a_auc = macro_auc(Y, ba); b_auc = macro_auc(Y, bb)
            if best_3way is None or a_auc > best_3way["val_a"]:
                best_3way = {"wp": float(wp), "w29": float(w29), "w31": float(w31),
                             "val_a": a_auc, "val_b": b_auc}
    print(f"  best 3-way: wp={best_3way['wp']:.1f} w29={best_3way['w29']:.1f} w31={best_3way['w31']:.1f}  "
          f"Val-A {best_3way['val_a']:.4f}  Val-B {best_3way['val_b']:.4f}")
    results.append({"name": "best_3way", **best_3way})

    print("\n[T6] Perch+Platt blended with SED29")
    perch_a_platt_z = zscore(perch_a_platt)
    perch_b_platt_z = zscore(perch_b_platt)
    for alpha in [0.7, 0.8, 0.9, 0.95, 1.0]:
        ba = alpha * perch_a_platt_z + (1 - alpha) * sed29_z
        bb = alpha * perch_b_platt_z + (1 - alpha) * sed29_z
        log(f"platt+sed29 a{alpha:.2f}", ba, bb)

    # Save best pipeline scores for later
    best_overall = max((r for r in results if r.get("val_a") is not None),
                       key=lambda r: r["val_a"])
    print(f"\nBEST OVERALL: {best_overall['name']}  Val-A {best_overall['val_a']:.4f}")
    print(f"baseline (perch_smoothed): 0.8943")

    (OUT / "results.json").write_text(json.dumps({
        "elapsed_s": time.time() - t0,
        "results": results,
        "best_overall": best_overall,
    }, indent=2))
    print(f"\nDone in {(time.time()-t0):.1f}s")


if __name__ == "__main__":
    main()
