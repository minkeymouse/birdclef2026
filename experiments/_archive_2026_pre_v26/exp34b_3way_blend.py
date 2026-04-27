"""exp34b — 3-way blend with exp35 added. Reuse exp34 infra."""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
warnings.filterwarnings("ignore")

ROOT = Path("/data/birdclef2026")
CACHE = ROOT / "experiments/exp21_outputs/perch_cache"
DATA = ROOT / "data/birdclef-2026"
OUT = ROOT / "experiments/exp34_outputs"


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
    return meta, Y


def macro_auc(y_true, y_score):
    keep = y_true.sum(0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def zscore(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)


def main():
    t0 = time.time()
    meta, Y = load_truth()
    e28 = np.load(ROOT / "experiments/exp28_outputs/best_oof.npz")
    perch_a = e28["val_a_smoothed"].astype(np.float32)
    perch_b = e28["val_b_smoothed"].astype(np.float32)
    sed29 = np.load(ROOT / "experiments/exp29_outputs/val_scores.npz")["preds"].astype(np.float32)
    sed35 = np.load(ROOT / "experiments/exp35_outputs/val_scores.npz")["preds"].astype(np.float32)

    Pa, Pb = zscore(perch_a), zscore(perch_b)
    S29 = zscore(sed29)
    S35 = zscore(sed35)

    # Correlation between SED29 and SED35 (active classes only, z-scored)
    active = Y.sum(0) > 0
    corrs = []
    for c in np.where(active)[0]:
        c29 = S29[:, c]; c35 = S35[:, c]
        if c29.std() > 1e-6 and c35.std() > 1e-6:
            corrs.append(np.corrcoef(c29, c35)[0, 1])
    print(f"SED29 vs SED35 mean Pearson across {len(corrs)} active classes: {np.mean(corrs):.3f}")
    print(f"  (lower = more diverse; ≤0.6 suggests genuine diversity)")

    results = []
    def log(name, a, b):
        aa = macro_auc(Y, a); ab = macro_auc(Y, b)
        print(f"  {name:50s}  Val-A {aa:.4f}  Val-B {ab:.4f}")
        results.append({"name": name, "val_a": aa, "val_b": ab})
        return aa, ab

    print("\n[Reference]")
    log("perch_smoothed", Pa, Pb)
    log("sed29_alone", S29, S29)
    log("sed35_alone", S35, S35)

    print("\n[2-way Perch + SED29]  (best from exp34: α=0.8)")
    for a in [0.7, 0.75, 0.8, 0.85]:
        log(f"perch{a:.2f} + sed29{1-a:.2f}", a*Pa + (1-a)*S29, a*Pb + (1-a)*S29)

    print("\n[2-way Perch + SED35]")
    for a in [0.7, 0.8, 0.85, 0.9, 0.95]:
        log(f"perch{a:.2f} + sed35{1-a:.2f}", a*Pa + (1-a)*S35, a*Pb + (1-a)*S35)

    print("\n[3-way Perch + SED29 + SED35 simplex, wp ≥ 0.6]")
    best3 = None
    for wp in np.arange(0.6, 0.951, 0.05):
        for w29 in np.arange(0.0, 1.0 - wp + 0.001, 0.05):
            w35 = 1.0 - wp - w29
            if w35 < -1e-6: continue
            ba = wp*Pa + w29*S29 + w35*S35
            bb = wp*Pb + w29*S29 + w35*S35
            a_auc = macro_auc(Y, ba); b_auc = macro_auc(Y, bb)
            if best3 is None or a_auc > best3["val_a"]:
                best3 = {"wp": float(wp), "w29": float(w29), "w35": float(w35),
                         "val_a": a_auc, "val_b": b_auc}
    print(f"  best 3-way: wp={best3['wp']:.2f} w29={best3['w29']:.2f} w35={best3['w35']:.2f}  "
          f"Val-A {best3['val_a']:.4f}  Val-B {best3['val_b']:.4f}")
    results.append({"name": "best_3way", **best3})

    # Also fine-grained grid around best
    print("\n[3-way finer grid around best]")
    for wp in np.arange(max(0.6, best3['wp']-0.05), min(0.95, best3['wp']+0.051), 0.02):
        for w29 in np.arange(max(0, best3['w29']-0.05), min(1-wp, best3['w29']+0.051), 0.02):
            w35 = 1.0 - wp - w29
            if w35 < -1e-6 or w35 > 0.5: continue
            ba = wp*Pa + w29*S29 + w35*S35
            bb = wp*Pb + w29*S29 + w35*S35
            a_auc = macro_auc(Y, ba); b_auc = macro_auc(Y, bb)
            if a_auc > best3["val_a"]:
                best3 = {"wp": float(wp), "w29": float(w29), "w35": float(w35),
                         "val_a": a_auc, "val_b": b_auc}
    print(f"  refined: wp={best3['wp']:.2f} w29={best3['w29']:.2f} w35={best3['w35']:.2f}  "
          f"Val-A {best3['val_a']:.4f}  Val-B {best3['val_b']:.4f}")
    results.append({"name": "best_3way_refined", **best3})

    # Also: two Gauss/EMA smoothing tests on the blended output
    from scipy.ndimage import gaussian_filter1d
    best_blend_a = best3['wp']*Pa + best3['w29']*S29 + best3['w35']*S35
    best_blend_b = best3['wp']*Pb + best3['w29']*S29 + best3['w35']*S35
    # Group by file to smooth temporally
    seq_a = np.stack([best_blend_a[meta[meta.filename==fn].index.values] for fn in meta.filename.unique()])
    seq_b = np.stack([best_blend_b[meta[meta.filename==fn].index.values] for fn in meta.filename.unique()])
    for sigma in [0.5, 0.75, 1.0]:
        sa = gaussian_filter1d(seq_a, sigma=sigma, axis=1, mode="nearest").reshape(-1, 234)
        sb = gaussian_filter1d(seq_b, sigma=sigma, axis=1, mode="nearest").reshape(-1, 234)
        # Reorder to meta original row order
        row_order = []
        for fn in meta.filename.unique():
            row_order.extend(meta[meta.filename==fn].index.values)
        row_order = np.array(row_order)
        sa_o = np.zeros_like(sa); sa_o[row_order] = sa
        sb_o = np.zeros_like(sb); sb_o[row_order] = sb
        log(f"3way_smooth_sigma{sigma}", sa_o, sb_o)

    print(f"\nDone {(time.time()-t0):.1f}s")
    (OUT / "results_3way.json").write_text(json.dumps({
        "elapsed_s": time.time() - t0,
        "sed29_sed35_mean_pearson": float(np.mean(corrs)),
        "results": results,
        "best_3way_refined": best3,
    }, indent=2))


if __name__ == "__main__":
    main()
