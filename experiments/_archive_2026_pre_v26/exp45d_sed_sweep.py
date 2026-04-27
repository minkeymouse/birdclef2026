#!/usr/bin/env python3
"""exp45d — SED weight sweep + cross-teacher agreement audit.

Motivation (from exp45):
  SED41f alone macro 0.878 >> v12 z-blend 0.714 on 40 eval classes.
  Current notebook uses W_SED41F=0.20. Maybe higher weight is better locally.
  But: LB transfer unreliable. Use audit to map the weight × taxon trade-off.

Also:
  - Cross-teacher agreement per species: where do SED29 and SED41f diverge?
  - Diverse agreement = diversity source. Concordant disagreement = both wrong.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP41F = ROOT / "experiments/exp41f_outputs"
OUT = ROOT / "experiments/exp45d_outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
EVAL_N_FILES = 11


def build_eval():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["lbls"]):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_eval, Y, primary


def align_from_exp43a(sc_eval):
    scs = np.load(EXP43A / "perch_ss_all.npz")["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    out = np.zeros((len(sc_eval), scs.shape[1]), dtype=np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = scs[j]
    return out


def align_from_old_cache(sc_eval, npz_path):
    if not npz_path.exists(): return None
    p = np.load(npz_path)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != p.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(sc_eval), p.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = p[j]
    return out


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
def zs(X):
    m = X.mean(axis=0, keepdims=True); s = X.std(axis=0, keepdims=True) + 1e-8
    return (X - m) / s
def per_class_auc(Y, P):
    out = {}
    for c in range(Y.shape[1]):
        y = Y[:, c].astype(int); p = P[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
        if not np.isfinite(p).all(): continue
        try: out[c] = float(roc_auc_score(y, p))
        except Exception: pass
    return out
def macro(d): return float(np.mean(list(d.values()))) if d else 0.0


def gauss_block(scores, sc_eval, sigma=0.5):
    """Per-file Gauss smoothing over the 12 windows."""
    out = np.zeros_like(scores)
    for fname in sc_eval["filename"].unique():
        mask = (sc_eval["filename"] == fname).values
        block = scores[mask]
        for c in range(block.shape[1]):
            out[mask, c] = gaussian_filter1d(block[:, c], sigma=sigma, mode="nearest")
    return out


def main():
    sc_eval, Y, primary = build_eval()
    print(f"Eval: {len(sc_eval)} rows × {len(primary)} cols")

    perch = align_from_exp43a(sc_eval)           # logits
    perch_prob = sigmoid(perch)
    sed29 = align_from_old_cache(sc_eval, EXP29 / "val_scores.npz")
    sed41f = align_from_old_cache(sc_eval, EXP41F / "val_scores_full.npz")
    print(f"aligned: perch {perch.shape}  sed29 {None if sed29 is None else sed29.shape}  sed41f {None if sed41f is None else sed41f.shape}")

    # --- weight sweep over (W_PERCH, W_SED29, W_SED41F) with Gauss 0.5 ---
    print("\n[Sweep] Blend weights (rows=SED29, cols=SED41f; W_PERCH = 1 - S29 - S41f)")
    wg = np.arange(0.0, 1.01, 0.1)
    sed29_filled = np.nan_to_num(sed29, nan=0.0) if sed29 is not None else np.zeros_like(perch_prob)
    sed41f_filled = np.nan_to_num(sed41f, nan=0.0) if sed41f is not None else np.zeros_like(perch_prob)

    zP = zs(perch_prob); z29 = zs(sed29_filled); z41 = zs(sed41f_filled)

    best = {"macro": -1}
    grid = np.full((len(wg), len(wg)), np.nan)
    for i, w29 in enumerate(wg):
        for j, w41 in enumerate(wg):
            wP = 1.0 - w29 - w41
            if wP < -0.01: continue
            blend = wP * zP + w29 * z29 + w41 * z41
            smoothed = gauss_block(blend, sc_eval, sigma=0.5)
            aucs = per_class_auc(Y, smoothed)
            m = macro(aucs)
            grid[i, j] = m
            if m > best["macro"]:
                best = {"macro": m, "wP": float(wP), "w29": float(w29), "w41": float(w41)}

    # print grid
    print(f"\n{'w29\\w41':>8}", *[f"{w:>6.1f}" for w in wg])
    for i, w29 in enumerate(wg):
        print(f"{w29:>6.1f}  ", *[f"{grid[i,j]:>6.4f}" if not np.isnan(grid[i,j]) else "   nan" for j in range(len(wg))])

    print(f"\nBest: macro={best['macro']:.4f}  wP={best['wP']:.2f}  w29={best['w29']:.2f}  w41={best['w41']:.2f}")

    # v12, v17, exp45c comparison
    ref_v12 = 0.80 * zP + 0.20 * z29
    ref_v12_s = gauss_block(ref_v12, sc_eval, 0.5)
    ref_v17 = 0.80 * zP + 0.20 * z41
    ref_v17_s = gauss_block(ref_v17, sc_eval, 0.5)
    print(f"\nReference:")
    print(f"  v12 (0.8P + 0.2SED29 + Gauss):  macro={macro(per_class_auc(Y, ref_v12_s)):.4f}")
    print(f"  v17 (0.8P + 0.2SED41f + Gauss): macro={macro(per_class_auc(Y, ref_v17_s)):.4f}")

    # Cross-teacher agreement per class
    print(f"\n[Cross-teacher] Per-class Perch-vs-SED41f correlation on eval rows:")
    if sed41f is not None:
        aucs_p = per_class_auc(Y, perch_prob)
        aucs_s = per_class_auc(Y, sed41f_filled)
        agreements = []
        for c in aucs_p:
            if c not in aucs_s: continue
            try:
                r, _ = pearsonr(perch_prob[:, c], sed41f_filled[:, c])
            except Exception:
                continue
            agreements.append({
                "class_idx": c, "label": primary[c],
                "perch_auc": aucs_p[c], "sed41f_auc": aucs_s[c],
                "corr": r,
                "complement": aucs_s[c] - aucs_p[c],
            })
        agreements = sorted(agreements, key=lambda x: x["corr"] if not np.isnan(x["corr"]) else 1)
        print(f"\nTop-10 lowest-correlation (highest diversity):")
        print(f"  {'label':<12} {'Perch_AUC':>9} {'SED41f_AUC':>10} {'corr':>8} {'SED−Perch':>10}")
        for r in agreements[:10]:
            if np.isnan(r["corr"]): continue
            print(f"  {r['label']:<12} {r['perch_auc']:>9.3f} {r['sed41f_auc']:>10.3f} "
                  f"{r['corr']:>+8.3f} {r['complement']:>+10.3f}")

        print(f"\nTop-10 biggest SED41f - Perch AUC gain (SED41f specialty):")
        for r in sorted(agreements, key=lambda x: -x["complement"])[:10]:
            print(f"  {r['label']:<12} {r['perch_auc']:>9.3f} {r['sed41f_auc']:>10.3f} "
                  f"{r['corr']:>+8.3f} {r['complement']:>+10.3f}")

        # macro delta by blend ratio
        print(f"\nBlend ratio α (SED41f) macro on 40 classes:")
        for a in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]:
            p = sigmoid((1-a) * perch + a * sed41f_filled)   # logit-level blend
            pm = gauss_block(zs(p), sc_eval, 0.5) if False else p  # keep it simple
            aucs = per_class_auc(Y, p)
            print(f"  α={a:.1f}  macro={macro(aucs):.4f}  (Perch_prob*(1-α) + SED41f*α)")

    # Save
    with open(OUT / "sweep.json", "w") as fp:
        json.dump({
            "grid": {f"{wg[i]:.1f}_{wg[j]:.1f}": float(grid[i,j]) for i in range(len(wg)) for j in range(len(wg)) if not np.isnan(grid[i,j])},
            "best": best,
            "ref_v12": float(macro(per_class_auc(Y, ref_v12_s))),
            "ref_v17": float(macro(per_class_auc(Y, ref_v17_s))),
        }, fp, indent=2)
    print(f"\nSaved → {OUT}/sweep.json")


if __name__ == "__main__":
    main()
