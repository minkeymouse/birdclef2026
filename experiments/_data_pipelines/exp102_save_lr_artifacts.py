#!/usr/bin/env python3
"""exp102 — Train and save LR detectors as Kaggle artifacts.

Saves:
  model-weights/lr_fp_detector.npz   — coefficients + standardizer
  model-weights/lr_fn_detector.npz   — coefficients + standardizer
  model-weights/lr_correction_meta.npz — feature names, candidate_classes,
                                          alpha, beta hyperparameters

These are loaded in the Kaggle notebook to apply per-(row, class)
correction after v33's file-max coherence step.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "_audits_post_v26"))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend


def get_cached(name): return np.load(EXP80 / name)["scores"]


def main():
    print("=== exp102: train + save LR detectors ===\n")
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")

    # v33 baseline
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # File-level stats for exp50
    file_mean = np.zeros_like(exp50); file_std = np.zeros_like(exp50)
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = exp50[idx]
        m = sub.mean(0); sd = sub.std(0)
        for ii in idx: file_mean[ii] = m; file_std[ii] = sd

    col_var = perch_prob.var(axis=0)
    mapped_idx = np.where(col_var >= 1e-6)[0]
    aves_mask = sp_taxon == "Aves"
    candidate_classes = [c for c in range(N_CLS)
                          if aves_mask[c] and c in mapped_idx
                          and Y[:, c].sum() >= 5 and (Y[:, c] == 0).sum() >= 50]
    print(f"Candidate Aves classes: {len(candidate_classes)}")

    # 9 universal features (in fixed order)
    FEATURE_NAMES = [
        "perch_on_c", "exp50_on_c", "exp59_on_c", "perch_sed_disagree",
        "perch_low_sed_high", "file_mean_e50_c", "file_std_e50_c",
        "file_uniform_e50_c", "v33_on_c",
    ]

    def make_features(i, c):
        return np.array([
            perch_prob[i, c], exp50[i, c], exp59[i, c],
            abs(perch_prob[i, c] - exp50[i, c]),
            max(0, exp50[i, c] - perch_prob[i, c]),
            file_mean[i, c], file_std[i, c],
            1.0 - (file_std[i, c] / (file_mean[i, c] + 1e-6)),
            v33[i, c],
        ])

    # Build train rows for both detectors using ALL labeled data (train + eval)
    # Ah wait — to avoid leak, train on TRAIN only. Final LR coefficients must
    # not see eval rows.
    fp_X, fp_y = [], []
    fn_X, fn_y = [], []
    for i in range(len(sc_g)):
        if sc_g.iloc[i].split != "train": continue
        for c in candidate_classes:
            f = make_features(i, c)
            v = v33[i, c]
            if v > 0.5:
                fp_X.append(f)
                fp_y.append(int(Y[i, c] == 0))
            else:
                fn_X.append(f)
                fn_y.append(int(Y[i, c] == 1))
    fp_X = np.array(fp_X); fp_y = np.array(fp_y)
    fn_X = np.array(fn_X); fn_y = np.array(fn_y)
    print(f"FP train pairs: {len(fp_X)} (FP={fp_y.sum()}, TP={(fp_y==0).sum()})")
    print(f"FN train pairs: {len(fn_X)} (FN={fn_y.sum()}, TN={(fn_y==0).sum()})")

    # Fit on TRAIN only
    sc_fp = StandardScaler().fit(fp_X)
    sc_fn = StandardScaler().fit(fn_X)
    clf_fp = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42).fit(sc_fp.transform(fp_X), fp_y)
    clf_fn = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42).fit(sc_fn.transform(fn_X), fn_y)

    # Verify on eval pairs
    ev_fp_X, ev_fp_y = [], []
    ev_fn_X, ev_fn_y = [], []
    for i in range(len(sc_g)):
        if sc_g.iloc[i].split != "eval": continue
        for c in candidate_classes:
            f = make_features(i, c)
            v = v33[i, c]
            if v > 0.5:
                ev_fp_X.append(f); ev_fp_y.append(int(Y[i, c] == 0))
            else:
                ev_fn_X.append(f); ev_fn_y.append(int(Y[i, c] == 1))
    ev_fp_X = np.array(ev_fp_X); ev_fp_y = np.array(ev_fp_y)
    ev_fn_X = np.array(ev_fn_X); ev_fn_y = np.array(ev_fn_y)
    auc_fp_ev = roc_auc_score(ev_fp_y, clf_fp.predict_proba(sc_fp.transform(ev_fp_X))[:, 1])
    auc_fn_ev = roc_auc_score(ev_fn_y, clf_fn.predict_proba(sc_fn.transform(ev_fn_X))[:, 1])
    print(f"\nFP detector Eval AUC: {auc_fp_ev:.4f}")
    print(f"FN detector Eval AUC: {auc_fn_ev:.4f}")

    # Save artifacts
    MW = ROOT / "model-weights"
    MW.mkdir(exist_ok=True, parents=True)

    np.savez_compressed(MW / "lr_fp_detector.npz",
                         coef=clf_fp.coef_[0].astype(np.float32),
                         intercept=clf_fp.intercept_[0].astype(np.float32),
                         scaler_mean=sc_fp.mean_.astype(np.float32),
                         scaler_scale=sc_fp.scale_.astype(np.float32))
    np.savez_compressed(MW / "lr_fn_detector.npz",
                         coef=clf_fn.coef_[0].astype(np.float32),
                         intercept=clf_fn.intercept_[0].astype(np.float32),
                         scaler_mean=sc_fn.mean_.astype(np.float32),
                         scaler_scale=sc_fn.scale_.astype(np.float32))
    np.savez_compressed(MW / "lr_correction_meta.npz",
                         feature_names=np.array(FEATURE_NAMES),
                         candidate_classes=np.array(candidate_classes, dtype=np.int32),
                         primary_labels=np.array(primary),
                         alpha_fn=np.float32(0.3),    # FN boost strength
                         beta_fp=np.float32(0.1),     # FP suppress strength
                         note=np.array(["Best class-A: V_A α=0.3 β=0.1: macro +0.0035, sp_row 0.9996, Aves +0.016"]))
    print(f"\nSaved 3 artifacts to {MW}/lr_*.npz")
    for f in ["lr_fp_detector.npz", "lr_fn_detector.npz", "lr_correction_meta.npz"]:
        print(f"  {f}: {(MW / f).stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
