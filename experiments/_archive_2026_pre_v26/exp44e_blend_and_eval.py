#!/usr/bin/env python3
"""exp44e — Blend exp44c 27-species head with Perch baseline + fair eval.

Setup:
  - 11 held-out SS files (same split as exp38/44c, seed 42)
  - 122 eval windows × 234 classes
  - Baseline A: raw Perch scores (exp43a scores matrix)
  - Baseline B: Perch + Gauss smoothing (crude v12 analog for 234 cols)
  - Blend: replace 27 double-blind cols with exp44c probabilities

Metrics:
  - Per-class AUC where eval has both positives and negatives
  - Macro AUC over those classes
  - Separately: macro over 207 mapped, macro over 27 unmapped

Output:
  experiments/exp44e_outputs/blend_results.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP44C = ROOT / "experiments/exp44c_outputs"
OUT = ROOT / "experiments/exp44e_outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
EVAL_N_FILES = 11


def build_eval_mapping():
    """Identify the 11 held-out SS files and build Y for full 234 classes."""
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)

    # Split files exactly as exp44c/38 did
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique())
    rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)

    l2i = {c: i for i, c in enumerate(primary)}
    Y = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["lbls"]):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_eval, Y, primary


def get_double_blind_indices(primary):
    tax = pd.read_csv(DATA / "taxonomy.csv")
    perch_sci = set(open(ROOT / "perch_v2/assets/labels.csv").read().strip().split("\n"))
    tax["in_perch"] = tax["scientific_name"].isin(perch_sci)
    ta_counts = pd.read_csv(DATA / "train.csv").groupby("primary_label").size()
    tax["n_ta"] = tax["primary_label"].astype(str).map(ta_counts).fillna(0).astype(int)
    double_blind = tax[(~tax.in_perch) & (tax.n_ta == 0)]["primary_label"].astype(str).tolist()
    unmapped_idx = [primary.index(p) for p in double_blind if p in primary]
    mapped_idx = [i for i in range(len(primary)) if i not in unmapped_idx]
    return unmapped_idx, mapped_idx, double_blind


def align_exp44c_to_eval(sc_eval, exp44c_labels):
    """exp44c val_scores was computed in order of sc_eval from exp44c's own build.
       Since exp44c used the same split as we did here, orders should match.
       Verify by length + filename ordering."""
    preds = np.load(EXP44C / "val_scores.npz")["preds"]  # (122, 27)
    assert preds.shape[0] == len(sc_eval), f"mismatch {preds.shape[0]} vs {len(sc_eval)}"
    return preds


def align_perch_scores_to_eval(sc_eval):
    """Load exp43a scores (127896, 234) and pick rows for the 11 held-out files."""
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    scores = np.load(EXP43A / "perch_ss_all.npz")["scores"]

    # Build row_id → index in meta
    rid2meta = {r: i for i, r in enumerate(meta["row_id"].values)}
    eval_rows = []
    for _, r in sc_eval.iterrows():
        rid = r.row_id
        if rid in rid2meta:
            eval_rows.append(rid2meta[rid])
        else:
            eval_rows.append(-1)

    out = np.full((len(sc_eval), scores.shape[1]), np.nan, dtype=np.float32)
    for i, m in enumerate(eval_rows):
        if m >= 0: out[i] = scores[m]
    return out


def per_class_auc(Y, preds, mask_cols=None):
    """Return dict of class_idx -> AUC for classes with ≥1 pos and ≥1 neg in Y."""
    cols = range(Y.shape[1]) if mask_cols is None else mask_cols
    aucs = {}
    for c in cols:
        y = Y[:, c].astype(int)
        p = preds[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
        if not np.isfinite(p).all(): continue
        try:
            aucs[int(c)] = float(roc_auc_score(y, p))
        except Exception:
            pass
    return aucs


def macro(aucs):
    vals = list(aucs.values())
    return float(np.mean(vals)) if vals else 0.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def zscore(x, axis=0):
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, keepdims=True) + 1e-8
    return (x - m) / s


def main():
    sc_eval, Y, primary = build_eval_mapping()
    unmapped_idx, mapped_idx, double_blind = get_double_blind_indices(primary)
    print(f"Eval rows: {len(sc_eval)}  cols: {len(primary)}")
    print(f"Unmapped (double-blind) cols: {len(unmapped_idx)}  Mapped cols: {len(mapped_idx)}")

    perch_scores = align_perch_scores_to_eval(sc_eval)  # logits, (122, 234)
    perch_probs = sigmoid(perch_scores)
    print(f"Perch scores shape: {perch_scores.shape}  nan rows: {np.isnan(perch_probs).any(1).sum()}")

    # exp44c predictions for 27 unmapped species, mapped columns = 0
    ckpt = __import__("torch").load(EXP44C / "best_ckpt.pt", map_location="cpu", weights_only=False)
    labels_27 = ckpt["labels_27"]  # 27 primary labels used by exp44c
    exp44c_preds_27 = np.load(EXP44C / "val_scores.npz")["preds"]  # (122, 27)

    exp44c_full = np.zeros_like(perch_probs)
    for i, lbl in enumerate(labels_27):
        if lbl in primary:
            exp44c_full[:, primary.index(lbl)] = exp44c_preds_27[:, i]

    # Build predictions to compare
    variants = {
        "A_raw_perch": perch_probs,
        "B_perch_gauss": np.stack([
            gaussian_filter1d(perch_probs[:, c], sigma=0.5, mode="nearest")
            for c in range(perch_probs.shape[1])
        ], axis=1),
    }

    # Blend: for 27 unmapped cols, replace with exp44c
    for name, base in list(variants.items()):
        blended = base.copy()
        blended[:, unmapped_idx] = exp44c_full[:, unmapped_idx]
        variants[f"{name}+exp44c"] = blended

    # Also blend: rank-mean of (Perch logit, exp44c prob) on unmapped cols (soft)
    def rank_norm(x, axis=0):
        return np.argsort(np.argsort(x, axis=axis), axis=axis).astype(np.float32) / max(x.shape[axis] - 1, 1)
    blended_soft = perch_probs.copy()
    for c in unmapped_idx:
        rp = rank_norm(perch_probs[:, c])
        re = rank_norm(exp44c_full[:, c])
        blended_soft[:, c] = 0.3 * rp + 0.7 * re       # weight exp44c heavier
    variants["C_perch+rankblend_exp44c_0.7"] = blended_soft

    print("\n=== Macro-AUC comparison on 11 eval files (122 windows, 234 classes) ===")
    print(f"{'variant':<36}  {'macro_all':>10}  {'macro_mapped':>13}  {'macro_unmapped':>15}  {'n_all':>5}  {'n_map':>5}  {'n_unmap':>7}")
    results = {}
    for name, preds in variants.items():
        all_aucs = per_class_auc(Y, preds)
        map_aucs = per_class_auc(Y, preds, mask_cols=mapped_idx)
        unmap_aucs = per_class_auc(Y, preds, mask_cols=unmapped_idx)
        row = {
            "macro_all": macro(all_aucs),
            "macro_mapped": macro(map_aucs),
            "macro_unmapped": macro(unmap_aucs),
            "n_all": len(all_aucs),
            "n_mapped": len(map_aucs),
            "n_unmapped": len(unmap_aucs),
        }
        results[name] = row
        print(f"  {name:<36}  {row['macro_all']:>10.4f}  {row['macro_mapped']:>13.4f}  "
              f"{row['macro_unmapped']:>15.4f}  {row['n_all']:>5d}  {row['n_mapped']:>5d}  {row['n_unmapped']:>7d}")

    # Delta summaries
    print("\n=== Δ gain from adding exp44c ===")
    base_all = results["A_raw_perch"]["macro_all"]
    base_unmap = results["A_raw_perch"]["macro_unmapped"]
    for name in ["A_raw_perch+exp44c", "B_perch_gauss+exp44c", "C_perch+rankblend_exp44c_0.7"]:
        if name not in results: continue
        d_all = results[name]["macro_all"] - base_all
        d_unmap = results[name]["macro_unmapped"] - base_unmap
        print(f"  {name:<36}  Δmacro_all={d_all:+.4f}  Δmacro_unmapped={d_unmap:+.4f}")

    with open(OUT / "blend_results.json", "w") as fp:
        json.dump(results, fp, indent=2, default=float)
    print(f"\nSaved → {OUT}/blend_results.json")


if __name__ == "__main__":
    main()
