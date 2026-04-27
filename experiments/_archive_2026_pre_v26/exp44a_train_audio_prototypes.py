#!/usr/bin/env python3
"""exp44a — train_audio Perch prototypes for SS impostor detection.

Rationale:
  exp43o built per-species centroid from teacher-c-positive labeled SS windows
  (5-20 clips each, 14 classes qualified).  train_audio has 100-500 CLEAN clips
  per abundant species → much stronger centroid, coverage up to 206 classes.

  Test: does centroid-distance to train_audio prototype detect impostors in
  teacher-c-positive SS groups better than SS-only centroid?  Since train_audio
  is clean solo recordings and SS is field, domain shift penalizes absolute
  recognition (exp22 probe AUC 0.677) but should preserve RELATIVE ranking of
  impostor vs true positive.

Data:
  - exp22_outputs/train_audio_perch.npz : 35549 × 1536 (TF-CPU Perch)
  - exp43a_outputs/perch_ss_all.npz     : 127896 × 1536 (ONNX-GPU Perch)
  NOTE documented embedding drift ≈ −0.037 between these.  Run A (current)
  uses existing caches for quick signal.  If promising, re-extract train_audio
  with ONNX for clean numbers (run B).

Outputs:
  - per-class train_audio prototype (mean Perch emb)
  - per-class impostor-detection AUC using train_audio centroid (labeled SS as test)
  - compare to exp43o SS-only centroid baseline
  - count of qualifying classes (vs 14 in exp43o)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP22 = ROOT / "experiments/exp22_outputs"
EXP43A = ROOT / "experiments/exp43a_outputs"
OUT = ROOT / "experiments/exp44a_outputs"
OUT.mkdir(exist_ok=True)

TAU_SCORE = 0.3
MIN_TP = 3
MIN_IMP = 3
MIN_TA_CLIPS = 5       # need at least 5 clean clips to form a prototype


def load_everything():
    # train_audio Perch cache
    d = np.load(EXP22 / "train_audio_perch.npz", allow_pickle=True)
    ta_emb = d["emb"].astype(np.float32)
    ta_yidx = d["y_idx"].astype(np.int64)
    ta_valid = d["valid"].astype(bool)
    print(f"train_audio emb: {ta_emb.shape}  valid: {ta_valid.sum()}/{len(ta_valid)}")

    # SS data
    ss_emb = np.load(EXP43A / "perch_ss_all.npz")["emb"].astype(np.float32)
    ss_scores = np.load(EXP43A / "perch_ss_all.npz")["scores"].astype(np.float32)
    ss_meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")

    # labeled SS mask + Y
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    rid2lbls = {r.row_id: r.lbls for _, r in sc_g.iterrows()}
    mask = np.zeros(len(ss_meta), dtype=bool)
    Y = np.zeros((len(ss_meta), len(primary)), dtype=np.uint8)
    for i, rid in enumerate(ss_meta["row_id"].values):
        if rid in rid2lbls:
            mask[i] = True
            for l in rid2lbls[rid]:
                if l in l2i: Y[i, l2i[l]] = 1

    # taxonomy + Perch mapping
    tax = pd.read_csv(DATA / "taxonomy.csv")
    perch_sci = set(open(ROOT / "perch_v2/assets/labels.csv").read().strip().split("\n"))
    tax["in_perch"] = tax["scientific_name"].isin(perch_sci)
    return ta_emb, ta_yidx, ta_valid, ss_emb, ss_scores, mask, Y, primary, tax


def build_prototypes(ta_emb, ta_yidx, ta_valid, n_classes=234):
    """Per-class L2-normalized mean embedding (prototype). Returns (C, 1536) and count."""
    protos = np.zeros((n_classes, ta_emb.shape[1]), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        idx = np.where((ta_yidx == c) & ta_valid)[0]
        if len(idx) == 0: continue
        protos[c] = ta_emb[idx].mean(0)
        counts[c] = len(idx)
    # L2 normalize
    norms = np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8
    protos_n = protos / norms
    return protos, protos_n, counts


def impostor_auc_with_proto(ss_emb, ss_scores, Y, mask, protos, counts, in_perch,
                             tau=TAU_SCORE, min_tp=MIN_TP, min_imp=MIN_IMP):
    """Per-class impostor-detection AUC:
       teacher_hi = ss_scores[:, c] > tau for labeled SS windows
       true_pos / impostor split by Y
       distance(w, prototype_c) = cosine distance
       AUC(is_impostor | distance).
    """
    emb_lab = ss_emb[mask]
    Y_lab = Y[mask]
    scores_lab = ss_scores[mask]
    emb_lab_n = emb_lab / (np.linalg.norm(emb_lab, axis=1, keepdims=True) + 1e-8)
    protos_n = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)

    results = []
    for c in range(Y.shape[1]):
        if counts[c] < MIN_TA_CLIPS:
            continue
        teacher_hi = scores_lab[:, c] > tau
        tp_idx = np.where(teacher_hi & (Y_lab[:, c] == 1))[0]
        imp_idx = np.where(teacher_hi & (Y_lab[:, c] == 0))[0]
        if len(tp_idx) < min_tp or len(imp_idx) < min_imp:
            continue
        proto = protos_n[c:c+1]                                    # (1, 1536)
        all_idx = np.concatenate([tp_idx, imp_idx])
        is_imp = np.concatenate([np.zeros(len(tp_idx)), np.ones(len(imp_idx))])
        dists = 1.0 - (emb_lab_n[all_idx] @ proto.T).ravel()
        try:
            auc = roc_auc_score(is_imp, dists)
        except Exception:
            continue
        results.append({
            "class_idx": int(c),
            "in_perch": bool(in_perch[c]),
            "n_train_audio_clips": int(counts[c]),
            "n_tp": int(len(tp_idx)),
            "n_imp": int(len(imp_idx)),
            "auc": float(auc),
        })
    return results


def aggregate(res, key_filter=None):
    if key_filter: res = [r for r in res if key_filter(r)]
    if not res: return {}
    aucs = [r["auc"] for r in res]
    return {
        "n_classes": len(aucs),
        "mean_auc": float(np.mean(aucs)),
        "median_auc": float(np.median(aucs)),
        "frac_above_0.7": float(np.mean([a > 0.7 for a in aucs])),
        "frac_above_0.8": float(np.mean([a > 0.8 for a in aucs])),
    }


def main():
    ta_emb, ta_yidx, ta_valid, ss_emb, ss_scores, mask, Y, primary, tax = load_everything()
    C = Y.shape[1]

    # Per-class Perch mapping flags
    in_perch = np.array([
        tax[tax["primary_label"].astype(str) == p]["in_perch"].iloc[0]
        if (tax["primary_label"].astype(str) == p).any() else False
        for p in primary
    ])
    n_perch = int(in_perch.sum())
    print(f"Classes: {n_perch} Perch-mapped, {C - n_perch} unmapped")

    print("\nBuilding train_audio prototypes...")
    protos, protos_n, counts = build_prototypes(ta_emb, ta_yidx, ta_valid, n_classes=C)
    print(f"  classes with ≥{MIN_TA_CLIPS} train_audio clips: {(counts >= MIN_TA_CLIPS).sum()}")
    print(f"  classes with ≥50 clips: {(counts >= 50).sum()}")
    print(f"  classes with ≥100 clips: {(counts >= 100).sum()}")

    print(f"\nImpostor detection with train_audio prototype (τ={TAU_SCORE}, ≥{MIN_TP} tp, ≥{MIN_IMP} imp)")
    results = impostor_auc_with_proto(ss_emb, ss_scores, Y, mask, protos, counts, in_perch)
    print(f"  qualifying classes: {len(results)}")
    agg_all = aggregate(results)
    print(f"  ALL         : {agg_all}")
    agg_mapped = aggregate(results, key_filter=lambda r: r["in_perch"])
    print(f"  Perch-mapped: {agg_mapped}")
    agg_unmapped = aggregate(results, key_filter=lambda r: not r["in_perch"])
    print(f"  unmapped    : {agg_unmapped}")

    # Distribution by train_audio clip count (is the prototype quality dependent?)
    print(f"\n  AUC by train_audio clip count bucket:")
    for lo, hi, name in [(5, 20, "5-20"), (20, 100, "20-100"), (100, 500, "100-500"), (500, 100000, "500+")]:
        sub = [r for r in results if lo <= r["n_train_audio_clips"] < hi]
        if not sub: continue
        aucs = [r["auc"] for r in sub]
        print(f"    {name:<8}  n_cls={len(sub):3d}  mean={np.mean(aucs):.3f}  median={np.median(aucs):.3f}")

    # Per-class details sorted by AUC (what's the failure modes?)
    print(f"\n  Worst 10 (AUC low → prototype ≠ SS window, likely domain shift):")
    for r in sorted(results, key=lambda x: x["auc"])[:10]:
        print(f"    {primary[r['class_idx']]:<10}  AUC={r['auc']:.3f}  "
              f"n_ta={r['n_train_audio_clips']:4d}  n_tp={r['n_tp']:3d}  n_imp={r['n_imp']:3d}  "
              f"Perch={r['in_perch']}")
    print(f"\n  Best 10:")
    for r in sorted(results, key=lambda x: -x["auc"])[:10]:
        print(f"    {primary[r['class_idx']]:<10}  AUC={r['auc']:.3f}  "
              f"n_ta={r['n_train_audio_clips']:4d}  n_tp={r['n_tp']:3d}  n_imp={r['n_imp']:3d}  "
              f"Perch={r['in_perch']}")

    np.savez_compressed(OUT / "prototypes.npz", protos=protos, counts=counts)
    with open(OUT / "impostor_auc.json", "w") as fp:
        json.dump({"per_class": results, "agg_all": agg_all,
                   "agg_mapped": agg_mapped, "agg_unmapped": agg_unmapped}, fp,
                  indent=2, default=float)
    print(f"\nSaved → {OUT}/")


if __name__ == "__main__":
    main()
