#!/usr/bin/env python3
"""exp89 — Does Perch "know it doesn't know"? Entropy + latent analysis.

Question: when Perch encounters a species OUTSIDE its 14,795 vocab (25
Insecta sonotypes, 3 Amphibia, 2 Mammalia, 1 Reptilia), does its output
distribution / embedding behave differently than on KNOWN species?

If yes → we have a Perch-internal OOD signal that's a NEW universal
source of site-invariance (uses model property, not train-SS lookup).

Row classification (labeled SS, 739 rows):
  KNOWN_HIT     : GT contains a Perch-mapped species, Perch sigmoid > 0.5 on it
  KNOWN_MISS    : GT contains mapped species, Perch top-1 NOT in GT
  UNK_PRESENT   : GT contains ≥1 unmapped species (regardless of mapped)
  BG_NEG        : no positive labels at all

Metrics per row:
  H_234         : Shannon entropy of softmax over our 234 sigmoid scores
  top1_prob     : max sigmoid prob
  top5_mass     : sum of top-5 sigmoid prob / sum-all
  emb_L2        : ||embedding|| (1536-d)
  emb_max       : max abs component
  emb_eff_dim   : participation ratio (sum(x²))² / sum(x⁴) — effective dim
  dist_unk_centroid  : L2 distance to mean embedding of UNK_PRESENT rows
  dist_known_centroid: L2 distance to mean embedding of KNOWN_HIT rows

Per-group: report mean ± std for each metric.
Also: separability test (AUC of each metric for UNK_PRESENT vs KNOWN_HIT).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, primary_labels, DATA, EXP80, N_CLS)


def entropy_normalized(probs, axis=-1):
    """Shannon entropy normalized by log(n_classes), so [0, 1]."""
    eps = 1e-12
    p = probs / (probs.sum(axis=axis, keepdims=True) + eps)
    H = -(p * np.log(p + eps)).sum(axis=axis)
    return H / np.log(probs.shape[axis])


def participation_ratio(emb, axis=-1):
    """Effective dimensionality of vector: PR = (Σx²)² / Σx⁴.
    For uniform distribution PR = n; for one-hot PR = 1. Normalize to [0, 1]."""
    sq = emb ** 2
    return (sq.sum(axis=axis) ** 2) / (sq.sum(axis=axis) ** 2 / emb.shape[axis] + (sq ** 2).sum(axis=axis))


def main():
    print("=== exp89: Perch entropy + latent analysis (does it know it doesn't know?) ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()  # sigmoid scores (n, 234)
    perch_emb = load_perch_emb_labeled()       # raw emb (n, 1536)

    # Identify mapped vs unmapped species per Perch's perspective.
    # Perch outputs 0 (logit) for unmapped → sigmoid = 0.5 constant.
    # We can detect: cols where perch_prob is constant 0.5 across many rows = unmapped
    col_var = perch_prob.var(axis=0)
    # Threshold: var < 1e-6 → essentially constant = unmapped
    unmapped_mask = col_var < 1e-6
    mapped_idx = np.where(~unmapped_mask)[0]
    unmapped_idx = np.where(unmapped_mask)[0]
    print(f"Mapped species (Perch outputs varying signal): {len(mapped_idx)}/{N_CLS}")
    print(f"Unmapped species (Perch constant): {len(unmapped_idx)}/{N_CLS}")
    print(f"Unmapped species labels: {[primary[i] for i in unmapped_idx]}")

    # === Row classification ===
    # KNOWN_HIT: GT has ≥1 mapped species, Perch sigmoid on at least one of them > 0.5
    # KNOWN_MISS: GT has ≥1 mapped, but Perch top-1 (over mapped) NOT in GT
    # UNK_PRESENT: GT contains any unmapped species
    # BG_NEG: no positives at all

    classes = []
    for i in range(len(sc_g)):
        gt = np.where(Y[i] == 1)[0]
        if len(gt) == 0:
            classes.append("BG_NEG"); continue
        gt_unmapped = np.intersect1d(gt, unmapped_idx)
        gt_mapped = np.intersect1d(gt, mapped_idx)
        # Cleaner partition: presence-of-unmapped, presence-of-mapped
        if len(gt_unmapped) > 0 and len(gt_mapped) == 0:
            classes.append("UNK_ONLY")        # only unmapped species — Perch fundamentally blind
        elif len(gt_unmapped) > 0 and len(gt_mapped) > 0:
            classes.append("UNK_MIXED")       # both present
        elif len(gt_mapped) > 0 and len(gt_unmapped) == 0:
            classes.append("KNOWN_ONLY")      # only mapped species — Perch should handle
        else:
            classes.append("BG_NEG")
    classes = np.array(classes)
    print(f"\nRow classification:")
    for c in ["KNOWN_ONLY", "UNK_ONLY", "UNK_MIXED", "BG_NEG"]:
        print(f"  {c:<14} {(classes == c).sum()}")

    # === Per-row metrics ===
    H_234 = entropy_normalized(perch_prob, axis=-1)
    top1_prob = perch_prob.max(axis=-1)
    sorted_p = np.sort(perch_prob, axis=-1)[:, ::-1]
    top5_mass = sorted_p[:, :5].sum(axis=-1) / (sorted_p.sum(axis=-1) + 1e-12)
    emb_L2 = np.linalg.norm(perch_emb, axis=-1)
    emb_max = np.abs(perch_emb).max(axis=-1)
    emb_var = perch_emb.var(axis=-1)

    # Centroid distances (TRAIN split only to avoid leak)
    tr_mask = sc_g.split.values == "train"
    unk_train = ((classes == "UNK_ONLY") | (classes == "UNK_MIXED")) & tr_mask
    hit_train = (classes == "KNOWN_ONLY") & tr_mask
    unk_centroid = perch_emb[unk_train].mean(axis=0) if unk_train.sum() > 0 else perch_emb.mean(axis=0)
    hit_centroid = perch_emb[hit_train].mean(axis=0) if hit_train.sum() > 0 else perch_emb.mean(axis=0)
    dist_unk = np.linalg.norm(perch_emb - unk_centroid, axis=-1)
    dist_hit = np.linalg.norm(perch_emb - hit_centroid, axis=-1)
    dist_ratio = dist_unk / (dist_hit + 1e-6)  # < 1 means closer to UNK centroid

    # Build per-row dataframe
    df = pd.DataFrame({
        "row_id": sc_g.row_id.values,
        "site": sc_g.site.values,
        "split": sc_g.split.values,
        "class": classes,
        "H_234": H_234,
        "top1_prob": top1_prob,
        "top5_mass": top5_mass,
        "emb_L2": emb_L2,
        "emb_max": emb_max,
        "emb_var": emb_var,
        "dist_unk": dist_unk,
        "dist_hit": dist_hit,
        "dist_ratio": dist_ratio,
    })

    print("\n=== Per-group statistics (mean ± std) ===")
    metrics = ["H_234", "top1_prob", "top5_mass", "emb_L2", "emb_max", "emb_var", "dist_unk", "dist_hit", "dist_ratio"]
    print(f"  {'group':<14} {'n':>5}  " + " ".join(f"{m:>12}" for m in metrics))
    for c in ["KNOWN_ONLY", "UNK_ONLY", "UNK_MIXED", "BG_NEG"]:
        sub = df[df["class"] == c]
        if len(sub) == 0: continue
        means = [sub[m].mean() for m in metrics]
        stds = [sub[m].std() for m in metrics]
        s = " ".join(f"{mn:>6.3f}±{sd:.3f}" for mn, sd in zip(means, stds))
        print(f"  {c:<14} ({len(sub):>4})  {s}")

    # === Separability AUC: UNK_ONLY (definitely Perch-blind) vs KNOWN_ONLY (definitely in vocab) ===
    print("\n=== Separability AUC: each metric for UNK_ONLY (1) vs KNOWN_ONLY (0) ===")
    mask = (classes == "UNK_ONLY") | (classes == "KNOWN_ONLY")
    y = (classes[mask] == "UNK_ONLY").astype(int)
    print(f"  n_pos (UNK_ONLY)={y.sum()}, n_neg (KNOWN_ONLY)={len(y)-y.sum()}")
    for m in metrics:
        try:
            score = df[mask][m].values
            auc = roc_auc_score(y, score)
            # Higher AUC = metric is HIGHER on UNK rows
            interpret = "↑ on UNK" if auc > 0.55 else ("↓ on UNK" if auc < 0.45 else "no signal")
            print(f"  {m:<14} AUC = {auc:.4f}  ({interpret})")
        except Exception as e:
            print(f"  {m:<14} ERROR: {e}")

    # === Per-site sanity: is UNK_ONLY just S08/S15/S19/S23 fingerprint? ===
    print("\n=== UNK_ONLY rows by site (test if 'UNK signal' is actually 'Insecta site fingerprint') ===")
    unk_only_sites = df[df["class"] == "UNK_ONLY"].site.value_counts()
    known_only_sites = df[df["class"] == "KNOWN_ONLY"].site.value_counts()
    all_sites = sorted(set(unk_only_sites.index) | set(known_only_sites.index))
    print(f"  {'site':<6} {'UNK_ONLY':>10} {'KNOWN_ONLY':>12}")
    for s in all_sites:
        u = unk_only_sites.get(s, 0); k = known_only_sites.get(s, 0)
        print(f"  {s:<6} {u:>10} {k:>12}")

    # Save
    df.to_csv(EXP80 / "exp89_perch_latent.csv", index=False)
    print(f"\nSaved → {EXP80}/exp89_perch_latent.csv")


if __name__ == "__main__":
    main()
