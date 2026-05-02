#!/usr/bin/env python3
"""exp130 — Cross-source verified pseudo-labels.

Problem with exp129: pseudo from v33 → train new SED → CIRCULAR distillation.
v33's noise (saturating Aves, cluster confusion) gets amplified. False
negatives (multi-label rows missing true positives) hurt Aves AUC.

Solution: verify each pseudo-positive (row, class) with INDEPENDENT signals
that don't share v33's biases:

  1. Train_audio Perch embedding similarity:
     For each class c with train_audio examples, compute centroid in Perch
     embedding space. For unlabeled SS row r, check cosine sim of its Perch
     emb vs centroid. > τ → confirmed.
     Independent because: train_audio = xeno-canto/iNat recordings (NOT
     the SS that v33 was tuned on). Different acoustic conditions.

  2. Per-class score (raw, no v33 transform):
     If Perch[r, c] > τ_perch → independent confirmation.

  3. exp50 score (separate model, separately trained):
     If exp50[r, c] > τ_exp50 → independent confirmation.

Refinement strategy:
  KEEP pseudo-positive if v33[r,c] > 0.5 AND ≥1 independent signal confirms.
  DROP if no independent signal supports.
  ADD pseudo-positive (boost coverage) for rare taxa (Mammalia/Insecta) when
  emb similarity is strong (> τ_emb_strict) regardless of v33.

This produces a CLEANER pseudo set with PRECISION over RECALL.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
PERCH_UNLABELED = ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz"
EXP50_UNLABELED = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
V33_SCORES = ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz"
TA_2026_PERCH = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
TA_2025_PERCH = ROOT / "experiments/_data_pipelines/exp120_outputs/ta25_perch.npz"
PSEUDO_REFINED = DATA / "pseudo_soundscapes_labels_refined.csv"
OUT = ROOT / "experiments/_data_pipelines/exp130_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA / "pseudo_soundscapes_labels_verified.csv"

N_CLS = 234

# Verification thresholds
TAU_V33 = 0.5            # v33 score threshold (initial pseudo filter)
TAU_PERCH = 0.4          # raw Perch sigmoid threshold
TAU_EXP50 = 0.3          # exp50 threshold
TAU_EMB_SIM = 0.5        # Perch emb cosine similarity threshold
TAU_EMB_SIM_STRICT = 0.7 # for rare-taxa boost
MIN_TA_PER_CLASS = 5     # need ≥k train_audio examples to build centroid

RARE_TAXA = {"Mammalia", "Insecta", "Reptilia"}


def load_taxon_array(primary):
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    return np.array([sp2tax.get(p, "Aves") for p in primary])


def cosine_sim(a, b):
    """a: (N, D), b: (M, D) → (N, M)"""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_n @ b_n.T


def main():
    print("=== exp130: Cross-source verified pseudo-labels ===\n", flush=True)

    # 1. Load unlabeled SS data
    print("Loading caches...")
    perch_unlab = np.load(PERCH_UNLABELED, mmap_mode="r")
    perch_emb_unlab = np.array(perch_unlab["emb"])
    perch_logits_unlab = np.array(perch_unlab["scores"])
    perch_probs_unlab = (1.0 / (1.0 + np.exp(-np.clip(perch_logits_unlab, -30, 30)))).astype(np.float32)
    print(f"  Unlabeled SS Perch emb: {perch_emb_unlab.shape}")

    exp50_unlab = np.load(EXP50_UNLABELED, allow_pickle=True)
    exp50_scores = exp50_unlab["scores"]
    filenames = exp50_unlab["filenames"].astype(str)
    row_ids = exp50_unlab["row_ids"].astype(str)
    print(f"  Unlabeled SS exp50: {exp50_scores.shape}")

    v33_data = np.load(V33_SCORES, allow_pickle=True)
    v33 = v33_data["v33"].astype(np.float32)
    print(f"  v33 scores: {v33.shape}")

    # 2. Build train_audio centroids in Perch embedding space
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    sp2idx = {s: i for i, s in enumerate(primary)}
    sp_taxon = load_taxon_array(primary)

    print("\nBuilding per-class train_audio centroids...")
    ta_2026 = np.load(TA_2026_PERCH)
    ta_emb_26 = ta_2026["emb"]
    ta_y_26 = ta_2026["y_idx"]
    ta_valid_26 = ta_2026["valid"]
    valid_26 = (ta_valid_26 == 1) & (ta_y_26 >= 0) & (ta_y_26 < N_CLS)

    # Combine 2026 + 2025 if present
    ta_emb_all = ta_emb_26[valid_26]
    ta_y_all = ta_y_26[valid_26]
    if TA_2025_PERCH.exists():
        ta_2025 = np.load(TA_2025_PERCH)
        ta_emb_25 = ta_2025["emb"]
        ta_y_25 = ta_2025["y_idx"]
        ta_valid_25 = ta_2025["valid"]
        valid_25 = (ta_valid_25 == 1) & (ta_y_25 >= 0) & (ta_y_25 < N_CLS)
        ta_emb_all = np.concatenate([ta_emb_all, ta_emb_25[valid_25]], axis=0)
        ta_y_all = np.concatenate([ta_y_all, ta_y_25[valid_25]], axis=0)
    print(f"  Train_audio total: {len(ta_emb_all)}")

    # Per-class centroid + count
    centroids = np.zeros((N_CLS, 1536), dtype=np.float32)
    n_per_class = np.zeros(N_CLS, dtype=np.int32)
    for c in range(N_CLS):
        mask = ta_y_all == c
        n_per_class[c] = mask.sum()
        if mask.sum() > 0:
            centroids[c] = ta_emb_all[mask].mean(axis=0)
    print(f"  Classes with ≥{MIN_TA_PER_CLASS} train_audio: {(n_per_class >= MIN_TA_PER_CLASS).sum()} / {N_CLS}")
    print(f"  Classes with 0 train_audio: {(n_per_class == 0).sum()}")
    has_centroid = n_per_class >= MIN_TA_PER_CLASS

    # 3. Compute Perch emb cosine similarity (unlabeled rows vs class centroids)
    print("\nComputing emb similarity (127k rows × 234 centroids)...")
    # Normalize once
    unlab_emb_n = perch_emb_unlab / (np.linalg.norm(perch_emb_unlab, axis=1, keepdims=True) + 1e-9)
    cent_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    emb_sim = unlab_emb_n @ cent_n.T  # (n_rows, N_CLS)
    emb_sim = emb_sim.astype(np.float32)
    print(f"  emb_sim: {emb_sim.shape}, range [{emb_sim.min():.3f}, {emb_sim.max():.3f}], mean {emb_sim.mean():.3f}")

    # 4. Per-class similarity stats — diagnostic
    print("\n  Per-taxon mean emb_sim distribution:")
    for tx in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        mask = sp_taxon == tx
        valid_classes = has_centroid & mask
        if valid_classes.sum() == 0:
            print(f"    {tx}: no centroids")
            continue
        # Mean cosine sim across rows for this taxon's classes
        avg_sim = emb_sim[:, valid_classes].mean()
        max_sim = emb_sim[:, valid_classes].max()
        n_valid = valid_classes.sum()
        print(f"    {tx}: {n_valid} classes with centroid, avg_sim {avg_sim:.3f}, max_sim {max_sim:.3f}")

    # 5. Build verified pseudo-labels
    print(f"\n=== Building verified pseudo-labels ===")
    print(f"  τ_v33={TAU_V33}, τ_emb_sim={TAU_EMB_SIM}, τ_perch={TAU_PERCH}, τ_exp50={TAU_EXP50}")

    # For each (row, class), evaluate independent signals
    # Initial filter: v33 > τ_v33
    initial = v33 > TAU_V33
    print(f"  Initial v33 > {TAU_V33}: {int(initial.sum())} (row, class) entries")

    # Independent signal counts per (row, class)
    n_signals = np.zeros_like(v33, dtype=np.int8)
    n_signals += (perch_probs_unlab > TAU_PERCH).astype(np.int8)
    n_signals += (exp50_scores > TAU_EXP50).astype(np.int8)
    # emb_sim only meaningful for classes with centroid
    has_centroid_mask = has_centroid[None, :]  # (1, N_CLS)
    emb_confirm = (emb_sim > TAU_EMB_SIM) & has_centroid_mask
    n_signals += emb_confirm.astype(np.int8)
    print(f"  At τ thresholds, distribution of n_independent_signals (initial filter ON):")
    initial_signals = n_signals[initial]
    for k in range(4):
        n_k = (initial_signals == k).sum()
        print(f"    n_signals={k}: {int(n_k)} entries ({100*n_k/max(initial.sum(),1):.1f}%)")

    # Verified: v33 > 0.5 AND ≥1 independent signal
    verified = initial & (n_signals >= 1)
    print(f"\n  Verified pseudo-positives (v33 + ≥1 indep): {int(verified.sum())}")

    # 6. Boost rare taxa using emb similarity (independent of v33)
    print(f"\n  Rare-taxa boost: emb_sim > {TAU_EMB_SIM_STRICT} for Mammalia/Insecta/Reptilia...")
    rare_mask_class = np.array([sp_taxon[c] in RARE_TAXA for c in range(N_CLS)])
    rare_boost = (emb_sim > TAU_EMB_SIM_STRICT) & has_centroid_mask & rare_mask_class[None, :]
    # Only add NEW (not already verified)
    new_rare = rare_boost & ~verified
    print(f"  New rare-taxa pseudo-positives via emb similarity: {int(new_rare.sum())}")

    # Combined final pseudo
    final = verified | new_rare
    print(f"  Total final verified pseudo: {int(final.sum())}")

    # Per-taxon final
    print(f"\n  Final per-taxon:")
    for tx in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        mask = sp_taxon == tx
        n_t = final[:, mask].sum()
        n_classes = (final[:, mask].sum(axis=0) > 0).sum()
        print(f"    {tx}: {int(n_t)} entries, {int(n_classes)} classes")

    # 7. Save CSV
    print("\nBuilding verified CSV...")
    rows_csv = []
    for r_idx in range(len(v33)):
        positive_classes = np.where(final[r_idx])[0]
        if len(positive_classes) == 0: continue
        for c in positive_classes:
            # Parse end_sec from row_id: "BC2026_..._5"
            rid = row_ids[r_idx]
            # extract trailing number
            import re
            m = re.search(r"_(\d+)$", str(rid))
            end_sec = int(m.group(1)) if m else 0
            rows_csv.append({
                "filename": filenames[r_idx],
                "start": str(end_sec - 5),
                "end": str(end_sec),
                "primary_label": primary[c],
                "v33_score": float(v33[r_idx, c]),
                "perch_score": float(perch_probs_unlab[r_idx, c]),
                "exp50_score": float(exp50_scores[r_idx, c]),
                "emb_sim": float(emb_sim[r_idx, c]),
                "n_signals": int(n_signals[r_idx, c]),
            })
    df_final = pd.DataFrame(rows_csv)
    print(f"  Total CSV rows: {len(df_final)}")
    df_final.to_csv(OUT_CSV, index=False)
    print(f"  Saved → {OUT_CSV}")

    # Top-15 in final
    print("\n  Top-15 most-fired in VERIFIED pseudo:")
    for lbl, n in df_final.primary_label.value_counts().head(15).items():
        taxon = next((sp_taxon[i] for i, p in enumerate(primary) if p == lbl), "?")
        print(f"    {lbl:<14} {taxon:<10} {n:>5}")


if __name__ == "__main__":
    main()
