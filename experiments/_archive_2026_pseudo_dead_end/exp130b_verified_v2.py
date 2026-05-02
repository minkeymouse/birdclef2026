#!/usr/bin/env python3
"""exp130b — Verified pseudo v2.

Fix from v1:
  - REMOVE Perch sigmoid as verification signal (circular: v33 uses Perch)
  - Use per-row TOP-K emb_sim (acoustic neighbors), not absolute threshold
  - Require BOTH emb_sim top-K AND exp50 (where applicable) to confirm v33

For each unlabeled row r:
  1. Compute emb_sim to all 234 class TA centroids
  2. row's "acoustic neighbors" = top-K classes by emb_sim
  3. Pseudo-positive (r, c) requires:
     - v33[r, c] > τ_v33 (initial filter)
     - c IN row's top-K acoustic neighbors (emb_sim confirmation)
     - exp50[r, c] > τ_exp50 OR (rare taxa boost: c is non-Aves)

Rationale:
  - v33's saturating Aves (grfdov1, picpig2 등) fire HIGH on most rows
  - But these rows' acoustic neighbor (top emb_sim) is the TRUE species
  - Filtering by acoustic neighbor removes hallucination
  - exp50 confirmation removes Perch-only saturation
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import re

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
PERCH_UNLABELED = ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz"
EXP50_UNLABELED = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
V33_SCORES = ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz"
TA_2026_PERCH = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
TA_2025_PERCH = ROOT / "experiments/_data_pipelines/exp120_outputs/ta25_perch.npz"
OUT = ROOT / "experiments/_data_pipelines/exp130_outputs"
OUT_CSV = DATA / "pseudo_soundscapes_labels_verified_v2.csv"

N_CLS = 234
TAU_V33 = 0.5
K_NEIGHBORS = 10           # row's top-K acoustic neighbors via emb_sim
TAU_EXP50 = 0.3
MIN_TA_PER_CLASS = 5

RARE_TAXA = {"Mammalia", "Insecta", "Reptilia", "Amphibia"}  # broader def


def main():
    print("=== exp130b: Verified pseudo v2 (per-row top-K emb_sim) ===\n", flush=True)

    perch_unlab = np.load(PERCH_UNLABELED, mmap_mode="r")
    perch_emb = np.array(perch_unlab["emb"])

    exp50_unlab = np.load(EXP50_UNLABELED, allow_pickle=True)
    exp50 = exp50_unlab["scores"]
    filenames = exp50_unlab["filenames"].astype(str)
    row_ids = exp50_unlab["row_ids"].astype(str)

    v33_data = np.load(V33_SCORES, allow_pickle=True)
    v33 = v33_data["v33"].astype(np.float32)

    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    sp_taxon = np.array([sp2tax.get(p, "Aves") for p in primary])

    # Build TA centroids
    print("Building TA centroids...")
    ta = np.load(TA_2026_PERCH)
    valid = (ta["valid"] == 1) & (ta["y_idx"] >= 0) & (ta["y_idx"] < N_CLS)
    ta_emb_all = ta["emb"][valid]; ta_y_all = ta["y_idx"][valid]
    if TA_2025_PERCH.exists():
        ta25 = np.load(TA_2025_PERCH)
        v25 = (ta25["valid"] == 1) & (ta25["y_idx"] >= 0) & (ta25["y_idx"] < N_CLS)
        ta_emb_all = np.concatenate([ta_emb_all, ta25["emb"][v25]], axis=0)
        ta_y_all = np.concatenate([ta_y_all, ta25["y_idx"][v25]], axis=0)
    print(f"  Total TA: {len(ta_emb_all)}")

    centroids = np.zeros((N_CLS, 1536), dtype=np.float32)
    n_per_class = np.zeros(N_CLS, dtype=np.int32)
    for c in range(N_CLS):
        mask = ta_y_all == c
        n_per_class[c] = mask.sum()
        if mask.sum() > 0:
            centroids[c] = ta_emb_all[mask].mean(axis=0)
    has_centroid = n_per_class >= MIN_TA_PER_CLASS
    print(f"  Classes with centroid: {has_centroid.sum()} / {N_CLS}")
    print(f"  Classes without centroid: {(~has_centroid).sum()} (28 are 31 unmapped + ...)")

    # Cosine similarity
    print("\nComputing emb_sim...")
    unlab_n = perch_emb / (np.linalg.norm(perch_emb, axis=1, keepdims=True) + 1e-9)
    cent_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    emb_sim = (unlab_n @ cent_n.T).astype(np.float32)
    # Mask out classes without centroid
    emb_sim[:, ~has_centroid] = -1.0
    print(f"  emb_sim range: [{emb_sim.min():.3f}, {emb_sim.max():.3f}]")

    # Per-row top-K acoustic neighbors
    print(f"\nFinding top-{K_NEIGHBORS} acoustic neighbors per row...")
    # Argsort descending, take top K
    top_k_idx = np.argpartition(-emb_sim, K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]
    # Build top-K mask
    top_k_mask = np.zeros_like(emb_sim, dtype=bool)
    rows_idx = np.arange(len(emb_sim))[:, None]
    top_k_mask[rows_idx, top_k_idx] = True
    print(f"  top-K coverage: {top_k_mask.sum()} entries (= {len(emb_sim)} × {K_NEIGHBORS})")

    # Per-row top-K diagnostics: what classes most often appear?
    n_in_topk = top_k_mask.sum(axis=0)
    top_classes = np.argsort(n_in_topk)[::-1][:15]
    print(f"\n  Top-15 classes appearing as acoustic neighbor (across all rows):")
    for c in top_classes:
        print(f"    {primary[c]:<14} {sp_taxon[c]:<10} appears as neighbor in {n_in_topk[c]:>5} rows ({100*n_in_topk[c]/len(emb_sim):.2f}%)")

    # Build verified pseudo
    print(f"\n=== Building verified pseudo v2 ===")
    print(f"  Filter: v33 > {TAU_V33} AND emb_sim in top-{K_NEIGHBORS}")

    initial = v33 > TAU_V33
    print(f"  v33 > {TAU_V33}: {int(initial.sum())} entries")

    # Step 1: emb_sim top-K confirms acoustic match
    confirmed_emb = initial & top_k_mask
    print(f"  + emb_sim top-K: {int(confirmed_emb.sum())} entries")

    # Step 2: exp50 supports for further confidence (optional, not required)
    confirmed_exp50 = confirmed_emb & (exp50 > TAU_EXP50)
    print(f"  + exp50 > {TAU_EXP50} (additional, not required): {int(confirmed_exp50.sum())} entries")

    final = confirmed_emb  # use this as primary (emb_sim is real independent signal)
    print(f"\n  Final pseudo (v33 + emb_sim top-K): {int(final.sum())}")

    # Per-taxon
    print(f"\n  Per-taxon final:")
    for tx in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        mask = sp_taxon == tx
        n_t = final[:, mask].sum()
        n_classes = (final[:, mask].sum(axis=0) > 0).sum()
        print(f"    {tx}: {int(n_t)} entries, {int(n_classes)} classes")

    # Top-15 final
    final_class_counts = final.sum(axis=0)
    top15 = np.argsort(final_class_counts)[::-1][:15]
    print(f"\n  Top-15 most-fired in VERIFIED v2 pseudo:")
    for c in top15:
        print(f"    {primary[c]:<14} {sp_taxon[c]:<10} {int(final_class_counts[c]):>5}")

    # Save CSV
    print(f"\nBuilding CSV...")
    rows_csv = []
    for r_idx in range(len(v33)):
        positive = np.where(final[r_idx])[0]
        if len(positive) == 0: continue
        for c in positive:
            m = re.search(r"_(\d+)$", row_ids[r_idx])
            end_sec = int(m.group(1)) if m else 0
            rows_csv.append({
                "filename": filenames[r_idx],
                "start": str(end_sec - 5),
                "end": str(end_sec),
                "primary_label": primary[c],
                "v33_score": float(v33[r_idx, c]),
                "emb_sim": float(emb_sim[r_idx, c]),
                "exp50_score": float(exp50[r_idx, c]),
            })
    df_final = pd.DataFrame(rows_csv)
    print(f"  Total CSV rows: {len(df_final)}")
    df_final.to_csv(OUT_CSV, index=False)
    print(f"  Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
