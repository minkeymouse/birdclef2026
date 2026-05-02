#!/usr/bin/env python3
"""exp134 — Build sonotype pseudo via confusion-signature inverse mapping.

Uses exp133 confusion_data.npz:
  - signatures (234, 234): for each species c, mean Perch profile when c is positive
  - target_classes: 29 unmapped species (25 Insecta + 2 Mam + 1 Rept + 3 Amp)
  - cos_sim (n_unlab, 29): per-row cosine sim to each target's signature

For each unlabeled row r:
  1. Find target with highest cos_sim above τ → primary sonotype/non-mapped match
  2. Conflict resolution: shared-signature targets (son21=22=23) → keep all (multi-label)
     because their signatures are IDENTICAL — can't disambiguate.

Output: pseudo_soundscapes_labels_sonotype.csv

Combined with v2: pseudo_soundscapes_labels_v3.csv (v2 + sonotype)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import re

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
EXP133_DATA = ROOT / "experiments/_data_pipelines/exp133_outputs/confusion_data.npz"
EXP125_UNLAB = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
V2_CSV = DATA / "pseudo_soundscapes_labels_verified_v2.csv"
OUT_SONOTYPE = DATA / "pseudo_soundscapes_labels_sonotype.csv"
OUT_V3 = DATA / "pseudo_soundscapes_labels_v3.csv"

TAU_COS_SIM = 0.5      # cosine sim threshold for sonotype match
TAU_HIGH = 0.7         # very confident sonotype match


def main():
    print("=== exp134: Sonotype pseudo via confusion mapping ===\n", flush=True)

    confusion = np.load(EXP133_DATA)
    cos_sim = confusion["cos_sim"].astype(np.float32)  # (n_rows, n_targets)
    target_classes = confusion["target_classes"]  # indices into 234
    print(f"  cos_sim: {cos_sim.shape}, target classes: {len(target_classes)}")

    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    sp2idx = {s: i for i, s in enumerate(primary)}
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))

    # Load unlabeled meta (filenames + row_ids)
    exp50_data = np.load(EXP125_UNLAB, allow_pickle=True)
    filenames = exp50_data["filenames"].astype(str)
    row_ids = exp50_data["row_ids"].astype(str)

    # For each row, build sonotype matches
    print("\n=== Per-row sonotype matching ===\n")
    rows_csv = []
    n_per_target = np.zeros(len(target_classes), dtype=int)
    n_rows_with_match = 0

    for r in range(len(cos_sim)):
        sims = cos_sim[r]
        # Filter targets above threshold
        high_targets = np.where(sims > TAU_COS_SIM)[0]
        if len(high_targets) == 0: continue
        n_rows_with_match += 1

        # Sort by cos_sim, descending
        order = np.argsort(sims[high_targets])[::-1]
        # Keep only top-3 targets (avoid over-labeling on shared signatures)
        # But for shared-signature groups (son21=22=23), keep all if all match
        kept_targets = high_targets[order[:5]]

        # Parse end_sec from row_id
        m = re.search(r"_(\d+)$", row_ids[r])
        end_sec = int(m.group(1)) if m else 0

        for t_idx in kept_targets:
            c_global = int(target_classes[t_idx])
            sim = float(sims[t_idx])
            rows_csv.append({
                "filename": filenames[r],
                "start": str(end_sec - 5),
                "end": str(end_sec),
                "primary_label": primary[c_global],
                "cos_sim": sim,
                "v33_score": 0.0,  # sonotype pseudo doesn't have v33 high
                "perch_score": 0.0,
                "exp50_score": 0.0,
            })
            n_per_target[t_idx] += 1

    print(f"  Total rows with ≥1 match: {n_rows_with_match} / {len(cos_sim)}")
    print(f"  Total sonotype pseudo entries: {len(rows_csv)}")
    print(f"\n  Per-target counts:")
    print(f"  {'target':<14} {'taxon':<10} {'count':>6}")
    for i, c in enumerate(target_classes):
        c_global = int(c)
        n = n_per_target[i]
        if n > 0:
            print(f"  {primary[c_global]:<14} {sp_taxon[c_global] if False else sp2tax.get(primary[c_global],'?'):<10} {n:>6}")

    df_sonotype = pd.DataFrame(rows_csv)
    df_sonotype.to_csv(OUT_SONOTYPE, index=False)
    print(f"\n  Saved sonotype CSV → {OUT_SONOTYPE}")

    # Combine with v2
    print("\n=== Combining with v2 (v2 ∪ sonotype) ===")
    df_v2 = pd.read_csv(V2_CSV)
    print(f"  v2 entries: {len(df_v2)}")
    print(f"  sonotype entries: {len(df_sonotype)}")

    # Dedup: same (filename, start, end, primary_label) — keep v2 if duplicate
    df_v2["source"] = "v2"
    df_sonotype["source"] = "sonotype"
    df_v2["start"] = df_v2["start"].astype(str)
    df_v2["end"] = df_v2["end"].astype(str)
    df_sonotype["start"] = df_sonotype["start"].astype(str)
    df_sonotype["end"] = df_sonotype["end"].astype(str)

    df_v3 = pd.concat([df_v2, df_sonotype], ignore_index=True)
    df_v3 = df_v3.drop_duplicates(subset=["filename", "start", "end", "primary_label"],
                                       keep="first")
    print(f"  v3 (combined, dedup): {len(df_v3)}")

    # Per-taxon distribution
    df_v3["taxon"] = df_v3.primary_label.map(sp2tax).fillna("?")
    print(f"\n  v3 per-taxon distribution:")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        n_t = (df_v3.taxon == t).sum()
        n_classes = df_v3[df_v3.taxon == t].primary_label.nunique()
        print(f"    {t}: {n_t} entries, {n_classes} classes")

    # Per-source breakdown
    print(f"\n  v3 by source:")
    print(df_v3.source.value_counts())

    df_v3.to_csv(OUT_V3, index=False)
    print(f"\n  Saved v3 CSV → {OUT_V3}")


if __name__ == "__main__":
    main()
