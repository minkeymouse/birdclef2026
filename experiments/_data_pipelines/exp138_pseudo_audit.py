#!/usr/bin/env python3
"""exp138 — Audit specific likely-wrong patterns in v3 pseudo."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
V3 = DATA / "pseudo_soundscapes_labels_v3.csv"

def main():
    print("=== exp138: v3 pseudo - likely wrong patterns ===\n", flush=True)
    df = pd.read_csv(V3)
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    df["taxon"] = df.primary_label.map(sp2tax).fillna("?")

    n_rows_unlab = 127896  # from exp43a cache
    print(f"Total entries: {len(df)}, by source: {df.source.value_counts().to_dict()}\n")

    # ============ Pattern 1: Saturating Aves (>5% of all unlabeled rows) ============
    print("=== Pattern 1: Saturating Aves (suspiciously frequent) ===")
    counts = df[df.taxon == "Aves"].primary_label.value_counts()
    saturating = counts[counts > n_rows_unlab * 0.05]
    print(f"Aves species firing on >5% of unlabeled rows ({len(saturating)} species):")
    for sp, n in saturating.head(15).items():
        pct = 100 * n / n_rows_unlab
        print(f"  {sp:<14} {n:>5} ({pct:>5.1f}% of unlabeled rows)")

    # Compare expected: typical Pantanal common species occur in maybe 1-3% of rows
    # These 5%+ are likely Perch saturation/false positive
    print(f"\n  Total entries from suspected saturating Aves: {saturating.sum()}")
    print(f"  Estimated false-positive Aves entries: {saturating.sum() * 0.7:.0f}-{saturating.sum() * 0.9:.0f}")
    print(f"  (assuming 70-90% of these are noise)\n")

    # ============ Pattern 2: Sonotype shared-signature collisions ============
    print("=== Pattern 2: Sonotype groups sharing signatures ===")
    # son21=22=23 in same row = same signature, can't disambiguate
    df_son = df[df.primary_label.str.startswith("47158son")].copy()
    print(f"Total sonotype entries: {len(df_son)}")
    df_son_per_row = (df_son.groupby(["filename", "end"]).primary_label.apply(set).reset_index())
    df_son_per_row["n_sonotypes"] = df_son_per_row.primary_label.apply(len)
    print(f"\n  Rows with multiple sonotypes:")
    n_dist = df_son_per_row.n_sonotypes.value_counts().sort_index()
    for n, count in n_dist.items():
        print(f"    {n} sonotype(s) per row: {count} rows")

    # Specifically check shared signature groups
    shared_signatures = [
        {"47158son21", "47158son22", "47158son23"},
        {"47158son15", "47158son16"},
    ]
    for grp in shared_signatures:
        n_with_all = df_son_per_row.primary_label.apply(lambda s: grp.issubset(s)).sum()
        if n_with_all > 0:
            print(f"\n  Rows with ALL of {sorted(grp)} co-labeled: {n_with_all}")
            print(f"    (signatures identical → can't tell which is real)")

    # ============ Pattern 3: Cross-taxon outliers ============
    print("\n=== Pattern 3: Cross-taxon outlier rows ===")
    # Rows where pseudo says BOTH Aves saturating species AND Insecta sonotype
    # If row has compot1+greantt+sonotype, the sonotype is LIKELY the real species
    # (Aves are saturating false positives)
    sat_set = set(saturating.head(10).index.tolist())
    df["is_saturating_aves"] = df.primary_label.isin(sat_set)
    df["is_sonotype"] = df.primary_label.str.startswith("47158son")

    per_row = df.groupby(["filename", "end"]).agg(
        n_saturating=("is_saturating_aves", "sum"),
        n_sonotype=("is_sonotype", "sum"),
        labels=("primary_label", lambda s: list(s)),
    ).reset_index()
    co_occur = per_row[(per_row.n_saturating >= 3) & (per_row.n_sonotype >= 1)]
    print(f"Rows with ≥3 saturating Aves AND ≥1 sonotype: {len(co_occur)}")
    print(f"  → Likely interpretation: row is Insecta-dominant, saturating Aves are FP")
    print(f"  → ~{len(co_occur) * 3:,} saturating-Aves entries are likely wrong in these rows\n")

    # ============ Pattern 4: Single-window outliers ============
    print("=== Pattern 4: Single-window outliers (suspicious) ===")
    # Real species typically calls for >1 window in a 60-sec file
    # If pseudo-labels show class C in only 1 window of file → suspect
    file_class_count = df.groupby(["filename", "primary_label"]).size().reset_index(name="n_win")
    single_win = file_class_count[file_class_count.n_win == 1]
    print(f"Single-window labels (only 1 of 12 windows in file): {len(single_win)} (file, class) pairs")
    by_taxon = single_win.merge(df[["primary_label", "taxon"]].drop_duplicates(), on="primary_label")
    print(f"  By taxon:")
    print(by_taxon.taxon.value_counts())

    # ============ Summary ============
    print("\n=== Summary: estimated likely-wrong entries ===")
    n_sat_likely_wrong = int(saturating.sum() * 0.8)  # ~80% noise
    n_son_collision = sum(df_son_per_row.n_sonotypes > 2) * 2  # at least 2 of 3 shared-sig must be wrong
    n_cross_taxon_aves = len(co_occur) * 3  # ~3 saturating Aves per row
    print(f"  Saturating Aves (~80% noise rate):       ~{n_sat_likely_wrong:>6,} entries")
    print(f"  Sonotype shared-sig over-labeling:        ~{n_son_collision:>6,} entries")
    print(f"  Cross-taxon Aves co-occur with sonotype:  ~{n_cross_taxon_aves:>6,} entries")
    print(f"  TOTAL ESTIMATED LIKELY-WRONG:             ~{n_sat_likely_wrong + n_son_collision + n_cross_taxon_aves:>6,}")
    print(f"  out of {len(df):,} total v3 entries ({100*(n_sat_likely_wrong + n_son_collision + n_cross_taxon_aves)/len(df):.1f}%)")


if __name__ == "__main__":
    main()
