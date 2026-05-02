#!/usr/bin/env python3
"""exp127 — Pseudo-label forensics: find obvious mislabels and likely-correct alternatives.

Diagnostic axes:
  1. Per-class over-firing rate (compared to expected occurrence)
  2. File-level over-presence (>k of 12 windows per file)
  3. Confusion cluster signatures (multiple cluster members co-fire)
  4. Frequency profile vs predicted taxon mismatch

For suspicious pseudo-labels, identify likely correct alternative species.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
PERCH_UNLABELED = ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz"
EXP50_UNLABELED = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
V33_SCORES = ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz"
PSEUDO_CSV = DATA / "pseudo_soundscapes_labels.csv"
OUT = ROOT / "experiments/_data_pipelines/exp127_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=== exp127: Pseudo-label forensics ===\n", flush=True)

    # Load
    print("Loading data...")
    df_pseudo = pd.read_csv(PSEUDO_CSV)
    print(f"  Pseudo-label CSV: {len(df_pseudo)} entries")

    v33_data = np.load(V33_SCORES, allow_pickle=True)
    v33 = v33_data["v33"].astype(np.float32)
    filenames = v33_data["filenames"].astype(str)
    end_secs = v33_data["end_secs"].astype(int)
    print(f"  v33 scores: {v33.shape}")

    perch = np.load(PERCH_UNLABELED, mmap_mode="r")
    perch_emb = np.array(perch["emb"])
    perch_logits = np.array(perch["scores"])
    perch_probs = (1.0 / (1.0 + np.exp(-np.clip(perch_logits, -30, 30)))).astype(np.float32)
    exp50_data = np.load(EXP50_UNLABELED, allow_pickle=True)
    exp50 = exp50_data["scores"]

    # Taxonomy
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    sp2idx = {s: i for i, s in enumerate(primary)}
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    sp_taxon = np.array([sp2tax.get(p, "Aves") for p in primary])

    n_rows = len(v33)
    n_files = len(set(filenames))
    print(f"  Files: {n_files}, rows: {n_rows}")

    # ===== 1. Per-class over-firing =====
    print("\n=== 1. Per-class over-firing analysis ===\n")
    pseudo_class_counts = df_pseudo.primary_label.value_counts()
    print(f"  Top-15 most-fired classes in pseudo-labels:")
    print(f"  {'class':<14} {'taxon':<10} {'n':>6} {'%rows':>7}")
    for lbl, n in pseudo_class_counts.head(15).items():
        pct = 100 * n / n_rows
        taxon = sp2tax.get(lbl, "?")
        print(f"  {lbl:<14} {taxon:<10} {n:>6} {pct:>6.2f}%")

    # Pseudo-classes that fire on >5% of rows are suspicious
    sat_threshold = 0.05  # 5% of rows
    saturating = pseudo_class_counts[pseudo_class_counts > n_rows * sat_threshold]
    print(f"\n  Suspicious saturating species (>5% of rows): {len(saturating)}")
    for lbl, n in saturating.items():
        taxon = sp2tax.get(lbl, "?")
        print(f"    {lbl} ({taxon}): {n} rows ({100*n/n_rows:.2f}%)")

    # ===== 2. File-level over-presence =====
    print("\n=== 2. File-level over-presence ===\n")
    # For each (file, class) pair, count number of windows
    df_pseudo["file_id"] = df_pseudo.filename.values
    file_class_counts = df_pseudo.groupby(["file_id", "primary_label"]).size().reset_index(name="n_windows")
    print(f"  Distribution of windows per (file, class):")
    print(file_class_counts.n_windows.describe())

    # File-class pairs where class appears in >8 of 12 windows = over-presence (>67%)
    over_present = file_class_counts[file_class_counts.n_windows > 8]
    print(f"\n  Over-present (>8 windows): {len(over_present)} (file, class) pairs")
    over_class_counts = over_present.primary_label.value_counts()
    print(f"  Top-10 over-present classes:")
    for lbl, n in over_class_counts.head(10).items():
        taxon = sp2tax.get(lbl, "?")
        print(f"    {lbl} ({taxon}): {n} files")

    # ===== 3. Confusion cluster signatures =====
    print("\n=== 3. Confusion cluster signatures (rows with all top-5 in cluster) ===\n")
    # Known clusters from exp108
    clusters = {
        "nightjar": ["litnig1", "nacnig1", "sptnig1", "47144", "compau"],
        "saturating_default": ["picpig2", "47144", "grfdov1", "baffal1", "compot1", "grepot1", "toctou1", "whiwoo1"],
        "Amphib_cluster_A": ["22973", "compau", "grepot1", "23158", "326272"],
    }

    for name, members in clusters.items():
        valid = [m for m in members if m in sp2idx]
        if len(valid) < 2: continue
        member_idx = [sp2idx[m] for m in valid]
        # Rows where ≥3 cluster members are in pseudo-positive
        pseudo_mask = (v33[:, member_idx] > 0.5).sum(axis=1)
        co_fire_rows = (pseudo_mask >= 3).sum()
        print(f"  Cluster '{name}' ({valid}): {co_fire_rows} rows with ≥3 members co-firing")

    # ===== 4. Suspicious row-level patterns =====
    print("\n=== 4. Suspicious row patterns (all top-5 are Aves, but Aves cluster) ===\n")
    # For each row, check if all 5 pseudo-labels are Aves AND from saturation list
    saturation_set = set(saturating.index.tolist())
    suspicious_rows = []
    for _, row in df_pseudo.iterrows():
        if row.primary_label in saturation_set:
            suspicious_rows.append(row)
    df_susp = pd.DataFrame(suspicious_rows)
    print(f"  Pseudo-labels involving saturating species: {len(df_susp)} / {len(df_pseudo)} ({100*len(df_susp)/len(df_pseudo):.1f}%)")

    # ===== 5. Likely-correct alternatives =====
    print("\n=== 5. Likely-correct alternative for suspicious labels ===\n")
    # For each suspicious (file, class) over-present pair, look at the SAME file's
    # rows and see if another class has consistently lower but non-trivial v33 → likely real species
    print("  Sample over-present cases (top 5):")

    fname_to_idx = defaultdict(list)
    for i, fn in enumerate(filenames):
        fname_to_idx[fn].append(i)

    for k, (_, row) in enumerate(over_present.head(5).iterrows()):
        fn = row.file_id
        susp_class = row.primary_label
        susp_class_idx = sp2idx[susp_class]
        n_w = row.n_windows
        susp_taxon = sp2tax.get(susp_class, "?")

        # File rows
        file_indices = fname_to_idx[fn]
        if not file_indices: continue
        file_v33 = v33[file_indices]  # (12, 234)
        # Mean v33 across file (excluding the suspicious class itself)
        per_class_mean = file_v33.mean(axis=0)
        # Sort by mean, find non-saturating high candidates
        top_indices = np.argsort(per_class_mean)[::-1][:10]
        print(f"\n  File: {fn}, suspicious: {susp_class} ({susp_taxon}, in {n_w}/12 windows)")
        print(f"    Susp class file-mean v33: {per_class_mean[susp_class_idx]:.3f}")
        print(f"    Top-10 file-mean v33 (suspicious=excluded):")
        for ti in top_indices:
            if ti == susp_class_idx: continue
            cls = primary[ti]; tx = sp2tax.get(cls, "?")
            sat = cls in saturation_set
            sat_str = " (sat!)" if sat else ""
            print(f"      {cls:<14} {tx:<10} {per_class_mean[ti]:.3f}{sat_str}")

    # Save flags
    df_pseudo["is_saturating"] = df_pseudo.primary_label.isin(saturation_set)
    df_pseudo["is_over_present"] = False
    over_set = set((r.file_id, r.primary_label) for _, r in over_present.iterrows())
    df_pseudo["is_over_present"] = df_pseudo.apply(lambda r: (r.file_id, r.primary_label) in over_set, axis=1)
    df_pseudo["suspicious"] = df_pseudo.is_saturating | df_pseudo.is_over_present

    n_susp = df_pseudo.suspicious.sum()
    print(f"\n=== Summary ===")
    print(f"  Total pseudo-labels: {len(df_pseudo)}")
    print(f"  Suspicious (saturating OR over-present): {n_susp} ({100*n_susp/len(df_pseudo):.1f}%)")
    print(f"  Clean: {len(df_pseudo) - n_susp} ({100*(1 - n_susp/len(df_pseudo)):.1f}%)")

    df_pseudo.to_csv(OUT / "pseudo_with_flags.csv", index=False)
    print(f"\n  Flagged CSV → {OUT}/pseudo_with_flags.csv")


if __name__ == "__main__":
    main()
