#!/usr/bin/env python3
"""exp128 — Refine pseudo-labels using accumulated experiment patterns.

Refinement rules (from exp43r/45/108/109 findings):

A. Drop saturation Aves with n_pos=0 in labeled eval:
   - 47144*, baffal1, compot1, grepot1, grfdov1, linwoo1, picpig2, shtnig1, strher2*,
     toctou1, whiwoo1, smbani, greyel, blttit1, ragmac1, schpar1, ...
   - These are the 37 saturating Aves from exp109. Filter strictly.
   - (* = some have n_pos>0, keep with stricter threshold)

B. Confusion cluster dedup — keep only top-1 per file from each cluster:
   - Amphibia cluster (22973/24279/555146/65377)
   - Default-Aves cluster (47144/picpig2/grfdov1/...)
   - Nightjar cluster (litnig1/47144/nacnig1/sptnig1/compau)
   - 326272 cluster (476521/23158/23154/23150)
   - 67107 cluster (65377/horscr1/555145/brnowl/70711)

C. File-level outlier suppression:
   - If class fires in >10/12 windows of file AND it's in saturation list → drop entire file
   - If class fires in 12/12 AND no other cluster member fires → could be real, keep

D. Confusion-target row drop:
   - If row's pseudo-positive is one of {22961, 22930, bcwfin2} (default-Amphibia targets)
     AND row has no other Aves/Amphibia high → likely confusion artifact, drop

Output: pseudo_soundscapes_labels_refined.csv
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import re

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
V33_SCORES = ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz"
PSEUDO_CSV = DATA / "pseudo_soundscapes_labels.csv"
OUT = ROOT / "experiments/_data_pipelines/exp128_outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = DATA / "pseudo_soundscapes_labels_refined.csv"

# A. Saturation Aves with n_pos=0 in 122 eval (from exp109)
# These are species that fire >70% of rows in Perch but were never positive
# in labeled SS eval — pure default-Aves predictions.
SATURATION_AVES_DROP = {
    "baffal1", "compot1", "grepot1", "grfdov1", "linwoo1", "picpig2",
    "shtnig1", "toctou1", "whiwoo1", "smbani", "greyel", "blttit1",
    "strowl1", "epaori4", "rubthr1", "swtman1", "schpar1", "ragmac1",
    "whlspi1", "horscr1", "y00678", "ocecra1", "squcuc1", "sobcac1",
    "fusfly1", "orbtro3", "greani1", "swthum1", "souant1", "yebcar",
    "sptnig1", "baymac", "plcjay1", "undtin1", "watjac1",
}

# Saturation species with n_pos>0 in 122 eval — keep but require stricter v33
SATURATION_AVES_STRICT = {"47144", "strher2"}

# B. Confusion clusters — within each, keep only top-1 v33 per file
CLUSTERS = {
    "Amphibia_cluster_A": ["22973", "24279", "555146", "65377", "23158", "326272"],
    "Amphibia_cluster_B": ["476521", "23154", "23150"],
    "Amphibia_cluster_C": ["555145", "24287", "555123", "67107", "65380", "70711"],
    "Nightjar_Aves": ["litnig1", "nacnig1", "sptnig1", "compau", "47144"],
    "Default_Aves": ["picpig2", "grfdov1", "baffal1", "compot1", "grepot1",
                       "toctou1", "whiwoo1"],
}

# D. Default-Amphibia confusion targets (from exp108: 67107/326272/25092/47158son11
# all confused with these Aves)
CONFUSION_TARGET_AVES = {"22961", "22930", "bcwfin2"}

V33_STRICT = 0.7  # stricter v33 threshold for saturation strict list


def main():
    print("=== exp128: Refine pseudo-labels using experiment patterns ===\n", flush=True)

    df = pd.read_csv(PSEUDO_CSV)
    print(f"  Initial pseudo-labels: {len(df)}")

    v33_data = np.load(V33_SCORES, allow_pickle=True)
    v33 = v33_data["v33"].astype(np.float32)
    filenames_full = v33_data["filenames"].astype(str)
    end_secs_full = v33_data["end_secs"].astype(int)

    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    sp2idx = {s: i for i, s in enumerate(primary)}
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))

    # Build row index lookup
    row_lookup = {}
    for i, (fn, es) in enumerate(zip(filenames_full, end_secs_full)):
        row_lookup[(fn, int(es))] = i

    # ==================== Rule A: Saturation drop ====================
    n_before = len(df)
    drop_sat = df.primary_label.isin(SATURATION_AVES_DROP)
    print(f"\n  [A] Saturation Aves drop ({len(SATURATION_AVES_DROP)} species):")
    for sp in sorted(SATURATION_AVES_DROP):
        n = (df.primary_label == sp).sum()
        if n > 0: print(f"    {sp}: {n}")
    df_a = df[~drop_sat].copy()
    print(f"  → after rule A: {len(df_a)} ({n_before - len(df_a)} dropped)")

    # ==================== Rule A2: Strict-keep saturation (require v33 > 0.7) ====================
    strict_mask = df_a.primary_label.isin(SATURATION_AVES_STRICT)
    if strict_mask.sum() > 0:
        n_strict_before = strict_mask.sum()
        keep_strict = strict_mask & (df_a.v33_score > V33_STRICT)
        df_a = df_a[(~strict_mask) | keep_strict].copy()
        n_strict_dropped = n_strict_before - keep_strict.sum()
        print(f"  [A2] Strict-saturation (require v33>{V33_STRICT}): dropped {n_strict_dropped} of {n_strict_before}")

    # ==================== Rule B: Cluster top-1 per file ====================
    print(f"\n  [B] Confusion-cluster dedup (keep top-1 v33 per file):")
    keep_mask = np.ones(len(df_a), dtype=bool)

    for cname, members in CLUSTERS.items():
        members_present = [m for m in members if m in sp2idx]
        if len(members_present) < 2:
            continue
        # Per file, find members in pseudo-labels of this file
        for fname, group in df_a.groupby("filename"):
            cluster_rows_idx = group.primary_label.isin(members_present)
            if cluster_rows_idx.sum() <= 1: continue
            # Multiple cluster members in same file's pseudo-labels
            cluster_subset = group[cluster_rows_idx]
            # For each window in this file, keep only top-1 v33 from cluster
            for end_sec, end_group in cluster_subset.groupby("end"):
                if len(end_group) <= 1: continue
                # Find window's row in v33 lookup
                key = (fname, int(end_sec))
                if key not in row_lookup: continue
                row_i = row_lookup[key]
                # v33 scores for cluster members in this row
                member_idx = [sp2idx[m] for m in members_present]
                scores = v33[row_i, member_idx]
                top_member = members_present[int(np.argmax(scores))]
                # Mark non-top members in this row's group as DROP
                for idx in end_group.index:
                    if df_a.loc[idx, "primary_label"] != top_member:
                        keep_mask[df_a.index.get_loc(idx)] = False
        n_dropped = (~keep_mask).sum()
        print(f"    {cname}: cluster_size={len(members_present)}, current cumulative drop={n_dropped}")

    df_b = df_a[keep_mask].copy()
    n_dropped_b = len(df_a) - len(df_b)
    print(f"  → after rule B: {len(df_b)} ({n_dropped_b} dropped)")

    # ==================== Rule C: File-level outlier ====================
    print(f"\n  [C] File-level outlier (>10/12 windows AND not in saturation drop):")
    file_class_counts = df_b.groupby(["filename", "primary_label"]).size().reset_index(name="n_windows")
    over_present = file_class_counts[file_class_counts.n_windows > 10]
    print(f"    Over-present (file, class) pairs: {len(over_present)}")

    # For each over-present, check if class is also in cluster — if yes, drop
    drop_op = set()
    for _, op in over_present.iterrows():
        cls = op.primary_label
        # If class is in any cluster, mark as suspicious file
        in_cluster = any(cls in m for m in CLUSTERS.values())
        if in_cluster:
            drop_op.add((op.filename, cls))
    print(f"    Cluster-related over-presence: {len(drop_op)} (file, class) pairs flagged")

    keep_c_mask = ~df_b.apply(lambda r: (r.filename, r.primary_label) in drop_op, axis=1)
    df_c = df_b[keep_c_mask].copy()
    print(f"  → after rule C: {len(df_c)} ({len(df_b) - len(df_c)} dropped)")

    # ==================== Rule D: Confusion-target Aves drop ====================
    print(f"\n  [D] Default-Amphibia confusion target Aves drop ({CONFUSION_TARGET_AVES}):")
    for sp in CONFUSION_TARGET_AVES:
        n = (df_c.primary_label == sp).sum()
        if n > 0: print(f"    {sp}: {n}")
    drop_d = df_c.primary_label.isin(CONFUSION_TARGET_AVES)
    df_final = df_c[~drop_d].copy()
    print(f"  → after rule D: {len(df_final)} ({len(df_c) - len(df_final)} dropped)")

    # ==================== Save + summary ====================
    print(f"\n=== Final refined pseudo-labels ===")
    print(f"  Initial: {n_before}")
    print(f"  Final:   {len(df_final)}")
    print(f"  Dropped: {n_before - len(df_final)} ({100*(1 - len(df_final)/n_before):.1f}%)")

    # Per-taxon final
    df_final["taxon"] = df_final.primary_label.map(sp2tax).fillna("?")
    print(f"\n  Final per-taxon distribution:")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        n_t = (df_final.taxon == t).sum()
        n_classes = df_final[df_final.taxon == t].primary_label.nunique()
        print(f"    {t}: {n_t} entries, {n_classes} classes")

    # Top-15 final pseudo-labels
    print(f"\n  Top-15 most-fired in REFINED pseudo:")
    for lbl, n in df_final.primary_label.value_counts().head(15).items():
        taxon = sp2tax.get(lbl, "?")
        print(f"    {lbl:<14} {taxon:<10} {n:>5}")

    df_final.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
