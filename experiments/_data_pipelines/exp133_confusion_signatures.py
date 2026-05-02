#!/usr/bin/env python3
"""exp133 — Confusion signature analysis: inverse mapping for unmapped species.

Insight: 31 species are unmapped to Perch (25 Insecta sonotypes + Mammalia +
Reptilia + 3 Amphibia). Perch outputs 0 for these species' columns. But when
they're actually present, Perch CONFIDENTLY predicts whatever mapped species
sounds nearest. This confusion is consistent (exp43r, exp108).

Forward (build signature):
  For each labeled SS row where species c is positive:
    Compute model's prediction profile (Perch sigmoid scores on all 234 classes).
    Mean across positive rows = c's confusion signature.

Reverse (apply to unlabeled):
  For each unlabeled SS row:
    Compute its Perch profile.
    Cosine-similarity to each species c's signature.
    If high to unmapped c → suspect c is present.

This recovers Insecta/Mammalia/Reptilia pseudo-labels independent of v33.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "_audits_post_v26"))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)

DATA = ROOT / "data" / "birdclef-2026"
PERCH_DIR = ROOT / "perch_v2"
PERCH_UNLABELED = ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz"
EXP50_UNLABELED = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
OUT = ROOT / "experiments/_data_pipelines/exp133_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def get_perch_mapped(primary):
    """Identify which species are mapped to Perch's 14k head."""
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2idx = {s: i for i, s in enumerate(primary)}
    sci2pl = dict(zip(tax.scientific_name, tax.primary_label))
    perch_labels = open(PERCH_DIR / "assets" / "labels.csv").read().strip().split("\n")
    is_mapped = np.zeros(N_CLS, dtype=bool)
    for pname in perch_labels:
        if pname in sci2pl and sci2pl[pname] in sp2idx:
            is_mapped[sp2idx[sci2pl[pname]]] = True
    return is_mapped


def main():
    print("=== exp133: Confusion signatures (inverse mapping) ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()  # (739, 234) sigmoid
    exp50_lab = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]

    is_mapped = get_perch_mapped(primary)
    n_mapped = is_mapped.sum()
    n_unmapped = (~is_mapped).sum()
    print(f"  Perch mapping: {n_mapped} mapped, {n_unmapped} unmapped")

    unmapped_per_taxon = defaultdict(list)
    for c in range(N_CLS):
        if not is_mapped[c]:
            unmapped_per_taxon[sp_taxon[c]].append(primary[c])
    for tx, lst in unmapped_per_taxon.items():
        print(f"    {tx} unmapped: {len(lst)} → {lst[:5]}{'...' if len(lst)>5 else ''}")

    # ============= 1. Forward: build per-species signature =============
    print("\n=== 1. Forward: per-species confusion signature ===\n")
    print("  For each species c, mean Perch profile on rows where c is positive.\n")

    n_pos_per_class = Y.sum(axis=0)
    print(f"  Classes with ≥1 positive in labeled SS: {(n_pos_per_class > 0).sum()} / {N_CLS}")
    print(f"  Classes with ≥3 positives: {(n_pos_per_class >= 3).sum()}")

    # Baseline Perch profile (mean over ALL labeled SS rows)
    perch_baseline = perch_prob.mean(axis=0)  # (234,) — average prediction across all rows

    # Per-class signature: mean Perch profile on rows where class is positive,
    # MINUS baseline (so signature is the lift over baseline)
    signatures = np.zeros((N_CLS, N_CLS), dtype=np.float32)
    for c in range(N_CLS):
        pos_rows = Y[:, c] > 0
        if pos_rows.sum() == 0:
            continue
        signatures[c] = perch_prob[pos_rows].mean(axis=0) - perch_baseline

    # Diagnostic: for each unmapped species with ≥3 positives, show top-5 confusion targets
    print("\n  Confusion signatures (top-5 over-fired species when target is positive):\n")
    diagnostic_classes = [
        c for c in range(N_CLS)
        if (not is_mapped[c]) and (n_pos_per_class[c] >= 3)
    ]
    print(f"  Unmapped + ≥3 positives: {len(diagnostic_classes)} classes\n")

    for c in diagnostic_classes:
        sig = signatures[c]
        n_pos = int(n_pos_per_class[c])
        # Top-5 most over-fired (excluding the target itself if it would fire)
        order = np.argsort(sig)[::-1]
        order_filtered = [i for i in order if i != c][:5]
        partners = ", ".join(f"{primary[i]}({sp_taxon[i][:3]}, +{sig[i]:.3f})" for i in order_filtered)
        print(f"  {primary[c]:<14} {sp_taxon[c]:<10} (n_pos={n_pos:>3}): top-5 confusion → {partners}")

    # Also for MAPPED species with ≥10 positives (sanity: should self-fire highest)
    print("\n  Sanity (mapped species with ≥10 positives, top-3 includes self?):\n")
    mapped_diag = [c for c in range(N_CLS) if is_mapped[c] and n_pos_per_class[c] >= 10]
    for c in mapped_diag[:8]:
        sig = signatures[c]
        order = np.argsort(sig)[::-1][:3]
        partners = ", ".join(f"{primary[i]}({sp_taxon[i][:3]},+{sig[i]:.2f})" for i in order)
        is_self_top1 = "✓" if order[0] == c else "✗"
        print(f"  {primary[c]:<14} {sp_taxon[c]:<10} self_top1={is_self_top1}: {partners}")

    # ============= 2. Reverse: apply signatures to unlabeled =============
    print("\n\n=== 2. Reverse: apply signatures to unlabeled SS ===\n")
    perch_unlab = np.load(PERCH_UNLABELED, mmap_mode="r")
    perch_logits = np.array(perch_unlab["scores"])
    perch_unlab_probs = (1.0 / (1.0 + np.exp(-np.clip(perch_logits, -30, 30)))).astype(np.float32)
    print(f"  Unlabeled SS Perch probs: {perch_unlab_probs.shape}")

    # For each unlabeled row r, compute its profile (lift over baseline)
    unlab_profile = perch_unlab_probs - perch_baseline[None, :]  # (n_rows, 234)

    # Cosine similarity between unlabeled profile and each unmapped species' signature
    # Restrict to unmapped + has-signature classes
    has_sig = (n_pos_per_class >= 3) & (~is_mapped)
    target_classes = np.where(has_sig)[0]
    print(f"  Targets (unmapped + ≥3 positives in labeled SS): {len(target_classes)}")
    for c in target_classes:
        print(f"    {primary[c]} ({sp_taxon[c]}, n_pos={int(n_pos_per_class[c])})")

    # Compute cos sim
    target_sigs = signatures[target_classes]  # (n_targets, 234)
    target_norm = np.linalg.norm(target_sigs, axis=1, keepdims=True) + 1e-9
    target_n = target_sigs / target_norm
    profile_norm = np.linalg.norm(unlab_profile, axis=1, keepdims=True) + 1e-9
    profile_n = unlab_profile / profile_norm

    # cosine sim shape (n_unlab, n_targets)
    cos_sim = (profile_n @ target_n.T).astype(np.float32)
    print(f"\n  cos_sim: {cos_sim.shape}, range [{cos_sim.min():.3f}, {cos_sim.max():.3f}], mean {cos_sim.mean():.3f}")

    # Per-target distribution
    print(f"\n  Per-target cos_sim distribution:")
    print(f"  {'target':<14} {'taxon':<10} {'mean':>8} {'p90':>8} {'p99':>8} {'max':>8}")
    for i, c in enumerate(target_classes):
        s = cos_sim[:, i]
        print(f"  {primary[c]:<14} {sp_taxon[c]:<10} {s.mean():>8.3f} {np.percentile(s, 90):>8.3f} "
              f"{np.percentile(s, 99):>8.3f} {s.max():>8.3f}")

    # Suggested pseudo: cos_sim > 0.5 → strong match
    print(f"\n  Pseudo-positive candidates (cos_sim > 0.5):")
    for i, c in enumerate(target_classes):
        n_strong = (cos_sim[:, i] > 0.5).sum()
        print(f"  {primary[c]:<14} {sp_taxon[c]:<10}: {n_strong} unlabeled rows match (sig)")

    # Save signatures + cos_sim for downstream use
    np.savez_compressed(OUT / "confusion_data.npz",
                          signatures=signatures.astype(np.float16),
                          target_classes=target_classes,
                          cos_sim=cos_sim.astype(np.float16),
                          perch_baseline=perch_baseline.astype(np.float16))
    print(f"\nSaved → {OUT}/confusion_data.npz")


if __name__ == "__main__":
    main()
