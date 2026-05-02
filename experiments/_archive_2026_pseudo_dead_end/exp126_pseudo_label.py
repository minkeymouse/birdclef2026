#!/usr/bin/env python3
"""exp126 — Pseudo-label all unlabeled SS using v33 best model.

Pipeline:
  1. Load Perch scores + embeddings on unlabeled SS (from exp43a)
  2. Load exp50 scores on unlabeled SS (from exp125)
  3. Apply v33 transformation: 0.7P + 0.3 exp50 → V9 gate → file-max coherence
  4. Filter:
     - HIGH confidence positives: v33[r,c] > τ_pos (default 0.5)
     - HIGH confidence negatives: v33[r,c] < τ_neg (default 0.05)
     - Mid range: uncertain, exclude
  5. Save pseudo-label CSV in same format as train_soundscapes_labels.csv:
     filename, start, end, primary_label

Output: data/birdclef-2026/pseudo_soundscapes_labels.csv
        + diagnostic stats
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "_audits_post_v26"))
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
PERCH_UNLABELED = ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz"
PERCH_META = ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all_meta.parquet"
EXP50_UNLABELED = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
OUT_CSV = DATA / "pseudo_soundscapes_labels.csv"
OUT_STATS = ROOT / "experiments/_data_pipelines/exp126_outputs/pseudo_label_stats.json"
OUT_STATS.parent.mkdir(parents=True, exist_ok=True)

N_CLS = 234
N_WINDOWS = 12
WINDOW_SEC = 5

# Pseudo-label thresholds — STRICT filter for high-precision pseudo-labels
TAU_V33 = 0.5       # v33 score > τ → candidate
TAU_PERCH = 0.5     # Perch sigmoid score > τ → required for ensemble agreement
TAU_EXP50 = 0.3     # exp50 score > τ → required (lower since exp50 polarized at 0)
TOP_K_PER_ROW = 5   # max pseudo-labels per row (keep top-K by v33)
TAU_NEG = 0.05      # v33 < τ_neg → high-conf negative (for completeness, not in CSV)


def main():
    print("=== exp126: Pseudo-label generation on unlabeled SS ===\n", flush=True)

    # 1. Load Perch on unlabeled
    print("Loading Perch unlabeled cache...")
    perch_unlab = np.load(PERCH_UNLABELED, mmap_mode="r")
    perch_emb = np.array(perch_unlab["emb"])
    # NOTE: unlabeled Perch scores are LOGITS (range ~-7 to +14)
    # labeled Perch scores (load_perch_scores_labeled) are sigmoid PROBS [0, 1]
    # Apply sigmoid to align with labeled cache + exp50 sigmoid output
    perch_logits_unlab = np.array(perch_unlab["scores"])
    perch_scores = (1.0 / (1.0 + np.exp(-np.clip(perch_logits_unlab, -30, 30)))).astype(np.float32)
    print(f"  emb {perch_emb.shape}, scores {perch_scores.shape}")
    print(f"  perch (post-sigmoid) range: [{perch_scores.min():.4f}, {perch_scores.max():.4f}]")

    perch_meta = pd.read_parquet(PERCH_META)
    print(f"  meta: {len(perch_meta)} rows, columns: {list(perch_meta.columns)}")

    # 2. Load exp50 on unlabeled
    print("\nLoading exp50 unlabeled scores...")
    exp50_data = np.load(EXP50_UNLABELED, allow_pickle=True)
    exp50_scores = exp50_data["scores"]
    exp50_filenames = exp50_data["filenames"]
    exp50_row_ids = exp50_data["row_ids"]
    print(f"  exp50_scores {exp50_scores.shape}")

    # Verify alignment between Perch cache and exp50 cache
    assert len(exp50_scores) == len(perch_scores), \
        f"row mismatch: perch {len(perch_scores)}, exp50 {len(exp50_scores)}"

    # 3. Build minimal sc_g for v33 transformation (file-max needs filename grouping)
    sc_g_unlab = pd.DataFrame({
        "filename": exp50_filenames.astype(str),
        "row_id": exp50_row_ids.astype(str),
    })
    # Extract end_sec from row_id
    def extract_end_sec(rid):
        m = re.search(r"_(\d+)$", str(rid))
        return int(m.group(1)) if m else -1
    sc_g_unlab["end_sec"] = sc_g_unlab["row_id"].apply(extract_end_sec)
    sc_g_unlab["start"] = sc_g_unlab["end_sec"] - WINDOW_SEC

    # Species taxon array
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    sp_taxon = np.array([sp2tax.get(p, "Aves") for p in primary])

    # 4. Apply v33 transformation
    print("\nBuilding v33 predictions on unlabeled SS...")
    base = 0.7 * perch_scores + 0.3 * exp50_scores
    print(f"  base: range [{base.min():.4f}, {base.max():.4f}]")
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    print(f"  gated: range [{gated.min():.4f}, {gated.max():.4f}]")
    v33 = file_max_blend(gated, sc_g_unlab, alpha=0.10)
    print(f"  v33: range [{v33.min():.4f}, {v33.max():.4f}]")
    print(f"  v33 mean: {v33.mean():.4f}")

    # 5. Pseudo-label generation: ensemble agreement + top-K
    print(f"\nGenerating pseudo-labels:")
    print(f"  τ_v33={TAU_V33}, τ_perch={TAU_PERCH}, τ_exp50={TAU_EXP50}, top_K_per_row={TOP_K_PER_ROW}")

    # Step 1: ensemble agreement
    ensemble_agree = (v33 > TAU_V33) & (perch_scores > TAU_PERCH) & (exp50_scores > TAU_EXP50)
    print(f"  ensemble-agreement candidates: {int(ensemble_agree.sum())}")

    # Step 2: top-K per row from v33
    high_conf_pos = np.zeros_like(ensemble_agree)
    for r in range(len(v33)):
        # Among candidates that pass ensemble agreement, keep top-K by v33
        candidates = np.where(ensemble_agree[r])[0]
        if len(candidates) == 0: continue
        if len(candidates) <= TOP_K_PER_ROW:
            high_conf_pos[r, candidates] = True
        else:
            # Top-K by v33 score
            top_k_idx = candidates[np.argsort(v33[r, candidates])[::-1][:TOP_K_PER_ROW]]
            high_conf_pos[r, top_k_idx] = True
    print(f"  after top-K filter: {int(high_conf_pos.sum())} pseudo-positives")

    # Per-class distribution
    n_pos_per_class = high_conf_pos.sum(axis=0)
    n_classes_with_pseudo = (n_pos_per_class > 0).sum()
    print(f"  classes with ≥1 pseudo-label: {int(n_classes_with_pseudo)} / {N_CLS}")

    # Per-row distribution
    n_pos_per_row = high_conf_pos.sum(axis=1)
    print(f"  rows with ≥1 pseudo-label: {int((n_pos_per_row > 0).sum())} / {len(v33)}")
    print(f"  mean pseudo-labels per row: {n_pos_per_row.mean():.2f}")
    print(f"  max pseudo-labels per row: {int(n_pos_per_row.max())}")

    # Per-taxon distribution
    print(f"\n  Pseudo-labels per taxon:")
    for taxon in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        mask = sp_taxon == taxon
        n_in_taxon = high_conf_pos[:, mask].sum()
        n_classes_in_taxon = (high_conf_pos[:, mask].sum(axis=0) > 0).sum()
        print(f"    {taxon}: {int(n_in_taxon)} pseudo entries, {int(n_classes_in_taxon)} classes")

    # 6. Build CSV in train_soundscapes_labels format
    print(f"\nBuilding CSV...")
    rows_csv = []
    for r_idx, (fname, end_sec) in enumerate(zip(sc_g_unlab.filename.values, sc_g_unlab.end_sec.values)):
        positive_classes = np.where(high_conf_pos[r_idx])[0]
        for c in positive_classes:
            rows_csv.append({
                "filename": fname,
                "start": f"{end_sec - WINDOW_SEC:.0f}",
                "end": f"{end_sec:.0f}",
                "primary_label": primary[c],
                "v33_score": float(v33[r_idx, c]),
                "perch_score": float(perch_scores[r_idx, c]),
                "exp50_score": float(exp50_scores[r_idx, c]),
            })
    df_pseudo = pd.DataFrame(rows_csv)
    print(f"  Total CSV rows (pseudo-positives): {len(df_pseudo)}")
    df_pseudo.to_csv(OUT_CSV, index=False)
    print(f"  Saved → {OUT_CSV}")

    # Save full v33 score matrix for flexible threshold tuning later
    full_scores_path = OUT_STATS.parent / "v33_unlabeled_scores.npz"
    np.savez_compressed(full_scores_path,
                          v33=v33.astype(np.float16),
                          filenames=sc_g_unlab.filename.values.astype("U200"),
                          end_secs=sc_g_unlab.end_sec.values.astype(np.int16))
    print(f"  Full v33 scores → {full_scores_path} ({full_scores_path.stat().st_size/1e6:.1f} MB)")

    # 7. Stats
    import json
    stats = {
        "n_unlabeled_rows": int(len(v33)),
        "n_files": int(sc_g_unlab.filename.nunique()),
        "tau_v33": TAU_V33,
        "tau_perch": TAU_PERCH,
        "tau_exp50": TAU_EXP50,
        "top_k_per_row": TOP_K_PER_ROW,
        "n_pseudo_pos_entries": int(high_conf_pos.sum()),
        "n_classes_with_pseudo": int(n_classes_with_pseudo),
        "n_rows_with_pseudo": int((n_pos_per_row > 0).sum()),
        "mean_pseudo_per_row": float(n_pos_per_row.mean()),
        "max_pseudo_per_row": int(n_pos_per_row.max()),
        "per_taxon": {t: {
            "n_entries": int(high_conf_pos[:, sp_taxon == t].sum()),
            "n_classes": int((high_conf_pos[:, sp_taxon == t].sum(axis=0) > 0).sum())
        } for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]},
    }
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats → {OUT_STATS}")


if __name__ == "__main__":
    main()
