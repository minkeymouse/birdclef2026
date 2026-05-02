#!/usr/bin/env python3
"""exp135 — Systematic error detection via embedding outlier + meta-learner.

Two filters applied to v3 pseudo:

1. EMBEDDING OUTLIER FILTER:
   For each pseudo-positive (row, class c):
     compute distance from row's Perch emb to class c's TA centroid
     compare to "expected distance distribution" (from labeled SS positives of c)
   If row's distance is far above that distribution (e.g. > p95) → systematic FP

2. META-LEARNER FILTER:
   Train a logistic regression that predicts P(label is correct) using:
     v33_score, perch_score, exp50_score, emb_sim, n_signals, file_freq_dom, ...
   Trained on labeled SS where we know GT.
   Apply to unlabeled to filter pseudo.

Output: pseudo_soundscapes_labels_v4.csv (v3 - systematic FPs)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "_audits_post_v26"))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)

DATA = ROOT / "data" / "birdclef-2026"
PERCH_UNLABELED = ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz"
EXP125_UNLAB = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
V126_SCORES = ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz"
V3_CSV = DATA / "pseudo_soundscapes_labels_v3.csv"
TA_2026_PERCH = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
TA_2025_PERCH = ROOT / "experiments/_data_pipelines/exp120_outputs/ta25_perch.npz"
OUT_V4 = DATA / "pseudo_soundscapes_labels_v4.csv"
OUT_DIAG = ROOT / "experiments/_data_pipelines/exp135_outputs"
OUT_DIAG.mkdir(parents=True, exist_ok=True)


def main():
    print("=== exp135: Systematic error filter for v3 pseudo ===\n", flush=True)

    # Load all signals
    print("Loading signals...")
    df_v3 = pd.read_csv(V3_CSV)
    print(f"  v3 entries: {len(df_v3)}")

    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    sp2idx = {s: i for i, s in enumerate(primary)}
    sp_taxon = species_taxon_array()

    # Load unlabeled features
    perch_unlab = np.load(PERCH_UNLABELED, mmap_mode="r")
    perch_emb_unlab = np.array(perch_unlab["emb"])
    perch_logits_unlab = np.array(perch_unlab["scores"])
    perch_probs_unlab = (1.0 / (1.0 + np.exp(-np.clip(perch_logits_unlab, -30, 30)))).astype(np.float32)

    exp50_data = np.load(EXP125_UNLAB, allow_pickle=True)
    exp50_unlab = exp50_data["scores"]
    filenames_unlab = exp50_data["filenames"].astype(str)

    v33_data = np.load(V126_SCORES, allow_pickle=True)
    v33 = v33_data["v33"].astype(np.float32)

    # Load labeled SS
    sc_g, Y, primary_lab, _ = build_ss()
    perch_emb_lab = load_perch_emb_labeled()
    perch_prob_lab = load_perch_scores_labeled()
    exp50_lab = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]

    # ==================== FILTER 1: Embedding outlier ====================
    print("\n=== Filter 1: Embedding outlier (TA centroid distance) ===\n")

    # Build TA centroids
    ta = np.load(TA_2026_PERCH)
    valid = (ta["valid"] == 1) & (ta["y_idx"] >= 0) & (ta["y_idx"] < N_CLS)
    ta_emb_all = ta["emb"][valid]; ta_y_all = ta["y_idx"][valid]
    if TA_2025_PERCH.exists():
        ta25 = np.load(TA_2025_PERCH)
        v25 = (ta25["valid"] == 1) & (ta25["y_idx"] >= 0) & (ta25["y_idx"] < N_CLS)
        ta_emb_all = np.concatenate([ta_emb_all, ta25["emb"][v25]], axis=0)
        ta_y_all = np.concatenate([ta_y_all, ta25["y_idx"][v25]], axis=0)

    centroids = np.zeros((N_CLS, 1536), dtype=np.float32)
    centroid_within_dist = np.full(N_CLS, np.inf, dtype=np.float32)  # within-class spread
    n_per_class = np.zeros(N_CLS, dtype=np.int32)
    for c in range(N_CLS):
        mask = ta_y_all == c
        n_per_class[c] = mask.sum()
        if mask.sum() >= 5:
            cent = ta_emb_all[mask].mean(axis=0)
            centroids[c] = cent
            # within-class p95 distance (cosine distance = 1 - cos sim)
            cent_n = cent / (np.linalg.norm(cent) + 1e-9)
            ta_n = ta_emb_all[mask] / (np.linalg.norm(ta_emb_all[mask], axis=1, keepdims=True) + 1e-9)
            within_sim = ta_n @ cent_n
            within_dist = 1 - within_sim
            centroid_within_dist[c] = np.percentile(within_dist, 95)

    # For each unlabeled row, compute cosine distance to each class centroid
    print("Computing emb sim (unlab vs centroids)...")
    unlab_n = perch_emb_unlab / (np.linalg.norm(perch_emb_unlab, axis=1, keepdims=True) + 1e-9)
    cent_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    cos_sim_all = unlab_n @ cent_n.T  # (n_rows, N_CLS)
    cos_dist_all = 1 - cos_sim_all  # smaller = more similar

    # For each row in v3, compute systematic_outlier flag
    # row's cos_dist[c] > centroid_within_dist[c] * 1.5 → outlier
    print("Building filename→row index map...")
    rid_to_idx = {(filenames_unlab[i], int(re.search(r"_(\d+)$", filenames_unlab[i] + "_0").group(1)) if False else None): i for i in range(0)}
    # Actually use exp50_data row_ids
    row_ids = exp50_data["row_ids"].astype(str)
    import re
    rid_to_idx = {}
    for i, rid in enumerate(row_ids):
        m = re.search(r"_(\d+)$", rid)
        if m:
            rid_to_idx[(filenames_unlab[i], int(m.group(1)))] = i

    n_outlier = 0
    n_kept_outlier = 0
    keep_mask = np.ones(len(df_v3), dtype=bool)

    df_v3["end"] = df_v3["end"].astype(int)
    for i, row in df_v3.iterrows():
        key = (row.filename, int(row.end))
        if key not in rid_to_idx: continue
        r_idx = rid_to_idx[key]
        c_idx = sp2idx.get(row.primary_label, -1)
        if c_idx < 0: continue
        if n_per_class[c_idx] < 5: continue  # no centroid

        # row's distance vs class within-class threshold
        row_dist = cos_dist_all[r_idx, c_idx]
        threshold = centroid_within_dist[c_idx] * 1.5  # allow 1.5x within-class spread
        if row_dist > threshold:
            n_outlier += 1
            keep_mask[i] = False  # drop
        else:
            n_kept_outlier += 0

    print(f"  Embedding-outlier flagged: {n_outlier} / {len(df_v3)} ({100*n_outlier/len(df_v3):.1f}%)")

    df_v3_filtered = df_v3[keep_mask].copy()
    print(f"  After embedding outlier filter: {len(df_v3_filtered)}")

    # ==================== FILTER 2: Meta-learner ====================
    print("\n=== Filter 2: Meta-learner (LogReg P(correct|features)) ===\n")

    # Build features for labeled SS rows (with GT)
    # For each (row, class) where v33 > 0.3 (interesting candidates), compute features + GT
    print("Building features for labeled SS training set...")
    feature_rows = []
    for r in range(len(perch_prob_lab)):
        for c in range(N_CLS):
            v33_lab = perch_prob_lab[r, c] * 0.7 + exp50_lab[r, c] * 0.3  # approx v33
            if v33_lab < 0.2: continue  # skip very low (most negative)
            # Compute emb_sim to centroid c
            row_emb_n = perch_emb_lab[r] / (np.linalg.norm(perch_emb_lab[r]) + 1e-9)
            if n_per_class[c] >= 5:
                cent_c_n = centroids[c] / (np.linalg.norm(centroids[c]) + 1e-9)
                emb_sim_lab = float(row_emb_n @ cent_c_n)
            else:
                emb_sim_lab = -1.0
            feature_rows.append({
                "v33": float(v33_lab),
                "perch": float(perch_prob_lab[r, c]),
                "exp50": float(exp50_lab[r, c]),
                "emb_sim": emb_sim_lab,
                "is_aves": int(sp_taxon[c] == "Aves"),
                "label": int(Y[r, c] > 0),
            })
    df_feat = pd.DataFrame(feature_rows)
    print(f"  Feature rows: {len(df_feat)}, positives: {int(df_feat.label.sum())}")

    # Train meta-learner
    feat_cols = ["v33", "perch", "exp50", "emb_sim", "is_aves"]
    X_train = df_feat[feat_cols].values
    y_train = df_feat["label"].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_s, y_train)
    train_acc = model.score(X_train_s, y_train)
    print(f"  Meta-learner train accuracy: {train_acc:.4f}")
    print(f"  Coefs: {dict(zip(feat_cols, model.coef_[0].round(3)))}")

    # Apply to v3 (post embedding-outlier filter)
    print("\n  Applying meta-learner to v3_filtered...")
    feature_v3 = []
    valid_indices = []
    for i, row in df_v3_filtered.iterrows():
        key = (row.filename, int(row.end))
        if key not in rid_to_idx: continue
        r_idx = rid_to_idx[key]
        c_idx = sp2idx.get(row.primary_label, -1)
        if c_idx < 0: continue
        v33_r = float(v33[r_idx, c_idx])
        perch_r = float(perch_probs_unlab[r_idx, c_idx])
        exp50_r = float(exp50_unlab[r_idx, c_idx])
        if n_per_class[c_idx] >= 5:
            row_emb_n = perch_emb_unlab[r_idx] / (np.linalg.norm(perch_emb_unlab[r_idx]) + 1e-9)
            cent_c_n = centroids[c_idx] / (np.linalg.norm(centroids[c_idx]) + 1e-9)
            emb_sim_r = float(row_emb_n @ cent_c_n)
        else:
            emb_sim_r = -1.0
        is_aves_r = int(sp_taxon[c_idx] == "Aves")
        feature_v3.append([v33_r, perch_r, exp50_r, emb_sim_r, is_aves_r])
        valid_indices.append(i)

    X_pseudo = np.array(feature_v3)
    X_pseudo_s = scaler.transform(X_pseudo)
    p_correct = model.predict_proba(X_pseudo_s)[:, 1]
    print(f"  Meta-learner P(correct) range: [{p_correct.min():.3f}, {p_correct.max():.3f}]")
    print(f"  Distribution: <0.3={int((p_correct<0.3).sum())}, 0.3-0.5={int(((p_correct>=0.3)&(p_correct<0.5)).sum())}, "
          f">=0.5={int((p_correct>=0.5).sum())}, >=0.7={int((p_correct>=0.7).sum())}")

    # Filter: keep p_correct >= 0.5
    final_keep = np.zeros(len(df_v3_filtered), dtype=bool)
    for k, i in enumerate(valid_indices):
        if p_correct[k] >= 0.5:
            df_idx = df_v3_filtered.index.get_loc(i) if i in df_v3_filtered.index else -1
            if df_idx >= 0:
                final_keep[df_idx] = True

    df_v4 = df_v3_filtered[final_keep].copy()
    print(f"\n  After meta-learner filter (p_correct >= 0.5): {len(df_v4)}")

    # Per-taxon
    df_v4["taxon"] = df_v4.primary_label.map({primary[i]: sp_taxon[i] for i in range(N_CLS)}).fillna("?")
    print(f"\n  v4 per-taxon distribution:")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        n_t = (df_v4.taxon == t).sum()
        n_classes = df_v4[df_v4.taxon == t].primary_label.nunique()
        print(f"    {t}: {n_t} entries, {n_classes} classes")

    df_v4.to_csv(OUT_V4, index=False)
    print(f"\n  Saved v4 → {OUT_V4}")

    # Save diagnostics
    np.savez_compressed(OUT_DIAG / "meta_learner_diagnostics.npz",
                          coef=model.coef_[0],
                          feat_cols=np.array(feat_cols),
                          p_correct_v3=p_correct,
                          centroids_within_dist=centroid_within_dist)
    print(f"  Saved diag → {OUT_DIAG}/meta_learner_diagnostics.npz")


if __name__ == "__main__":
    main()
