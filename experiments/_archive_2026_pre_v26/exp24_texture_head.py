#!/usr/bin/env python3
"""
exp24 — dedicated texture-taxa (Insecta + Amphibia) head with richer features.

Hypothesis (from exp21): texture taxa suffer most under the v1 pipeline:
  - Insecta sonotypes (47158sonXX): AUC 0.500 → 0.108 with prior fusion
  - Amphibia: 0.804 → 0.524 with prior fusion
The v1 probes use PCA(32) + scalar features per window. For texture taxa, which
are sustained calls with clear file-level structure, longer context and richer
embedding features should help.

Method: for each texture-class with ≥5 positives in Y_FULL, train a binary
LogReg on:
  - Full 1536-d Perch embedding of current window
  - File-level mean embedding (averaged over 12 windows of same file)
  - File-level max-pooled Perch logits for mapped texture classes (context)
Compare to v1's PCA(32)+scalar baseline on per-class AUC, OOF GroupKFold.

Outputs:
  experiments/exp24_outputs/results.json
  experiments/exp24_outputs/per_class_compare.csv
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
PERCH_DIR = ROOT / "perch_v2"
OUT = ROOT / "experiments" / "exp24_outputs"
OUT.mkdir(parents=True, exist_ok=True)
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"

N_WINDOWS = 12
PROBE_PCA_DIM = 32
PROBE_C = 0.25
TEXTURE_TAXA = {"Amphibia", "Insecta"}


def per_class_auc(y_true, y_score):
    aucs = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        if 0 < y_true[:, j].sum() < len(y_true):
            try: aucs[j] = roc_auc_score(y_true[:, j], y_score[:, j])
            except ValueError: pass
    return aucs


def file_mean_emb(emb, meta_full):
    """For each row, compute mean embedding of all 12 windows of the same file."""
    fm = np.zeros_like(emb)
    by_file = meta_full.groupby("filename").indices
    for fn, idx in by_file.items():
        m = emb[idx].mean(axis=0)
        fm[idx] = m
    return fm


def main():
    t0 = time.time()
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary)}
    n_classes = len(primary)

    fre = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
    def parse_lbls(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_clean = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
                .apply(lambda s: sorted({lbl for x in s for lbl in parse_lbls(x)}))
                .reset_index(name="label_list"))
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (sc_clean["filename"].str.replace(".ogg", "", regex=False)
                          + "_" + sc_clean["end_sec"].astype(str))
    Y_SC = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, labs in enumerate(sc_clean["label_list"]):
        for lbl in labs:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    # Load exp21 cache
    meta_full = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    arr = np.load(EXP21 / "full_perch_arrays.npz")
    scores_full = arr["scores"].astype(np.float32)
    emb_full = arr["emb"].astype(np.float32)
    sites_full = meta_full["site"].to_numpy()
    sc_idx = sc_clean.set_index("row_id")
    Y_FULL = np.stack([Y_SC[sc_idx.index.get_loc(rid)] for rid in meta_full["row_id"]])

    cn_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    texture_classes = [c for c in primary if cn_map.get(c) in TEXTURE_TAXA]
    texture_idx = np.array([label_to_idx[c] for c in texture_classes], dtype=np.int32)
    print(f"Texture classes (Insecta+Amphibia): {len(texture_classes)}")
    print(f"Active texture classes in Y_FULL: {(Y_FULL[:, texture_idx].sum(0) > 0).sum()}")

    # File-level mean embedding (context)
    file_emb = file_mean_emb(emb_full, meta_full)

    # Get max-pooled mapped texture logits per file (context for related classes)
    # But we don't have a mapped-texture mask directly. Use texture classes that have nonzero Perch.
    perch_mapped_tex_mask = (np.abs(scores_full[:, texture_idx]).sum(0) > 1e-3)
    mapped_tex_idx = texture_idx[perch_mapped_tex_mask]
    print(f"Mapped texture (Perch fires): {len(mapped_tex_idx)}")

    file_max_logits = np.zeros((len(meta_full), len(mapped_tex_idx)), dtype=np.float32)
    by_file = meta_full.groupby("filename").indices
    for fn, idx in by_file.items():
        m = scores_full[idx][:, mapped_tex_idx].max(axis=0)
        file_max_logits[idx] = m

    # Build feature matrices per fold
    gkf = GroupKFold(n_splits=5)
    oof_v1 = np.zeros_like(scores_full, dtype=np.float32)
    oof_rich = np.zeros_like(scores_full, dtype=np.float32)

    for fi, (tr_idx, va_idx) in enumerate(gkf.split(scores_full, groups=sites_full)):
        tr_idx = np.sort(tr_idx); va_idx = np.sort(va_idx)
        # v1 probe features: PCA32 + raw + base + seq features (mirror exp21)
        scaler_v1 = StandardScaler()
        Et = scaler_v1.fit_transform(emb_full[tr_idx])
        Ev = scaler_v1.transform(emb_full[va_idx])
        n_comp = min(PROBE_PCA_DIM, Et.shape[0] - 1, Et.shape[1])
        pca = PCA(n_components=n_comp)
        Zt = pca.fit_transform(Et).astype(np.float32)
        Zv = pca.transform(Ev).astype(np.float32)

        # rich features: full emb (scaled) + file_emb (scaled) + file_max_logits
        scaler_full = StandardScaler()
        Ft_emb = scaler_full.fit_transform(emb_full[tr_idx]).astype(np.float32)
        Fv_emb = scaler_full.transform(emb_full[va_idx]).astype(np.float32)
        scaler_fe = StandardScaler()
        Ft_fe = scaler_fe.fit_transform(file_emb[tr_idx]).astype(np.float32)
        Fv_fe = scaler_fe.transform(file_emb[va_idx]).astype(np.float32)
        scaler_fl = StandardScaler()
        Ft_fl = scaler_fl.fit_transform(file_max_logits[tr_idx]).astype(np.float32)
        Fv_fl = scaler_fl.transform(file_max_logits[va_idx]).astype(np.float32)

        oof_v1[va_idx] = scores_full[va_idx]
        oof_rich[va_idx] = scores_full[va_idx]

        for cls in texture_idx:
            yc = Y_FULL[tr_idx, cls].astype(np.float32)
            if yc.sum() < 5 or yc.sum() == len(yc):
                continue
            # v1 probe
            X_v1_t = np.concatenate([Zt, scores_full[tr_idx, cls:cls+1]], axis=1)
            X_v1_v = np.concatenate([Zv, scores_full[va_idx, cls:cls+1]], axis=1)
            try:
                clf1 = LogisticRegression(C=PROBE_C, max_iter=300, solver="liblinear", class_weight="balanced")
                clf1.fit(X_v1_t, yc)
                pred1 = clf1.decision_function(X_v1_v).astype(np.float32)
                oof_v1[va_idx, cls] = 0.5 * scores_full[va_idx, cls] + 0.5 * pred1
            except Exception:
                pass
            # rich probe (full emb + file_emb + file_max_logits)
            X_rich_t = np.concatenate([Ft_emb, Ft_fe, Ft_fl], axis=1)
            X_rich_v = np.concatenate([Fv_emb, Fv_fe, Fv_fl], axis=1)
            try:
                clf2 = LogisticRegression(C=0.05, max_iter=300, solver="liblinear", class_weight="balanced")
                clf2.fit(X_rich_t, yc)
                pred2 = clf2.decision_function(X_rich_v).astype(np.float32)
                oof_rich[va_idx, cls] = 0.5 * scores_full[va_idx, cls] + 0.5 * pred2
            except Exception:
                pass

    # Compare per texture class
    pc_v1 = per_class_auc(Y_FULL[:, texture_idx], oof_v1[:, texture_idx])
    pc_rich = per_class_auc(Y_FULL[:, texture_idx], oof_rich[:, texture_idx])
    pc_raw = per_class_auc(Y_FULL[:, texture_idx], scores_full[:, texture_idx])

    df = pd.DataFrame({
        "primary": texture_classes,
        "class_name": [cn_map[c] for c in texture_classes],
        "n_pos": Y_FULL[:, texture_idx].sum(0),
        "auc_raw_perch": pc_raw,
        "auc_v1_probe": pc_v1,
        "auc_rich_probe": pc_rich,
    })
    df["delta_rich_vs_v1"] = df["auc_rich_probe"] - df["auc_v1_probe"]
    df = df.sort_values("n_pos", ascending=False)
    df.to_csv(OUT / "per_class_compare.csv", index=False)

    active = df[df["n_pos"] > 0]
    print(f"\nActive texture classes: {len(active)}")
    print(active.head(20).to_string())

    # Macro AUC over texture-only
    keep = (Y_FULL[:, texture_idx].sum(0) > 0)
    y_tex = Y_FULL[:, texture_idx][:, keep]
    auc_raw = float(roc_auc_score(y_tex, scores_full[:, texture_idx][:, keep], average="macro"))
    auc_v1 = float(roc_auc_score(y_tex, oof_v1[:, texture_idx][:, keep], average="macro"))
    auc_rich = float(roc_auc_score(y_tex, oof_rich[:, texture_idx][:, keep], average="macro"))
    print(f"\n=== Texture-only Macro AUC (OOF) ===")
    print(f"  raw Perch     : {auc_raw:.4f}")
    print(f"  v1 probe      : {auc_v1:.4f}  (Δ {auc_v1 - auc_raw:+.4f})")
    print(f"  rich probe    : {auc_rich:.4f}  (Δ {auc_rich - auc_raw:+.4f})")
    print(f"  rich vs v1    : Δ {auc_rich - auc_v1:+.4f}")

    # Group by Insecta vs Amphibia
    by_taxa = active.groupby("class_name")[["auc_raw_perch", "auc_v1_probe", "auc_rich_probe"]].mean().round(3)
    print(f"\nBy taxa (mean per-class AUC, active only):")
    print(by_taxa)

    results = {
        "n_texture_classes": len(texture_classes),
        "n_active_texture": int(keep.sum()),
        "auc_raw_perch_texture": auc_raw,
        "auc_v1_probe_texture": auc_v1,
        "auc_rich_probe_texture": auc_rich,
        "delta_rich_vs_v1": auc_rich - auc_v1,
        "by_taxa": by_taxa.to_dict(),
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT/'results.json'}  Wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
