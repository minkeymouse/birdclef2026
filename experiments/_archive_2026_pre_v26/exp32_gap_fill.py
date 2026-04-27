"""
exp32 — evidence gap fills for exp24/25/26.

A. exp25 caveat fix: site/hour classifier AUCs under FILE-GROUPED CV (not window-level).
B. exp26 R3 revisit: taxa-CONDITIONAL per-file centering (only apply to taxa where it helped).
C. exp24 follow-up: rich-head (1536d + file_mean + context) under Val-A (only Val-B tested before).
D. exp22 follow-up: train_audio → SS domain bridge via per-class mean shift in embedding space.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, StratifiedKFold

ROOT = Path("/data/birdclef2026")
CACHE = ROOT / "experiments/exp21_outputs/perch_cache"
EXP28 = ROOT / "experiments/exp28_outputs"
OUT = ROOT / "experiments/exp32_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/birdclef-2026"


def load_all():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sample_sub.columns[1:].tolist()
    lab2idx = {c: i for i, c in enumerate(primary)}
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    class_name = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"]))

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    meta = pd.read_parquet(CACHE / "full_perch_meta.parquet")
    Y = np.zeros((len(meta), len(primary)), dtype=np.uint8)
    by_rowid = sc.set_index("row_id")
    for i, rid in enumerate(meta["row_id"]):
        if rid in by_rowid.index:
            for l in by_rowid.loc[rid, "lbls"]:
                if l in lab2idx:
                    Y[i, lab2idx[l]] = 1

    arr = np.load(CACHE / "full_perch_arrays.npz")
    scores_raw = arr["scores"]      # (708, 234)  Perch raw
    emb = arr["emb"]                # (708, 1536) Perch embeddings

    return meta, Y, primary, class_name, scores_raw, emb


def macro_auc(y_true, y_score, mask_extra=None):
    keep = y_true.sum(0) > 0
    if mask_extra is not None:
        keep = keep & mask_extra
    if keep.sum() < 2:
        return float("nan")
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def val_a_folds(meta):
    files = meta.drop_duplicates("filename").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f2f = {}
    for fold, (_, vi) in enumerate(skf.split(files["filename"], files["site"])):
        for f in files.iloc[vi]["filename"].values:
            f2f[f] = fold
    return meta["filename"].map(f2f).values


def val_b_folds(meta):
    gkf = GroupKFold(n_splits=min(5, meta["site"].nunique()))
    folds = np.full(len(meta), -1)
    for fold, (_, vi) in enumerate(gkf.split(meta, groups=meta["site"])):
        folds[vi] = fold
    return folds


# ═══ A. exp25 caveat fix: file-grouped site/hour classifier ══════════════

def a_site_hour_honest(meta, emb):
    print("\n[A] Honest site/hour classifier under FILE-grouped CV")
    files = meta["filename"].values
    sites = meta["site"].values
    hours = meta["hour_utc"].values

    # Site classifier — file-grouped 5-fold
    from sklearn.preprocessing import LabelEncoder
    le_s = LabelEncoder(); le_h = LabelEncoder()
    y_site = le_s.fit_transform(sites)
    y_hour = le_h.fit_transform(hours)
    n_site = len(le_s.classes_); n_hour = len(le_h.classes_)

    from collections import Counter
    def file_grouped_eval(y):
        # Stratified by first appearance per file to keep site/hour distribution balanced
        file_df = pd.DataFrame({"file": files, "y": y}).drop_duplicates("file")
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros((len(y),), dtype=np.int64)
        for fold, (ti, vi) in enumerate(sgkf.split(emb, y, groups=files)):
            clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", n_jobs=-1)
            clf.fit(emb[ti], y[ti])
            preds[vi] = clf.predict(emb[vi])
        return (preds == y).mean()

    site_acc = file_grouped_eval(y_site)
    hour_acc = file_grouped_eval(y_hour)
    print(f"  site acc (file-grouped): {site_acc:.4f}  vs chance 1/{n_site}={1/n_site:.4f}")
    print(f"  hour acc (file-grouped): {hour_acc:.4f}  vs chance 1/{n_hour}={1/n_hour:.4f}")
    return {
        "site_acc_file_grouped": site_acc, "site_chance": 1/n_site, "n_sites": int(n_site),
        "hour_acc_file_grouped": hour_acc, "hour_chance": 1/n_hour, "n_hours": int(n_hour),
    }


# ═══ B. exp26 R3 revisit: taxa-conditional per-file centering ═════════════

def b_taxa_conditional_centering(meta, Y, scores_raw, emb, primary, class_name):
    """
    Start from exp28's best probe output (val_a_probe in best_oof.npz), apply per-file centering
    ONLY to classes in selected taxa. Grid over: {Reptilia only, Reptilia+Mammalia, all-non-Insecta}.
    """
    print("\n[B] Taxa-conditional per-file centering on exp28 probe output")
    d = np.load(EXP28 / "best_oof.npz")
    probe_a = d["val_a_probe"].copy()
    probe_b = d["val_b_probe"].copy()
    base_a = macro_auc(Y, probe_a)
    base_b = macro_auc(Y, probe_b)
    print(f"  baseline (no center): Val-A {base_a:.4f}  Val-B {base_b:.4f}")

    # taxa masks over 234 classes
    class_of = [class_name.get(c, "unknown") for c in primary]
    TAXA_SETS = {
        "Reptilia_only": {"Reptilia"},
        "Reptilia+Amphibia": {"Reptilia", "Amphibia"},
        "non_Aves_non_Insecta": {"Amphibia", "Reptilia", "Mammalia"},
        "non_Insecta": {"Amphibia", "Reptilia", "Mammalia", "Aves"},
        "all": {"Amphibia", "Reptilia", "Mammalia", "Aves", "Insecta"},
    }

    # Per-file means
    def apply_center(x, taxa_mask, alpha):
        """x: (708, 234). Subtract (alpha × per-file mean) only on taxa_mask cols."""
        out = x.copy()
        for fn, g in meta.groupby("filename", sort=False):
            idx = g.index.values
            chunk = out[idx]  # (12, 234)
            mean = chunk.mean(axis=0, keepdims=True)  # (1, 234)
            chunk[:, taxa_mask] = chunk[:, taxa_mask] - alpha * mean[:, taxa_mask]
            out[idx] = chunk
        return out

    results = []
    for name, taxa in TAXA_SETS.items():
        mask = np.array([c in taxa for c in class_of])
        for alpha in [0.3, 0.5, 0.7, 1.0]:
            pa = apply_center(probe_a, mask, alpha)
            pb = apply_center(probe_b, mask, alpha)
            auc_a = macro_auc(Y, pa); auc_b = macro_auc(Y, pb)
            results.append({"taxa": name, "alpha": alpha, "val_a": auc_a, "val_b": auc_b,
                            "delta_a": auc_a - base_a, "delta_b": auc_b - base_b,
                            "n_classes_centered": int(mask.sum())})
            print(f"  {name:25s} α={alpha:.1f}  Val-A {auc_a:.4f} ({auc_a-base_a:+.4f})  Val-B {auc_b:.4f} ({auc_b-base_b:+.4f})")
    return {"baseline_val_a": base_a, "baseline_val_b": base_b, "grid": results}


# ═══ C. exp24 follow-up: rich head under Val-A ═════════════════════════════

def c_rich_head_val_a(meta, Y, emb, scores_raw):
    """
    Rich head = [raw Perch score for class c, per-file mean of scores, per-file max, PCA32 emb, file_mean_emb].
    Train LogReg probe per class under Val-A 5-fold (file-stratified by site).
    Compare to PCA32-only baseline (= exp28's LB910_freeze).
    """
    print("\n[C] Rich head vs PCA32-only under Val-A")
    fold_a = val_a_folds(meta)
    # PCA32
    pca = PCA(n_components=32, random_state=42).fit(emb)
    emb_pca = pca.transform(emb).astype(np.float32)

    # Per-file stats
    mean_emb = np.zeros_like(emb)
    mean_score = np.zeros_like(scores_raw)
    max_score = np.zeros_like(scores_raw)
    for fn, g in meta.groupby("filename", sort=False):
        idx = g.index.values
        mean_emb[idx] = emb[idx].mean(0, keepdims=True)
        mean_score[idx] = scores_raw[idx].mean(0, keepdims=True)
        max_score[idx] = scores_raw[idx].max(0, keepdims=True)
    mean_emb_pca = pca.transform(mean_emb).astype(np.float32)

    MIN_POS = 8
    C_REG = 0.25
    N = len(Y)

    def train_probes(X_builder, label="var"):
        preds = np.zeros_like(scores_raw, dtype=np.float32)
        n_trained = 0
        for c in range(Y.shape[1]):
            if Y[:, c].sum() < MIN_POS: continue
            Xc = X_builder(c)
            for f in range(5):
                ti = fold_a != f; vi = ~ti
                if Y[ti, c].sum() < 3: continue
                try:
                    clf = LogisticRegression(max_iter=500, C=C_REG)
                    clf.fit(Xc[ti], Y[ti, c])
                    preds[vi, c] = clf.decision_function(Xc[vi])
                except Exception:
                    pass
            n_trained += 1
        auc = macro_auc(Y, preds)
        print(f"  {label:25s}  n_classes_trained {n_trained}  Val-A {auc:.4f}")
        return auc, n_trained

    # Baseline: PCA32 only (similar to exp28)
    base_auc, base_n = train_probes(lambda c: emb_pca, label="PCA32_only")

    # Rich head: PCA32 + per-class scalar score + mean/max + mean_emb_pca16
    pca_m = PCA(n_components=16, random_state=42).fit(emb)
    mean_emb_pca16 = pca_m.transform(mean_emb).astype(np.float32)

    rich_auc, rich_n = train_probes(
        lambda c: np.concatenate([
            emb_pca,
            scores_raw[:, [c]].astype(np.float32),
            mean_score[:, [c]].astype(np.float32),
            max_score[:, [c]].astype(np.float32),
            mean_emb_pca16,
        ], axis=1),
        label="PCA32+scalar+mean_emb"
    )

    # Even richer: full 1536d + mean_emb (what exp24 found worst under Val-B)
    full_auc, full_n = train_probes(
        lambda c: np.concatenate([emb, mean_emb], axis=1),
        label="full1536+mean_emb_full"
    )
    return {"PCA32_only": base_auc, "PCA32_scalar_meanembpca16": rich_auc,
            "full1536+mean_emb_full": full_auc,
            "delta_rich_vs_base": rich_auc - base_auc,
            "delta_full_vs_base": full_auc - base_auc}


# ═══ D. train_audio domain bridge via per-class embedding mean shift ══════
# (Skipping for now — would need train_audio Perch cache which is separate.)


def main():
    t0 = time.time()
    meta, Y, primary, class_name, scores_raw, emb = load_all()
    print(f"Loaded: meta {meta.shape}, Y {Y.shape}, emb {emb.shape}, primary {len(primary)}")

    out = {"elapsed_s": None}
    out["A_site_hour"] = a_site_hour_honest(meta, emb)
    out["B_taxa_centering"] = b_taxa_conditional_centering(meta, Y, scores_raw, emb, primary, class_name)
    out["C_rich_head"] = c_rich_head_val_a(meta, Y, emb, scores_raw)
    out["elapsed_s"] = time.time() - t0

    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nDone. Elapsed {(time.time()-t0)/60:.1f} min. Saved {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
