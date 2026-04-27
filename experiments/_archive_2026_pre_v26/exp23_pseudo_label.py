#!/usr/bin/env python3
"""
exp23 — pseudo-labeling on partially-labeled train_soundscapes.

Hypothesis: 147 partially-labeled SS files (out of 206) are unused by the
0.910 pipeline because their windows are not all annotated. Self-training
with pseudo-labels can expand the effective training set for probes.

Method:
  1. Extract Perch on the 147 partial-labeled files (cache).
  2. Build a "teacher" prediction by running exp21's full pipeline (in-sample
     priors + LogReg probes trained on Y_FULL) on the partial windows.
  3. Generate soft pseudo-labels: keep top-K predictions per window, scaled.
  4. Retrain probes on (Y_FULL + Y_pseudo) with sample weights (real=1, pseudo=w).
  5. Evaluate via GroupKFold-by-site on Y_FULL real labels.

Outputs:
  experiments/exp23_outputs/partial_perch.npz       (Perch cache for partial files)
  experiments/exp23_outputs/results.json
"""
from __future__ import annotations
import gc, json, os, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
PERCH_DIR = ROOT / "perch_v2"
OUT = ROOT / "experiments" / "exp23_outputs"
OUT.mkdir(parents=True, exist_ok=True)
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"

SR = 32_000
WINDOW_SAMPLES = 5 * SR
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 8
PROBE_PCA_DIM = 32
PROBE_C = 0.25
PROBE_MIN_POS = 8
LAMBDA_EVENT = 0.4
LAMBDA_TEXTURE = 1.0

# Pseudo-label settings
PSEUDO_TOPK = 3        # keep top-K predicted classes per window
PSEUDO_THRESH = 0.0    # logit threshold; only labels above this become pseudo positive
PSEUDO_WEIGHT = 0.3    # sample weight for pseudo rows (vs 1.0 for real)


def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


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
    def parse_meta(name):
        m = fre.match(name)
        if not m: return None, -1
        _, site, _, hms = m.groups()
        return site, int(hms[:2])

    sc_clean = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
                .apply(lambda s: sorted({lbl for x in s for lbl in parse_lbls(x)}))
                .reset_index(name="label_list"))
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (sc_clean["filename"].str.replace(".ogg", "", regex=False)
                          + "_" + sc_clean["end_sec"].astype(str))
    meta_cols = sc_clean["filename"].apply(lambda n: pd.Series(dict(zip(("site","hour_utc"), parse_meta(n)))))
    sc_clean = pd.concat([sc_clean, meta_cols], axis=1)
    wpf = sc_clean.groupby("filename").size()
    full_files = sorted(wpf[wpf == N_WINDOWS].index.tolist())
    partial_files = sorted([f for f in wpf.index if f not in set(full_files)])

    Y_SC = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, labs in enumerate(sc_clean["label_list"]):
        for lbl in labs:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    print(f"Full files: {len(full_files)}  Partial files: {len(partial_files)}")
    print(f"Total sc_clean rows: {len(sc_clean)}  with positive label: {(Y_SC.sum(1) > 0).sum()}")

    # Load exp21 Perch cache (fully-labeled)
    meta_full = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    arr = np.load(EXP21 / "full_perch_arrays.npz")
    scores_full = arr["scores"].astype(np.float32)
    emb_full = arr["emb"].astype(np.float32)

    sc_idx = sc_clean.set_index("row_id")
    Y_FULL = np.stack([Y_SC[sc_idx.index.get_loc(rid)] for rid in meta_full["row_id"]])
    sites_full = meta_full["site"].to_numpy()
    hours_full = meta_full["hour_utc"].to_numpy()

    # Phase 1: Perch on partial files
    cache_p = OUT / "partial_perch.npz"
    cache_m = OUT / "partial_perch_meta.parquet"
    if cache_p.exists() and cache_m.exists():
        meta_p = pd.read_parquet(cache_m)
        d = np.load(cache_p)
        scores_p = d["scores"].astype(np.float32)
        emb_p = d["emb"].astype(np.float32)
        print(f"Loaded partial cache: {scores_p.shape}, {emb_p.shape}")
    else:
        # Need taxonomy mapping for partial Perch inference
        bc_labels = (pd.read_csv(PERCH_DIR / "assets" / "labels.csv").reset_index()
                     .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
        no_label = len(bc_labels)
        tax = taxonomy.copy(); tax["scientific_name"] = tax["scientific_name"].astype(str)
        mapping = tax.merge(bc_labels[["scientific_name", "bc_index"]], on="scientific_name", how="left")
        mapping["bc_index"] = mapping["bc_index"].fillna(no_label).astype(int)
        BC_INDICES = np.array([int(mapping.set_index("primary_label")["bc_index"].loc[c]) for c in primary], dtype=np.int32)
        MAPPED_MASK = BC_INDICES != no_label
        MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
        MAPPED_BC = BC_INDICES[MAPPED_MASK].astype(np.int32)

        # Genus proxy
        unmapped_df = mapping[mapping["bc_index"] == no_label].copy()
        unmapped_non_son = unmapped_df[~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)].copy()
        proxy_pos_to_bc = {}
        CLASS_NAME = tax.set_index("primary_label")["class_name"].to_dict()
        for _, row in unmapped_non_son.iterrows():
            if CLASS_NAME.get(row["primary_label"]) != "Amphibia":
                continue
            genus = str(row["scientific_name"]).split()[0]
            hits = bc_labels[bc_labels["scientific_name"].str.match(rf"^{re.escape(genus)}\s", na=False)]
            if len(hits) > 0:
                proxy_pos_to_bc[label_to_idx[row["primary_label"]]] = hits["bc_index"].astype(int).to_numpy()

        print("Loading Perch v2 ...")
        model = tf.saved_model.load(str(PERCH_DIR))
        infer = model.signatures["serving_default"]

        n_files = len(partial_files)
        n_rows = n_files * N_WINDOWS
        meta_p = pd.DataFrame({
            "row_id": np.empty(n_rows, dtype=object),
            "filename": np.empty(n_rows, dtype=object),
            "site": np.empty(n_rows, dtype=object),
            "hour_utc": np.zeros(n_rows, dtype=np.int16),
        })
        scores_p = np.zeros((n_rows, n_classes), dtype=np.float32)
        emb_p = np.zeros((n_rows, 1536), dtype=np.float32)

        write = 0
        for start in tqdm(range(0, n_files, BATCH_FILES), desc="partial Perch"):
            batch = partial_files[start:start + BATCH_FILES]
            bn = len(batch)
            x = np.zeros((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            bstart = write
            for bi, fn in enumerate(batch):
                p = DATA / "train_soundscapes" / fn
                y, sr = sf.read(p, dtype="float32", always_2d=False)
                if y.ndim == 2: y = y.mean(axis=1)
                if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
                y = y[:FILE_SAMPLES]
                x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                site, hour = parse_meta(fn)
                for j, t in enumerate(range(5, 65, 5)):
                    rid = f"{Path(fn).stem}_{t}"
                    meta_p.iloc[write + j, meta_p.columns.get_loc("row_id")] = rid
                    meta_p.iloc[write + j, meta_p.columns.get_loc("filename")] = fn
                    meta_p.iloc[write + j, meta_p.columns.get_loc("site")] = site
                    meta_p.iloc[write + j, meta_p.columns.get_loc("hour_utc")] = hour
                write += N_WINDOWS

            out = infer(inputs=tf.convert_to_tensor(x))
            logits = out["label"].numpy().astype(np.float32)
            em = out["embedding"].numpy().astype(np.float32)
            scores_p[bstart:write, MAPPED_POS] = logits[:, MAPPED_BC]
            emb_p[bstart:write] = em
            for pos, arr in proxy_pos_to_bc.items():
                scores_p[bstart:write, pos] = logits[:, arr].max(axis=1)
            del x, out, logits, em
            gc.collect()

        meta_p.to_parquet(cache_m, index=False)
        np.savez_compressed(cache_p, scores=scores_p, emb=emb_p)
        print(f"Cached partial → {cache_p}")

    # Phase 2: Build teacher predictions on PARTIAL using full+probes (trained on Y_FULL)
    # Simple teacher = raw Perch scores on partial (as a starting point — could elaborate)
    teacher_p = scores_p.copy()
    print(f"Teacher predictions on partial: shape {teacher_p.shape}")

    # Phase 3: Generate pseudo-labels: top-K per window where logit > thresh
    Y_PSEUDO = np.zeros_like(teacher_p, dtype=np.float32)
    # For each window, take top-K logits above threshold
    topk = np.argpartition(-teacher_p, PSEUDO_TOPK, axis=1)[:, :PSEUDO_TOPK]
    for i in range(teacher_p.shape[0]):
        for k in topk[i]:
            if teacher_p[i, k] > PSEUDO_THRESH:
                Y_PSEUDO[i, k] = 1.0
    # Also include any REAL labels from sc_clean for these partial windows
    for i, rid in enumerate(meta_p["row_id"]):
        if rid in sc_idx.index:
            real_y = Y_SC[sc_idx.index.get_loc(rid)]
            Y_PSEUDO[i] = np.maximum(Y_PSEUDO[i], real_y.astype(np.float32))

    print(f"Pseudo positives per row (mean): {Y_PSEUDO.sum(axis=1).mean():.2f}")
    print(f"Pseudo positives per class (top 10):")
    pp = Y_PSEUDO.sum(axis=0)
    for ci in np.argsort(-pp)[:10]:
        print(f"  {primary[ci]:20s} {int(pp[ci])}")

    # Phase 4: Train probes on (Y_FULL + Y_PSEUDO) with weights
    # Build combined embedding + label sets per fold
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros_like(scores_full, dtype=np.float32)

    for fi, (tr_idx, va_idx) in enumerate(tqdm(list(gkf.split(scores_full, groups=sites_full)), desc="folds")):
        tr_idx = np.sort(tr_idx); va_idx = np.sort(va_idx)
        val_sites = set(sites_full[va_idx].tolist())
        # Filter pseudo to exclude val sites
        partial_sites = meta_p["site"].to_numpy()
        partial_mask = ~pd.Series(partial_sites).isin(val_sites).to_numpy()

        emb_tr = np.concatenate([emb_full[tr_idx], emb_p[partial_mask]], axis=0)
        Y_tr = np.concatenate([Y_FULL[tr_idx].astype(np.float32), Y_PSEUDO[partial_mask]], axis=0)
        sw_tr = np.concatenate([
            np.ones(len(tr_idx), dtype=np.float32),
            np.full(int(partial_mask.sum()), PSEUDO_WEIGHT, dtype=np.float32),
        ])

        scaler = StandardScaler()
        Et = scaler.fit_transform(emb_tr)
        Ev = scaler.transform(emb_full[va_idx])
        n_comp = min(PROBE_PCA_DIM, Et.shape[0] - 1, Et.shape[1])
        pca = PCA(n_components=n_comp)
        Zt = pca.fit_transform(Et).astype(np.float32)
        Zv = pca.transform(Ev).astype(np.float32)

        oof[va_idx] = scores_full[va_idx]  # start from raw Perch as base
        # Train per-class LogReg on COMBINED data
        for cls in range(n_classes):
            yc = Y_tr[:, cls]
            if yc.sum() < PROBE_MIN_POS or yc.sum() == len(yc):
                continue
            X_t = np.concatenate([Zt, scores_full[tr_idx][:, cls:cls+1] if False else np.zeros((len(Zt), 1))], axis=1)
            # Just use embedding features for simplicity
            clf = LogisticRegression(C=PROBE_C, max_iter=300, solver="liblinear", class_weight="balanced")
            try:
                clf.fit(Zt, yc, sample_weight=sw_tr)
            except Exception:
                continue
            pred = clf.decision_function(Zv).astype(np.float32)
            oof[va_idx, cls] = 0.5 * scores_full[va_idx, cls] + 0.5 * pred  # naive blend

    # Evaluate
    auc_oof = macro_auc(Y_FULL, oof)
    auc_raw = macro_auc(Y_FULL, scores_full)
    print(f"\n=== exp23 results ===")
    print(f"Raw Perch (baseline)          : {auc_raw:.4f}")
    print(f"+ pseudo-labels probes (OOF)  : {auc_oof:.4f}  (Δ {auc_oof - auc_raw:+.4f})")

    # Compare with NO pseudo (use real Y_FULL only for probes)
    oof_real_only = np.zeros_like(scores_full, dtype=np.float32)
    for fi, (tr_idx, va_idx) in enumerate(gkf.split(scores_full, groups=sites_full)):
        tr_idx = np.sort(tr_idx); va_idx = np.sort(va_idx)
        scaler = StandardScaler()
        Et = scaler.fit_transform(emb_full[tr_idx])
        Ev = scaler.transform(emb_full[va_idx])
        n_comp = min(PROBE_PCA_DIM, Et.shape[0] - 1, Et.shape[1])
        pca = PCA(n_components=n_comp)
        Zt = pca.fit_transform(Et).astype(np.float32)
        Zv = pca.transform(Ev).astype(np.float32)
        oof_real_only[va_idx] = scores_full[va_idx]
        for cls in range(n_classes):
            yc = Y_FULL[tr_idx, cls].astype(np.float32)
            if yc.sum() < PROBE_MIN_POS or yc.sum() == len(yc):
                continue
            clf = LogisticRegression(C=PROBE_C, max_iter=300, solver="liblinear", class_weight="balanced")
            try: clf.fit(Zt, yc)
            except Exception: continue
            pred = clf.decision_function(Zv).astype(np.float32)
            oof_real_only[va_idx, cls] = 0.5 * scores_full[va_idx, cls] + 0.5 * pred
    auc_real_only = macro_auc(Y_FULL, oof_real_only)
    print(f"+ real-only probes (OOF)      : {auc_real_only:.4f}  (Δ {auc_real_only - auc_raw:+.4f})")
    print(f"Pseudo-label gain over real-only: {auc_oof - auc_real_only:+.4f}")

    results = {
        "n_partial_files": len(partial_files),
        "n_partial_windows": int(scores_p.shape[0]),
        "n_pseudo_pos_total": float(Y_PSEUDO.sum()),
        "auc_raw_perch": auc_raw,
        "auc_real_only_probes_oof": auc_real_only,
        "auc_pseudo_probes_oof": auc_oof,
        "delta_pseudo_vs_real": auc_oof - auc_real_only,
        "settings": {"PSEUDO_TOPK": PSEUDO_TOPK, "PSEUDO_THRESH": PSEUDO_THRESH, "PSEUDO_WEIGHT": PSEUDO_WEIGHT},
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT/'results.json'}  Wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
