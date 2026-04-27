#!/usr/bin/env python3
"""
exp21 — OOF measurement + ablation harness for the 0.910 LB pipeline.

Goal: decompose where the 0.910 score comes from. Run each component on the same
GroupKFold(site) splits over fully-labeled train_soundscapes and report macro-AUC.

Conditions (cumulative, each adds one component):
  A: raw Perch logit (mapped classes only, others = -8)
  B: + genus proxy for unmapped Amphibia
  C: + Bayesian site/hour prior fusion (mapped+proxy+prioronly classes)
  D: + texture-class temporal smoothing (Amphibia/Insecta only)
  E: + PCA(32) LogReg probes blended with dynamic alpha
  F: + Gaussian smoothing across 12 windows per file (= full v1 pipeline)

Outputs:
  experiments/exp21_outputs/oof_ablation.json
  experiments/exp21_outputs/per_class_auc.csv
  experiments/exp21_outputs/perch_cache/full_perch_{meta.parquet,arrays.npz}
    (re-used by exp22-25)
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.ndimage import convolve1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Perch v2 SavedModel here is CPU-only

import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
PERCH_DIR = ROOT / "perch_v2"
OUT = ROOT / "experiments" / "exp21_outputs"
CACHE = OUT / "perch_cache"
CACHE.mkdir(parents=True, exist_ok=True)

SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12
BATCH_FILES = 8
SEED = 42

# Frozen v1 hyperparams (LB 0.910)
LAMBDA_EVENT = 0.4
LAMBDA_TEXTURE = 1.0
LAMBDA_PROXY_TEXTURE = 0.8
SMOOTH_TEXTURE_ALPHA = 0.35
PROBE_PCA_DIM = 32
PROBE_MIN_POS = 8
PROBE_C = 0.25
GAUSS_W = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

np.random.seed(SEED)


# ───────────────────────── data loading ─────────────────────────

def load_metadata():
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    soundscape_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    sc = soundscape_raw.drop_duplicates().reset_index(drop=True)

    primary = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary)}
    n_classes = len(primary)

    fname_re = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

    def parse_labels(x):
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    def union_labels(s):
        return sorted({lbl for x in s for lbl in parse_labels(x)})

    def parse_fname(name):
        m = fname_re.match(name)
        if not m:
            return {"site": None, "hour_utc": -1}
        _, site, _, hms = m.groups()
        return {"site": site, "hour_utc": int(hms[:2])}

    sc_clean = (
        sc.groupby(["filename", "start", "end"])["primary_label"]
        .apply(union_labels)
        .reset_index(name="label_list")
    )
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (
        sc_clean["filename"].str.replace(".ogg", "", regex=False)
        + "_" + sc_clean["end_sec"].astype(str)
    )
    meta_cols = sc_clean["filename"].apply(parse_fname).apply(pd.Series)
    sc_clean = pd.concat([sc_clean, meta_cols], axis=1)

    wpf = sc_clean.groupby("filename").size()
    full_files = sorted(wpf[wpf == N_WINDOWS].index.tolist())
    sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

    Y_SC = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, labels in enumerate(sc_clean["label_list"]):
        for lbl in labels:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    full_truth = (
        sc_clean[sc_clean["file_fully_labeled"]]
        .sort_values(["filename", "end_sec"])
        .reset_index(drop=False)
    )

    return dict(
        taxonomy=taxonomy,
        primary=primary,
        label_to_idx=label_to_idx,
        n_classes=n_classes,
        sc_clean=sc_clean,
        Y_SC=Y_SC,
        full_files=full_files,
        full_truth=full_truth,
    )


# ───────────────────────── Perch model ─────────────────────────

def load_perch():
    print("Loading Perch v2 ...")
    t0 = time.time()
    model = tf.saved_model.load(str(PERCH_DIR))
    infer = model.signatures["serving_default"]
    print(f"Perch loaded in {time.time() - t0:.1f}s")
    bc_labels = (
        pd.read_csv(PERCH_DIR / "assets" / "labels.csv")
        .reset_index()
        .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"})
    )
    return model, infer, bc_labels


def build_taxonomy_mapping(taxonomy, primary, label_to_idx, bc_labels, sc_clean, Y_SC):
    no_label = len(bc_labels)
    tax = taxonomy.copy()
    tax["scientific_name"] = tax["scientific_name"].astype(str)
    mapping = tax.merge(bc_labels[["scientific_name", "bc_index"]], on="scientific_name", how="left")
    mapping["bc_index"] = mapping["bc_index"].fillna(no_label).astype(int)

    label_to_bc = mapping.set_index("primary_label")["bc_index"]
    BC_INDICES = np.array([int(label_to_bc.loc[c]) for c in primary], dtype=np.int32)
    MAPPED_MASK = BC_INDICES != no_label
    MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
    UNMAPPED_POS = np.where(~MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC = BC_INDICES[MAPPED_MASK].astype(np.int32)

    CLASS_NAME = tax.set_index("primary_label")["class_name"].to_dict()
    TEXTURE_TAXA = {"Amphibia", "Insecta"}
    ACTIVE = [primary[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]

    idx_active_texture = np.array(
        [label_to_idx[c] for c in ACTIVE if CLASS_NAME.get(c) in TEXTURE_TAXA], dtype=np.int32
    )
    idx_active_event = np.array(
        [label_to_idx[c] for c in ACTIVE if CLASS_NAME.get(c) not in TEXTURE_TAXA], dtype=np.int32
    )
    idx_mapped_active_texture = idx_active_texture[MAPPED_MASK[idx_active_texture]]
    idx_mapped_active_event = idx_active_event[MAPPED_MASK[idx_active_event]]
    idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
    idx_unmapped_active_event = idx_active_event[~MAPPED_MASK[idx_active_event]]
    idx_unmapped_inactive = np.array(
        [i for i in UNMAPPED_POS if primary[i] not in ACTIVE], dtype=np.int32
    )

    # Genus proxy for unmapped Amphibia
    unmapped_df = mapping[mapping["bc_index"] == no_label].copy()
    unmapped_non_son = unmapped_df[
        ~unmapped_df["primary_label"].astype(str).str.contains("son", na=False)
    ].copy()
    proxy_map = {}
    for _, row in unmapped_non_son.iterrows():
        genus = str(row["scientific_name"]).split()[0]
        hits = bc_labels[bc_labels["scientific_name"].str.match(rf"^{re.escape(genus)}\s", na=False)]
        if len(hits) > 0:
            proxy_map[str(row["primary_label"])] = hits["bc_index"].astype(int).tolist()

    SELECTED_PROXY = sorted([t for t in proxy_map if CLASS_NAME.get(t) == "Amphibia"])
    proxy_pos = np.array([label_to_idx[c] for c in SELECTED_PROXY], dtype=np.int32)
    proxy_pos_to_bc = {label_to_idx[t]: np.array(proxy_map[t], dtype=np.int32) for t in SELECTED_PROXY}

    idx_selected_proxy_active_texture = np.intersect1d(proxy_pos, idx_active_texture)
    idx_prioronly_active_event = np.setdiff1d(idx_unmapped_active_event, proxy_pos)
    idx_prioronly_active_texture = np.setdiff1d(idx_unmapped_active_texture, proxy_pos)

    return dict(
        no_label=no_label,
        BC_INDICES=BC_INDICES,
        MAPPED_MASK=MAPPED_MASK,
        MAPPED_POS=MAPPED_POS,
        MAPPED_BC=MAPPED_BC,
        UNMAPPED_POS=UNMAPPED_POS,
        proxy_pos_to_bc=proxy_pos_to_bc,
        idx_active_texture=idx_active_texture,
        idx_active_event=idx_active_event,
        idx_mapped_active_event=idx_mapped_active_event,
        idx_mapped_active_texture=idx_mapped_active_texture,
        idx_selected_proxy_active_texture=idx_selected_proxy_active_texture,
        idx_prioronly_active_event=idx_prioronly_active_event,
        idx_prioronly_active_texture=idx_prioronly_active_texture,
        idx_unmapped_inactive=idx_unmapped_inactive,
    )


# ───────────────────────── Perch inference ─────────────────────────

def read_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    return y[:FILE_SAMPLES]


def perch_infer_files(infer, paths, n_classes, m):
    paths = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows = n_files * N_WINDOWS

    fname_re = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

    def parse_meta(name):
        mm = fname_re.match(name)
        if not mm:
            return None, -1
        _, site, _, hms = mm.groups()
        return site, int(hms[:2])

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, n_classes), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write = 0
    for start in tqdm(range(0, n_files, BATCH_FILES), desc="Perch"):
        batch = paths[start:start + BATCH_FILES]
        bn = len(batch)
        x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        bstart = write
        for bi, p in enumerate(batch):
            audio = read_60s(p)
            x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
            site, hour = parse_meta(p.name)
            row_ids[write:write + N_WINDOWS] = [f"{p.stem}_{t}" for t in range(5, 65, 5)]
            filenames[write:write + N_WINDOWS] = p.name
            sites[write:write + N_WINDOWS] = site
            hours[write:write + N_WINDOWS] = hour
            write += N_WINDOWS

        out = infer(inputs=tf.convert_to_tensor(x))
        logits = out["label"].numpy().astype(np.float32)
        emb = out["embedding"].numpy().astype(np.float32)

        scores[bstart:write, m["MAPPED_POS"]] = logits[:, m["MAPPED_BC"]]
        embeddings[bstart:write] = emb
        for pos, arr in m["proxy_pos_to_bc"].items():
            scores[bstart:write, pos] = logits[:, arr].max(axis=1)

        del x, out, logits, emb
        gc.collect()

    meta = pd.DataFrame({
        "row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours,
    })
    return meta, scores, embeddings


# ───────────────────────── prior fusion ─────────────────────────

def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)
    n_cls = Y_prior.shape[1]
    global_p = Y_prior.mean(axis=0).astype(np.float32)

    site_keys = sorted(prior_df["site"].dropna().astype(str).unique())
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique())

    site_to_i, site_n, site_p = {}, [], []
    site_arr = prior_df["site"].astype(str).values
    for s in site_keys:
        mask = site_arr == s
        site_to_i[s] = len(site_n)
        site_n.append(mask.sum())
        site_p.append(Y_prior[mask].mean(axis=0))
    site_n = np.array(site_n, dtype=np.float32)
    site_p = np.stack(site_p).astype(np.float32) if site_p else np.zeros((0, n_cls), np.float32)

    hour_to_i, hour_n, hour_p = {}, [], []
    hour_arr = prior_df["hour_utc"].astype(int).values
    for h in hour_keys:
        mask = hour_arr == h
        hour_to_i[h] = len(hour_n)
        hour_n.append(mask.sum())
        hour_p.append(Y_prior[mask].mean(axis=0))
    hour_n = np.array(hour_n, dtype=np.float32)
    hour_p = np.stack(hour_p).astype(np.float32) if hour_p else np.zeros((0, n_cls), np.float32)

    sh_to_i, sh_n_l, sh_p_l = {}, [], []
    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_l)
        idx = np.array(list(idx))
        sh_n_l.append(len(idx))
        sh_p_l.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n_l, dtype=np.float32)
    sh_p = np.stack(sh_p_l).astype(np.float32) if sh_p_l else np.zeros((0, n_cls), np.float32)

    return dict(
        global_p=global_p,
        site_to_i=site_to_i, site_n=site_n, site_p=site_p,
        hour_to_i=hour_to_i, hour_n=hour_n, hour_p=hour_p,
        sh_to_i=sh_to_i, sh_n=sh_n, sh_p=sh_p,
    )


def prior_logits(sites, hours, T, eps=1e-4):
    n = len(sites)
    p = np.repeat(T["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)
    si = np.fromiter((T["site_to_i"].get(str(s), -1) for s in sites), np.int32, n)
    hi = np.fromiter(
        (T["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        np.int32, n,
    )
    shi = np.fromiter(
        (T["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1
         for s, h in zip(sites, hours)),
        np.int32, n,
    )
    valid = hi >= 0
    if valid.any():
        nh = T["hour_n"][hi[valid]][:, None]
        p[valid] = nh / (nh + 8.0) * T["hour_p"][hi[valid]] + (1 - nh / (nh + 8.0)) * p[valid]
    valid = si >= 0
    if valid.any():
        ns = T["site_n"][si[valid]][:, None]
        p[valid] = ns / (ns + 8.0) * T["site_p"][si[valid]] + (1 - ns / (ns + 8.0)) * p[valid]
    valid = shi >= 0
    if valid.any():
        nsh = T["sh_n"][shi[valid]][:, None]
        p[valid] = nsh / (nsh + 4.0) * T["sh_p"][shi[valid]] + (1 - nsh / (nsh + 4.0)) * p[valid]
    np.clip(p, eps, 1 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def smooth_cols(scores, cols, alpha):
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s = scores.copy()
    view = s.reshape(-1, N_WINDOWS, s.shape[1])
    x = view[:, :, cols]
    prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    nxt = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1 - alpha) * x + 0.5 * alpha * (prev + nxt)
    return s


def fuse_scores(base, sites, hours, tables, m, conditions):
    """Apply prior fusion subset based on conditions dict."""
    scores = base.copy()
    if not conditions.get("prior", False):
        # cond A or B: still need to suppress unmapped_inactive to a sane baseline
        scores[:, m["idx_unmapped_inactive"]] = -8.0
        return scores
    prior = prior_logits(sites, hours, tables)
    if len(m["idx_mapped_active_event"]):
        scores[:, m["idx_mapped_active_event"]] += LAMBDA_EVENT * prior[:, m["idx_mapped_active_event"]]
    if len(m["idx_mapped_active_texture"]):
        scores[:, m["idx_mapped_active_texture"]] += LAMBDA_TEXTURE * prior[:, m["idx_mapped_active_texture"]]
    if len(m["idx_selected_proxy_active_texture"]):
        scores[:, m["idx_selected_proxy_active_texture"]] += (
            LAMBDA_PROXY_TEXTURE * prior[:, m["idx_selected_proxy_active_texture"]])
    if len(m["idx_prioronly_active_event"]):
        scores[:, m["idx_prioronly_active_event"]] = (
            LAMBDA_EVENT * prior[:, m["idx_prioronly_active_event"]])
    if len(m["idx_prioronly_active_texture"]):
        scores[:, m["idx_prioronly_active_texture"]] = (
            LAMBDA_TEXTURE * prior[:, m["idx_prioronly_active_texture"]])
    scores[:, m["idx_unmapped_inactive"]] = -8.0
    return scores


# ───────────────────────── ablation ─────────────────────────

def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def per_class_auc(y_true, y_score):
    aucs = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        if 0 < y_true[:, j].sum() < len(y_true):
            try:
                aucs[j] = roc_auc_score(y_true[:, j], y_score[:, j])
            except ValueError:
                pass
    return aucs


def gauss_smooth_final(scores):
    out = scores.reshape(-1, N_WINDOWS, scores.shape[1]).copy()
    for i in range(out.shape[0]):
        out[i] = convolve1d(out[i], GAUSS_W, axis=0, mode="nearest")
    return out.reshape(-1, scores.shape[1])


def seq_features_1d(v):
    x = v.reshape(-1, N_WINDOWS)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    nxt = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    return prev, nxt, np.repeat(x.mean(1), N_WINDOWS), np.repeat(x.max(1), N_WINDOWS)


def build_class_features(Z, raw_col, prior_col, base_col):
    p, n, mn, mx = seq_features_1d(base_col)
    return np.concatenate(
        [Z, raw_col[:, None], prior_col[:, None], base_col[:, None],
         p[:, None], n[:, None], mn[:, None], mx[:, None]],
        axis=1,
    ).astype(np.float32)


def fit_apply_probes(Z_train, Z_val, raw_t, raw_v, prior_t, prior_v, base_t, base_v, Y_t):
    """Train probes per class on (train), predict on (val). Returns final blended scores on val."""
    final = base_v.copy()
    pos_counts = Y_t.sum(axis=0)
    probe_idx = np.where(pos_counts >= PROBE_MIN_POS)[0].astype(np.int32)
    for cls in probe_idx:
        y = Y_t[:, cls]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        Xt = build_class_features(Z_train, raw_t[:, cls], prior_t[:, cls], base_t[:, cls])
        Xv = build_class_features(Z_val, raw_v[:, cls], prior_v[:, cls], base_v[:, cls])
        clf = LogisticRegression(C=PROBE_C, max_iter=400, solver="liblinear", class_weight="balanced")
        clf.fit(Xt, y)
        pred = clf.decision_function(Xv).astype(np.float32)
        n_pos = int(y.sum())
        alpha = 0.25 + 0.25 * min(1.0, (n_pos - PROBE_MIN_POS) / 40.0)
        final[:, cls] = (1 - alpha) * base_v[:, cls] + alpha * pred
    return final


# ───────────────────────── main ─────────────────────────

def main():
    t0 = time.time()
    md = load_metadata()
    print(f"Loaded metadata in {time.time() - t0:.1f}s")

    # Cache check / Perch inference
    meta_p = CACHE / "full_perch_meta.parquet"
    arr_p = CACHE / "full_perch_arrays.npz"
    if meta_p.exists() and arr_p.exists():
        print(f"Loading cached Perch from {CACHE}")
        meta_full = pd.read_parquet(meta_p)
        arr = np.load(arr_p)
        scores_raw = arr["scores"].astype(np.float32)
        emb = arr["emb"].astype(np.float32)
        # We need taxonomy mapping for ablation regardless
        _model, _infer, bc_labels = load_perch()
        m = build_taxonomy_mapping(md["taxonomy"], md["primary"], md["label_to_idx"],
                                   bc_labels, md["sc_clean"], md["Y_SC"])
    else:
        _model, infer, bc_labels = load_perch()
        m = build_taxonomy_mapping(md["taxonomy"], md["primary"], md["label_to_idx"],
                                   bc_labels, md["sc_clean"], md["Y_SC"])
        full_paths = [DATA / "train_soundscapes" / f for f in md["full_files"]]
        print(f"Running Perch on {len(full_paths)} fully-labeled soundscapes ...")
        meta_full, scores_raw, emb = perch_infer_files(infer, full_paths, md["n_classes"], m)
        meta_full.to_parquet(meta_p, index=False)
        np.savez_compressed(arr_p, scores=scores_raw, emb=emb)
        print(f"Cached to {CACHE}")

    # Align Y_FULL to meta_full row order
    full_truth_aligned = (
        md["full_truth"].set_index("row_id").loc[meta_full["row_id"]].reset_index(drop=False)
    )
    Y_FULL = md["Y_SC"][full_truth_aligned["index"].to_numpy()]
    print(f"scores={scores_raw.shape}  emb={emb.shape}  Y_FULL={Y_FULL.shape}")
    print(f"Active classes: {(Y_FULL.sum(0) > 0).sum()} / {md['n_classes']}")
    print(f"Wall time: {time.time() - t0:.1f}s")

    # Set up GroupKFold by site
    gkf = GroupKFold(n_splits=5)
    groups = meta_full["site"].to_numpy()

    # Pre-compute embedding scaler+PCA on full data (used for E,F)
    # For probe condition we re-fit per fold on train portion.
    sites_arr = meta_full["site"].to_numpy()
    hours_arr = meta_full["hour_utc"].to_numpy()

    # ───── Run conditions ─────
    # We compute OOF predictions per condition, then macro_auc once.
    conditions = {
        "A_raw_perch": {"proxy": False, "prior": False, "smooth": False, "probe": False, "gauss": False},
        "B_proxy":     {"proxy": True,  "prior": False, "smooth": False, "probe": False, "gauss": False},
        "C_prior":     {"proxy": True,  "prior": True,  "smooth": False, "probe": False, "gauss": False},
        "D_smooth":    {"proxy": True,  "prior": True,  "smooth": True,  "probe": False, "gauss": False},
        "E_probe":     {"proxy": True,  "prior": True,  "smooth": True,  "probe": True,  "gauss": False},
        "F_full":      {"proxy": True,  "prior": True,  "smooth": True,  "probe": True,  "gauss": True},
    }

    # For "no proxy" condition, zero out proxy-derived columns
    proxy_cols = np.array(list(m["proxy_pos_to_bc"].keys()), dtype=np.int32)

    results = {}
    per_class_results = {}

    for cname, conds in conditions.items():
        print(f"\n=== Condition {cname} {conds} ===")
        oof = np.zeros_like(scores_raw, dtype=np.float32)
        for fi, (tr_idx, va_idx) in enumerate(gkf.split(scores_raw, groups=groups)):
            tr_idx = np.sort(tr_idx)
            va_idx = np.sort(va_idx)
            val_sites = set(meta_full.iloc[va_idx]["site"].tolist())
            prior_m = ~md["sc_clean"]["site"].isin(val_sites).values
            tables = fit_prior_tables(
                md["sc_clean"].loc[prior_m].reset_index(drop=True), md["Y_SC"][prior_m]
            )

            # Start from raw Perch scores (with proxy already applied during inference).
            # Strip proxy if condition disables it.
            base_v = scores_raw[va_idx].copy()
            base_t = scores_raw[tr_idx].copy()
            if not conds["proxy"]:
                base_v[:, proxy_cols] = -8.0
                base_t[:, proxy_cols] = -8.0

            # Prior fusion (or skip)
            fused_v = fuse_scores(
                base_v, sites_arr[va_idx], hours_arr[va_idx], tables, m, conds
            )
            fused_t = fuse_scores(
                base_t, sites_arr[tr_idx], hours_arr[tr_idx], tables, m, conds
            )

            # Texture smoothing
            if conds["smooth"]:
                fused_v = smooth_cols(fused_v, m["idx_active_texture"], SMOOTH_TEXTURE_ALPHA)
                fused_t = smooth_cols(fused_t, m["idx_active_texture"], SMOOTH_TEXTURE_ALPHA)

            # Probe
            if conds["probe"]:
                # PCA+scaler fitted on TRAIN embeddings only
                scaler = StandardScaler()
                emb_t = scaler.fit_transform(emb[tr_idx])
                emb_v = scaler.transform(emb[va_idx])
                n_comp = min(PROBE_PCA_DIM, emb_t.shape[0] - 1, emb_t.shape[1])
                pca = PCA(n_components=n_comp)
                Z_t = pca.fit_transform(emb_t).astype(np.float32)
                Z_v = pca.transform(emb_v).astype(np.float32)

                # We need separate raw/prior for probes too
                raw_t = scores_raw[tr_idx]; raw_v = scores_raw[va_idx]
                prior_t = prior_logits(sites_arr[tr_idx], hours_arr[tr_idx], tables)
                prior_v = prior_logits(sites_arr[va_idx], hours_arr[va_idx], tables)
                final_v = fit_apply_probes(
                    Z_t, Z_v, raw_t, raw_v, prior_t, prior_v, fused_t, fused_v, Y_FULL[tr_idx]
                )
                oof[va_idx] = final_v
            else:
                oof[va_idx] = fused_v

        # Gaussian smoothing across all rows (applied per-file)
        if conds["gauss"]:
            # Sort by file then time to apply gaussian per file
            order = np.lexsort((meta_full["row_id"].apply(
                lambda x: int(x.rsplit("_", 1)[1])).values,
                meta_full["filename"].values))
            inv = np.argsort(order)
            oof_sorted = oof[order]
            oof_sorted = gauss_smooth_final(oof_sorted)
            oof = oof_sorted[inv]

        auc = macro_auc(Y_FULL, oof)
        results[cname] = auc
        per_class_results[cname] = per_class_auc(Y_FULL, oof)
        print(f"{cname}: macro AUC = {auc:.6f}")

    pc_df = pd.DataFrame(per_class_results, index=md["primary"])
    pc_df["class_name"] = [m_ for m_ in md["taxonomy"].set_index("primary_label")["class_name"].reindex(pc_df.index).values]
    pc_df["n_pos"] = Y_FULL.sum(axis=0)

    # ───── In-sample upper bound: prior tables fit on ALL labeled SS ─────
    # (no site holdout). This simulates the LB scenario where prior tables are
    # built on the full training set and applied to (potentially overlapping) test sites.
    print("\n=== In-sample upper bound (no fold split) ===")
    tables_all = fit_prior_tables(md["sc_clean"].reset_index(drop=True), md["Y_SC"])
    base_all = scores_raw.copy()
    fused_all = fuse_scores(base_all, sites_arr, hours_arr, tables_all, m,
                            {"proxy": True, "prior": True})
    fused_all = smooth_cols(fused_all, m["idx_active_texture"], SMOOTH_TEXTURE_ALPHA)
    insample_C = macro_auc(Y_FULL, fused_all)
    insample_A = macro_auc(Y_FULL, scores_raw)
    print(f"  in_sample_A_raw   AUC = {insample_A:.4f}")
    print(f"  in_sample_D_smooth AUC = {insample_C:.4f}  (Δ {insample_C - insample_A:+.4f})")
    results["in_sample_A_raw"] = insample_A
    results["in_sample_D_smooth"] = insample_C

    # Save
    with open(OUT / "oof_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    pc_df.to_csv(OUT / "per_class_auc.csv")

    print("\n=== Summary ===")
    prev = None
    for k, v in results.items():
        delta = "" if prev is None else f"  (Δ {v - prev:+.4f})"
        print(f"  {k:18s}  AUC = {v:.4f}{delta}")
        prev = v

    print(f"\nWall time: {time.time() - t0:.1f}s")
    print(f"Wrote {OUT/'oof_ablation.json'} and {OUT/'per_class_auc.csv'}")


if __name__ == "__main__":
    main()
