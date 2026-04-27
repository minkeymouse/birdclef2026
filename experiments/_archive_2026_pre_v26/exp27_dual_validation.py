#!/usr/bin/env python3
"""
exp27 — Dual validation: Val-A (seen-site, new-file) vs Val-B (unseen-site).

Motivation (from lb928.md, Tucker Arrants):
  "GKF on site is overly harsh because the test set has site overlap with
   training. GKF models can't learn site-level species priors that would help
   at test time."
  Recommendation: dual validation —
    Val-A: hold out specific recordings from seen sites (matches actual test regime)
    Val-B: hold out an entire site (hedge against unseen sites)
  Combine, weighted toward Val-A.

Method (5-fold each, on cached exp21 features):
  Val-A: stratified-by-site file-level 5-fold (each site distributed across folds)
  Val-B: GroupKFold by site (same as exp21–26)

Recipes evaluated under both:
  R0  raw Perch
  PR  raw + Bayesian prior fusion only (LB 0.910 minus probes)
  R1  probes-only (no priors), 50/50 blend with raw
  PP  priors + probes (LB 0.910 frozen config)
  R5  R1 + Gaussian temporal smoothing

Output:
  experiments/exp27_outputs/results.json
  experiments/exp27_outputs/recipes_table.csv
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"
OUT = ROOT / "experiments" / "exp27_outputs"
OUT.mkdir(parents=True, exist_ok=True)

N_WINDOWS = 12
PROBE_PCA_DIM = 32
PROBE_C = 0.25
PROBE_MIN_POS = 5
SEED = 42
GAUSS_W = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

# LB 0.910 frozen prior weights
LAMBDA_EVENT = 0.4
LAMBDA_TEXTURE = 1.0
LAMBDA_PROXY_TEXTURE = 0.8
SMOOTH_TEXTURE_ALPHA = 0.35


# ─────────────── Data ───────────────

def load_metadata():
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
    meta_cols = sc_clean["filename"].apply(lambda n: pd.Series(dict(zip(("site", "hour_utc"), parse_meta(n)))))
    sc_clean = pd.concat([sc_clean, meta_cols], axis=1)

    Y_SC = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, labs in enumerate(sc_clean["label_list"]):
        for lbl in labs:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1
    return taxonomy, primary, label_to_idx, n_classes, sc_clean, Y_SC


# ─────────────── Folds ───────────────

def stratified_file_folds(files, sites, k=5, seed=SEED):
    """Round-robin assign files to k folds within each site → each site appears
    in most train folds (Val-A regime: site is seen at train time)."""
    rng = np.random.default_rng(seed)
    folds = -np.ones(len(files), dtype=int)
    sites_arr = np.array(sites)
    for site in np.unique(sites_arr):
        idxs = np.where(sites_arr == site)[0]
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            folds[i] = j % k
    return folds


# ─────────────── Priors (same as exp26) ───────────────

def fit_priors(prior_df, Y_prior):
    n_cls = Y_prior.shape[1]
    global_p = Y_prior.mean(axis=0).astype(np.float32)

    site_keys = sorted(prior_df["site"].dropna().astype(str).unique())
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique())

    site_to_i, site_n, site_p = {}, [], []
    arr_s = prior_df["site"].astype(str).values
    for s in site_keys:
        mk = arr_s == s
        site_to_i[s] = len(site_n)
        site_n.append(mk.sum())
        site_p.append(Y_prior[mk].mean(axis=0))
    site_n = np.array(site_n, dtype=np.float32)
    site_p = (np.stack(site_p).astype(np.float32) if site_p
              else np.zeros((0, n_cls), np.float32))

    hour_to_i, hour_n, hour_p = {}, [], []
    arr_h = prior_df["hour_utc"].astype(int).values
    for h in hour_keys:
        mk = arr_h == h
        hour_to_i[h] = len(hour_n)
        hour_n.append(mk.sum())
        hour_p.append(Y_prior[mk].mean(axis=0))
    hour_n = np.array(hour_n, dtype=np.float32)
    hour_p = (np.stack(hour_p).astype(np.float32) if hour_p
              else np.zeros((0, n_cls), np.float32))

    sh_to_i, sh_n_l, sh_p_l = {}, [], []
    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_l)
        idx = np.array(list(idx))
        sh_n_l.append(len(idx))
        sh_p_l.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n_l, dtype=np.float32)
    sh_p = (np.stack(sh_p_l).astype(np.float32) if sh_p_l
            else np.zeros((0, n_cls), np.float32))

    return dict(global_p=global_p,
                site_to_i=site_to_i, site_n=site_n, site_p=site_p,
                hour_to_i=hour_to_i, hour_n=hour_n, hour_p=hour_p,
                sh_to_i=sh_to_i, sh_n=sh_n, sh_p=sh_p)


def prior_logits(sites, hours, T, k_hour=8.0, k_site=8.0, k_sh=4.0, eps=1e-4):
    n = len(sites)
    p = np.repeat(T["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)
    si = np.fromiter((T["site_to_i"].get(str(s), -1) for s in sites), np.int32, n)
    hi = np.fromiter(
        (T["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        np.int32, n)
    shi = np.fromiter(
        (T["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1
         for s, h in zip(sites, hours)),
        np.int32, n)
    valid = hi >= 0
    if valid.any():
        nh = T["hour_n"][hi[valid]][:, None]
        w = nh / (nh + k_hour)
        p[valid] = w * T["hour_p"][hi[valid]] + (1 - w) * p[valid]
    valid = si >= 0
    if valid.any():
        ns = T["site_n"][si[valid]][:, None]
        w = ns / (ns + k_site)
        p[valid] = w * T["site_p"][si[valid]] + (1 - w) * p[valid]
    valid = shi >= 0
    if valid.any():
        nsh = T["sh_n"][shi[valid]][:, None]
        w = nsh / (nsh + k_sh)
        p[valid] = w * T["sh_p"][shi[valid]] + (1 - w) * p[valid]
    np.clip(p, eps, 1 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


# ─────────────── Recipe components ───────────────

def fuse_priors(scores_raw, prior, idx_event, idx_texture):
    """Apply LB-0.910-style prior fusion: add λ·prior to logits."""
    out = scores_raw.copy()
    if len(idx_event):
        out[:, idx_event] = out[:, idx_event] + LAMBDA_EVENT * prior[:, idx_event]
    if len(idx_texture):
        out[:, idx_texture] = out[:, idx_texture] + LAMBDA_TEXTURE * prior[:, idx_texture]
    return out


def gauss_smooth_per_file(scores, meta_full):
    out = scores.copy()
    by_file = meta_full.groupby("filename").indices
    for fn, idx in by_file.items():
        end_secs = meta_full.iloc[idx]["row_id"].apply(
            lambda x: int(x.rsplit("_", 1)[1])).values
        order = np.argsort(end_secs)
        oi = np.array(idx)[order]
        out[oi] = convolve1d(out[oi], GAUSS_W, axis=0, mode="nearest")
    return out


def fit_probes_pca_only(emb, Y, tr_idx, va_idx, scores_raw, base_v, alpha=0.5):
    """exp26 R1 recipe: PCA32 + LogReg, 50/50 blend with `base_v` on val."""
    scaler = StandardScaler()
    Et = scaler.fit_transform(emb[tr_idx])
    Ev = scaler.transform(emb[va_idx])
    n_comp = min(PROBE_PCA_DIM, Et.shape[0] - 1, Et.shape[1])
    pca = PCA(n_components=n_comp)
    Zt = pca.fit_transform(Et).astype(np.float32)
    Zv = pca.transform(Ev).astype(np.float32)

    out = base_v.copy()
    pos_counts = Y[tr_idx].sum(axis=0)
    for cls in range(Y.shape[1]):
        if pos_counts[cls] < PROBE_MIN_POS or pos_counts[cls] == len(tr_idx):
            continue
        y = Y[tr_idx, cls].astype(np.float32)
        try:
            clf = LogisticRegression(C=PROBE_C, max_iter=400,
                                      solver="liblinear", class_weight="balanced")
            clf.fit(Zt, y)
        except Exception:
            continue
        pred = clf.decision_function(Zv).astype(np.float32)
        out[:, cls] = (1 - alpha) * base_v[:, cls] + alpha * pred
    return out


# ─────────────── Eval ───────────────

def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


# ─────────────── Main ───────────────

def main():
    t0 = time.time()
    taxonomy, primary, label_to_idx, n_classes, sc_clean, Y_SC = load_metadata()

    meta_full = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    arr = np.load(EXP21 / "full_perch_arrays.npz")
    scores_raw = arr["scores"].astype(np.float32)
    emb = arr["emb"].astype(np.float32)
    sites_full = meta_full["site"].to_numpy()
    hours_full = meta_full["hour_utc"].to_numpy()
    sc_idx = sc_clean.set_index("row_id")
    Y_FULL = np.stack([Y_SC[sc_idx.index.get_loc(rid)] for rid in meta_full["row_id"]])

    cn_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    TEXTURE_TAXA = {"Amphibia", "Insecta"}
    idx_texture = np.array([i for i, c in enumerate(primary)
                             if cn_map.get(c) in TEXTURE_TAXA], dtype=np.int32)
    idx_event = np.array([i for i, c in enumerate(primary)
                           if cn_map.get(c) not in TEXTURE_TAXA], dtype=np.int32)

    # ── Build folds ──────────────────────────────────────────────────────────
    files = sorted(meta_full["filename"].unique().tolist())
    file_to_site = meta_full.drop_duplicates("filename").set_index("filename")["site"].to_dict()
    file_sites = [file_to_site[f] for f in files]

    # Val-A: file-stratified by site (5 folds)
    file_fold_a = stratified_file_folds(files, file_sites, k=5, seed=SEED)
    file_to_fold_a = dict(zip(files, file_fold_a))
    rowfold_a = np.array([file_to_fold_a[f] for f in meta_full["filename"]])

    # Val-B: site-grouped (5 folds, GroupKFold)
    gkf = GroupKFold(n_splits=5)
    rowfold_b = np.full(len(meta_full), -1, dtype=int)
    for fi, (_, va_idx) in enumerate(gkf.split(scores_raw, groups=sites_full)):
        rowfold_b[va_idx] = fi

    print(f"Eval rows: {len(emb)}  files: {len(files)}  sites: {meta_full['site'].nunique()}")
    print(f"Active classes: {(Y_FULL.sum(0) > 0).sum()}")
    print()
    print("Val-A fold sizes (rows):", np.bincount(rowfold_a))
    print("Val-B fold sizes (rows):", np.bincount(rowfold_b))
    print()
    print("Val-A: each fold's site composition (n_files):")
    for fi in range(5):
        comp = pd.Series([file_sites[i] for i in range(len(files))
                          if file_fold_a[i] == fi]).value_counts().sort_index().to_dict()
        print(f"  fold{fi}: {comp}")
    print()

    # ── Run all recipes under both fold schemes ─────────────────────────────
    def run_under(fold_name, rowfold):
        print(f"\n=== {fold_name} ===")
        # Pre-build prior tables per fold (TRAIN sites only is wrong for Val-A;
        # for Val-A we should use all rows except val rows — same logic, just
        # different held-out indices).
        fold_priors = []
        for fi in range(5):
            va_mask_global = sc_clean["row_id"].isin(meta_full[rowfold == fi]["row_id"])
            tr_for_prior = sc_clean.loc[~va_mask_global].reset_index(drop=True)
            Y_for_prior = Y_SC[~va_mask_global.values]
            fold_priors.append(fit_priors(tr_for_prior, Y_for_prior))

        recipes_oof = {
            "R0_raw":   scores_raw.copy(),  # no folds needed
            "PR_priors": np.zeros_like(scores_raw),
            "R1_probes": np.zeros_like(scores_raw),
            "PP_priors+probes": np.zeros_like(scores_raw),
            "R5_R1+gauss": None,  # filled after R1
        }

        for fi in range(5):
            tr_idx = np.where(rowfold != fi)[0]
            va_idx = np.where(rowfold == fi)[0]
            T = fold_priors[fi]
            prior_v = prior_logits(sites_full[va_idx], hours_full[va_idx], T)

            # PR: raw + prior fusion
            recipes_oof["PR_priors"][va_idx] = fuse_priors(
                scores_raw[va_idx], prior_v, idx_event, idx_texture)

            # R1: probes only on raw base
            recipes_oof["R1_probes"][va_idx] = fit_probes_pca_only(
                emb, Y_FULL, tr_idx, va_idx, scores_raw,
                base_v=scores_raw[va_idx])

            # PP: probes on (raw + prior) base
            base_pp = fuse_priors(scores_raw[va_idx], prior_v, idx_event, idx_texture)
            recipes_oof["PP_priors+probes"][va_idx] = fit_probes_pca_only(
                emb, Y_FULL, tr_idx, va_idx, scores_raw, base_v=base_pp)

        recipes_oof["R5_R1+gauss"] = gauss_smooth_per_file(
            recipes_oof["R1_probes"], meta_full)

        out = {}
        for name, sc in recipes_oof.items():
            auc = macro_auc(Y_FULL, sc)
            out[name] = auc
            print(f"  {name:22s}  AUC = {auc:.4f}")
        return out

    res_a = run_under("Val-A (seen-site, new-file)", rowfold_a)
    res_b = run_under("Val-B (unseen-site, GroupKFold)", rowfold_b)

    # ── Side-by-side table ─────────────────────────────────────────────────
    print(f"\n=== Side-by-side ===")
    print(f"{'Recipe':22s}  {'Val-A':>8s}  {'Val-B':>8s}  {'A−B':>8s}")
    rows = []
    for name in res_a:
        a, b = res_a[name], res_b[name]
        diff = a - b
        rows.append({"recipe": name, "val_A_seen_site": a, "val_B_unseen_site": b,
                     "delta_A_minus_B": diff})
        print(f"  {name:20s}  {a:.4f}    {b:.4f}    {diff:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recipes_table.csv", index=False)

    # ── Verdict ────────────────────────────────────────────────────────────
    print(f"\n=== Interpretation ===")
    print(f"  PR-priors gain over raw (Val-A): {res_a['PR_priors']-res_a['R0_raw']:+.4f}")
    print(f"  PR-priors gain over raw (Val-B): {res_b['PR_priors']-res_b['R0_raw']:+.4f}")
    print(f"  R1-probes gain over raw (Val-A): {res_a['R1_probes']-res_a['R0_raw']:+.4f}")
    print(f"  R1-probes gain over raw (Val-B): {res_b['R1_probes']-res_b['R0_raw']:+.4f}")
    print(f"  PP vs R1 (Val-A): {res_a['PP_priors+probes']-res_a['R1_probes']:+.4f}")
    print(f"  PP vs R1 (Val-B): {res_b['PP_priors+probes']-res_b['R1_probes']:+.4f}")

    summary = {
        "n_files": len(files), "n_rows": len(emb),
        "val_A": res_a, "val_B": res_b,
        "interpretation": {
            "priors_gain_seen_site":   res_a["PR_priors"] - res_a["R0_raw"],
            "priors_gain_unseen_site": res_b["PR_priors"] - res_b["R0_raw"],
            "PP_vs_R1_seen_site":   res_a["PP_priors+probes"] - res_a["R1_probes"],
            "PP_vs_R1_unseen_site": res_b["PP_priors+probes"] - res_b["R1_probes"],
        }
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT/'results.json'}  Wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
