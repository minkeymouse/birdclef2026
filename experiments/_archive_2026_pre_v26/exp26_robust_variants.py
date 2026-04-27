#!/usr/bin/env python3
"""
exp26 — Local validation of robust recipe + variants.

Goal: before submitting to Kaggle, decide which exact recipe goes into exp30
v1–v5 by measuring OOF (5-fold GroupKFold by site) macro-AUC for several
candidate variants.

Variants (all built on the cached exp21 Perch features, 708 windows × 234 cls):

  R0  Raw Perch (baseline)                                   = exp21 stage A
  R1  Probes only, no priors, 50/50 blend with raw Perch     = exp23 robust
  R2a Probes + weak priors  (k_h=k_s=50, k_sh=25)
  R2b Probes + weak priors  (k_h=k_s=100, k_sh=50)
  R2c Probes + weak priors  (k_h=k_s=200, k_sh=100)
  R2d Probes + weak priors  (k_h=k_s=500, k_sh=250)
  R3  R1 + per-file embedding centering (whitening)
  R4  R3 + best-k priors (selected after R2 sweep)
  R5  R1 + Gaussian temporal smoothing (σ=1.0)
  R6  R3 + best-k priors + Gaussian smoothing (full robust+)

Reproduce exp23 result: R1 should hit OOF ≈ 0.816.

Outputs:
  experiments/exp26_outputs/results.json
  experiments/exp26_outputs/per_class.csv
  experiments/exp26_outputs/run.log
"""
from __future__ import annotations

import json
import os
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
OUT = ROOT / "experiments" / "exp26_outputs"
OUT.mkdir(parents=True, exist_ok=True)

N_WINDOWS = 12
PROBE_PCA_DIM = 32
PROBE_C = 0.25
PROBE_MIN_POS = 8
LAMBDA_EVENT = 0.4
LAMBDA_TEXTURE = 1.0
LAMBDA_PROXY_TEXTURE = 0.8
SMOOTH_TEXTURE_ALPHA = 0.35
GAUSS_W = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

# Frozen prior shrinkage (exp21 v1): k_h=k_s=8, k_sh=4 → strong (LB 0.910)
WEAK_PRIORS = {
    "R2a": dict(k_hour=50,  k_site=50,  k_sh=25),
    "R2b": dict(k_hour=100, k_site=100, k_sh=50),
    "R2c": dict(k_hour=200, k_site=200, k_sh=100),
    "R2d": dict(k_hour=500, k_site=500, k_sh=250),
}


# ──────────────────────────── Data ────────────────────────────

def load_metadata():
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary)}
    n_classes = len(primary)

    fre = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

    def parse_lbls(x):
        if pd.isna(x):
            return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    def parse_meta(name):
        m = fre.match(name)
        if not m:
            return None, -1
        _, site, _, hms = m.groups()
        return site, int(hms[:2])

    sc_clean = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
                .apply(lambda s: sorted({lbl for x in s for lbl in parse_lbls(x)}))
                .reset_index(name="label_list"))
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (sc_clean["filename"].str.replace(".ogg", "", regex=False)
                          + "_" + sc_clean["end_sec"].astype(str))
    meta = sc_clean["filename"].apply(lambda n: pd.Series(dict(zip(("site", "hour_utc"), parse_meta(n)))))
    sc_clean = pd.concat([sc_clean, meta], axis=1)

    Y_SC = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, labs in enumerate(sc_clean["label_list"]):
        for lbl in labs:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1
    return taxonomy, primary, label_to_idx, n_classes, sc_clean, Y_SC


# ──────────────────────────── Priors ────────────────────────────

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
    site_p = np.stack(site_p).astype(np.float32) if site_p else np.zeros((0, n_cls), np.float32)

    hour_to_i, hour_n, hour_p = {}, [], []
    arr_h = prior_df["hour_utc"].astype(int).values
    for h in hour_keys:
        mk = arr_h == h
        hour_to_i[h] = len(hour_n)
        hour_n.append(mk.sum())
        hour_p.append(Y_prior[mk].mean(axis=0))
    hour_n = np.array(hour_n, dtype=np.float32)
    hour_p = np.stack(hour_p).astype(np.float32) if hour_p else np.zeros((0, n_cls), np.float32)

    sh_to_i, sh_n, sh_p = {}, [], []
    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n)
        idx = np.array(list(idx))
        sh_n.append(len(idx))
        sh_p.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n, dtype=np.float32)
    sh_p = np.stack(sh_p).astype(np.float32) if sh_p else np.zeros((0, n_cls), np.float32)

    return dict(
        global_p=global_p,
        site_to_i=site_to_i, site_n=site_n, site_p=site_p,
        hour_to_i=hour_to_i, hour_n=hour_n, hour_p=hour_p,
        sh_to_i=sh_to_i, sh_n=sh_n, sh_p=sh_p,
    )


def prior_logits(sites, hours, T, k_hour=8.0, k_site=8.0, k_sh=4.0, eps=1e-4):
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


# ──────────────────────────── Smoothing ────────────────────────────

def gauss_smooth_per_file(scores, meta_full):
    """Apply 5-tap Gaussian smoothing across the 12 windows of each file."""
    out = scores.copy()
    by_file = meta_full.groupby("filename").indices
    for fn, idx in by_file.items():
        # Sort by end_sec
        end_secs = meta_full.iloc[idx]["row_id"].apply(lambda x: int(x.rsplit("_", 1)[1])).values
        order = np.argsort(end_secs)
        ordered_idx = np.array(idx)[order]
        seg = out[ordered_idx]
        seg = convolve1d(seg, GAUSS_W, axis=0, mode="nearest")
        out[ordered_idx] = seg
    return out


# ──────────────────────────── Whitening ────────────────────────────

def per_file_center(emb, meta_full):
    """Subtract file-mean embedding from each window."""
    centered = emb.copy()
    by_file = meta_full.groupby("filename").indices
    for fn, idx in by_file.items():
        m = emb[idx].mean(axis=0)
        centered[idx] = emb[idx] - m
    return centered


# ──────────────────────────── Probe ────────────────────────────

def fit_apply_probes_v1(emb_full, scores_raw, Y_FULL, tr_idx, va_idx,
                         add_prior=False, prior_t=None, prior_v=None,
                         lam_event=0.0, lam_texture=0.0,
                         idx_event=None, idx_texture=None):
    """Train PCA(32)+LogReg per class, return blended OOF scores on val.

    add_prior: if True, prior_logits added to base before blending; AND a
      `prior_col` feature is appended to the probe input.
    """
    scaler = StandardScaler()
    Et = scaler.fit_transform(emb_full[tr_idx])
    Ev = scaler.transform(emb_full[va_idx])
    n_comp = min(PROBE_PCA_DIM, Et.shape[0] - 1, Et.shape[1])
    pca = PCA(n_components=n_comp)
    Zt = pca.fit_transform(Et).astype(np.float32)
    Zv = pca.transform(Ev).astype(np.float32)

    # Base predictions (raw Perch + optional prior fusion)
    base_v = scores_raw[va_idx].copy()
    if add_prior:
        if idx_event is not None and len(idx_event):
            base_v[:, idx_event] = base_v[:, idx_event] + lam_event * prior_v[:, idx_event]
        if idx_texture is not None and len(idx_texture):
            base_v[:, idx_texture] = base_v[:, idx_texture] + lam_texture * prior_v[:, idx_texture]

    out = base_v.copy()
    pos_counts = Y_FULL[tr_idx].sum(axis=0)
    n_classes = scores_raw.shape[1]
    for cls in range(n_classes):
        if pos_counts[cls] < PROBE_MIN_POS or pos_counts[cls] == len(tr_idx):
            continue
        y = Y_FULL[tr_idx, cls].astype(np.float32)
        # Probe input: just PCA features (matches exp23)
        clf = LogisticRegression(C=PROBE_C, max_iter=400, solver="liblinear",
                                  class_weight="balanced")
        try:
            clf.fit(Zt, y)
        except Exception:
            continue
        pred = clf.decision_function(Zv).astype(np.float32)
        # 50/50 blend with base (raw Perch ± optional prior)
        out[:, cls] = 0.5 * base_v[:, cls] + 0.5 * pred
    return out


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


# ──────────────────────────── Main ────────────────────────────

def main():
    t0 = time.time()
    taxonomy, primary, label_to_idx, n_classes, sc_clean, Y_SC = load_metadata()

    # Load exp21 Perch cache
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
    idx_texture = np.array([i for i, c in enumerate(primary) if cn_map.get(c) in TEXTURE_TAXA], dtype=np.int32)
    idx_event = np.array([i for i, c in enumerate(primary) if cn_map.get(c) not in TEXTURE_TAXA], dtype=np.int32)

    print(f"Eval rows: {len(emb)}  classes: {n_classes}  active: {(Y_FULL.sum(0) > 0).sum()}")
    print(f"Texture cls: {len(idx_texture)}  Event cls: {len(idx_event)}")

    # Per-file centered embeddings (for R3, R4, R6)
    emb_centered = per_file_center(emb, meta_full)
    print(f"Built per-file-centered embeddings  ({time.time()-t0:.1f}s)")

    # Sites for GroupKFold
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(scores_raw, groups=sites_full))

    # Pre-compute per-fold prior tables (fit on TRAIN sites only, for OOF honesty)
    fold_priors = []
    for tr_idx, va_idx in splits:
        val_sites = set(sites_full[va_idx].tolist())
        prior_m = ~sc_clean["site"].isin(val_sites).values
        T = fit_priors(sc_clean.loc[prior_m].reset_index(drop=True), Y_SC[prior_m])
        fold_priors.append(T)

    results = {}
    per_class_oof = {}

    def run_recipe(name, *, use_centered=False, add_prior=False,
                   k_hour=None, k_site=None, k_sh=None, gauss=False):
        oof = np.zeros_like(scores_raw, dtype=np.float32)
        E = emb_centered if use_centered else emb
        for fi, (tr_idx, va_idx) in enumerate(splits):
            tr_idx = np.sort(tr_idx); va_idx = np.sort(va_idx)
            if add_prior:
                T = fold_priors[fi]
                prior_t = prior_logits(sites_full[tr_idx], hours_full[tr_idx], T,
                                        k_hour=k_hour, k_site=k_site, k_sh=k_sh)
                prior_v = prior_logits(sites_full[va_idx], hours_full[va_idx], T,
                                        k_hour=k_hour, k_site=k_site, k_sh=k_sh)
            else:
                prior_t = prior_v = None
            oof[va_idx] = fit_apply_probes_v1(
                E, scores_raw, Y_FULL, tr_idx, va_idx,
                add_prior=add_prior,
                prior_t=prior_t, prior_v=prior_v,
                lam_event=LAMBDA_EVENT, lam_texture=LAMBDA_TEXTURE,
                idx_event=idx_event, idx_texture=idx_texture,
            )
        if gauss:
            oof = gauss_smooth_per_file(oof, meta_full)
        auc = macro_auc(Y_FULL, oof)
        pc = per_class_auc(Y_FULL, oof)
        results[name] = auc
        per_class_oof[name] = pc
        print(f"  {name:6s}  OOF macro-AUC = {auc:.4f}")
        return auc, oof

    # R0 — raw Perch baseline
    auc0 = macro_auc(Y_FULL, scores_raw)
    pc0 = per_class_auc(Y_FULL, scores_raw)
    results["R0"] = auc0
    per_class_oof["R0"] = pc0
    print(f"\n=== R0 raw Perch ===")
    print(f"  R0     OOF macro-AUC = {auc0:.4f}")

    # R1 — probes only, no priors
    print(f"\n=== R1 probes only (reproduce exp23) ===")
    auc1, oof1 = run_recipe("R1", use_centered=False, add_prior=False, gauss=False)

    # R2a–d — probes + weak priors at multiple k
    print(f"\n=== R2 probes + weak priors (k sweep) ===")
    for tag, kk in WEAK_PRIORS.items():
        run_recipe(tag, use_centered=False, add_prior=True,
                   k_hour=kk["k_hour"], k_site=kk["k_site"], k_sh=kk["k_sh"], gauss=False)

    # Pick best k from R2
    best_tag = max(WEAK_PRIORS, key=lambda t: results[t])
    best_kk = WEAK_PRIORS[best_tag]
    print(f"  Best R2 variant: {best_tag} (AUC {results[best_tag]:.4f})")

    # R3 — probes (no priors) + per-file embedding centering
    print(f"\n=== R3 probes + per-file centering (no priors) ===")
    auc3, oof3 = run_recipe("R3", use_centered=True, add_prior=False, gauss=False)

    # R4 — R3 + best-k priors
    print(f"\n=== R4 R3 + best-k priors ({best_tag}) ===")
    run_recipe("R4", use_centered=True, add_prior=True,
               k_hour=best_kk["k_hour"], k_site=best_kk["k_site"], k_sh=best_kk["k_sh"], gauss=False)

    # R5 — R1 + Gaussian temporal smoothing
    print(f"\n=== R5 R1 + Gaussian temporal smoothing ===")
    run_recipe("R5", use_centered=False, add_prior=False, gauss=True)

    # R6 — R3 + best-k priors + Gaussian
    print(f"\n=== R6 full robust+ ===")
    run_recipe("R6", use_centered=True, add_prior=True,
               k_hour=best_kk["k_hour"], k_site=best_kk["k_site"], k_sh=best_kk["k_sh"], gauss=True)

    # ──────── Per-taxa breakdown for headline recipes ────────
    print(f"\n=== Per-taxa breakdown (R0, R1, R3, R6) ===")
    keep = Y_FULL.sum(0) > 0
    rows = []
    for r in ["R0", "R1", "R3", "R6"]:
        pc = per_class_oof[r]
        for taxa in ["Aves", "Mammalia", "Amphibia", "Insecta", "Reptilia"]:
            mask = np.array([cn_map.get(c) == taxa for c in primary]) & keep
            if mask.sum() == 0:
                continue
            sub = pc[mask]
            sub = sub[~np.isnan(sub)]
            if len(sub) == 0:
                continue
            rows.append({"recipe": r, "taxa": taxa, "n_cls": len(sub),
                         "auc_mean": float(sub.mean())})
    df_taxa = pd.DataFrame(rows).pivot(index="taxa", columns="recipe", values="auc_mean")
    print(df_taxa.round(3).to_string())

    # ──────── Save results ────────
    print(f"\n=== Summary (sorted by OOF AUC) ===")
    sorted_results = dict(sorted(results.items(), key=lambda x: -x[1]))
    for k, v in sorted_results.items():
        delta = v - results["R0"]
        print(f"  {k:6s} {v:.4f}  (Δ vs R0 raw {delta:+.4f})")

    # Per-class CSV
    pc_df = pd.DataFrame({k: v for k, v in per_class_oof.items()}, index=primary)
    pc_df["class_name"] = [cn_map.get(c, "?") for c in primary]
    pc_df["n_pos"] = Y_FULL.sum(0)
    pc_df.to_csv(OUT / "per_class.csv")

    out_results = {
        "n_active_classes": int((Y_FULL.sum(0) > 0).sum()),
        "n_eval_rows": int(len(emb)),
        "macro_auc_oof": results,
        "best_R2": best_tag,
        "best_R2_k": best_kk,
        "by_taxa": df_taxa.round(4).to_dict(),
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(out_results, f, indent=2)
    print(f"\nWrote {OUT/'results.json'}  Wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
