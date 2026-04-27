#!/usr/bin/env python3
"""
exp28 — Perch probe v2: hyperparameter re-baseline + per-class rank normalization.

Grounding (from lb928.md + perch-v2-starter-train-infer.ipynb):
  The starter's frozen probe is (PCA=64, C=0.50, alpha=0.40), but our exp20 v1
  LB 0.910 used (PCA=32, C=0.25, alpha=0.40). hengck23's 0.93 decomposition
  adds "per-class normalisation" on top of probe output. exp28 measures:
    (A) probe hyperparameter sensitivity under Val-A and Val-B
    (B) per-class rank normalization lift on top of best probe
    (C) alternative smoothers (Gauss σ sweep, median filter)

Baseline lock: exp27 reports R5 = Val-A 0.892 / Val-B 0.842. This run re-measures
R5 under identical conditions and locks the number in results.json.

Reuses exp27 helpers in-script (self-contained).
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"
OUT = ROOT / "experiments" / "exp28_outputs"
OUT.mkdir(parents=True, exist_ok=True)

N_WINDOWS = 12
SEED = 42
GAUSS_W_DEFAULT = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # R5 kernel


# ─────────────── Data (subset of exp27) ───────────────

def load_metadata():
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
    return primary, label_to_idx, n_classes, sc_clean, Y_SC


def stratified_file_folds(files, sites, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    folds = -np.ones(len(files), dtype=int)
    sites_arr = np.array(sites)
    for site in np.unique(sites_arr):
        idxs = np.where(sites_arr == site)[0]
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            folds[i] = j % k
    return folds


def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def fit_probes(emb, Y, tr_idx, va_idx, base_va, pca_dim, C, alpha, min_pos=5):
    scaler = StandardScaler()
    Et = scaler.fit_transform(emb[tr_idx])
    Ev = scaler.transform(emb[va_idx])
    n_comp = min(pca_dim, Et.shape[0] - 1, Et.shape[1])
    pca = PCA(n_components=n_comp)
    Zt = pca.fit_transform(Et).astype(np.float32)
    Zv = pca.transform(Ev).astype(np.float32)

    out = base_va.copy()
    pos_counts = Y[tr_idx].sum(axis=0)
    for cls in range(Y.shape[1]):
        if pos_counts[cls] < min_pos or pos_counts[cls] == len(tr_idx):
            continue
        y = Y[tr_idx, cls].astype(np.float32)
        try:
            clf = LogisticRegression(C=C, max_iter=400,
                                     solver="liblinear", class_weight="balanced")
            clf.fit(Zt, y)
        except Exception:
            continue
        pred = clf.decision_function(Zv).astype(np.float32)
        out[:, cls] = (1 - alpha) * base_va[:, cls] + alpha * pred
    return out


def gauss_smooth_per_file(scores, meta_full, kernel):
    out = scores.copy()
    by_file = meta_full.groupby("filename").indices
    for fn, idx in by_file.items():
        end_secs = meta_full.iloc[idx]["row_id"].apply(
            lambda x: int(x.rsplit("_", 1)[1])).values
        order = np.argsort(end_secs)
        oi = np.array(idx)[order]
        out[oi] = convolve1d(out[oi], kernel, axis=0, mode="nearest")
    return out


def gauss_kernel(sigma, halfwidth=2):
    xs = np.arange(-halfwidth, halfwidth + 1, dtype=np.float32)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    k /= k.sum()
    return k


def per_class_rank_norm(scores):
    """Transform each class column to [0, 1] uniform ranks. Preserves within-class
    ordering; removes cross-class logit-scale differences (what hengck23 cites)."""
    out = np.empty_like(scores, dtype=np.float32)
    n = scores.shape[0]
    for c in range(scores.shape[1]):
        out[:, c] = (rankdata(scores[:, c], method="average") - 1.0) / max(1, n - 1)
    return out


# ─────────────── Main ───────────────

def main():
    t0 = time.time()
    primary, label_to_idx, n_classes, sc_clean, Y_SC = load_metadata()
    meta_full = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    arr = np.load(EXP21 / "full_perch_arrays.npz")
    scores_raw = arr["scores"].astype(np.float32)
    emb = arr["emb"].astype(np.float32)

    sc_idx = sc_clean.set_index("row_id")
    Y_FULL = np.stack([Y_SC[sc_idx.index.get_loc(rid)] for rid in meta_full["row_id"]])

    files = sorted(meta_full["filename"].unique().tolist())
    file_to_site = meta_full.drop_duplicates("filename").set_index("filename")["site"].to_dict()
    file_sites = [file_to_site[f] for f in files]

    file_fold_a = stratified_file_folds(files, file_sites, k=5, seed=SEED)
    file_to_fold_a = dict(zip(files, file_fold_a))
    rowfold_a = np.array([file_to_fold_a[f] for f in meta_full["filename"]])

    sites_full = meta_full["site"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    rowfold_b = np.full(len(meta_full), -1, dtype=int)
    for fi, (_, va_idx) in enumerate(gkf.split(scores_raw, groups=sites_full)):
        rowfold_b[va_idx] = fi

    print(f"Rows: {len(emb)}  Files: {len(files)}  Sites: {meta_full['site'].nunique()}")
    print(f"Active classes: {(Y_FULL.sum(0) > 0).sum()}\n")

    results = []

    def run_probe_config(name, pca_dim, C, alpha, min_pos=5):
        # Compute probe OOF under both Val-A and Val-B
        out = {"config": name, "pca_dim": pca_dim, "C": C, "alpha": alpha}
        for reg_name, rowfold in [("val_a", rowfold_a), ("val_b", rowfold_b)]:
            oof = np.zeros_like(scores_raw)
            for fi in range(5):
                tr = np.where(rowfold != fi)[0]
                va = np.where(rowfold == fi)[0]
                oof[va] = fit_probes(emb, Y_FULL, tr, va,
                                     base_va=scores_raw[va],
                                     pca_dim=pca_dim, C=C, alpha=alpha,
                                     min_pos=min_pos)
            out[f"{reg_name}_auc"] = macro_auc(Y_FULL, oof)
            out[f"{reg_name}_oof"] = oof
        return out

    # ═══════ A. probe hyperparameter sweep ═══════
    print("=== Phase A: probe hyperparameter sweep ===")
    probe_configs = [
        ("R1_baseline",   32, 0.25, 0.50),   # exp26/27 R1
        ("LB910_freeze",  32, 0.25, 0.40),   # exp20 v1 frozen
        ("starter_freeze",64, 0.50, 0.40),   # perch-v2-starter frozen
        ("PCA64_C025",    64, 0.25, 0.50),
        ("PCA32_C050",    32, 0.50, 0.50),
        ("PCA96_C050",    96, 0.50, 0.40),
    ]
    probe_results = []
    for name, pca_d, C, alpha in probe_configs:
        tA = time.time()
        r = run_probe_config(name, pca_d, C, alpha)
        r["time_s"] = time.time() - tA
        probe_results.append(r)
        print(f"  {name:20s}  Val-A={r['val_a_auc']:.4f}  Val-B={r['val_b_auc']:.4f}  ({r['time_s']:.1f}s)")

    best_probe = max(probe_results, key=lambda x: x["val_a_auc"])
    print(f"\nBest probe by Val-A: {best_probe['config']} "
          f"(Val-A {best_probe['val_a_auc']:.4f}, Val-B {best_probe['val_b_auc']:.4f})")

    # ═══════ B. Smoother sweep on best probe (Val-A only) ═══════
    print("\n=== Phase B: smoother sweep on best probe ===")
    smoother_results = []
    for sigma in [0.5, 0.75, 1.0, 1.25, 1.5]:
        k = gauss_kernel(sigma)
        for reg_name, oof_key in [("val_a", "val_a_oof"), ("val_b", "val_b_oof")]:
            sm = gauss_smooth_per_file(best_probe[oof_key], meta_full, k)
            auc = macro_auc(Y_FULL, sm)
            smoother_results.append({
                "kernel": f"gauss_sigma{sigma}",
                "regime": reg_name,
                "auc": auc,
            })
    # R5 default
    for reg_name, oof_key in [("val_a", "val_a_oof"), ("val_b", "val_b_oof")]:
        sm = gauss_smooth_per_file(best_probe[oof_key], meta_full, GAUSS_W_DEFAULT)
        auc = macro_auc(Y_FULL, sm)
        smoother_results.append({
            "kernel": "R5_default",
            "regime": reg_name,
            "auc": auc,
        })
    for r in smoother_results:
        print(f"  {r['kernel']:25s} {r['regime']}  {r['auc']:.4f}")

    # Pick best smoother by Val-A
    va_sm = [r for r in smoother_results if r["regime"] == "val_a"]
    best_sm = max(va_sm, key=lambda x: x["auc"])
    # Re-apply for persistent arr
    if best_sm["kernel"] == "R5_default":
        best_kernel = GAUSS_W_DEFAULT
    else:
        sigma = float(best_sm["kernel"].replace("gauss_sigma", ""))
        best_kernel = gauss_kernel(sigma)
    best_smooth_a = gauss_smooth_per_file(best_probe["val_a_oof"], meta_full, best_kernel)
    best_smooth_b = gauss_smooth_per_file(best_probe["val_b_oof"], meta_full, best_kernel)

    # ═══════ C. Per-class rank norm (applied to smoothed probe) ═══════
    print("\n=== Phase C: per-class rank normalization ===")
    rn_results = []
    for label, oof, reg_name in [
        ("probe_only",   best_probe["val_a_oof"], "val_a"),
        ("probe+smooth", best_smooth_a,            "val_a"),
        ("probe_only",   best_probe["val_b_oof"], "val_b"),
        ("probe+smooth", best_smooth_b,            "val_b"),
    ]:
        rn = per_class_rank_norm(oof)
        auc_plain = macro_auc(Y_FULL, oof)
        auc_rn = macro_auc(Y_FULL, rn)
        rn_results.append({
            "base": label, "regime": reg_name,
            "auc_plain": auc_plain, "auc_rank_norm": auc_rn,
            "delta": auc_rn - auc_plain,
        })
        print(f"  {label:15s} {reg_name}  plain={auc_plain:.4f}  rn={auc_rn:.4f}  Δ={auc_rn-auc_plain:+.4f}")

    # ═══════ Save ═══════
    trim_probe_results = [{k: v for k, v in r.items() if not k.endswith("_oof")}
                           for r in probe_results]
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": time.time() - t0,
        "eval_rows": int(len(emb)),
        "active_classes": int((Y_FULL.sum(0) > 0).sum()),
        "probe_sweep": trim_probe_results,
        "best_probe": {k: v for k, v in best_probe.items() if not k.endswith("_oof")},
        "smoother_sweep": smoother_results,
        "best_smoother": best_sm,
        "rank_norm": rn_results,
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        OUT / "best_oof.npz",
        val_a_probe=best_probe["val_a_oof"],
        val_b_probe=best_probe["val_b_oof"],
        val_a_smoothed=best_smooth_a,
        val_b_smoothed=best_smooth_b,
    )
    print(f"\nSaved: {OUT/'results.json'}  ({summary['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
