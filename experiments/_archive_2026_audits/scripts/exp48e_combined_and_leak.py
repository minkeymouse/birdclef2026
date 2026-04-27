#!/usr/bin/env python3
"""exp48e — Leak-free cluster validation + combined site-prior × cluster-rewrite.

Address two risks from exp48d:
  1. Cluster definitions (triggers → targets) were derived from the 11-file EVAL
     → possible eval leak. Re-derive clusters from TRAIN 55-file labeled SS
     and re-evaluate on eval.
  2. Site prior and cluster rewrite may be redundant. Test combined.

Approach:
  (A) Use train labeled SS (55 files, 617 rows) to find mean Aves prediction
      on each rare class' positive rows. Take top-3 Aves per rare target as
      triggers. Apply to eval.
  (B) Combine with site prior (best-perf cfg from 48b).
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
OUT = ROOT / "experiments/exp48_outputs"
SEED = 42; EVAL_N = 11
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")


def parse_site(fn):
    m = FNAME_RE.match(fn)
    return m.group(2) if m else None


def build_splits():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g["site"] = sc_g["filename"].apply(parse_site)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N])
    train_files = set(files[EVAL_N:])
    sc_train = sc_g[sc_g.filename.isin(train_files)].reset_index(drop=True)
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    l2i = {p: i for i, p in enumerate(primary)}
    def Y(df):
        Y_ = np.zeros((len(df), len(primary)), dtype=np.uint8)
        for i, labs in enumerate(df["lbls"]):
            for l in labs:
                if l in l2i: Y_[i, l2i[l]] = 1
        return Y_
    return sc_train, Y(sc_train), sc_eval, Y(sc_eval), primary, l2i


def align_43a(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(df), scs.shape[1]), np.float32)
    for i, rid in enumerate(df["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]
    return S


def align_old(df, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(df), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(df["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s
def gauss_pf(scores, df, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in df["filename"].unique():
        m = (df["filename"] == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out


def per_class_auc(Y, P):
    ev = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    return {c: float(roc_auc_score(Y[:, c], P[:, c])) for c in ev
            if np.isfinite(P[:, c]).all()}


def derive_clusters(Y_train, P_train, primary, l2i, species_taxon, top_k=3, min_pos=3):
    """For each rare target (non-Aves + low-performing), find top-K Aves
    classes with highest mean prediction on positive rows.
    """
    target_cls = []
    for c in range(Y_train.shape[1]):
        if species_taxon[c] == "Aves" or species_taxon[c] == "?": continue
        if Y_train[:, c].sum() >= min_pos:
            target_cls.append(c)
    aves_idx = np.array([c for c in range(Y_train.shape[1]) if species_taxon[c] == "Aves"])

    cluster_map = {}  # target_idx → list of Aves trigger idx
    for tc in target_cls:
        pos_rows = np.where(Y_train[:, tc] == 1)[0]
        mean_aves_pred = P_train[pos_rows][:, aves_idx].mean(axis=0)
        top = np.argsort(mean_aves_pred)[-top_k:]
        cluster_map[tc] = aves_idx[top].tolist()
    return cluster_map


def main():
    sc_train, Y_train, sc_eval, Y_eval, primary, l2i = build_splits()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    # Base pipeline on both
    def v12(df):
        S_perch = align_43a(df)
        perch_prob = sigmoid(S_perch)
        S29 = align_old(df, EXP29 / "val_scores.npz")
        zP = zs(perch_prob); z29 = zs(np.nan_to_num(S29, nan=0))
        return sigmoid(gauss_pf(0.8*zP + 0.2*z29, df, 0.5))

    P_train = v12(sc_train); P_eval = v12(sc_eval)
    eval_aucs = per_class_auc(Y_eval, P_eval)
    base_macro = np.mean(list(eval_aucs.values()))
    print(f"v12 on train: rows={len(sc_train)}  on eval: rows={len(sc_eval)}  macro={base_macro:.4f}")

    # Derive clusters from TRAIN (leak-free)
    cluster_map = derive_clusters(Y_train, P_train, primary, l2i, species_taxon, top_k=3)
    print(f"\n[Clusters derived from TRAIN ({len(cluster_map)} targets)]")
    for tc, trig_idx in cluster_map.items():
        print(f"  {primary[tc]:<12} ({species_taxon[tc]:<8}) triggers={[primary[t] for t in trig_idx]}")

    # Apply cluster-rewrite derived from TRAIN on EVAL
    print(f"\n[Eval macro with TRAIN-derived clusters]")
    best_macro = base_macro; best_cfg = None
    for alpha in [0.5, 1.0, 2.0, 4.0]:
        p_new = P_eval.copy()
        for tc, trig_idx in cluster_map.items():
            sc_arr = P_eval[:, trig_idx].min(axis=1)  # min-agg
            p_new[:, tc] = p_new[:, tc] * (1 + alpha * sc_arr)
        aucs = per_class_auc(Y_eval, p_new)
        m = np.mean([aucs[c] for c in eval_aucs if c in aucs])
        print(f"  alpha={alpha}  macro={m:.4f}  Δ={m-base_macro:+.4f}")
        if m > best_macro: best_macro = m; best_cfg = ("cluster", alpha, p_new)

    # Build site prior from TRAIN (also leak-free)
    site_prior_norm = np.zeros((16, len(primary)), dtype=np.float32)  # max 16 sites safety
    site_idx = {}
    for s in sorted(sc_train.site.unique()):
        site_idx[s] = len(site_idx)
    for site, grp in sc_train.groupby("site"):
        si = site_idx[site]
        cnt = np.zeros(len(primary), dtype=np.float32)
        for _, r in grp.iterrows():
            for l in r.lbls:
                if l in l2i: cnt[l2i[l]] += 1
        site_prior_norm[si] = cnt / (cnt.max() + 1e-8)

    # Apply site prior (soft tau=0.5 from 48b)
    print(f"\n[Site prior alone (TRAIN-derived, soft tau variants)]")
    eval_site_vec = np.ones((len(sc_eval), len(primary)), dtype=np.float32)
    for i, row in sc_eval.iterrows():
        si = site_idx.get(row.site)
        if si is not None:
            eval_site_vec[i] = site_prior_norm[si]
    for tau in [0.25, 0.5, 0.75]:
        p_new = P_eval * (tau * eval_site_vec + (1 - tau))
        aucs = per_class_auc(Y_eval, p_new)
        m = np.mean([aucs[c] for c in eval_aucs if c in aucs])
        print(f"  tau={tau}  macro={m:.4f}  Δ={m-base_macro:+.4f}")

    # COMBINED: site prior AND cluster rewrite
    print(f"\n[COMBINED: site-prior (soft tau) × cluster-rewrite (alpha)]")
    for tau in [0.25, 0.5, 0.75]:
        for alpha in [0.5, 1.0, 2.0, 4.0]:
            p_new = P_eval * (tau * eval_site_vec + (1 - tau))
            for tc, trig_idx in cluster_map.items():
                sc_arr = P_eval[:, trig_idx].min(axis=1)
                p_new[:, tc] = p_new[:, tc] * (1 + alpha * sc_arr)
            aucs = per_class_auc(Y_eval, p_new)
            m = np.mean([aucs[c] for c in eval_aucs if c in aucs])
            print(f"  tau={tau}  alpha={alpha}  macro={m:.4f}  Δ={m-base_macro:+.4f}")

    # Do the CLEANEST combined (tau=0.5, alpha=2) per-class breakdown
    print(f"\n[Per-class breakdown @ tau=0.5, alpha=2.0]")
    tau, alpha = 0.5, 2.0
    p_new = P_eval * (tau * eval_site_vec + (1 - tau))
    for tc, trig_idx in cluster_map.items():
        sc_arr = P_eval[:, trig_idx].min(axis=1)
        p_new[:, tc] = p_new[:, tc] * (1 + alpha * sc_arr)
    aucs = per_class_auc(Y_eval, p_new)
    # Show biggest Δ up + biggest Δ down
    deltas = [(c, aucs.get(c, 0) - eval_aucs[c]) for c in eval_aucs if c in aucs]
    deltas.sort(key=lambda x: x[1])
    print("  BIGGEST DROPS (worst-case damage):")
    for c, d in deltas[:5]:
        print(f"    {primary[c]:<12} ({species_taxon[c]:<8}) {eval_aucs[c]:.3f} → {aucs[c]:.3f}  Δ{d:+.3f}")
    print("  BIGGEST GAINS:")
    for c, d in deltas[-10:]:
        print(f"    {primary[c]:<12} ({species_taxon[c]:<8}) {eval_aucs[c]:.3f} → {aucs[c]:.3f}  Δ{d:+.3f}")

    # Per-taxon summary
    print(f"\n[Per-taxon Δ @ tau=0.5, alpha=2.0]")
    for tname in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        cls_t = [c for c in eval_aucs if species_taxon[c] == tname]
        cls_t = [c for c in cls_t if c in aucs]
        if cls_t:
            b = np.mean([eval_aucs[c] for c in cls_t])
            n = np.mean([aucs[c] for c in cls_t])
            print(f"  {tname:<10} n={len(cls_t):2d}  {b:.3f} → {n:.3f}  Δ{n-b:+.3f}")

    # Check on train: anti-leak check (train macro shouldn't bump from clusters it produced)
    p_t = P_train.copy()
    for tc, trig_idx in cluster_map.items():
        sc_arr = P_train[:, trig_idx].min(axis=1)
        p_t[:, tc] = p_t[:, tc] * (1 + 2.0 * sc_arr)
    train_aucs_base = per_class_auc(Y_train, P_train)
    train_aucs_rewrite = per_class_auc(Y_train, p_t)
    t_base = np.mean(list(train_aucs_base.values()))
    t_rewrite = np.mean([train_aucs_rewrite[c] for c in train_aucs_base if c in train_aucs_rewrite])
    print(f"\n[Sanity: TRAIN macro base={t_base:.4f}  train-rewrite={t_rewrite:.4f} (should be HIGHER on train since clusters derived here)]")

    with open(OUT / "48e_combined.json", "w") as f:
        json.dump({
            "base_macro_eval": float(base_macro),
            "n_clusters": len(cluster_map),
            "cluster_map": {primary[tc]: [primary[t] for t in trig] for tc, trig in cluster_map.items()},
        }, f, indent=2, default=float)


if __name__ == "__main__":
    main()
