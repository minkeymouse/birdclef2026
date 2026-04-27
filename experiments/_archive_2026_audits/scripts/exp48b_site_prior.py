#!/usr/bin/env python3
"""exp48b — Site-conditional prior.

Build P(species | site) from all AVAILABLE labeled signals:
  (a) labeled SS train split (55 files): P(species present in labeled window | site)
  (b) train_audio metadata: if available per-file site info (may not exist, skip if not)

Apply site prior to v12 predictions on 11-file held-out eval:
  final = base × (site_prior × tau + (1 - tau))        # soft gate
  (tau=1.0: hard site prior; tau=0: no-op)

Test multiple tau values {0, 0.1, 0.25, 0.5, 0.75, 1.0} and report:
  - macro AUC change per tau
  - per-site macro AUC change
  - per-taxon macro AUC change
  - bottom-8 species recovery

Also test a "hard zero" variant: classes with site_prior == 0 get zeroed.
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
OUT.mkdir(exist_ok=True)
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
    Y_eval = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["lbls"]):
        for l in labs:
            if l in l2i: Y_eval[i, l2i[l]] = 1
    return sc_eval, sc_train, Y_eval, primary, l2i


def build_site_prior(sc_train, l2i, laplace_alpha=1.0):
    """P(species | site) with Laplace smoothing.

    For a window at site s, prior[c] = (count of windows with species c at site s + alpha)
                                       / (total windows at site s + alpha * n_cls)
    """
    sites = sorted(sc_train.site.unique())
    n_cls = len(l2i)
    prior = np.zeros((len(sites), n_cls), dtype=np.float32)
    site_idx = {s: i for i, s in enumerate(sites)}
    totals = np.zeros(len(sites), dtype=np.int64)
    for _, row in sc_train.iterrows():
        si = site_idx[row.site]
        totals[si] += 1
        for l in row.lbls:
            if l in l2i: prior[si, l2i[l]] += 1
    # Laplace smoothing
    prior_sm = (prior + laplace_alpha) / (totals[:, None] + laplace_alpha * n_cls)
    # also compute normalized-by-max per-site so values in [0,1]
    prior_norm = prior / (prior.max(axis=1, keepdims=True) + 1e-8)
    # indicator (any positive in site) + Laplace 0-prob
    prior_ind = (prior > 0).astype(np.float32)
    return prior, prior_sm, prior_norm, prior_ind, site_idx, sites


def align_43a(sc_eval):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(sc_eval), scs.shape[1]), np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]
    return S


def align_old(sc_eval, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(sc_eval), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s

def gauss_pf(scores, sc_eval, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in sc_eval["filename"].unique():
        m = (sc_eval["filename"] == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out


def per_class_auc(Y, P, sc_eval=None, by="macro"):
    evaluable = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    aucs = {}
    for c in evaluable:
        try: aucs[c] = float(roc_auc_score(Y[:, c], P[:, c]))
        except Exception: pass
    return aucs


def main():
    sc_eval, sc_train, Y, primary, l2i = build_splits()
    print(f"Train SS: {len(sc_train)} rows from {len(sc_train.filename.unique())} files, "
          f"sites {sorted(sc_train.site.unique())}")
    print(f"Eval SS: {len(sc_eval)} rows, sites {sorted(sc_eval.site.unique())}")

    # Overlap check: which eval sites are represented in train?
    train_sites = set(sc_train.site.unique())
    eval_sites = set(sc_eval.site.unique())
    print(f"Eval sites in train: {eval_sites & train_sites}")
    print(f"Eval sites NOT in train: {eval_sites - train_sites}")

    # Build priors
    pr_raw, pr_sm, pr_norm, pr_ind, site_idx, sites_list = build_site_prior(sc_train, l2i)
    print(f"\nSite prior shape: {pr_raw.shape}. n_cls with positive at >=1 site: "
          f"{int((pr_raw.sum(0) > 0).sum())}/{pr_raw.shape[1]}")

    # Per-site species counts
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    # Show which rare species have positive prior at eval sites
    for es in sorted(eval_sites & train_sites):
        si = site_idx[es]
        p_site = pr_ind[si]
        n_pos = int(p_site.sum())
        pos_cls = [primary[c] for c in np.where(p_site > 0)[0]]
        insect_at_site = [p for p in pos_cls if p.startswith("47158son")]
        mammalia_at_site = [p for p in pos_cls if species_taxon[l2i[p]] == "Mammalia"]
        reptilia_at_site = [p for p in pos_cls if species_taxon[l2i[p]] == "Reptilia"]
        amphibia_at_site = [p for p in pos_cls if species_taxon[l2i[p]] == "Amphibia"]
        print(f"  {es}: {n_pos} species observed. Insecta {len(insect_at_site)}: {insect_at_site[:5]}... "
              f"Mammalia {len(mammalia_at_site)}: {mammalia_at_site}, "
              f"Reptilia {len(reptilia_at_site)}: {reptilia_at_site}, "
              f"Amphibia {len(amphibia_at_site)}: {amphibia_at_site[:5]}")

    # v12 pipeline
    S_perch = align_43a(sc_eval)
    perch_prob = sigmoid(S_perch)
    S_sed29 = align_old(sc_eval, EXP29 / "val_scores.npz")
    zP = zs(perch_prob); z29 = zs(np.nan_to_num(S_sed29, nan=0))
    v12_raw = 0.8*zP + 0.2*z29
    v12_s = gauss_pf(v12_raw, sc_eval, 0.5)
    v12_prob = sigmoid(v12_s)

    base_aucs = per_class_auc(Y, v12_prob)
    base_macro = np.mean(list(base_aucs.values()))
    print(f"\nv12 base macro AUC: {base_macro:.4f} over {len(base_aucs)} classes")

    # Build per-row site prior vector
    eval_site_vec = np.zeros((len(sc_eval), len(primary)), dtype=np.float32)
    eval_site_ind = np.zeros((len(sc_eval), len(primary)), dtype=np.float32)
    for i, row in sc_eval.iterrows():
        si = site_idx.get(row.site)
        if si is not None:
            eval_site_vec[i] = pr_norm[si]
            eval_site_ind[i] = pr_ind[si]
        else:
            eval_site_vec[i] = 1.0  # no info — identity
            eval_site_ind[i] = 1.0

    # Variant A: soft gate  final = base * (tau * prior_norm + (1 - tau))
    print("\n=== Variant A: soft site prior (normalized by max per site) ===")
    print(f"  {'tau':>5}  {'macro':>6}  {'Δ':>7}")
    for tau in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        p_new = v12_prob * (tau * eval_site_vec + (1 - tau))
        aucs = per_class_auc(Y, p_new)
        m = np.mean([aucs[c] for c in base_aucs if c in aucs])
        print(f"  {tau:>5.2f}  {m:.4f}  {m-base_macro:+.4f}")

    # Variant B: hard zero using indicator
    print("\n=== Variant B: hard indicator (zero out classes with prior=0 at site) ===")
    print(f"  {'tau':>5}  {'macro':>6}  {'Δ':>7}")
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # final = base * (tau * indicator + (1 - tau))
        p_new = v12_prob * (tau * eval_site_ind + (1 - tau))
        aucs = per_class_auc(Y, p_new)
        m = np.mean([aucs[c] for c in base_aucs if c in aucs])
        print(f"  {tau:>5.2f}  {m:.4f}  {m-base_macro:+.4f}")

    # Per-taxon breakdown for tau=1 indicator (the strongest)
    print("\n=== Per-taxon Δ under Variant B tau=1 ===")
    p_hard = v12_prob * eval_site_ind
    aucs_hard = per_class_auc(Y, p_hard)
    for tname in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        cls_t = [c for c in base_aucs if species_taxon[c] == tname]
        cls_both = [c for c in cls_t if c in aucs_hard]
        if cls_both:
            b = np.mean([base_aucs[c] for c in cls_both])
            h = np.mean([aucs_hard[c] for c in cls_both])
            print(f"  {tname:<10} n={len(cls_both):2d}  base={b:.3f} → site-hard={h:.3f}  Δ={h-b:+.3f}")

    # Per-class breakdown for bottom-8 (classes known to be AUC < 0.5)
    print("\n=== Bottom-8 species under multiple tau (soft variant) ===")
    bottom8 = ["516975", "67107", "326272", "bafcur1", "74113", "25073", "116570", "47158son11"]
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p_new = v12_prob * (tau * eval_site_vec + (1 - tau))
        aucs_new = per_class_auc(Y, p_new)
        rec = []
        for lbl in bottom8:
            c = primary.index(lbl) if lbl in primary else -1
            if c >= 0 and c in aucs_new:
                rec.append(f"{lbl}:{aucs_new[c]:.2f}")
        print(f"  tau={tau}  " + "  ".join(rec))

    # Per-site macro under Variant B tau=1
    print("\n=== Per-site macro under Variant B tau=1 (site indicator hard) ===")
    for site in sorted(eval_sites):
        mask = (sc_eval.site == site).values
        if mask.sum() == 0: continue
        Y_s = Y[mask]; P_s_base = v12_prob[mask]; P_s_hard = p_hard[mask]
        cls_s = [c for c in range(Y.shape[1]) if 0 < Y_s[:, c].sum() < len(Y_s)]
        if not cls_s: continue
        try:
            b_auc = np.mean([roc_auc_score(Y_s[:, c], P_s_base[:, c]) for c in cls_s])
            h_auc = np.mean([roc_auc_score(Y_s[:, c], P_s_hard[:, c]) for c in cls_s])
            print(f"  {site}  n_rows={mask.sum():3d}  n_cls={len(cls_s):2d}  base={b_auc:.3f} → hard={h_auc:.3f}  Δ={h_auc-b_auc:+.3f}")
        except Exception: pass

    # Save
    with open(OUT / "48b_site_prior.json", "w") as f:
        json.dump({
            "base_macro": float(base_macro),
            "train_sites": list(sorted(train_sites)),
            "eval_sites": list(sorted(eval_sites)),
            "sites_in_both": list(sorted(eval_sites & train_sites)),
        }, f, indent=2, default=float)


if __name__ == "__main__":
    main()
