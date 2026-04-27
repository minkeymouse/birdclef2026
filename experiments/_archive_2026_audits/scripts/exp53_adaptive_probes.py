#!/usr/bin/env python3
"""exp53 — Adaptive inference-time lever comparison.

Tests 5 inference-time schemes that DO NOT depend on train-SS-derived lookup:
  53a: per-row confidence-weighted blend between Perch and SED29
  53b: ensemble disagreement smoothing (pull high-disagreement toward row mean)
  53c: test-time per-class temperature (quantile normalization of logits)
  53d: per-file bias removal (file mean centering, priors-OFF style)
  53e: SSM on embedding sequence (per-file Kalman on pooled Perch emb then re-probe via class centroids)

Each scheme is scored by:
  (1) macro AUC on 66 labeled SS files
  (2) Spearman correlation (v12_prob, new_prob) per-row and per-class
  (3) per-taxon Δ AUC

Only schemes with Spearman ≥ 0.95 (small perturbation) are LB-safe candidates.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, rankdata

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
OUT = ROOT / "experiments/exp53_outputs"
OUT.mkdir(exist_ok=True)
SEED = 42


def build_all():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g.lbls):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_g, Y, primary, l2i


def align_43a(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]; embs = d["emb"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(df), scs.shape[1]), np.float32)
    E = np.zeros((len(df), embs.shape[1]), np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]; E[i] = embs[j]
    return S, E


def align_old(df, p):
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(df), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s

def gauss_pf(scores, df, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in df.filename.unique():
        m = (df.filename == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out


def per_class_auc(Y, P):
    ev = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    return {c: float(roc_auc_score(Y[:, c], P[:, c])) for c in ev
            if np.isfinite(P[:, c]).all()}


def spearman_summary(A, B, label):
    r_row = []; r_col = []
    for i in range(A.shape[0]):
        r, _ = spearmanr(A[i], B[i])
        if np.isfinite(r): r_row.append(r)
    for c in range(A.shape[1]):
        r, _ = spearmanr(A[:, c], B[:, c])
        if np.isfinite(r): r_col.append(r)
    return {"label": label,
            "per_row_mean": float(np.mean(r_row)), "per_row_min": float(np.min(r_row)),
            "per_col_mean": float(np.mean(r_col)), "per_col_min": float(np.min(r_col))}


def eval_and_compare(Y, P, P_ref, species_taxon, label):
    aucs = per_class_auc(Y, P)
    aucs_ref = per_class_auc(Y, P_ref)
    common = set(aucs) & set(aucs_ref)
    macro = np.mean([aucs[c] for c in common])
    macro_ref = np.mean([aucs_ref[c] for c in common])
    per_tax = {}
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        cls = [c for c in common if species_taxon[c] == t]
        if cls:
            per_tax[t] = np.mean([aucs[c] - aucs_ref[c] for c in cls])
    sp = spearman_summary(P_ref, P, label)
    return {"label": label, "macro": macro, "delta": macro - macro_ref,
            **{f"tx_{t}_delta": per_tax.get(t, np.nan) for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]},
            **{f"sp_{k}": sp[k] for k in ["per_row_mean","per_row_min","per_col_mean","per_col_min"]}}


# ─── Scheme implementations ───

def scheme_53a(perch_prob, sed_prob, beta=0.2):
    """Per-row confidence-weighted blend.
    Perch confidence = top1 - top2 probability gap.
    Higher gap → trust Perch more → w_perch increases above 0.8.
    Lower gap → lean on SED → w_perch decreases below 0.8.
    Maps gap ∈ [0, 0.5] → δ ∈ [-beta, +beta], centered on 0.
    """
    sorted_p = np.sort(perch_prob, axis=1)[:, -2:]
    gap = sorted_p[:, -1] - sorted_p[:, -2]  # (N,)
    # Normalize gap: if gap=0.25 (median-ish), δ=0
    gap_norm = np.clip((gap - gap.mean()) / (gap.std() + 1e-8), -2, 2)
    delta = beta * gap_norm / 2
    w_perch = 0.8 + delta  # (N,)
    w_perch = np.clip(w_perch, 0.5, 0.95)
    w_sed = 1 - w_perch
    # z-score normalize then blend
    zP = zs(perch_prob); zS = zs(sed_prob)
    blend = w_perch[:, None] * zP + w_sed[:, None] * zS
    return blend


def scheme_53b(perch_prob, sed_prob, k=0.3):
    """Ensemble disagreement smoothing.
    For rows where teachers strongly disagree, pull prediction toward row mean (shrink toward uniform).
    disagreement[r, c] = |perch[r, c] - sed[r, c]|
    For each row, if mean disagreement is high → shrink factor higher.
    """
    zP = zs(perch_prob); zS = zs(sed_prob)
    base = 0.8 * zP + 0.2 * zS  # v12-like blend
    # Disagreement per row (RMS over classes)
    disagree = np.sqrt(((perch_prob - sed_prob) ** 2).mean(axis=1))  # (N,)
    # Normalize to [0, 1]
    d_norm = (disagree - disagree.min()) / (disagree.max() - disagree.min() + 1e-8)
    shrink = k * d_norm[:, None]  # (N, 1)
    # Shrink toward row mean
    row_mean = base.mean(axis=1, keepdims=True)
    adjusted = (1 - shrink) * base + shrink * row_mean
    return adjusted


def scheme_53c(probs):
    """Test-time per-class temperature via quantile normalization.
    For each class, map its prediction distribution across rows to a standard
    normal distribution. This preserves within-class ranking but normalizes
    the distribution shape (equalizes cross-class scales).
    """
    N, C = probs.shape
    out = np.zeros_like(probs)
    for c in range(C):
        ranks = rankdata(probs[:, c]) / (N + 1)  # (0,1)
        # Map to standard normal quantile
        from scipy.stats import norm
        out[:, c] = norm.ppf(ranks)
    # Back to probability via sigmoid
    return sigmoid(out)


def scheme_53d(probs, df):
    """Per-file bias removal (priors-OFF / file-mean centering).
    For each file, subtract per-class mean within file, add global per-class mean.
    Removes file-level bias (e.g., "this file sounds overall quiet/noisy").
    """
    gmean = probs.mean(axis=0, keepdims=True)
    out = probs.copy()
    for fn in df.filename.unique():
        m = (df.filename == fn).values
        fmean = probs[m].mean(axis=0, keepdims=True)
        out[m] = probs[m] - fmean + gmean
    return np.clip(out, 0, 1)


def scheme_53e(perch_prob, perch_emb, df):
    """SSM on embedding sequence — per-file Kalman smoothing on pooled Perch
    embeddings, then re-combine with original predictions via cosine similarity
    weighting."""
    # For each file, apply exponential smoothing on the embedding sequence
    # (poor-man's Kalman: EMA with adaptive gain).
    smoothed_emb = perch_emb.copy()
    for fn in df.filename.unique():
        m = (df.filename == fn).values
        idx = np.where(m)[0]
        if len(idx) < 2: continue
        # EMA with alpha 0.3
        alpha = 0.3
        emb_file = perch_emb[idx].copy()
        ema = emb_file[0].copy()
        for k in range(1, len(emb_file)):
            ema = alpha * emb_file[k] + (1 - alpha) * ema
            smoothed_emb[idx[k]] = 0.5 * ema + 0.5 * emb_file[k]
    # Similarity between original and smoothed emb per row
    sim = (perch_emb * smoothed_emb).sum(axis=1) / \
          (np.linalg.norm(perch_emb, axis=1) * np.linalg.norm(smoothed_emb, axis=1) + 1e-8)
    # Rows with high similarity: trust original. Low: pull toward file mean.
    gamma = 0.3
    file_mean_prob = np.zeros_like(perch_prob)
    for fn in df.filename.unique():
        m = (df.filename == fn).values
        file_mean_prob[m] = perch_prob[m].mean(axis=0)
    w = np.clip(sim, 0, 1)[:, None]  # (N, 1)
    out = w * perch_prob + (1 - w) * (gamma * file_mean_prob + (1 - gamma) * perch_prob)
    return out


# ─── main ───

def main():
    print("Loading data and base preds...")
    sc_all, Y_all, primary, l2i = build_all()
    S_p, E_p = align_43a(sc_all)
    perch_prob = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)
    sed_prob = sigmoid(S29)  # also sigmoid sed for symmetry
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    # v12 reference
    zP = zs(perch_prob); z29 = zs(S29)
    v12_raw = 0.8 * zP + 0.2 * z29
    v12_prob = sigmoid(gauss_pf(v12_raw, sc_all, 0.5))
    print(f"Rows: {len(sc_all)}  classes evaluable (v12): {len(per_class_auc(Y_all, v12_prob))}")

    # Apply each scheme (then Gauss smoothing + sigmoid where needed to match v12 pipeline)
    results = [eval_and_compare(Y_all, v12_prob, v12_prob, species_taxon, "v12 (reference)")]

    print("\n--- 53a: per-row confidence-weighted blend ---")
    for beta in [0.05, 0.1, 0.2, 0.3]:
        b = scheme_53a(perch_prob, sed_prob, beta=beta)
        p_sm = sigmoid(gauss_pf(b, sc_all, 0.5))
        r = eval_and_compare(Y_all, p_sm, v12_prob, species_taxon, f"53a_conf_β={beta}")
        results.append(r)
        print(f"  β={beta}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  "
              f"sp_row {r['sp_per_row_mean']:.3f}  sp_col {r['sp_per_col_mean']:.3f}  "
              f"Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 53b: ensemble disagreement smoothing ---")
    for k in [0.1, 0.2, 0.3, 0.5]:
        b = scheme_53b(perch_prob, sed_prob, k=k)
        p_sm = sigmoid(gauss_pf(b, sc_all, 0.5))
        r = eval_and_compare(Y_all, p_sm, v12_prob, species_taxon, f"53b_disagree_k={k}")
        results.append(r)
        print(f"  k={k}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  "
              f"sp_row {r['sp_per_row_mean']:.3f}  sp_col {r['sp_per_col_mean']:.3f}  "
              f"Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 53c: per-class quantile normalization ---")
    # Apply quantile norm on the blend z-logit before sigmoid (not on v12_prob, since that ruins Gauss)
    for variant in ["on_logit", "on_prob"]:
        if variant == "on_logit":
            qn = scheme_53c(sigmoid(gauss_pf(v12_raw, sc_all, 0.5)))  # v12_prob
        else:
            qn = scheme_53c(v12_prob)
        r = eval_and_compare(Y_all, qn, v12_prob, species_taxon, f"53c_quantile_{variant}")
        results.append(r)
        print(f"  variant={variant}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  "
              f"sp_row {r['sp_per_row_mean']:.3f}  sp_col {r['sp_per_col_mean']:.3f}  "
              f"Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 53d: per-file bias removal ---")
    for which in ["v12_prob", "v12_raw"]:
        if which == "v12_prob":
            out = scheme_53d(v12_prob, sc_all)
        else:
            # Apply centering on z-blend before sigmoid
            out = sigmoid(gauss_pf(scheme_53d(v12_raw, sc_all), sc_all, 0.5))
        r = eval_and_compare(Y_all, out, v12_prob, species_taxon, f"53d_file_center_{which}")
        results.append(r)
        print(f"  which={which}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  "
              f"sp_row {r['sp_per_row_mean']:.3f}  sp_col {r['sp_per_col_mean']:.3f}  "
              f"Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 53e: SSM on embeddings — file-level EMA + confidence weighting ---")
    ssm_probs = scheme_53e(perch_prob, E_p, sc_all)
    # Blend with SED29 at v12 ratios
    ssm_raw = 0.8 * zs(ssm_probs) + 0.2 * z29
    ssm_sm = sigmoid(gauss_pf(ssm_raw, sc_all, 0.5))
    r = eval_and_compare(Y_all, ssm_sm, v12_prob, species_taxon, "53e_SSM_emb_EMA")
    results.append(r)
    print(f"  SSM EMA  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  "
          f"sp_row {r['sp_per_row_mean']:.3f}  sp_col {r['sp_per_col_mean']:.3f}  "
          f"Aves_Δ {r['tx_Aves_delta']:+.4f}")

    # Save
    df = pd.DataFrame(results).sort_values("delta", ascending=False)
    df.to_csv(OUT / "53_results.csv", index=False)
    print("\n=== SUMMARY (sorted by macro Δ) ===")
    print(df[["label","macro","delta","sp_per_row_mean","sp_per_col_mean","tx_Aves_delta"]].to_string(index=False))

    # Best LB-safe candidates: sp_per_row_mean ≥ 0.95 AND delta > 0
    safe = df[(df.sp_per_row_mean >= 0.95) & (df.delta > 0) & (df.label != "v12 (reference)")]
    if len(safe):
        print(f"\n=== LB-SAFE CANDIDATES (Spearman≥0.95, Δ>0) ===")
        print(safe[["label","macro","delta","sp_per_row_mean","tx_Aves_delta"]].to_string(index=False))
    else:
        print(f"\nNo LB-safe candidates found (no scheme preserves rank≥0.95 AND improves macro)")

    print(f"\nSaved → {OUT}/53_results.csv")


if __name__ == "__main__":
    main()
