"""
exp30 — State-space smoothing on Perch probe output

Motivation (hengck23): `lb 0.93 = perch2 + proxy + temporal + PCA + prior + state space + perclass_norm`.
We already have proxy/temporal/PCA/probe baked in (exp28 best: Val-A 0.8943 with Gaussian σ=0.75).
This experiment replaces the Gaussian smoother with proper state-space models across the 12
windows of each file, for each class independently.

Smoothers tested (all per-(file, class) on 12-length sequences):
  S1  Gaussian σ ∈ {0.5, 0.75, 1.0, 1.25}            (baseline)
  S2  1D Kalman RTS smoother (random-walk state)     — sweep Q/R
  S3  EMA (exponential moving average, two-pass)      — cheap baseline
  S4  Median filter (k=3, k=5)                        — robust baseline
  S5  HMM binary, transitions from labels.csv         — hengck23's spec

Inputs: exp28 probe logits per-class, shape (708, 234). File grouping from meta (12 windows/file).
Outputs: Val-A / Val-B macro-AUC per smoother. Writes results.json + best scores.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d, median_filter

ROOT = Path("/data/birdclef2026")
EXP28 = ROOT / "experiments/exp28_outputs"
CACHE = ROOT / "experiments/exp21_outputs/perch_cache"
OUT = ROOT / "experiments/exp30_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/birdclef-2026"


def load_truth():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sample_sub.columns[1:].tolist()

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    meta = pd.read_parquet(CACHE / "full_perch_meta.parquet")
    Y = np.zeros((len(meta), len(primary)), dtype=np.uint8)
    lab2idx = {c: i for i, c in enumerate(primary)}
    by_rowid = sc.set_index("row_id")
    for i, rid in enumerate(meta["row_id"]):
        if rid in by_rowid.index:
            for l in by_rowid.loc[rid, "lbls"]:
                if l in lab2idx:
                    Y[i, lab2idx[l]] = 1
    return meta, Y, primary


def macro_auc(y_true, y_score):
    keep = y_true.sum(0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


# ─── Smoothers ─────────────────────────────────────────────────────────────

def smooth_gauss(seq, sigma):
    """seq: (N_files, 12, 234) → smoothed same shape."""
    return gaussian_filter1d(seq, sigma=sigma, axis=1, mode="nearest")


def smooth_ema(seq, alpha):
    """Two-pass EMA (forward + backward, averaged)."""
    T = seq.shape[1]
    fwd = np.empty_like(seq); fwd[:, 0] = seq[:, 0]
    for t in range(1, T):
        fwd[:, t] = alpha * seq[:, t] + (1 - alpha) * fwd[:, t-1]
    bwd = np.empty_like(seq); bwd[:, -1] = seq[:, -1]
    for t in range(T - 2, -1, -1):
        bwd[:, t] = alpha * seq[:, t] + (1 - alpha) * bwd[:, t+1]
    return 0.5 * (fwd + bwd)


def smooth_median(seq, k):
    """Median filter along axis=1 (12 windows), size=k."""
    size = [1, k, 1]
    return median_filter(seq, size=size, mode="nearest")


def smooth_kalman_rts(seq, Q, R):
    """1D random-walk Kalman RTS smoother per (file, class).
    State model: x_t = x_{t-1} + w,  w ~ N(0, Q)
    Obs model:   z_t = x_t + v,       v ~ N(0, R)
    Input shape: (N, T, C).  Vectorized across (N, C).
    """
    N, T, C = seq.shape
    # Forward
    x_f = np.empty((T, N, C), dtype=np.float32)
    P_f = np.empty((T, N, C), dtype=np.float32)
    # init with observation
    x_f[0] = seq[:, 0]
    P_f[0] = 1e6  # wide prior
    for t in range(1, T):
        x_pred = x_f[t-1]
        P_pred = P_f[t-1] + Q
        K = P_pred / (P_pred + R)
        x_f[t] = x_pred + K * (seq[:, t] - x_pred)
        P_f[t] = (1 - K) * P_pred
    # Backward (RTS)
    x_s = x_f.copy()
    P_s = P_f.copy()
    for t in range(T - 2, -1, -1):
        P_pred = P_f[t] + Q
        A = P_f[t] / P_pred
        x_s[t] = x_f[t] + A * (x_s[t+1] - x_f[t])
        P_s[t] = P_f[t] + A * A * (P_s[t+1] - P_pred)
    return x_s.transpose(1, 0, 2)  # (N, T, C)


def smooth_hmm_binary(seq_prob, p01, p10, prior_pos=0.1):
    """Binary HMM forward-backward, per (file, class). seq_prob in [0,1] = P(obs=1 | state=1)·somewhat.
    We treat seq_prob as emission likelihood directly (approx sigmoid of logit).
    Returns posterior P(s_t = 1 | z_{1..T}) shape (N, T, C).
    Transitions: A = [[1-p01, p01], [p10, 1-p10]]
    Emissions (Bernoulli-ish): b_t(1) = eps + (1-2eps)*seq_prob;  b_t(0) = 1 - b_t(1)
    """
    eps = 1e-3
    N, T, C = seq_prob.shape
    b1 = np.clip(seq_prob, eps, 1 - eps).astype(np.float32)
    b0 = (1 - b1).astype(np.float32)

    # Forward
    a0 = np.empty((T, N, C), dtype=np.float32)
    a1 = np.empty((T, N, C), dtype=np.float32)
    a0[0] = (1 - prior_pos) * b0[:, 0]
    a1[0] = prior_pos * b1[:, 0]
    s = a0[0] + a1[0] + 1e-30
    a0[0] /= s; a1[0] /= s
    for t in range(1, T):
        na0 = ((1 - p01) * a0[t-1] + p10 * a1[t-1]) * b0[:, t]
        na1 = (p01 * a0[t-1] + (1 - p10) * a1[t-1]) * b1[:, t]
        s = na0 + na1 + 1e-30
        a0[t] = na0 / s
        a1[t] = na1 / s
    # Backward
    B0 = np.ones((T, N, C), dtype=np.float32)
    B1 = np.ones((T, N, C), dtype=np.float32)
    for t in range(T - 2, -1, -1):
        nB0 = (1 - p01) * b0[:, t+1] * B0[t+1] + p01 * b1[:, t+1] * B1[t+1]
        nB1 = p10 * b0[:, t+1] * B0[t+1] + (1 - p10) * b1[:, t+1] * B1[t+1]
        s = nB0 + nB1 + 1e-30
        B0[t] = nB0 / s
        B1[t] = nB1 / s
    g0 = a0 * B0
    g1 = a1 * B1
    s = g0 + g1 + 1e-30
    post = g1 / s  # P(s_t = 1 | all obs)
    return post.transpose(1, 0, 2)  # (N, T, C)


# ─── CV splits (matching exp28) ─────────────────────────────────────────────

def val_a_splits(meta):
    """File-stratified-by-site 5-fold (matches exp27/28)."""
    from sklearn.model_selection import StratifiedKFold
    files = meta.drop_duplicates("filename").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    file2fold = {}
    for fold, (_, vi) in enumerate(skf.split(files["filename"], files["site"])):
        for f in files.iloc[vi]["filename"].values:
            file2fold[f] = fold
    meta = meta.copy()
    meta["fold"] = meta["filename"].map(file2fold)
    return meta


def val_b_splits(meta):
    """GroupKFold by site."""
    from sklearn.model_selection import GroupKFold
    meta = meta.copy()
    sites = meta["site"].values
    gkf = GroupKFold(n_splits=min(5, meta["site"].nunique()))
    meta["fold"] = -1
    for fold, (_, vi) in enumerate(gkf.split(meta, groups=sites)):
        meta.iloc[vi, meta.columns.get_loc("fold")] = fold
    return meta


# ─── Pipeline ──────────────────────────────────────────────────────────────

def group_by_file(scores_708x234, meta):
    """Reshape (708, 234) → (N_files=59, 12, 234) in row_id order within each file."""
    out = []
    order = []
    for fn, g in meta.groupby("filename", sort=False):
        idx = g.index.values
        assert len(idx) == 12, f"file {fn} has {len(idx)} windows"
        out.append(scores_708x234[idx])
        order.append(idx)
    order_flat = np.concatenate(order)  # row order in seq
    seq = np.stack(out, axis=0)  # (59, 12, 234)
    return seq, order_flat


def ungroup(seq, order_flat, N_rows=708):
    """Reverse of group_by_file."""
    flat = seq.reshape(-1, seq.shape[-1])
    out = np.empty_like(flat)
    out[order_flat] = flat
    return out


def learn_hmm_transitions(Y, meta):
    """Estimate p(0→1), p(1→0) per class from SS labels. Returns two arrays shape (234,)."""
    p01 = np.zeros(Y.shape[1], dtype=np.float32)
    p10 = np.zeros(Y.shape[1], dtype=np.float32)
    n01 = np.zeros(Y.shape[1]); n0 = np.zeros(Y.shape[1])
    n10 = np.zeros(Y.shape[1]); n1 = np.zeros(Y.shape[1])
    for fn, g in meta.groupby("filename", sort=False):
        idx = g.index.values
        y = Y[idx]  # (12, 234)
        prev, curr = y[:-1], y[1:]
        n0 += (prev == 0).sum(0)
        n1 += (prev == 1).sum(0)
        n01 += ((prev == 0) & (curr == 1)).sum(0)
        n10 += ((prev == 1) & (curr == 0)).sum(0)
    p01 = n01 / np.maximum(n0, 1)
    p10 = n10 / np.maximum(n1, 1)
    # floor + shrinkage for classes with no positive observations
    p01 = np.clip(p01, 0.02, 0.5)
    p10 = np.clip(p10, 0.05, 0.9)
    return p01, p10


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading probe outputs from exp28…")
    d = np.load(EXP28 / "best_oof.npz")
    probes = {"val_a": d["val_a_probe"], "val_b": d["val_b_probe"]}
    print("Loading truth + meta…")
    meta, Y, primary = load_truth()
    meta_a = val_a_splits(meta)
    meta_b = val_b_splits(meta)

    # Group once per regime ordering — actually per-file reshaping is regime-independent
    # since files stay intact in both Val-A and Val-B (only fold assignment changes).
    seq_a, order_a = group_by_file(probes["val_a"], meta_a)
    seq_b, order_b = group_by_file(probes["val_b"], meta_b)
    print(f"  seq_a shape: {seq_a.shape}, seq_b shape: {seq_b.shape}")

    # HMM transition priors (class-level, from all SS labels)
    p01, p10 = learn_hmm_transitions(Y, meta)
    print(f"  HMM p01 median={np.median(p01):.3f}, p10 median={np.median(p10):.3f}")

    results = []

    def evaluate(name, smoothed_a, smoothed_b, **extra):
        flat_a = ungroup(smoothed_a, order_a)
        flat_b = ungroup(smoothed_b, order_b)
        auc_a = macro_auc(Y, flat_a)
        auc_b = macro_auc(Y, flat_b)
        r = {"name": name, "val_a": auc_a, "val_b": auc_b, **extra}
        print(f"  {name:32s}  Val-A {auc_a:.4f}  Val-B {auc_b:.4f}")
        results.append(r)
        return r, flat_a, flat_b

    # Baseline: raw probe
    evaluate("raw_probe", seq_a, seq_b)

    # S1 Gaussian sweep
    print("\n[S1 Gaussian]")
    for sigma in [0.5, 0.75, 1.0, 1.25]:
        evaluate(f"gauss_s{sigma}", smooth_gauss(seq_a, sigma), smooth_gauss(seq_b, sigma), sigma=sigma)

    # S3 EMA sweep
    print("\n[S3 EMA]")
    for alpha in [0.3, 0.5, 0.7]:
        evaluate(f"ema_a{alpha}", smooth_ema(seq_a, alpha), smooth_ema(seq_b, alpha), alpha=alpha)

    # S4 Median
    print("\n[S4 Median]")
    for k in [3, 5]:
        evaluate(f"median_k{k}", smooth_median(seq_a, k), smooth_median(seq_b, k), k=k)

    # S2 Kalman RTS sweep
    print("\n[S2 Kalman RTS]")
    best_kalman = None
    for Q in [0.1, 0.5, 1.0, 2.0]:
        for R in [0.5, 1.0, 2.0, 4.0]:
            r, fa, fb = evaluate(f"kalman_Q{Q}_R{R}", smooth_kalman_rts(seq_a, Q, R),
                                 smooth_kalman_rts(seq_b, Q, R), Q=Q, R=R)
            if best_kalman is None or r["val_a"] > best_kalman[0]["val_a"]:
                best_kalman = (r, fa, fb)

    # S5 HMM binary — sigmoid to probs, apply, use logit of posterior as score
    print("\n[S5 HMM binary]")
    # Sigmoid first, then HMM on probs
    seq_a_p = sigmoid(seq_a)
    seq_b_p = sigmoid(seq_b)
    # Broadcast per-class transitions to (T, N, C) — already shape-compat after reshape
    # Evaluate with uniform transitions first (sanity), then class-specific
    for (p01_val, p10_val, tag) in [(0.15, 0.4, "unif_0.15_0.4"),
                                      (0.1, 0.3, "unif_0.1_0.3"),
                                      (0.2, 0.5, "unif_0.2_0.5")]:
        post_a = smooth_hmm_binary(seq_a_p, p01_val, p10_val)
        post_b = smooth_hmm_binary(seq_b_p, p01_val, p10_val)
        evaluate(f"hmm_{tag}", post_a, post_b, p01=p01_val, p10=p10_val)

    # Class-specific
    # Need broadcasting: expand p01/p10 to (N, T, C) — handled by broadcasting in smooth_hmm_binary
    # Our impl multiplies a1, a0 which have shape (T, N, C) by scalars. For class-specific,
    # we need p01[c] etc. Rewrite inline:
    def hmm_class_specific(seq_prob, p01_c, p10_c, prior_c=0.1):
        eps = 1e-3
        N, T, C = seq_prob.shape
        b1 = np.clip(seq_prob, eps, 1 - eps).astype(np.float32)
        b0 = 1 - b1
        a0 = np.empty((T, N, C), dtype=np.float32)
        a1 = np.empty((T, N, C), dtype=np.float32)
        a0[0] = (1 - prior_c) * b0[:, 0]
        a1[0] = prior_c * b1[:, 0]
        s = a0[0] + a1[0] + 1e-30
        a0[0] /= s; a1[0] /= s
        for t in range(1, T):
            na0 = ((1 - p01_c) * a0[t-1] + p10_c * a1[t-1]) * b0[:, t]
            na1 = (p01_c * a0[t-1] + (1 - p10_c) * a1[t-1]) * b1[:, t]
            s = na0 + na1 + 1e-30
            a0[t] = na0 / s; a1[t] = na1 / s
        B0 = np.ones_like(a0); B1 = np.ones_like(a1)
        for t in range(T - 2, -1, -1):
            nB0 = (1 - p01_c) * b0[:, t+1] * B0[t+1] + p01_c * b1[:, t+1] * B1[t+1]
            nB1 = p10_c * b0[:, t+1] * B0[t+1] + (1 - p10_c) * b1[:, t+1] * B1[t+1]
            s = nB0 + nB1 + 1e-30
            B0[t] = nB0 / s; B1[t] = nB1 / s
        g0 = a0 * B0; g1 = a1 * B1
        s = g0 + g1 + 1e-30
        return (g1 / s).transpose(1, 0, 2)

    post_a = hmm_class_specific(seq_a_p, p01, p10)
    post_b = hmm_class_specific(seq_b_p, p01, p10)
    evaluate("hmm_class_specific", post_a, post_b)

    # Save best
    best = max(results, key=lambda r: r["val_a"])
    print(f"\nBEST: {best['name']}  Val-A {best['val_a']:.4f}  Val-B {best['val_b']:.4f}")

    (OUT / "results.json").write_text(json.dumps({
        "elapsed_s": time.time() - t0,
        "reference": {"gauss_s0.75 (exp28)": {"val_a": 0.8943, "val_b": 0.8437}},
        "results": results,
        "best": best,
    }, indent=2))

    # Best smoothed outputs as score
    print(f"\nDone. {len(results)} configs tested. {(time.time()-t0):.1f}s.")


if __name__ == "__main__":
    main()
