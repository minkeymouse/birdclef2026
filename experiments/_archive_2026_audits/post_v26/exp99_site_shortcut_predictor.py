#!/usr/bin/env python3
"""exp99 — Can we identify site-shortcut FP/TN cases pre-prediction?

Hypothesis: among (row, class) pairs where v33 fires high, the FPs are
disproportionately driven by site-shortcut SED firings. We can build a
pre-prediction classifier of FP-likelihood using:

  1. perch_sed_disagree |perch[i,c] - exp50[i,c]| (site-invariant teacher
     vs site-conflated teacher gap)
  2. Within-file uniformity of exp50 on class c (high uniformity → site
     fingerprint pulling all windows up; low uniformity → real call)
  3. Class-site Gini in training labels (how concentrated is class c
     at one site historically?). If high Gini AND row's site matches
     historical site → site shortcut likely.
  4. Perch-on-c (low Perch, high exp50 → SED firing alone, suspect)
  5. Hour-of-day mismatch with class historical hour distribution

Train this LR on TRAIN rows' (i, c) pairs, eval on EVAL rows. AUC of
predicting FP.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend


def get_cached(name): return np.load(EXP80 / name)["scores"]


def class_site_gini(Y, sites, n_sites):
    """Per-class Gini coefficient of site distribution in training positives.
    1.0 = single-site, 0.0 = uniform across sites."""
    out = np.zeros(N_CLS, dtype=np.float32)
    for c in range(N_CLS):
        site_counts = np.zeros(n_sites)
        for i in np.where(Y[:, c] == 1)[0]:
            s_idx = np.where(sites == np.unique(sites))[0]
            # simpler:
            pass
        # Recompute via groupby logic
    return out


def main():
    print("=== exp99: Pre-prediction site-shortcut FP detector ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")

    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    sites_arr = sc_g.site.values
    unique_sites = sorted(set(sites_arr))
    n_sites = len(unique_sites)
    s2i = {s: i for i, s in enumerate(unique_sites)}

    # === Compute per-class historical Gini (site concentration) ===
    print("Computing per-class site Gini (TRAIN rows only)...", flush=True)
    tr_mask = sc_g.split.values == "train"
    Y_tr = Y[tr_mask]
    sites_tr = sites_arr[tr_mask]
    class_gini = np.zeros(N_CLS, dtype=np.float32)
    class_dom_site = np.zeros(N_CLS, dtype=np.int32)
    for c in range(N_CLS):
        site_counts = np.zeros(n_sites, dtype=np.float64)
        for i in np.where(Y_tr[:, c] == 1)[0]:
            si = s2i.get(sites_tr[i], -1)
            if si >= 0: site_counts[si] += 1
        if site_counts.sum() == 0:
            class_gini[c] = 0
            class_dom_site[c] = -1
            continue
        # Gini-like concentration: max share - 1/n_sites, normalized
        share = site_counts / site_counts.sum()
        class_dom_site[c] = share.argmax()
        # Use a simple "concentration" metric: max share
        class_gini[c] = share.max()

    # Per-class historical hour distribution
    print("Computing per-class historical hour distribution (TRAIN)...", flush=True)
    hour_counts = np.zeros((N_CLS, 24), dtype=np.float32)
    for c in range(N_CLS):
        for i in np.where(Y_tr[:, c] == 1)[0]:
            h = int(sc_g.iloc[np.where(tr_mask)[0][i]].hour)
            if 0 <= h < 24:
                hour_counts[c, h] += 1
    hour_norm = hour_counts / (hour_counts.sum(axis=1, keepdims=True) + 1e-6)

    # === Within-file uniformity (per row, class) ===
    # For each row i, class c: compute std of exp50 on c across windows of same file
    print("Computing within-file uniformity of exp50 per (row, class)...", flush=True)
    file_std_exp50 = np.zeros_like(exp50)
    file_mean_exp50 = np.zeros_like(exp50)
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = exp50[idx]
        m = sub.mean(axis=0)
        sd = sub.std(axis=0)
        for i in idx:
            file_std_exp50[i] = sd
            file_mean_exp50[i] = m

    # === Build (row, class) features for FP prediction ===
    # Restrict to (i, c) pairs where v33[i,c] > 0.5 (model says YES)
    # Label: TP=0 if Y[i,c]=1, FP=1 if Y[i,c]=0
    print("\nBuilding (row, class) feature matrix...", flush=True)

    # Class filter: mapped Aves with both pos/neg in 739
    col_var = perch_prob.var(axis=0)
    mapped_idx = np.where(col_var >= 1e-6)[0]
    aves_mask = sp_taxon == "Aves"
    candidate_classes = [c for c in range(N_CLS)
                          if aves_mask[c] and c in mapped_idx
                          and Y[:, c].sum() >= 5 and (Y[:, c] == 0).sum() >= 50]
    print(f"  candidate Aves classes: {len(candidate_classes)}")

    feat_rows = []
    for i in range(len(sc_g)):
        for c in candidate_classes:
            if v33[i, c] <= 0.5: continue   # only "model says YES" pairs
            label = 0 if Y[i, c] == 1 else 1   # 1 = FP
            same_site_as_dom = 1 if class_dom_site[c] != -1 and s2i.get(sites_arr[i], -1) == class_dom_site[c] else 0
            row_hour = int(sc_g.iloc[i].hour)
            hour_match = float(hour_norm[c, row_hour] if 0 <= row_hour < 24 else 0)
            feat_rows.append({
                "row_idx": i, "class": c, "label_FP": label,
                "split": sc_g.iloc[i].split,
                # ---- features ----
                "perch_on_c": perch_prob[i, c],
                "exp50_on_c": exp50[i, c],
                "exp59_on_c": exp59[i, c],
                "perch_sed_disagree": abs(perch_prob[i, c] - exp50[i, c]),
                "perch_low_sed_high": max(0, exp50[i, c] - perch_prob[i, c]),  # SED fires while Perch doesn't
                "file_mean_e50_c": file_mean_exp50[i, c],
                "file_std_e50_c": file_std_exp50[i, c],
                "file_uniform_e50_c": 1.0 - (file_std_exp50[i, c] / (file_mean_exp50[i, c] + 1e-6)),
                "class_dom_site_match": same_site_as_dom,
                "class_site_concentration": class_gini[c],
                "hour_match_strength": hour_match,
                "v33_on_c": v33[i, c],
            })

    df = pd.DataFrame(feat_rows)
    print(f"  total (row, class) pairs (v33 > 0.5): {len(df)}")
    print(f"  TPs: {(df.label_FP == 0).sum()}, FPs: {(df.label_FP == 1).sum()}")

    # === Univariate AUC of each feature for FP prediction ===
    print("\n=== Univariate AUC for FP detection (eval rows) ===")
    df_ev = df[df.split == "eval"]
    print(f"  eval (row, class) pairs: {len(df_ev)} (TP={int((df_ev.label_FP==0).sum())}, FP={int((df_ev.label_FP==1).sum())})")
    feature_names = ["perch_on_c", "exp50_on_c", "exp59_on_c", "perch_sed_disagree",
                      "perch_low_sed_high", "file_mean_e50_c", "file_std_e50_c",
                      "file_uniform_e50_c", "class_dom_site_match",
                      "class_site_concentration", "hour_match_strength", "v33_on_c"]
    print(f"  {'feature':<28} {'AUC':>7}")
    for f in feature_names:
        try:
            auc = roc_auc_score(df_ev.label_FP, df_ev[f])
            print(f"  {f:<28} {auc:>7.3f}")
        except: pass

    # === Multivariate LR ===
    print("\n=== Multivariate LR (train on TRAIN pairs, eval on EVAL pairs) ===")
    df_tr = df[df.split == "train"]
    print(f"  train (row, class) pairs: {len(df_tr)} (TP={int((df_tr.label_FP==0).sum())}, FP={int((df_tr.label_FP==1).sum())})")

    from sklearn.preprocessing import StandardScaler
    X_tr = df_tr[feature_names].values
    y_tr = df_tr.label_FP.values
    X_ev = df_ev[feature_names].values
    y_ev = df_ev.label_FP.values

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_ev_s = scaler.transform(X_ev)

    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(X_tr_s, y_tr)
    p_tr = clf.predict_proba(X_tr_s)[:, 1]
    p_ev = clf.predict_proba(X_ev_s)[:, 1]
    auc_tr = roc_auc_score(y_tr, p_tr) if y_tr.sum() > 0 else np.nan
    auc_ev = roc_auc_score(y_ev, p_ev) if y_ev.sum() > 0 else np.nan
    print(f"  Train AUC = {auc_tr:.3f}")
    print(f"  Eval AUC = {auc_ev:.3f}")

    # Coefficient inspection
    print("\n  Top features by |coefficient| (LR with standardized features):")
    coefs = pd.DataFrame({"feature": feature_names, "coef": clf.coef_[0]})
    coefs["abs_coef"] = coefs["coef"].abs()
    coefs = coefs.sort_values("abs_coef", ascending=False)
    for _, r in coefs.iterrows():
        sign = "↑FP" if r["coef"] > 0 else "↓FP"
        print(f"    {r['feature']:<28} {r['coef']:+.3f} ({sign})")

    # === Same for TN/FN at low predictions ===
    print("\n=== Same analysis for FN detection (v33 < 0.5 pairs) ===")
    fn_rows = []
    for i in range(len(sc_g)):
        for c in candidate_classes:
            if v33[i, c] >= 0.5: continue
            label = 1 if Y[i, c] == 1 else 0   # 1 = FN
            same_site = 1 if class_dom_site[c] != -1 and s2i.get(sites_arr[i], -1) == class_dom_site[c] else 0
            row_hour = int(sc_g.iloc[i].hour)
            hour_match = float(hour_norm[c, row_hour] if 0 <= row_hour < 24 else 0)
            fn_rows.append({
                "row_idx": i, "class": c, "label_FN": label, "split": sc_g.iloc[i].split,
                "perch_on_c": perch_prob[i, c],
                "exp50_on_c": exp50[i, c],
                "exp59_on_c": exp59[i, c],
                "perch_sed_disagree": abs(perch_prob[i, c] - exp50[i, c]),
                "perch_low_sed_high": max(0, exp50[i, c] - perch_prob[i, c]),
                "file_mean_e50_c": file_mean_exp50[i, c],
                "file_std_e50_c": file_std_exp50[i, c],
                "class_dom_site_match": same_site,
                "class_site_concentration": class_gini[c],
                "hour_match_strength": hour_match,
                "v33_on_c": v33[i, c],
            })
    df_fn = pd.DataFrame(fn_rows)
    df_fn_tr = df_fn[df_fn.split == "train"]
    df_fn_ev = df_fn[df_fn.split == "eval"]
    print(f"  train (i,c) pairs (v33<0.5): {len(df_fn_tr)} (FN={int(df_fn_tr.label_FN.sum())}, TN={int((df_fn_tr.label_FN==0).sum())})")
    print(f"  eval pairs: {len(df_fn_ev)} (FN={int(df_fn_ev.label_FN.sum())}, TN={int((df_fn_ev.label_FN==0).sum())})")

    fn_feats = ["perch_on_c", "exp50_on_c", "exp59_on_c", "perch_sed_disagree",
                 "perch_low_sed_high", "file_mean_e50_c", "file_std_e50_c",
                 "class_dom_site_match", "class_site_concentration", "hour_match_strength",
                 "v33_on_c"]
    print(f"\n  {'feature':<28} {'AUC':>7}")
    for f in fn_feats:
        try:
            auc = roc_auc_score(df_fn_ev.label_FN, df_fn_ev[f])
            print(f"  {f:<28} {auc:>7.3f}")
        except: pass

    print("\n  Multivariate LR (FN detection):")
    X_tr_fn = df_fn_tr[fn_feats].values
    y_tr_fn = df_fn_tr.label_FN.values
    X_ev_fn = df_fn_ev[fn_feats].values
    y_ev_fn = df_fn_ev.label_FN.values
    scaler_fn = StandardScaler().fit(X_tr_fn)
    clf_fn = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42)
    clf_fn.fit(scaler_fn.transform(X_tr_fn), y_tr_fn)
    auc_fn_tr = roc_auc_score(y_tr_fn, clf_fn.predict_proba(scaler_fn.transform(X_tr_fn))[:, 1])
    auc_fn_ev = roc_auc_score(y_ev_fn, clf_fn.predict_proba(scaler_fn.transform(X_ev_fn))[:, 1])
    print(f"    Train AUC = {auc_fn_tr:.3f}")
    print(f"    Eval AUC = {auc_fn_ev:.3f}")
    coefs_fn = pd.DataFrame({"feature": fn_feats, "coef": clf_fn.coef_[0]})
    coefs_fn["abs_coef"] = coefs_fn["coef"].abs()
    print("\n  Top features (FN model):")
    for _, r in coefs_fn.sort_values("abs_coef", ascending=False).iterrows():
        sign = "↑FN" if r["coef"] > 0 else "↓FN"
        print(f"    {r['feature']:<28} {r['coef']:+.3f} ({sign})")


if __name__ == "__main__":
    main()
