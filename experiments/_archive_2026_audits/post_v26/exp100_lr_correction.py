#!/usr/bin/env python3
"""exp100 — LR-based per-(row, class) correction lever.

Builds on exp99's discovery: file_uniformity, perch_sed_disagree, file_mean,
exp59_on_c form a near-perfect FP detector (AUC 0.95) and FN detector
(AUC 0.94) on labeled SS. This lever applies those LR predictions as
soft corrections to v33.

Variants:
  V1 — full features (incl. site-specific class_dom_site_match)
  V2 — universal features only (drops site identity, transfer-safe)
  V3 — V2 + LOSO-CV training to verify generalization
  V4 — sweep correction strengths α (FN boost), β (FP suppress)

The whole point: we have a near-perfect signal locally. The question
is whether universal-features-only LR retains the signal AND whether the
signal transfers to LB hidden sites.

Honest tests:
  T1. Does the LR detector still work with universal-only features?
      (Strip site-specific features, check Eval AUC.)
  T2. Does cross-site CV match cross-fold CV? (LOSO over labeled sites.)
      If yes → signal is genuinely site-invariant.
      If no → LR is implicitly using site identity even in universal features.
  T3. Apply the correction to v33 with various α, β strengths.
      Measure macro_d, sp_row, per-taxon Δ.
  T4. S08 sanity: did this fix the S08 regression in exp97?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman, per_class_auc
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name): return np.load(EXP80 / name)["scores"]


def build_features_for_pairs(sc_g, perch_prob, exp50, exp59, v33, sp_taxon,
                               candidate_classes, threshold_kind, mapped_idx,
                               class_dom_site, class_gini, hour_norm, sites_arr, s2i,
                               file_mean, file_std):
    """Build (row, class) feature dict for either FP (v33>0.5) or FN (v33<0.5) pairs."""
    rows = []
    for i in range(len(sc_g)):
        for c in candidate_classes:
            v = v33[i, c]
            if threshold_kind == "FP" and v <= 0.5: continue
            if threshold_kind == "FN" and v >= 0.5: continue
            row_hour = int(sc_g.iloc[i].hour)
            hour_match = float(hour_norm[c, row_hour] if 0 <= row_hour < 24 else 0)
            same_site = 1 if class_dom_site[c] != -1 and s2i.get(sites_arr[i], -1) == class_dom_site[c] else 0
            label = (1 if (Y_global[i, c] == 0) else 0) if threshold_kind == "FP" else (1 if Y_global[i, c] == 1 else 0)
            rows.append({
                "row_idx": i, "class": c, "label": label,
                "split": sc_g.iloc[i].split, "site": sites_arr[i],
                # universal features
                "perch_on_c": perch_prob[i, c],
                "exp50_on_c": exp50[i, c],
                "exp59_on_c": exp59[i, c],
                "perch_sed_disagree": abs(perch_prob[i, c] - exp50[i, c]),
                "perch_low_sed_high": max(0, exp50[i, c] - perch_prob[i, c]),
                "file_mean_e50_c": file_mean[i, c],
                "file_std_e50_c": file_std[i, c],
                "file_uniform_e50_c": 1.0 - (file_std[i, c] / (file_mean[i, c] + 1e-6)),
                "v33_on_c": v33[i, c],
                # site-specific features (kept separately so we can drop)
                "class_dom_site_match": same_site,
                "class_site_concentration": class_gini[c],
                "hour_match_strength": hour_match,
            })
    return pd.DataFrame(rows)


def fit_eval_lr(df, train_split, eval_split, feature_cols, label_col="label"):
    """Train LR on train_split rows, eval on eval_split rows."""
    df_tr = df[df.split == train_split]
    df_ev = df[df.split == eval_split]
    if df_tr[label_col].sum() < 5: return None, None, None
    X_tr, y_tr = df_tr[feature_cols].values, df_tr[label_col].values
    X_ev, y_ev = df_ev[feature_cols].values, df_ev[label_col].values
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(sc.transform(X_tr), y_tr)
    auc_tr = roc_auc_score(y_tr, clf.predict_proba(sc.transform(X_tr))[:, 1]) if y_tr.sum() > 0 else np.nan
    auc_ev = roc_auc_score(y_ev, clf.predict_proba(sc.transform(X_ev))[:, 1]) if y_ev.sum() > 0 else np.nan
    return clf, sc, (auc_tr, auc_ev)


def loso_site_cv(df, feature_cols, label_col="label"):
    """Leave-one-site-out CV: train on all sites except one, eval on holdout."""
    sites = sorted(set(df.site.unique()))
    aucs = []; n_evals = []
    for ho_site in sites:
        df_tr = df[df.site != ho_site]
        df_ho = df[df.site == ho_site]
        if df_tr[label_col].sum() < 5 or df_ho[label_col].sum() < 1 or df_ho[label_col].sum() == len(df_ho):
            continue
        X_tr, y_tr = df_tr[feature_cols].values, df_tr[label_col].values
        X_ho, y_ho = df_ho[feature_cols].values, df_ho[label_col].values
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=42)
        clf.fit(sc.transform(X_tr), y_tr)
        try:
            auc = roc_auc_score(y_ho, clf.predict_proba(sc.transform(X_ho))[:, 1])
            aucs.append((ho_site, auc, len(df_ho), int(y_ho.sum())))
        except: pass
    return aucs


# Globals (set in main)
Y_global = None


def main():
    global Y_global
    print("=== exp100: LR-based correction lever ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    Y_global = Y
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")

    # v33 baseline
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    sites_arr = sc_g.site.values
    unique_sites = sorted(set(sites_arr))
    s2i = {s: i for i, s in enumerate(unique_sites)}

    # Compute auxiliary statistics on TRAIN
    tr_mask = sc_g.split.values == "train"
    Y_tr = Y[tr_mask]
    sites_tr = sites_arr[tr_mask]
    class_gini = np.zeros(N_CLS, dtype=np.float32)
    class_dom_site = np.full(N_CLS, -1, dtype=np.int32)
    for c in range(N_CLS):
        site_counts = np.zeros(len(unique_sites))
        for ii in np.where(Y_tr[:, c] == 1)[0]:
            si = s2i.get(sites_tr[ii], -1)
            if si >= 0: site_counts[si] += 1
        if site_counts.sum() > 0:
            share = site_counts / site_counts.sum()
            class_gini[c] = share.max()
            class_dom_site[c] = share.argmax()

    hour_counts = np.zeros((N_CLS, 24), dtype=np.float32)
    tr_idx = np.where(tr_mask)[0]
    for ii_pos, c in enumerate(range(N_CLS)):
        for ii in np.where(Y_tr[:, c] == 1)[0]:
            h = int(sc_g.iloc[tr_idx[ii]].hour)
            if 0 <= h < 24: hour_counts[c, h] += 1
    hour_norm = hour_counts / (hour_counts.sum(axis=1, keepdims=True) + 1e-6)

    # File-level stats of exp50 per class
    file_mean = np.zeros_like(exp50)
    file_std = np.zeros_like(exp50)
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = exp50[idx]
        m = sub.mean(axis=0); sd = sub.std(axis=0)
        for ii in idx:
            file_mean[ii] = m
            file_std[ii] = sd

    # Mapped Aves classes with eval pos
    col_var = perch_prob.var(axis=0)
    mapped_idx = np.where(col_var >= 1e-6)[0]
    aves_mask = sp_taxon == "Aves"
    candidate_classes = [c for c in range(N_CLS)
                          if aves_mask[c] and c in mapped_idx
                          and Y[:, c].sum() >= 5 and (Y[:, c] == 0).sum() >= 50]
    print(f"Candidate Aves classes: {len(candidate_classes)}\n")

    # Build feature dataframes
    df_fp = build_features_for_pairs(sc_g, perch_prob, exp50, exp59, v33, sp_taxon,
                                       candidate_classes, "FP", mapped_idx,
                                       class_dom_site, class_gini, hour_norm,
                                       sites_arr, s2i, file_mean, file_std)
    df_fn = build_features_for_pairs(sc_g, perch_prob, exp50, exp59, v33, sp_taxon,
                                       candidate_classes, "FN", mapped_idx,
                                       class_dom_site, class_gini, hour_norm,
                                       sites_arr, s2i, file_mean, file_std)
    print(f"FP pairs: train {(df_fp.split=='train').sum()}  eval {(df_fp.split=='eval').sum()}")
    print(f"FN pairs: train {(df_fn.split=='train').sum()}  eval {(df_fn.split=='eval').sum()}")

    UNIVERSAL = ["perch_on_c", "exp50_on_c", "exp59_on_c", "perch_sed_disagree",
                  "perch_low_sed_high", "file_mean_e50_c", "file_std_e50_c",
                  "file_uniform_e50_c", "v33_on_c"]
    SITE_SPECIFIC = ["class_dom_site_match", "class_site_concentration", "hour_match_strength"]
    FULL = UNIVERSAL + SITE_SPECIFIC

    # === T1. AUC comparison: full vs universal features ===
    print("\n=== T1. FP detector AUC: full features vs universal-only ===")
    for label, feats in [("FULL (incl. site)", FULL), ("UNIVERSAL-only", UNIVERSAL)]:
        clf, sc_, aucs = fit_eval_lr(df_fp, "train", "eval", feats)
        if aucs:
            print(f"  {label:<20}  Train AUC={aucs[0]:.3f}  Eval AUC={aucs[1]:.3f}")
    print("\n=== T1b. FN detector AUC ===")
    for label, feats in [("FULL", FULL), ("UNIVERSAL-only", UNIVERSAL)]:
        clf, sc_, aucs = fit_eval_lr(df_fn, "train", "eval", feats)
        if aucs:
            print(f"  {label:<20}  Train AUC={aucs[0]:.3f}  Eval AUC={aucs[1]:.3f}")

    # === T2. LOSO-site CV (on labeled SS sites) ===
    print("\n=== T2. LOSO-site cross-validation (UNIVERSAL features, FP detector) ===")
    fp_loso = loso_site_cv(df_fp, UNIVERSAL)
    print(f"  {'site':<6} {'AUC':>7} {'n_eval':>7} {'n_FP':>7}")
    for s, auc, ne, nf in fp_loso:
        print(f"  {s:<6} {auc:>7.3f} {ne:>7} {nf:>7}")
    if fp_loso:
        mean_auc = np.mean([a for _, a, _, _ in fp_loso])
        print(f"  Mean LOSO-site AUC: {mean_auc:.3f}")
    print("\n=== T2b. LOSO FN detector ===")
    fn_loso = loso_site_cv(df_fn, UNIVERSAL)
    print(f"  {'site':<6} {'AUC':>7} {'n_eval':>7} {'n_FN':>7}")
    for s, auc, ne, nf in fn_loso:
        print(f"  {s:<6} {auc:>7.3f} {ne:>7} {nf:>7}")
    if fn_loso:
        mean_auc = np.mean([a for _, a, _, _ in fn_loso])
        print(f"  Mean LOSO-site AUC: {mean_auc:.3f}")

    # === T3. Apply correction to v33, sweep strengths ===
    print("\n=== T3. Apply LR correction (universal-only features), sweep α (FN boost) × β (FP suppress) ===")

    # Train final LRs on ALL train pairs (v9 universal feats)
    feats = UNIVERSAL
    clf_fp, sc_fp, _ = fit_eval_lr(df_fp, "train", "eval", feats)
    clf_fn, sc_fn, _ = fit_eval_lr(df_fn, "train", "eval", feats)

    # Predict P_FP on FP pairs (only v33>0.5 pairs); P_FN on FN pairs (v33<0.5)
    # We need to map back to (row, class) → P
    P_FP_grid = np.zeros((len(sc_g), N_CLS), dtype=np.float32)   # 0 if not in FP-pair domain
    P_FN_grid = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    if clf_fp is not None:
        df_fp_all = df_fp.copy()
        X = sc_fp.transform(df_fp_all[feats].values)
        df_fp_all["P_FP"] = clf_fp.predict_proba(X)[:, 1]
        for _, r in df_fp_all.iterrows():
            P_FP_grid[r["row_idx"], r["class"]] = r["P_FP"]
    if clf_fn is not None:
        df_fn_all = df_fn.copy()
        X = sc_fn.transform(df_fn_all[feats].values)
        df_fn_all["P_FN"] = clf_fn.predict_proba(X)[:, 1]
        for _, r in df_fn_all.iterrows():
            P_FN_grid[r["row_idx"], r["class"]] = r["P_FN"]

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    print(f"\n  α × β grid (FN boost × FP suppress):")
    for alpha_fn in [0.0, 0.10, 0.20, 0.30, 0.50]:
        for beta_fp in [0.0, 0.10, 0.20, 0.30, 0.50]:
            P = v33.copy()
            # FN boost on low-v33 pairs
            high_fn = (v33 < 0.5) & (P_FN_grid > 0)
            P[high_fn] = v33[high_fn] + alpha_fn * P_FN_grid[high_fn] * (1 - v33[high_fn])
            # FP suppress on high-v33 pairs
            high_fp = (v33 > 0.5) & (P_FP_grid > 0)
            P[high_fp] = v33[high_fp] * (1 - beta_fp * P_FP_grid[high_fp])
            P = np.clip(P, 0, 1).astype(np.float32)
            rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                                  f"LR α_fn={alpha_fn} β_fp={beta_fp}"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== Top 15 by macro_d ===")
    print(res.sort_values("macro_d", ascending=False)[cols].head(15).to_string(index=False))

    print("\n=== Top class-A (sp_row≥0.99 + Aves≥0) sorted by Aves Δ ===")
    safe = res[res.predicted.str.startswith("A") & (res.label != "v33 ref")]
    print(safe.sort_values("Aves", ascending=False).head(8)[cols].to_string(index=False))

    res.to_csv(EXP80 / "exp100_lr_correction.csv", index=False)

    # === T4. Per-site analysis of best variant ===
    print("\n=== T4. Per-site Δ for best class-A candidate ===")
    best = safe.sort_values("macro_d", ascending=False).iloc[0]
    print(f"  Best: {best['label']}, macro_d={best['macro_d']:.4f}, Aves={best['Aves']:.4f}")

    # Re-build the best variant
    parts = best['label']
    af = float(parts.split('α_fn=')[1].split(' ')[0])
    bf = float(parts.split('β_fp=')[1])
    P_best = v33.copy()
    high_fn = (v33 < 0.5) & (P_FN_grid > 0)
    P_best[high_fn] = v33[high_fn] + af * P_FN_grid[high_fn] * (1 - v33[high_fn])
    high_fp = (v33 > 0.5) & (P_FP_grid > 0)
    P_best[high_fp] = v33[high_fp] * (1 - bf * P_FP_grid[high_fp])
    P_best = np.clip(P_best, 0, 1)

    print(f"\n  {'site':<6} {'n_rows':>7} {'v33':>8} {'lever':>8} {'Δ':>8}")
    for s in sorted(set(sites_arr[ev_mask])):
        ms = ev_mask & (sites_arr == s)
        if ms.sum() < 5: continue
        v_aucs = per_class_auc(Y[ms], v33[ms])
        l_aucs = per_class_auc(Y[ms], P_best[ms])
        common = set(v_aucs) & set(l_aucs)
        if not common: continue
        v_macro = np.mean([v_aucs[c] for c in common])
        l_macro = np.mean([l_aucs[c] for c in common])
        print(f"  {s:<6} {ms.sum():>7} {v_macro:>8.4f} {l_macro:>8.4f} {l_macro-v_macro:>+8.4f}")


if __name__ == "__main__":
    main()
