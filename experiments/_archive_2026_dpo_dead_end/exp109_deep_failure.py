#!/usr/bin/env python3
"""exp109 — Deep dive on each failure mode identified in exp108.

Sections:
  A. Default-Aves saturation: which exact species fire on every row?
  B. Confusion cluster mapping: species that always co-fire when wrong
  C. Disagreement: productive vs noise. When (model_i, model_j) disagree,
     who's right? And does that correlate with row/class properties?
  D. Temperature calibration test on exp50 / P_NEW3 (47% confident-miss)
  E. S19 deep-dive: what about S19 makes all models collapse?
  F. Universal-miss row analysis: are these rows audio outliers?
  G. Disagreement-aware blending experiments
  H. Per-row top-K cap experiments
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import macro_auc
from sklearn.metrics import roc_auc_score


def get_cached(name):
    d = np.load(EXP80 / name)
    if "scores" in d.files: return d["scores"]
    if "predictions" in d.files: return d["predictions"]
    return d[d.files[0]]


def build_models():
    sc_g, Y, primary, l2i = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()
    perch_prob_ss = load_perch_scores_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")

    from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend
    base = 0.7 * perch_prob_ss + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb_ss, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # P_NEW3 LOSO predictions on eval
    from exp106_pnew_hybrid import build_perch_init, train_hybrid
    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0

    tr_mask = sc_g.split.values == "train"
    ev_mask = sc_g.split.values == "eval"
    X_train = np.concatenate([ta_emb[valid], perch_emb_ss[tr_mask]], axis=0)
    Y_train = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])
    W_init, b_init, _ = build_perch_init()
    _, p_new3_ev, _, _, _ = train_hybrid(
        X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y[ev_mask].astype(np.float32),
        W_init, b_init, n_epochs=12, verbose=False
    )

    return {
        "sc_g": sc_g, "Y": Y, "primary": primary, "l2i": l2i, "sp_taxon": sp_taxon,
        "perch_emb": perch_emb_ss, "perch_prob": perch_prob_ss, "exp50": exp50, "v33": v33,
        "p_new3_ev": p_new3_ev, "ev_mask": ev_mask, "tr_mask": tr_mask,
    }


def section_a_aves_saturation(D):
    """Identify species that get >0.7 on essentially every row in Perch."""
    print("\n" + "="*70)
    print("A. DEFAULT-AVES SATURATION")
    print("="*70)
    perch = D["perch_prob"][D["ev_mask"]]
    Y_ev = D["Y"][D["ev_mask"]]
    primary = D["primary"]; sp_taxon = D["sp_taxon"]

    # For each class, compute % of rows with score > 0.7
    high_rate = (perch > 0.7).mean(axis=0)  # (234,)
    saturating = []
    for c in range(N_CLS):
        if high_rate[c] > 0.5:  # >50% of rows score this class >0.7
            n_pos = int(Y_ev[:, c].sum())
            saturating.append((primary[c], sp_taxon[c], high_rate[c], n_pos))

    print(f"\n  {len(saturating)} species score >0.7 on >50% of 122 eval rows in Perch:")
    print(f"  {'label':<14} {'taxon':<10} {'high_rate':>10} {'n_pos':>6}")
    for lbl, t, r, np_ in sorted(saturating, key=lambda x: -x[2]):
        print(f"  {lbl:<14} {t:<10} {r:>10.3f} {np_:>6}")

    # Suppression test: per-row top-K cap
    print(f"\n  TOP-K CAP TEST: cap Perch's top-K species per row to suppress saturation.\n")
    print(f"  {'top_K':>6} {'macro_v33':>10} {'macro_perch_capped+exp50':>26}")
    base_v33_macro, _ = macro_auc(Y_ev.astype(np.float32), D["v33"][D["ev_mask"]])
    print(f"  v33 base: {base_v33_macro:.4f}")
    for K in [3, 5, 10, 15, 20, 30]:
        # For each row, keep top-K Perch scores, zero rest
        perch_capped = perch.copy()
        for r in range(len(perch_capped)):
            top_idx = np.argsort(perch_capped[r])[::-1][:K]
            mask = np.ones(N_CLS, dtype=bool); mask[top_idx] = False
            perch_capped[r, mask] = 0.0
        # Re-blend with exp50
        blended = 0.7 * perch_capped + 0.3 * D["exp50"][D["ev_mask"]]
        try:
            m, _ = macro_auc(Y_ev.astype(np.float32), blended)
            print(f"  K={K:>4}  base_capped+exp50: {m:.4f}")
        except Exception as e:
            print(f"  K={K:>4}  failed: {e}")


def section_b_confusion_cluster(D):
    """Find species pairs that frequently co-fire (Pearson on Perch outputs)."""
    print("\n" + "="*70)
    print("B. CONFUSION CLUSTER MAPPING")
    print("="*70)

    perch = D["perch_prob"]  # full 739 rows for stable correlations
    primary = D["primary"]; sp_taxon = D["sp_taxon"]

    # Per-class correlation across rows
    P_centered = perch - perch.mean(axis=0)
    P_norm = np.linalg.norm(P_centered, axis=0) + 1e-9
    corr = (P_centered.T @ P_centered) / np.outer(P_norm, P_norm)
    np.fill_diagonal(corr, 0)

    # For each Aves species that's saturating, find its top-3 correlation partners
    high_rate = (perch[D["ev_mask"]] > 0.7).mean(axis=0)
    saturating_idx = np.where((high_rate > 0.5) & (sp_taxon == "Aves"))[0]

    print(f"\n  {len(saturating_idx)} saturating Aves species. Their nearest neighbors in Perch output space:")
    for idx in saturating_idx[:15]:
        partners = np.argsort(corr[idx])[::-1][:5]
        partner_str = ", ".join(f"{primary[p]}({corr[idx, p]:.2f})" for p in partners)
        print(f"  {primary[idx]:<14} → {partner_str}")

    # Universal-miss species (from exp108): see who they confuse with
    universal_miss_classes = ["litnig1", "bafcur1", "326272", "74113", "1491113", "67107"]
    print(f"\n  Universal-miss species — Perch confusion partners:")
    for lbl in universal_miss_classes:
        if lbl in primary:
            idx = primary.index(lbl)
            partners = np.argsort(corr[idx])[::-1][:5]
            partner_str = ", ".join(f"{primary[p]}({corr[idx, p]:.2f},{sp_taxon[p][:3]})" for p in partners)
            print(f"  {lbl:<10} ({sp_taxon[idx]}) → {partner_str}")


def section_c_disagreement(D):
    """When models disagree, who's right? Productive vs noise."""
    print("\n" + "="*70)
    print("C. DISAGREEMENT ANALYSIS")
    print("="*70)

    Y_ev = D["Y"][D["ev_mask"]]
    perch = D["perch_prob"][D["ev_mask"]]
    exp50 = D["exp50"][D["ev_mask"]]
    v33 = D["v33"][D["ev_mask"]]
    pnew = D["p_new3_ev"]

    # Disagreement = max - min across models
    stack = np.stack([perch, exp50, v33, pnew])  # (4, n, c)
    disagreement = stack.max(axis=0) - stack.min(axis=0)

    # When disagreement is large, who's most often correct?
    pos_mask = Y_ev > 0
    neg_mask = Y_ev == 0

    print("\n  Mean prediction per model on POSITIVE entries by disagreement bucket:")
    for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]:
        sel = pos_mask & (disagreement >= lo) & (disagreement < hi)
        n = sel.sum()
        if n == 0: continue
        print(f"  Disagreement [{lo:.1f}, {hi:.1f}) n={n:>4}: "
              f"Perch {perch[sel].mean():.3f}, exp50 {exp50[sel].mean():.3f}, "
              f"v33 {v33[sel].mean():.3f}, P_NEW3 {pnew[sel].mean():.3f}, "
              f"max {stack[:,sel.nonzero()[0],sel.nonzero()[1]].max(axis=0).mean():.3f}")

    print("\n  Mean prediction per model on NEGATIVE entries by disagreement bucket:")
    for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]:
        sel = neg_mask & (disagreement >= lo) & (disagreement < hi)
        n = sel.sum()
        if n == 0: continue
        print(f"  Disagreement [{lo:.1f}, {hi:.1f}) n={n:>4}: "
              f"Perch {perch[sel].mean():.3f}, exp50 {exp50[sel].mean():.3f}, "
              f"v33 {v33[sel].mean():.3f}, P_NEW3 {pnew[sel].mean():.3f}, "
              f"max {stack[:,sel.nonzero()[0],sel.nonzero()[1]].max(axis=0).mean():.3f}")

    # Test "use max" and "use most-confident" blending
    print("\n  ALTERNATIVE BLENDING SCHEMES:")
    schemes = {
        "v33 baseline": v33,
        "max(v33, exp50, P_NEW3)": np.maximum(np.maximum(v33, exp50), pnew),
        "max(v33, P_NEW3)": np.maximum(v33, pnew),
        "0.5 v33 + 0.5 max(exp50, P_NEW3)": 0.5*v33 + 0.5*np.maximum(exp50, pnew),
        "v33 + 0.1 max(exp50, P_NEW3) - v33)": v33 + 0.1*(np.maximum(exp50, pnew) - v33),
        "v33 where conf, max otherwise":
            np.where(np.maximum(v33, np.maximum(exp50, pnew)) > 0.4,
                      np.maximum(v33, np.maximum(exp50, pnew)),
                      v33),
    }
    print(f"  {'scheme':<45} {'macro':>8}")
    for name, pred in schemes.items():
        try:
            m, _ = macro_auc(Y_ev.astype(np.float32), np.clip(pred, 0, 1).astype(np.float32))
            print(f"  {name:<45} {m:>8.4f}")
        except Exception:
            pass


def section_d_calibration(D):
    """Per-class temperature calibration on exp50 to lift confident-miss positives."""
    print("\n" + "="*70)
    print("D. TEMPERATURE CALIBRATION (exp50)")
    print("="*70)

    Y_ev = D["Y"][D["ev_mask"]]
    exp50_ev = D["exp50"][D["ev_mask"]]

    # Find global temperature that maximizes AUC AND lifts low predictions
    print("\n  exp50 temperature scaling on logits:")
    print(f"  {'temp':>6} {'macro':>8} {'mean_pos_pred':>14} {'fn_<0.1':>10}")
    eps = 1e-4
    logit = np.log(np.clip(exp50_ev, eps, 1-eps) / (1 - np.clip(exp50_ev, eps, 1-eps)))
    base_macro, _ = macro_auc(Y_ev.astype(np.float32), exp50_ev)
    print(f"  base   {base_macro:.4f}    pos_mean={exp50_ev[Y_ev>0].mean():.4f}, fn<0.1={(exp50_ev[Y_ev>0]<0.1).mean()*100:.1f}%")
    for T in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        prob = 1/(1 + np.exp(-logit/T))
        try:
            m, _ = macro_auc(Y_ev.astype(np.float32), prob.astype(np.float32))
            pos_mean = prob[Y_ev>0].mean()
            fn_low = (prob[Y_ev>0]<0.1).mean()*100
            print(f"  T={T:>4.1f}  {m:.4f}  pos_mean={pos_mean:.4f}, fn<0.1={fn_low:.1f}%")
        except Exception: pass

    # Per-class temperature: class with confident-miss gets aggressive temperature
    print("\n  Per-class temperature (computed from train-split, applied on eval):")
    Y_train = D["Y"][D["tr_mask"]].astype(np.float32)
    exp50_train = D["exp50"][D["tr_mask"]]
    # For each class, find T* that maximizes per-class AUC on train
    best_T = np.ones(N_CLS, dtype=np.float32)
    for c in range(N_CLS):
        if Y_train[:, c].sum() == 0: continue
        eps_l = np.log(np.clip(exp50_train[:, c], eps, 1-eps) / (1 - np.clip(exp50_train[:, c], eps, 1-eps)))
        # Skip classes where AUC is invariant (all same)
        if exp50_train[:, c].std() < 1e-6: continue
        # AUC is invariant to temperature, but for SED with confident-miss the issue is
        # post-blend ranking. We instead optimize: shift to maximize TP rate at fixed FPR.
        # Simplest: use sqrt(median pos / median neg) as scaling
        pos = exp50_train[Y_train[:, c]>0, c]
        neg = exp50_train[Y_train[:, c]==0, c]
        if len(pos) == 0 or pos.mean() < 1e-3: continue
        best_T[c] = max(0.3, min(3.0, np.sqrt(pos.mean() / max(neg.mean(), 1e-3))))

    # Apply on eval
    logit_ev = np.log(np.clip(exp50_ev, eps, 1-eps) / (1 - np.clip(exp50_ev, eps, 1-eps)))
    prob_corr = 1 / (1 + np.exp(-logit_ev / best_T[None, :]))
    m, _ = macro_auc(Y_ev.astype(np.float32), prob_corr.astype(np.float32))
    print(f"  Per-class temperature exp50 macro: {m:.4f}")
    print(f"  Best T range: [{best_T.min():.2f}, {best_T.max():.2f}], mean {best_T.mean():.2f}")

    # Same for P_NEW3 (using LOSO-clean train predictions: re-train on TR, eval on EV)
    pnew = D["p_new3_ev"]
    logit_p = np.log(np.clip(pnew, eps, 1-eps) / (1 - np.clip(pnew, eps, 1-eps)))
    print("\n  P_NEW3 global temperature scaling on logits:")
    base_pnew_macro, _ = macro_auc(Y_ev.astype(np.float32), pnew)
    print(f"  base T=1.0: {base_pnew_macro:.4f}")
    for T in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        prob = 1/(1 + np.exp(-logit_p/T))
        try:
            m, _ = macro_auc(Y_ev.astype(np.float32), prob.astype(np.float32))
            print(f"  T={T:>4.1f}  {m:.4f}  pos_mean={prob[Y_ev>0].mean():.4f}")
        except Exception: pass


def section_e_s19(D):
    """Why does S19 collapse?"""
    print("\n" + "="*70)
    print("E. S19 DEEP DIVE")
    print("="*70)

    sc_g = D["sc_g"]
    Y = D["Y"]
    primary = D["primary"]; sp_taxon = D["sp_taxon"]
    sites_arr = sc_g.site.values

    s19_mask = sites_arr == "S19"
    print(f"\n  S19 has {s19_mask.sum()} rows total in labeled SS")
    print(f"    of which train: {((s19_mask) & (sc_g.split == 'train')).sum()}, eval: {((s19_mask) & (sc_g.split == 'eval')).sum()}")
    other_sites = sorted(set(sites_arr[sites_arr != "S19"]))

    # Species in S19 but not other sites
    s19_species = set()
    other_species = set()
    for i in range(len(Y)):
        for c in np.where(Y[i])[0]:
            if sites_arr[i] == "S19":
                s19_species.add(primary[c])
            else:
                other_species.add(primary[c])
    s19_only = s19_species - other_species
    other_only = other_species - s19_species
    common = s19_species & other_species
    print(f"\n  Species occurring in S19: {len(s19_species)}")
    print(f"    S19-only: {len(s19_only)}, common with other sites: {len(common)}")
    s19_only_with_taxon = sorted([(s, sp_taxon[primary.index(s)]) for s in s19_only], key=lambda x: x[1])
    print(f"    S19-only species: {s19_only_with_taxon}")

    # On eval rows from S19, compare to S22 (similar sized eval site)
    s19_ev = s19_mask & (sc_g.split.values == "eval")
    s22_ev = (sites_arr == "S22") & (sc_g.split.values == "eval")
    print(f"\n  S19 eval Insecta proportion vs S22 eval:")
    for site_name, sm in [("S19", s19_ev), ("S22", s22_ev)]:
        Y_site = Y[sm]
        n = len(Y_site)
        per_taxon_pos = {t: 0 for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
        for r in range(n):
            for c in np.where(Y_site[r])[0]:
                per_taxon_pos[sp_taxon[c]] += 1
        print(f"    {site_name} (n={n}): " + ", ".join(f"{t}:{cnt}" for t, cnt in per_taxon_pos.items()))


def section_f_universal_misses(D):
    """Look at the 92 universal-miss (row, class) pairs in detail."""
    print("\n" + "="*70)
    print("F. UNIVERSAL-MISS ROW ANALYSIS")
    print("="*70)

    Y_ev = D["Y"][D["ev_mask"]]
    pos_idx = np.where(Y_ev > 0)
    perch = D["perch_prob"][D["ev_mask"]]
    exp50 = D["exp50"][D["ev_mask"]]
    v33 = D["v33"][D["ev_mask"]]
    pnew = D["p_new3_ev"]

    miss_arr = {}
    for name, P in [("Perch", perch), ("exp50", exp50), ("v33", v33), ("P_NEW3", pnew)]:
        miss_arr[name] = (P[pos_idx] < 0.5)
    all_miss = miss_arr["Perch"] & miss_arr["exp50"] & miss_arr["v33"] & miss_arr["P_NEW3"]

    primary = D["primary"]
    sp_taxon = D["sp_taxon"]
    sc_ev = D["sc_g"][D["ev_mask"]].reset_index(drop=True)

    # Universal misses with metadata
    print(f"\n  92 universal-miss instances by site:")
    site_counts = {}
    for i in range(len(pos_idx[0])):
        if all_miss[i]:
            r = pos_idx[0][i]
            site = sc_ev.site.values[r]
            site_counts[site] = site_counts.get(site, 0) + 1
    for s, n in sorted(site_counts.items(), key=lambda x: -x[1]):
        print(f"    {s}: {n}")

    print(f"\n  Universal-miss instances by hour bucket:")
    hour_counts = {}
    for i in range(len(pos_idx[0])):
        if all_miss[i]:
            r = pos_idx[0][i]
            hr = int(sc_ev.hour.values[r])
            bucket = "0-5" if hr < 6 else ("6-11" if hr < 12 else ("12-17" if hr < 18 else "18-23"))
            hour_counts[bucket] = hour_counts.get(bucket, 0) + 1
    for b, n in sorted(hour_counts.items()):
        print(f"    {b}: {n}")

    # Per-class universal-miss with cluster-level confusion
    print(f"\n  Universal-miss species: are confusion targets within Aves cluster?")
    cluster_stats = {}
    for i in range(len(pos_idx[0])):
        if all_miss[i]:
            r, c = pos_idx[0][i], pos_idx[1][i]
            top3 = np.argsort(v33[r])[::-1][:3]
            taxon_targets = [sp_taxon[t] for t in top3]
            true_taxon = sp_taxon[c]
            tax_str = ",".join(taxon_targets)
            key = f"true={true_taxon} → {tax_str}"
            cluster_stats[key] = cluster_stats.get(key, 0) + 1
    for k, v in sorted(cluster_stats.items(), key=lambda x: -x[1])[:10]:
        print(f"    {k}: {v}")


def section_g_blending_schemes(D):
    """Test more sophisticated blending schemes."""
    print("\n" + "="*70)
    print("G. ADVANCED BLENDING SCHEMES")
    print("="*70)

    Y_ev = D["Y"][D["ev_mask"]]
    perch = D["perch_prob"][D["ev_mask"]]
    exp50 = D["exp50"][D["ev_mask"]]
    v33 = D["v33"][D["ev_mask"]]
    pnew = D["p_new3_ev"]

    base_macro, _ = macro_auc(Y_ev.astype(np.float32), v33)
    print(f"\n  v33 baseline: {base_macro:.4f}\n")

    schemes = {}
    schemes["v33 (baseline)"] = v33

    # Conditional max: when min model says <0.05 (probably negative), use mean.
    # When max model says >0.5 (probably positive), use max.
    stack = np.stack([v33, exp50, pnew])  # (3, n, c)
    avg = stack.mean(axis=0)
    max_ = stack.max(axis=0)
    schemes["conditional_max (max if any>0.5 else avg)"] = np.where(stack.max(axis=0) > 0.5, max_, avg)

    # Confidence-weighted: weight by distance from 0.5
    confidence = np.abs(stack - 0.5) * 2  # in [0, 1]
    cw = (stack * confidence).sum(axis=0) / (confidence.sum(axis=0) + 1e-6)
    schemes["confidence_weighted_avg"] = cw

    # Soft max: log-sum-exp
    schemes["log_sum_exp_blend"] = np.log(np.exp(stack).sum(axis=0) + 1e-9)
    schemes["log_sum_exp_blend"] = (schemes["log_sum_exp_blend"] - schemes["log_sum_exp_blend"].min()) / \
                                       (schemes["log_sum_exp_blend"].max() - schemes["log_sum_exp_blend"].min() + 1e-6)

    # Ranking blend
    from scipy.stats import rankdata
    rank_v33 = rankdata(v33, axis=0) / v33.shape[0]
    rank_exp50 = rankdata(exp50, axis=0) / exp50.shape[0]
    rank_pnew = rankdata(pnew, axis=0) / pnew.shape[0]
    schemes["rank_avg(v33,exp50,pnew)"] = (rank_v33 + rank_exp50 + rank_pnew) / 3
    schemes["rank_max"] = np.maximum(rank_v33, np.maximum(rank_exp50, rank_pnew))

    # Top-K cap on Perch in v26 base
    perch_capped = perch.copy()
    K = 10
    for r in range(len(perch_capped)):
        top_idx = np.argsort(perch_capped[r])[::-1][:K]
        mask = np.ones(N_CLS, dtype=bool); mask[top_idx] = False
        perch_capped[r, mask] = 0.0
    base_capped = 0.7 * perch_capped + 0.3 * exp50
    from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend
    sp_taxon = D["sp_taxon"]
    perch_emb_full = D["perch_emb"]
    sc_g = D["sc_g"]
    # Apply gate on full SS (then take eval slice). For simplicity here just on eval slice
    schemes["v33-style with Perch top-10 cap"] = base_capped  # Skip gate/file-max for speed

    print(f"  {'scheme':<45} {'macro':>8} {'Δ vs v33':>10}")
    for name, pred in schemes.items():
        try:
            pred_clip = np.clip(pred, 0, 1).astype(np.float32)
            m, _ = macro_auc(Y_ev.astype(np.float32), pred_clip)
            print(f"  {name:<45} {m:>8.4f} {m-base_macro:>+10.4f}")
        except Exception as e:
            print(f"  {name:<45} ERROR: {e}")


def main():
    print("=== exp109: Deep dive on each failure mode ===", flush=True)
    print("\nLoading models and predictions ...", flush=True)
    D = build_models()

    section_a_aves_saturation(D)
    section_b_confusion_cluster(D)
    section_c_disagreement(D)
    section_d_calibration(D)
    section_e_s19(D)
    section_f_universal_misses(D)
    section_g_blending_schemes(D)

    print("\n=== exp109 complete ===")


if __name__ == "__main__":
    main()
