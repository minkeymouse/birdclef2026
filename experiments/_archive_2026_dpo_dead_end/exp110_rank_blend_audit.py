#!/usr/bin/env python3
"""exp110 — Proper validation of rank-blending (exp109 finding).

exp109 G found: rank_avg(v33, exp50, P_NEW3) = 0.8979 vs v33 = 0.8340 on
40 evaluable classes. +0.064 macro is huge — needs full audit.

Apply the standard evaluate() framework (sp_row, Aves Δ, per-taxon, class
A/B/C) used throughout exp82-106 for LB-readiness assessment.

Sweep model subsets and rank-blend variants:
  - Subsets: {v33}, {v33, exp50}, {v33, P_NEW3}, {v33, exp50, P_NEW3},
              {Perch, exp50, P_NEW3}, {Perch, exp50, v33, P_NEW3}
  - Variants: rank_avg, rank_max, weighted rank_avg, partial rank
              (rank only when disagreement high)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name):
    d = np.load(EXP80 / name)
    if "scores" in d.files: return d["scores"]
    if "predictions" in d.files: return d["predictions"]
    return d[d.files[0]]


def rank_normalize(P, axis=0):
    """Per-class ranking across rows: each column -> uniform [0, 1]."""
    return rankdata(P, axis=axis) / P.shape[axis]


def main():
    print("=== exp110: rank-blend audit (full evaluate, sp_row + class A/B/C) ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb = load_perch_emb_labeled()
    perch_prob = load_perch_scores_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")

    # v33 reference
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # P_NEW3 LOSO predictions (clean for eval rows)
    from exp106_pnew_hybrid import build_perch_init, train_hybrid
    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0
    tr_mask = sc_g.split.values == "train"
    ev_mask = sc_g.split.values == "eval"
    print("Building P_NEW3 LOSO predictions for eval rows ...", flush=True)
    X_train = np.concatenate([ta_emb[valid], perch_emb[tr_mask]], axis=0)
    Y_train = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])
    W_init, b_init, _ = build_perch_init()
    _, p_new3_ev, _, _, _ = train_hybrid(
        X_train, Y_train, src_w, perch_emb[ev_mask], Y[ev_mask].astype(np.float32),
        W_init, b_init, n_epochs=12, verbose=False
    )
    # P_NEW3 on train: trained on TA + SS_train_keep, predict train? Use TA-only model for clean preds
    # Simpler: use the same LOSO scheme — for each train row, leave its site out
    print("Building P_NEW3 LOSO for train rows (leave-each-site-out) ...", flush=True)
    p_new3_full = np.zeros((len(perch_emb), N_CLS), dtype=np.float32)
    p_new3_full[ev_mask] = p_new3_ev
    sites_arr = sc_g.site.values
    for ho_site in sorted(set(sites_arr[tr_mask])):
        ho_mask = (sites_arr == ho_site) & tr_mask
        if ho_mask.sum() < 5: continue
        keep_mask = (~(sites_arr == ho_site)) & tr_mask
        X_tr_ = np.concatenate([ta_emb[valid], perch_emb[keep_mask]], axis=0)
        Y_tr_ = np.concatenate([Y_ta[valid], Y[keep_mask].astype(np.float32)], axis=0)
        src_w_ = np.concatenate([np.ones(valid.sum()), np.full(keep_mask.sum(), 5.0)])
        _, ev_pred_, _, _, _ = train_hybrid(
            X_tr_, Y_tr_, src_w_, perch_emb[ho_mask], Y[ho_mask].astype(np.float32),
            W_init, b_init, n_epochs=12, verbose=False
        )
        p_new3_full[ho_mask] = ev_pred_
        print(f"  {ho_site}: n={ho_mask.sum()}", flush=True)

    # Diagnostic correlations
    from scipy.stats import pearsonr
    print(f"\nPearson correlations (full 739 × 234):")
    print(f"  v33    ↔ exp50:  {pearsonr(v33.flatten(), exp50.flatten())[0]:.3f}")
    print(f"  v33    ↔ P_NEW3: {pearsonr(v33.flatten(), p_new3_full.flatten())[0]:.3f}")
    print(f"  exp50  ↔ P_NEW3: {pearsonr(exp50.flatten(), p_new3_full.flatten())[0]:.3f}")
    print(f"  Perch  ↔ P_NEW3: {pearsonr(perch_prob.flatten(), p_new3_full.flatten())[0]:.3f}")

    # ===== Now run proper evaluate() on each blend variant =====
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    configs = []

    # ---- Per-class rank versions ----
    for label, models_list in [
        ("rank_avg(v33, exp50)", [v33, exp50]),
        ("rank_avg(v33, P_NEW3)", [v33, p_new3_full]),
        ("rank_avg(v33, exp50, P_NEW3)", [v33, exp50, p_new3_full]),
        ("rank_avg(Perch, exp50, P_NEW3)", [perch_prob, exp50, p_new3_full]),
        ("rank_avg(v33, Perch, exp50, P_NEW3)", [v33, perch_prob, exp50, p_new3_full]),
    ]:
        ranks = [rank_normalize(p, axis=0) for p in models_list]
        avg_rank = np.mean(ranks, axis=0).astype(np.float32)
        configs.append((label, avg_rank))

    # ---- Rank max ----
    for label, models_list in [
        ("rank_max(v33, exp50, P_NEW3)", [v33, exp50, p_new3_full]),
    ]:
        ranks = [rank_normalize(p, axis=0) for p in models_list]
        max_rank = np.maximum.reduce(ranks).astype(np.float32)
        configs.append((label, max_rank))

    # ---- Soft blend with rank ----
    for w in [0.05, 0.10, 0.20, 0.30]:
        ranks = [rank_normalize(p, axis=0) for p in [v33, exp50, p_new3_full]]
        avg_rank = np.mean(ranks, axis=0).astype(np.float32)
        # Mix v33 base with rank_avg at small weight (preserve v33 ranking)
        P = (1 - w) * v33 + w * avg_rank
        configs.append((f"v33 + {w} * rank_avg(v33,exp50,P_NEW3)", P))

    # ---- Z-score normalization (alternative to rank) ----
    def z_normalize(P, axis=0):
        return (P - P.mean(axis=axis, keepdims=True)) / (P.std(axis=axis, keepdims=True) + 1e-6)
    z_blend_3 = (z_normalize(v33) + z_normalize(exp50) + z_normalize(p_new3_full)) / 3
    # Map back to [0, 1] via sigmoid
    z_blend_prob = 1 / (1 + np.exp(-z_blend_3))
    configs.append(("z_avg(v33, exp50, P_NEW3)", z_blend_prob.astype(np.float32)))

    # ---- Restricted rank — only modify ranking when disagreement high ----
    # Use rank_avg result, but only for (row, class) where v33 / exp50 / P_NEW3 max-min > threshold
    stack = np.stack([v33, exp50, p_new3_full])
    disagree = stack.max(axis=0) - stack.min(axis=0)
    ranks = [rank_normalize(p, axis=0) for p in [v33, exp50, p_new3_full]]
    avg_rank = np.mean(ranks, axis=0).astype(np.float32)
    for thr in [0.2, 0.4, 0.6]:
        P = np.where(disagree > thr, avg_rank, v33).astype(np.float32)
        configs.append((f"v33-with-rank_avg if disagree>{thr}", P))

    # ---- Per-row rank: rank predictions WITHIN A ROW (cross-class) ----
    # exp42 was the across-rows form (per-class ranking). Here try cross-class ranking
    # within each row, then average across models.
    def rank_normalize_row(P):
        return rankdata(P, axis=1) / P.shape[1]
    cross_rank_avg = (rank_normalize_row(v33) + rank_normalize_row(exp50) + rank_normalize_row(p_new3_full)) / 3
    configs.append(("rank_avg_PER_ROW(v33,exp50,P_NEW3)", cross_rank_avg.astype(np.float32)))

    # ---- Evaluate all -----
    for label, P in configs:
        rows.append(evaluate(np.clip(P, 0, 1).astype(np.float32), v33, ev_mask, Y, sp_taxon, label))

    res = pd.DataFrame(rows)
    cols = ["label", "macro", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n  Sorted by macro_d desc (positive class A means LB-positive prediction):")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))

    # ===== Hold-out style validation: drop one site at a time, see if pattern stable =====
    print("\n\n=== Site-stratified validation (rank_avg vs v33) ===")
    print(f"  {'site':<6} {'n':>4} {'v33':>8} {'rank_avg':>10} {'Δ':>8}")
    sites = sorted(set(sc_g.site.values))
    for site in sites:
        sm = (sc_g.site.values == site) & ev_mask
        if sm.sum() < 5: continue
        try:
            v33_ev = v33[sm]
            ranks = [rank_normalize(p, axis=0) for p in [v33, exp50, p_new3_full]]
            avg_rank = np.mean(ranks, axis=0)[sm]
            from _lib.eval_metrics import macro_auc
            m1, _ = macro_auc(Y[sm].astype(np.float32), v33_ev)
            m2, _ = macro_auc(Y[sm].astype(np.float32), avg_rank.astype(np.float32))
            print(f"  {site:<6} {sm.sum():>4} {m1:>8.4f} {m2:>10.4f} {m2-m1:>+8.4f}")
        except Exception as e:
            print(f"  {site:<6} ERROR: {e}")


if __name__ == "__main__":
    main()
