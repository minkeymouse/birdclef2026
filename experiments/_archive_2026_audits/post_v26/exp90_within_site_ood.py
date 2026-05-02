#!/usr/bin/env python3
"""exp90 — Within-site UNK detection + uncertainty-routing test.

Two-stage:
A. Within-site UNK detection: for each site with both UNK_ONLY and
   KNOWN_ONLY rows (n≥3 each), compute per-row uncertainty signals
   and report UNK-vs-KNOWN AUC WITHIN that site. Aggregate across sites.

   If aggregate AUC > 0.7 → genuine acoustic OOD signal independent of
   site fingerprint. Justifies routing.

B. If signal real, build a per-row uncertainty score and test
   uncertainty-routed blend on 122 eval rows:
     final[i] = (1 - α(i)) × v33[i] + α(i) × exp50[i]
     where α(i) = sigmoid((U(i) - U_median) × scale)
   i.e., Perch trusted when certain, exp50 weighted up when uncertain.

   Compare macro_d, sp_row, per-taxon Δ vs v33.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS, TAXA)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name): return np.load(EXP80 / name)["scores"]


def entropy_norm(p, axis=-1):
    eps = 1e-12
    pp = p / (p.sum(axis=axis, keepdims=True) + eps)
    return -(pp * np.log(pp + eps)).sum(axis=axis) / np.log(p.shape[axis])


def build_v33_base(perch_prob, exp50, perch_emb, sc_g, sp_taxon):
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    return file_max_blend(gated, sc_g, alpha=0.10)


def main():
    print("=== exp90: within-site UNK detection + uncertainty routing ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")

    col_var = perch_prob.var(axis=0)
    unmapped_idx = np.where(col_var < 1e-6)[0]
    mapped_idx = np.where(col_var >= 1e-6)[0]

    # Class per row (UNK_ONLY / KNOWN_ONLY / UNK_MIXED / BG)
    classes = []
    for i in range(len(sc_g)):
        gt = np.where(Y[i] == 1)[0]
        if len(gt) == 0:
            classes.append("BG"); continue
        gt_unmapped = np.intersect1d(gt, unmapped_idx)
        gt_mapped = np.intersect1d(gt, mapped_idx)
        if len(gt_unmapped) > 0 and len(gt_mapped) == 0:
            classes.append("UNK_ONLY")
        elif len(gt_unmapped) > 0 and len(gt_mapped) > 0:
            classes.append("UNK_MIXED")
        elif len(gt_mapped) > 0:
            classes.append("KNOWN_ONLY")
        else:
            classes.append("BG")
    classes = np.array(classes)

    # Per-row pure features (no centroid → no leak)
    H_234 = entropy_norm(perch_prob, axis=-1)
    top1 = perch_prob.max(axis=-1)
    sorted_p = np.sort(perch_prob, axis=-1)[:, ::-1]
    top5_mass = sorted_p[:, :5].sum(axis=-1) / (sorted_p.sum(axis=-1) + 1e-12)
    emb_L2 = np.linalg.norm(perch_emb, axis=-1)
    emb_max = np.abs(perch_emb).max(axis=-1)
    emb_var = perch_emb.var(axis=-1)

    df = pd.DataFrame({
        "row_id": sc_g.row_id.values,
        "site": sc_g.site.values,
        "split": sc_g.split.values,
        "class": classes,
        "H_234": H_234, "top1": top1, "top5_mass": top5_mass,
        "emb_L2": emb_L2, "emb_max": emb_max, "emb_var": emb_var,
    })

    # === A. Within-site UNK_ONLY vs KNOWN_ONLY AUC ===
    print("=== A. Within-site UNK_ONLY vs KNOWN_ONLY AUC (no centroid features) ===")
    sites = sorted(df.site.unique())
    metrics = ["H_234", "top1", "top5_mass", "emb_L2", "emb_max", "emb_var"]

    print(f"  {'site':<6} {'n_UNK':>6} {'n_KNOWN':>7}  " + " ".join(f"{m:>10}" for m in metrics))
    per_site_aucs = {m: [] for m in metrics}
    for s in sites:
        sub = df[df.site == s]
        u = sub[sub["class"] == "UNK_ONLY"]
        k = sub[sub["class"] == "KNOWN_ONLY"]
        if len(u) < 3 or len(k) < 3: continue
        y = np.concatenate([np.ones(len(u)), np.zeros(len(k))])
        row = [f"  {s:<6} {len(u):>6} {len(k):>7}"]
        for m in metrics:
            scores = np.concatenate([u[m].values, k[m].values])
            try:
                auc = roc_auc_score(y, scores)
                # If most metrics are inverted (UNK has lower value), flip to consistent direction
                # Use deviation from 0.5 as "signal strength"
                row.append(f"{auc:>10.3f}")
                per_site_aucs[m].append(auc)
            except: row.append(f"{'---':>10}")
        print(" ".join(row))

    print(f"\n  Median across qualifying sites:")
    for m in metrics:
        if per_site_aucs[m]:
            med = np.median(per_site_aucs[m])
            # Direction-agnostic strength
            strength = abs(med - 0.5) * 2
            print(f"    {m:<14} median AUC = {med:.3f}  (strength = {strength:.3f})")

    # === A2. Combined uncertainty score (PCA over per-row features) ===
    # Build a "Perch-uncertainty score" U:
    # U high = uncertain, U low = certain
    # Use sign: emb_L2 LOW means uncertain → flip sign
    feats = np.stack([
        -emb_L2,          # higher when L2 lower (uncertain)
        -emb_max,
        -emb_var,
        -H_234,           # higher when entropy lower (UNK has lower H)... actually wait
                          # UNK has LOWER H according to exp89. We want U HIGH on UNK.
                          # so we negate H_234 (UNK has H=0.960, KNOWN H=0.963 → U=-0.960 vs -0.963 ⇒ UNK higher)
        top5_mass,        # UNK has higher top5_mass
    ], axis=1)
    # standardize
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
    # Combined: simple mean
    U_combined = feats.mean(axis=1)

    print("\n=== A3. Within-site AUC of combined uncertainty score U ===")
    print(f"  {'site':<6} {'n_UNK':>6} {'n_KNOWN':>7}  AUC")
    site_aucs_combined = []
    for s in sites:
        sub = df[df.site == s]
        u_idx = sub[sub["class"] == "UNK_ONLY"].index
        k_idx = sub[sub["class"] == "KNOWN_ONLY"].index
        if len(u_idx) < 3 or len(k_idx) < 3: continue
        y = np.concatenate([np.ones(len(u_idx)), np.zeros(len(k_idx))])
        scores = np.concatenate([U_combined[u_idx], U_combined[k_idx]])
        try:
            auc = roc_auc_score(y, scores)
            site_aucs_combined.append(auc)
            print(f"  {s:<6} {len(u_idx):>6} {len(k_idx):>7}  {auc:.3f}")
        except: pass
    if site_aucs_combined:
        med = np.median(site_aucs_combined)
        mean = np.mean(site_aucs_combined)
        print(f"\n  Median within-site AUC for U_combined: {med:.3f}")
        print(f"  Mean   within-site AUC for U_combined: {mean:.3f}")

    # === B. Routing test: blend more exp50 when U is high ===
    print("\n=== B. Uncertainty-routed blend test on 122 eval rows ===", flush=True)
    v33_ref = build_v33_base(perch_prob, exp50, perch_emb, sc_g, sp_taxon)
    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33_ref, v33_ref, ev_mask, Y, sp_taxon, "v33 ref")]

    # Build per-row α(i) ∈ [0, 1] from U
    # Standardize U on TRAIN, apply to ALL
    tr_mask = sc_g.split.values == "train"
    U_tr_mean = U_combined[tr_mask].mean()
    U_tr_std = U_combined[tr_mask].std() + 1e-6
    U_norm = (U_combined - U_tr_mean) / U_tr_std

    # Routing variants:
    # v33 mod: final[i] = (1 - extra(i)) * v33[i] + extra(i) * exp50_via_pipeline[i]
    # Where extra(i) = clip(σ(U_norm * scale) - 0.5, 0, 0.3)  — extra weight to exp50 component
    # But v33 already has exp50 at 0.3 weight. We want to BOOST exp50 weight on uncertain rows.
    # Simpler: re-blend per-row
    # base_certain[i] = (0.7P + 0.3 exp50) (current v33)
    # base_uncertain[i] = (0.5P + 0.5 exp50) (more SED on uncertain)
    # final = (1 - σ(U)) base_certain + σ(U) base_uncertain
    # Then V9 + file-max
    base_cert = 0.7 * perch_prob + 0.3 * exp50
    base_unc = 0.5 * perch_prob + 0.5 * exp50

    print("\n  routing scale sweep:")
    for scale in [0.5, 1.0, 2.0, 3.0]:
        alpha = 1.0 / (1.0 + np.exp(-U_norm * scale))   # σ in [0,1]
        base_per_row = (1 - alpha[:, None]) * base_cert + alpha[:, None] * base_unc
        gated = apply_v9_gate(base_per_row, perch_emb, sc_g, sp_taxon) if False else None
        # Use existing apply_v9_gate
        gated = apply_v9_gate(base_per_row, perch_emb, sp_taxon, offset=0.1)
        P_routed = file_max_blend(gated, sc_g, alpha=0.10)
        rows.append(evaluate(P_routed, v33_ref, ev_mask, Y, sp_taxon,
                              f"Uncert-route σ-sigmoid scale={scale}"))

    # Also test: hard threshold routing
    print("\n  hard-threshold sweep:")
    for thresh in [0.5, 1.0, 1.5]:
        is_uncertain = U_norm > thresh
        base_per_row = base_cert.copy()
        base_per_row[is_uncertain] = base_unc[is_uncertain]
        gated = apply_v9_gate(base_per_row, perch_emb, sp_taxon, offset=0.1)
        P_routed = file_max_blend(gated, sc_g, alpha=0.10)
        n_unc = is_uncertain.sum()
        rows.append(evaluate(P_routed, v33_ref, ev_mask, Y, sp_taxon,
                              f"Uncert-route hard U>{thresh} (n_unc={n_unc})"))

    # Also test: routing toward exp84b (external-finetune SED) on uncertain rows
    print("\n  routing to exp84b on uncertain rows:")
    exp84b = get_cached("exp84b_scores_labeled.npz")
    base_unc_84b = 0.5 * perch_prob + 0.5 * exp84b
    for scale in [1.0, 2.0]:
        alpha = 1.0 / (1.0 + np.exp(-U_norm * scale))
        base_per_row = (1 - alpha[:, None]) * base_cert + alpha[:, None] * base_unc_84b
        gated = apply_v9_gate(base_per_row, perch_emb, sp_taxon, offset=0.1)
        P_routed = file_max_blend(gated, sc_g, alpha=0.10)
        rows.append(evaluate(P_routed, v33_ref, ev_mask, Y, sp_taxon,
                              f"Uncert→exp84b scale={scale}"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== ROUTING RESULTS ===")
    print(res[cols].sort_values("macro_d", ascending=False).to_string(index=False))
    res.to_csv(EXP80 / "exp90_uncert_routing.csv", index=False)


if __name__ == "__main__":
    main()
