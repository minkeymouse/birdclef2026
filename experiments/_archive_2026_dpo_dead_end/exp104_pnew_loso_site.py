#!/usr/bin/env python3
"""exp104 — LOSO-site validation of P_NEW + blend test with v33.

Critical: P_NEW local macro 0.869 is +0.247 over Perch baseline 0.622, but
local-LB anti-correlation is well-established. Need to verify cross-site
generalization BEFORE LB submission.

Tests:
  1. LOSO-site CV: train P_NEW on N-1 labeled SS sites, eval on holdout.
     If Insecta/Reptilia gains COLLAPSE on holdout sites, that's site
     shortcut (same as exp80a iVAE finding).
  2. v33 + P_NEW blend test on 122 eval: does adding P_NEW help v33?
  3. Per-class delta breakdown: which classes drive gain on labeled vs
     hidden-likely.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import per_class_auc, macro_auc, per_taxon_macro, per_row_spearman
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate
from exp103_perch_head_finetune import PerchHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_cached(name):
    d = np.load(EXP80 / name)
    if "scores" in d.files: return d["scores"]
    if "predictions" in d.files: return d["predictions"]
    return d[d.files[0]]


def train_head_on_subset(X_train, Y_train, src_weight, X_eval, Y_eval, n_epochs=20):
    """Train P_NEW head; return (best_macro, best_predictions_on_eval)."""
    model = PerchHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    cls_pos_count = Y_train.sum(axis=0)
    pos_weight = np.where(cls_pos_count > 0,
                            np.sqrt(len(X_train) / (cls_pos_count * N_CLS + 1e-6)),
                            1.0).astype(np.float32)
    pos_weight = np.clip(pos_weight, 0.5, 50.0)
    pw_t = torch.from_numpy(pos_weight).to(DEVICE)

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    Y_t = torch.from_numpy(Y_train.astype(np.float32)).to(DEVICE)
    W_t = torch.from_numpy(src_weight.astype(np.float32)).to(DEVICE)
    Xev_t = torch.from_numpy(X_eval.astype(np.float32)).to(DEVICE)

    n = len(X_t); BATCH = 512
    best_auc = 0.0; best_pred = None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, BATCH):
            idx = perm[s:s+BATCH]
            x = X_t[idx]; y = Y_t[idx]; w = W_t[idx]
            opt.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw_t, reduction="none")
            loss = (loss.mean(dim=-1) * w).mean()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            ev_pred = torch.sigmoid(model(Xev_t)).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        if macro > best_auc:
            best_auc = macro; best_pred = ev_pred
    return best_auc, best_pred


def main():
    print("=== exp104: LOSO-site validation of P_NEW ===\n", flush=True)

    sc_g, Y, primary, l2i = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()
    perch_prob_ss = load_perch_scores_labeled()

    # Load TA cache
    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    valid_mask = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta[np.arange(len(ta_emb))[valid_mask], ta_y_idx[valid_mask]] = 1.0

    sites_arr = sc_g.site.values
    unique_sites = sorted(set(sites_arr))
    print(f"Sites in labeled SS: {unique_sites}\n", flush=True)

    # ===== Test 1: LOSO-site CV =====
    print("=== T1. LOSO-site CV — train P_NEW on N-1 sites, eval on holdout site ===")
    print(f"  {'holdout':<8} {'n_eval':>7} {'macro':>8} {'Aves':>8} {'Amphib':>8} {'Insecta':>8} {'Mamm':>8}")

    overall_macros = []
    overall_per_taxon = {t: [] for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}

    for ho_site in unique_sites:
        ho_mask = sites_arr == ho_site
        # Train on labeled SS rows EXCLUDING this site (regardless of train/eval split)
        # Plus all train_audio
        labeled_mask_keep = ~ho_mask
        # SS eval: hold-out = rows from this site
        if ho_mask.sum() < 5: continue

        X_ss_train = perch_emb_ss[labeled_mask_keep]
        Y_ss_train = Y[labeled_mask_keep].astype(np.float32)
        X_ss_eval = perch_emb_ss[ho_mask]
        Y_ss_eval = Y[ho_mask].astype(np.float32)

        # Skip if eval has no positives in any class (then no AUC available)
        if (Y_ss_eval.sum(axis=0) > 0).sum() < 1: continue

        # Combine TA + SS_keep
        X_train = np.concatenate([ta_emb[valid_mask], X_ss_train], axis=0)
        Y_train_combined = np.concatenate([Y_ta[valid_mask], Y_ss_train], axis=0)
        src_weight = np.concatenate([
            np.ones(valid_mask.sum()),
            np.full(len(X_ss_train), 5.0)
        ])

        best_auc, ev_pred = train_head_on_subset(X_train, Y_train_combined, src_weight,
                                                   X_ss_eval, Y_ss_eval, n_epochs=15)

        # Per-taxon on holdout
        pt = per_taxon_macro(Y_ss_eval, ev_pred, sp_taxon)
        cells = [f"{best_auc:>8.4f}"]
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
            v = pt[t] if not np.isnan(pt[t]) else None
            cells.append(f"{v:>8.4f}" if v is not None else f"{'--':>8}")
            if v is not None: overall_per_taxon[t].append(v)
        print(f"  {ho_site:<8} {ho_mask.sum():>7} " + " ".join(cells), flush=True)
        overall_macros.append(best_auc)

    print(f"\n  Mean LOSO macro: {np.mean(overall_macros):.4f}  (vs same-site 122 eval macro 0.869)")
    print("  Per-taxon LOSO mean:")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        if overall_per_taxon[t]:
            print(f"    {t}: {np.mean(overall_per_taxon[t]):.4f} (n={len(overall_per_taxon[t])})")

    # ===== Test 2: P_NEW + v33 blend test =====
    print("\n=== T2. P_NEW + v33 blend test on standard 122 eval ===")
    p_new_pred = get_cached("p_new_predictions.npz")
    print(f"  Loaded P_NEW predictions: {p_new_pred.shape}")

    exp50 = get_cached("exp50_scores_labeled.npz")
    base = 0.7 * perch_prob_ss + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb_ss, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    # P_NEW alone
    rows.append(evaluate(p_new_pred, v33, ev_mask, Y, sp_taxon, "P_NEW alone"))

    # Various blends
    for w_pnew in [0.10, 0.20, 0.30, 0.50]:
        # additive blend in prob space
        for keep_perch in [True, False]:
            if keep_perch:
                # v33 stays primary; P_NEW additive small weight
                P = (1 - w_pnew) * v33 + w_pnew * p_new_pred
                label = f"v33 + P_NEW w={w_pnew} (additive)"
            else:
                # Replace exp50 with P_NEW
                base_new = 0.7 * perch_prob_ss + 0.3 * p_new_pred
                gated_new = apply_v9_gate(base_new, perch_emb_ss, sp_taxon, offset=0.1)
                P = file_max_blend(gated_new, sc_g, alpha=0.10)
                label = f"v33-style with exp50→P_NEW (w_new={w_pnew}, ignored)"
                if w_pnew != 0.30: continue   # only one variant
            P = np.clip(P, 0, 1).astype(np.float32)
            rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, label))

    # Z-space blend (like v33 does)
    def z_blend(arr1, arr2, w1, w2, sc_g, sp_taxon, perch_emb):
        from scipy.stats import zscore
        z1 = (arr1 - arr1.mean(0)) / (arr1.std(0) + 1e-6)
        z2 = (arr2 - arr2.mean(0)) / (arr2.std(0) + 1e-6)
        blend_z = w1 * z1 + w2 * z2
        blend_back = blend_z * arr1.std(0) + arr1.mean(0)
        return np.clip(blend_back, 0, 1).astype(np.float32)

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n  Sorted by macro_d desc:")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
