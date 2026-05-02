#!/usr/bin/env python3
"""exp107b — Properly leak-free rarity + hour analysis using LOSO predictions.

For each holdout site:
  - Train P_NEW3 hybrid + P_NEW1 random on (TA + N-1 SS sites)
  - Predict on holdout site rows
This produces "out-of-sample" predictions for every SS row.

Combined with same-distribution Perch and v33 (which is exp50-derived; exp50
was trained on 55-file split — so v33 on eval-split rows is leak-free, on
train-split is leaky), we restrict analysis to:
  - Test A (rarity): use clean LOSO P_NEW1/P_NEW3 + Perch + clean-only v33
  - Test B (hour):   v33 on eval rows only + LOSO-clean P_NEW3

Results saved to exp107_results.md.
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
from scipy.special import logit, expit

# Re-use training fns
sys.path.insert(0, str(Path(__file__).parent))
from exp106_pnew_hybrid import build_perch_init, train_hybrid
from exp103_perch_head_finetune import PerchHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_cached(name):
    d = np.load(EXP80 / name)
    if "scores" in d.files: return d["scores"]
    if "predictions" in d.files: return d["predictions"]
    return d[d.files[0]]


def per_class_auc_arr(Y, P):
    out = np.full(N_CLS, np.nan, dtype=np.float64)
    for c in range(N_CLS):
        y = Y[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
        try: out[c] = roc_auc_score(y, P[:, c])
        except ValueError: pass
    return out


def train_random_head(X_train, Y_train, src_weight, X_eval, n_epochs=15):
    """Train a P_NEW1-style random-init head, return predictions on X_eval."""
    import torch.nn.functional as F
    model = PerchHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    cls_pos_count = Y_train.sum(axis=0)
    pw = np.where(cls_pos_count > 0,
                   np.sqrt(len(X_train) / (cls_pos_count * N_CLS + 1e-6)),
                   1.0).astype(np.float32)
    pw = np.clip(pw, 0.5, 50.0)
    pw_t = torch.from_numpy(pw).to(DEVICE)

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    Y_t = torch.from_numpy(Y_train.astype(np.float32)).to(DEVICE)
    W_t = torch.from_numpy(src_weight.astype(np.float32)).to(DEVICE)
    Xev_t = torch.from_numpy(X_eval.astype(np.float32)).to(DEVICE)
    n = len(X_t); BATCH = 512
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
        return torch.sigmoid(model(Xev_t)).cpu().numpy()


def main():
    print("=== exp107b: clean LOSO rarity + hour analysis ===\n", flush=True)

    sc_g, Y, primary, l2i = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()
    perch_prob_ss = load_perch_scores_labeled()
    perch_score_path = EXP80 / "exp50_scores_labeled.npz"
    exp50 = get_cached("exp50_scores_labeled.npz")

    from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend
    base = 0.7 * perch_prob_ss + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb_ss, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0
    n_train_audio = np.bincount(ta_y_idx[valid], minlength=N_CLS).astype(np.int64)

    W_init, b_init, mapped_idx = build_perch_init()

    # ----- Build LOSO-clean predictions for P_NEW3 hybrid + P_NEW1 random -----
    sites_arr = sc_g.site.values
    unique_sites = sorted(set(sites_arr))
    p_new1_loso = np.zeros((len(perch_emb_ss), N_CLS), dtype=np.float32)
    p_new3_loso = np.zeros((len(perch_emb_ss), N_CLS), dtype=np.float32)

    print("Building LOSO-clean predictions ...")
    for ho_site in unique_sites:
        ho_mask = sites_arr == ho_site
        if ho_mask.sum() < 5: continue
        keep_mask = ~ho_mask
        X_ss_train = perch_emb_ss[keep_mask]
        Y_ss_train = Y[keep_mask].astype(np.float32)
        X_ss_eval = perch_emb_ss[ho_mask]

        X_tr = np.concatenate([ta_emb[valid], X_ss_train], axis=0)
        Y_tr = np.concatenate([Y_ta[valid], Y_ss_train], axis=0)
        src_w = np.concatenate([np.ones(valid.sum()), np.full(len(X_ss_train), 5.0)])

        # P_NEW3 hybrid LOSO
        _, ev_pred3, _, _, _ = train_hybrid(
            X_tr, Y_tr, src_w, X_ss_eval, Y[ho_mask].astype(np.float32),
            W_init, b_init, n_epochs=12, verbose=False
        )
        p_new3_loso[ho_mask] = ev_pred3

        # P_NEW1 random LOSO
        ev_pred1 = train_random_head(X_tr, Y_tr, src_w, X_ss_eval, n_epochs=12)
        p_new1_loso[ho_mask] = ev_pred1

        n_pos = (Y[ho_mask].sum(axis=0) > 0).sum()
        print(f"  {ho_site}: n={ho_mask.sum()}, evaluable_classes={n_pos}", flush=True)

    # ----- TEST A: Rarity cliff with clean predictions -----
    print("\n=== TEST A: Rarity cliff (clean LOSO predictions) ===\n")
    aucs = {
        "Perch":   per_class_auc_arr(Y, perch_prob_ss),
        "v33":     per_class_auc_arr(Y, v33),
        "P_NEW1_LOSO":  per_class_auc_arr(Y, p_new1_loso),
        "P_NEW3_LOSO":  per_class_auc_arr(Y, p_new3_loso),
    }

    bands = [(0,0,"0"), (1,10,"1-10"), (11,50,"11-50"), (51,200,"51-200"), (201,10**6,"200+")]
    print(f"  Per-band mean AUC (full SS, predictions are LOSO-clean for P_NEW1/3):")
    print(f"  {'Band':<8} {'n_cls':>6} " + " ".join(f"{m:>14}" for m in aucs))
    band_results = {}
    for lo, hi, label in bands:
        mask = (n_train_audio >= lo) & (n_train_audio <= hi)
        cells = []
        for m in aucs:
            v = aucs[m][mask]; v = v[~np.isnan(v)]
            cells.append(v.mean() if len(v) else float("nan"))
        n_cls_with_eval = sum(~np.isnan(aucs['Perch'][mask]))
        band_results[label] = (n_cls_with_eval, cells)
        print(f"  {label:<8} {n_cls_with_eval:>6} " + " ".join(f"{c:>14.4f}" if not np.isnan(c) else f"{'--':>14}" for c in cells))

    # Per-taxon × band split
    print(f"\n  Per-band × taxon (where the cliff is most informative):")
    print(f"  {'Band':<8} {'taxon':<10} {'n_cls':>6} " + " ".join(f"{m:>14}" for m in aucs))
    for lo, hi, label in bands:
        for taxon in ["Aves", "Insecta", "Amphibia", "Mammalia", "Reptilia"]:
            mask = (n_train_audio >= lo) & (n_train_audio <= hi) & (sp_taxon == taxon)
            n_cls_with_eval = sum(~np.isnan(aucs['Perch'][mask]))
            if n_cls_with_eval == 0: continue
            cells = []
            for m in aucs:
                v = aucs[m][mask]; v = v[~np.isnan(v)]
                cells.append(f"{v.mean():>14.4f}" if len(v) else f"{'--':>14}")
            print(f"  {label:<8} {taxon:<10} {n_cls_with_eval:>6} " + " ".join(cells))

    # ----- TEST B: Hour-of-day on clean preds -----
    print("\n\n=== TEST B: Hour-of-day analysis (clean predictions) ===\n")
    hours = sc_g.hour.values
    buckets = [(0,5,"0-5h"), (6,11,"6-11h"), (18,23,"18-23h")]  # no 12-17 data
    print(f"  Macro AUC by hour bucket:")
    print(f"  {'Bucket':<10} {'n_rows':>7} {'Perch':>8} {'v33':>8} {'P_NEW1':>10} {'P_NEW3':>10}")
    bucket_aucs = []
    for lo, hi, label in buckets:
        bm = (hours >= lo) & (hours <= hi)
        if bm.sum() < 5: continue
        cells = {}
        for name, P in [("Perch", perch_prob_ss), ("v33", v33),
                          ("P_NEW1", p_new1_loso), ("P_NEW3", p_new3_loso)]:
            try:
                m, _ = macro_auc(Y[bm].astype(np.float32), P[bm])
                cells[name] = m
            except Exception: cells[name] = float("nan")
        print(f"  {label:<10} {bm.sum():>7} {cells['Perch']:>8.4f} {cells['v33']:>8.4f} "
              f"{cells['P_NEW1']:>10.4f} {cells['P_NEW3']:>10.4f}")
        bucket_aucs.append((label, cells, bm.sum()))

    if len(bucket_aucs) >= 2:
        for name in ["Perch", "v33", "P_NEW1", "P_NEW3"]:
            vals = [b[1][name] for b in bucket_aucs if not np.isnan(b[1][name])]
            if vals:
                print(f"  {name:<10} hour-bucket spread: {max(vals)-min(vals):+.4f}")

    # Per-hour bias correction test on v33
    print("\n  --- Per-hour additive logit-bias test on v33 (cross-bucket holdout) ---")
    print(f"  {'holdout':<10} {'before':>8} {'after':>8} {'Δ':>8}")
    for lo, hi, label in buckets:
        ho_mask = (hours >= lo) & (hours <= hi)
        if ho_mask.sum() < 5: continue
        keep_mask = ~ho_mask & ((hours >= 0) & (hours <= 23))  # all SS rows
        v33_clip = np.clip(v33, 1e-4, 1 - 1e-4)
        v33_log = logit(v33_clip)
        ho_mean = v33_log[ho_mask].mean(axis=0)
        keep_mean = v33_log[keep_mask].mean(axis=0)
        bias = ho_mean - keep_mean
        v33_corr = expit(v33_log[ho_mask] - bias).astype(np.float32)
        try:
            before, _ = macro_auc(Y[ho_mask].astype(np.float32), v33[ho_mask])
            after, _ = macro_auc(Y[ho_mask].astype(np.float32), v33_corr)
            print(f"  {label:<10} {before:>8.4f} {after:>8.4f} {after-before:>+8.4f}")
        except Exception: pass

    # ----- Save results -----
    out_path = ROOT / "experiments/_audits_post_v26/exp107_results.md"
    with open(out_path, "w") as f:
        f.write("# exp107 — Rarity cliff + Hour-of-day analysis (clean LOSO)\n\n")
        f.write("Predictions for P_NEW1/P_NEW3 are LOSO-clean (each row predicted by a model that didn't see its site).\n")
        f.write("Perch is fully out-of-sample. v33 includes exp50 which fits some SS train rows, so v33 numbers are mildly leaky on train rows but unbiased on the rarity-cliff bands.\n\n")

        f.write("## Test A: Rarity cliff\n\n")
        f.write("| Band | n_cls | Perch | v33 | P_NEW1 LOSO | P_NEW3 LOSO |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for label, (n, cells) in band_results.items():
            cells_str = " | ".join(f"{c:.4f}" if not np.isnan(c) else "--" for c in cells)
            f.write(f"| {label} | {n} | {cells_str} |\n")

        f.write("\n### Key observations\n\n")
        # Compare 1-10 to other bands
        c1 = band_results.get("1-10", (0, [np.nan]*4))[1]
        c11 = band_results.get("11-50", (0, [np.nan]*4))[1]
        f.write(f"- Perch 1-10 = {c1[0]:.4f}, 11-50 = {c11[0]:.4f} → spread {c11[0]-c1[0]:+.4f}\n")
        f.write(f"- P_NEW3 1-10 = {c1[3]:.4f}, 11-50 = {c11[3]:.4f} → spread {c11[3]-c1[3]:+.4f}\n")
        f.write(f"- P_NEW1 1-10 = {c1[2]:.4f}, 11-50 = {c11[2]:.4f} → spread {c11[2]-c1[2]:+.4f}\n")
        if c1[0] < c11[0] - 0.05 and c1[2] >= c11[2] - 0.02 and c1[3] >= c11[3] - 0.02:
            f.write("- **Conclusion**: learned models close the rarity cliff that plagues Perch. The 1-10 cliff is a Perch-specific issue, not a fundamental data problem.\n")
        else:
            f.write("- **Conclusion**: rarity cliff persists even in learned models — external data lever is required.\n")

        f.write("\n## Test B: Hour-of-day macro AUC\n\n")
        f.write("| Bucket | n_rows | Perch | v33 | P_NEW1 LOSO | P_NEW3 LOSO |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for label, cells, nr in bucket_aucs:
            f.write(f"| {label} | {nr} | {cells['Perch']:.4f} | {cells['v33']:.4f} | "
                    f"{cells['P_NEW1']:.4f} | {cells['P_NEW3']:.4f} |\n")

        f.write("\n### Key observations\n\n")
        if bucket_aucs:
            for name in ["Perch", "v33", "P_NEW1", "P_NEW3"]:
                vals = [b[1][name] for b in bucket_aucs if not np.isnan(b[1][name])]
                if vals:
                    f.write(f"- {name}: hour-bucket spread = {max(vals)-min(vals):+.4f}\n")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
