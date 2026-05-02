#!/usr/bin/env python3
"""exp112 — P_NEW3 trained with ranking loss instead of BCE.

Hypothesis (from exp108): 68/92 universal misses are within-taxon
sister-species confusion. BCE pushes individual logits independently
toward 0/1 — doesn't explicitly enforce relative ranking. Ranking loss
forces logit(positive) > logit(negative) directly.

Two ranking losses tested:
  (A) Per-CLASS soft AUC: for each class c, all pos rows rank above neg rows.
      This is directly the AUC target and aligns with our LB metric.
  (B) Per-ROW pairwise: within each row, all positive species rank above
      all negative species. Directly attacks confusion-cluster failure mode.
  (C) Mixed: BCE + λ * (A or B) with λ tuned

Same architecture as exp106 (frozen Perch-init Linear + trainable correction
MLP). Same data (TA + SS).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from exp106_pnew_hybrid import build_perch_init, PerchHybrid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def per_class_soft_auc_loss(logits, targets, pos_weight=None):
    """Soft AUC loss per class.
    For each class c: encourage logit[i,c] > logit[j,c] for all i in pos_c, j in neg_c.
    loss_c = mean over (i,j) pairs of softplus(logit[j,c] - logit[i,c]).
    """
    B, C = logits.shape
    losses = []
    for c in range(C):
        pos_mask = targets[:, c] > 0
        neg_mask = targets[:, c] == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        pos_l = logits[pos_mask, c]  # (P,)
        neg_l = logits[neg_mask, c]  # (N,)
        # All pairs: P*N matrix of (neg - pos)
        diffs = neg_l.unsqueeze(0) - pos_l.unsqueeze(1)  # (P, N)
        # softplus(neg - pos) = log(1 + exp(neg - pos))
        loss_c = F.softplus(diffs).mean()
        if pos_weight is not None:
            loss_c = loss_c * pos_weight[c]
        losses.append(loss_c)
    if not losses:
        return logits.sum() * 0.0  # zero gradient
    return torch.stack(losses).mean()


def per_row_pairwise_loss(logits, targets):
    """Per-row ranking: within each row, pos species rank above neg species."""
    B, C = logits.shape
    losses = []
    for i in range(B):
        pos_mask = targets[i] > 0
        neg_mask = targets[i] == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        pos_l = logits[i][pos_mask]
        neg_l = logits[i][neg_mask]
        diffs = neg_l.unsqueeze(0) - pos_l.unsqueeze(1)
        loss_i = F.softplus(diffs).mean()
        losses.append(loss_i)
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def train_with_ranking(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                        loss_type="mixed", lambda_rank=1.0,
                        n_epochs=20, lr=1e-3, verbose=False):
    """Train PerchHybrid with ranking loss.

    loss_type: one of {bce, class_auc, row_pair, mixed_class, mixed_row}
    lambda_rank: weight on ranking loss (vs BCE) when loss_type='mixed_*'
    """
    model = PerchHybrid(W_init, b_init).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
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

    best_auc = 0.0; best_pred = None; best_ep = -1

    model.eval()
    with torch.no_grad():
        ev_pred0 = torch.sigmoid(model(Xev_t)).cpu().numpy()
    macro0, _ = macro_auc(Y_eval, ev_pred0)
    if verbose: print(f"  ep -1 init  macro {macro0:.4f}")

    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_bce = 0.0; ep_rank = 0.0; nb = 0
        for s in range(0, n, BATCH):
            idx = perm[s:s+BATCH]
            x = X_t[idx]; y = Y_t[idx]; w = W_t[idx]
            opt.zero_grad()
            logits = model(x)

            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw_t, reduction="none")
            bce = (bce.mean(dim=-1) * w).mean()

            rank_loss = torch.tensor(0.0, device=DEVICE)
            if loss_type == "bce":
                loss = bce
            elif loss_type == "class_auc":
                rank_loss = per_class_soft_auc_loss(logits, y)
                loss = rank_loss
            elif loss_type == "row_pair":
                rank_loss = per_row_pairwise_loss(logits, y)
                loss = rank_loss
            elif loss_type == "mixed_class":
                rank_loss = per_class_soft_auc_loss(logits, y)
                loss = bce + lambda_rank * rank_loss
            elif loss_type == "mixed_row":
                rank_loss = per_row_pairwise_loss(logits, y)
                loss = bce + lambda_rank * rank_loss
            else:
                raise ValueError(f"unknown loss_type: {loss_type}")

            loss.backward()
            opt.step()
            ep_bce += bce.item(); ep_rank += float(rank_loss); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            ev_pred = torch.sigmoid(model(Xev_t)).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        if macro > best_auc:
            best_auc = macro; best_pred = ev_pred; best_ep = ep
        if verbose and (ep % 3 == 0 or ep == n_epochs - 1):
            print(f"  ep {ep:02d}  bce={ep_bce/nb:.4f} rank={ep_rank/nb:.4f}  eval_macro {macro:.4f}")

    return best_auc, best_pred, macro0, best_ep, model


def main():
    print("=== exp112: P_NEW3 with ranking loss ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()
    perch_prob_ss = load_perch_scores_labeled()

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0

    W_init, b_init, _ = build_perch_init()

    tr_mask = sc_g.split.values == "train"
    ev_mask = sc_g.split.values == "eval"
    Y_ss_ev = Y[ev_mask].astype(np.float32)

    X_train = np.concatenate([ta_emb[valid], perch_emb_ss[tr_mask]], axis=0)
    Y_train = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])
    print(f"  Train: TA {valid.sum()} + SS_train {tr_mask.sum()} = {len(X_train)}")
    print(f"  Eval (122 SS): {ev_mask.sum()} rows\n")

    # Compare loss types
    print("=== Loss type sweep on 122 eval (same architecture, same data) ===\n")
    loss_configs = [
        ("bce (baseline P_NEW3)", "bce", 0.0),
        ("class_auc (only)", "class_auc", 0.0),
        ("row_pair (only)", "row_pair", 0.0),
        ("mixed_class λ=0.5", "mixed_class", 0.5),
        ("mixed_class λ=1.0", "mixed_class", 1.0),
        ("mixed_class λ=2.0", "mixed_class", 2.0),
        ("mixed_row λ=0.5", "mixed_row", 0.5),
        ("mixed_row λ=1.0", "mixed_row", 1.0),
        ("mixed_row λ=2.0", "mixed_row", 2.0),
    ]

    results = []
    saved_preds = {}
    for name, loss_type, lam in loss_configs:
        print(f"\n--- {name} ---", flush=True)
        best, ev_pred, init_macro, best_ep, model = train_with_ranking(
            X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
            W_init, b_init, loss_type=loss_type, lambda_rank=lam,
            n_epochs=15, verbose=False
        )
        pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
        rec = {"loss": name, "macro": best, "best_ep": best_ep,
                "Aves": pt.get("Aves", float("nan")),
                "Amphibia": pt.get("Amphibia", float("nan")),
                "Insecta": pt.get("Insecta", float("nan")),
                "Mammalia": pt.get("Mammalia", float("nan")),
                "Reptilia": pt.get("Reptilia", float("nan"))}
        results.append(rec)
        saved_preds[name] = ev_pred
        print(f"  best macro {best:.4f} @ ep{best_ep:02d}, Aves {rec['Aves']:.4f} "
              f"Amphib {rec['Amphibia']:.4f} Insecta {rec['Insecta']:.4f} "
              f"Mam {rec['Mammalia']:.4f}", flush=True)

    df = pd.DataFrame(results)
    print("\n=== Summary (122 eval, sorted by macro) ===")
    print(df.sort_values("macro", ascending=False).to_string(index=False))

    # Save best loss type's predictions for blend testing
    best_row = df.loc[df.macro.idxmax()]
    best_name = best_row["loss"]
    np.savez_compressed(
        EXP80 / "p_new3_ranking_predictions.npz",
        predictions=saved_preds[best_name].astype(np.float32),
        loss_type=best_name,
    )
    print(f"\nSaved best-loss predictions ({best_name}) → {EXP80}/p_new3_ranking_predictions.npz")


if __name__ == "__main__":
    main()
