#!/usr/bin/env python3
"""exp119b — Fix: cross-taxon penalty only on PURELY non-Aves rows.

Bug in exp119: penalty triggered whenever ANY non-Aves positive present.
Multi-label SS rows often have BOTH Aves and non-Aves positive — penalty
suppressed Aves even when Aves was correctly positive.

Fix: only trigger when row has zero Aves positive AND ≥1 non-Aves positive.
This matches the actual failure mode (exp108): pure-Mammalia row predicted
as Aves top-1.

In TA: ~1% of rows are pure non-Aves (single non-Aves species)
In SS: rows where ALL truth is non-Aves (Insecta sonotypes, etc.)

These are exactly the cases where Perch's bird-bias is wrong.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        ROOT, N_CLS, TAXA)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from exp106_pnew_hybrid import build_perch_init, PerchHybrid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cross_taxon_penalty_v2(logits, y, sp_taxon_idx, margin=1.0):
    """Penalty only on rows with NO Aves positive AND ≥1 non-Aves positive."""
    aves_mask = (sp_taxon_idx == 0)
    has_aves_pos = (y[:, aves_mask] > 0).any(dim=1)  # (B,)
    has_non_aves_pos = (y[:, ~aves_mask] > 0).any(dim=1)
    trigger = (~has_aves_pos) & has_non_aves_pos

    if trigger.sum() == 0:
        return logits.sum() * 0.0

    rows = trigger.nonzero(as_tuple=False).squeeze(-1)
    aves_logits = logits[rows][:, aves_mask]
    non_aves_logits = logits[rows][:, ~aves_mask]
    non_aves_y = y[rows][:, ~aves_mask]

    max_aves = aves_logits.max(dim=1).values
    masked = non_aves_logits.masked_fill(non_aves_y == 0, -1e9)
    max_true_non_aves = masked.max(dim=1).values

    penalty = F.relu(max_aves - max_true_non_aves + margin)
    return penalty.mean()


def train_with_penalty(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                         sp_taxon_idx, lambda_ct=0.0, margin=1.0, n_epochs=15, lr=1e-3, verbose=False):
    model = PerchHybrid(W_init, b_init).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/10)

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
    sp_taxon_t = torch.from_numpy(sp_taxon_idx).to(DEVICE)

    n = len(X_t); BATCH = 512
    best_auc = 0.0; best_pred = None
    n_trigger_batches = 0
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_bce = 0.0; ep_ct = 0.0; n_trig = 0; nb = 0
        for s in range(0, n, BATCH):
            idx = perm[s:s+BATCH]
            x = X_t[idx]; y = Y_t[idx]; w = W_t[idx]
            opt.zero_grad()
            logits = model(x)
            bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw_t, reduction="none")
            bce = (bce.mean(dim=-1) * w).mean()
            loss = bce
            ct = torch.tensor(0.0, device=DEVICE)
            if lambda_ct > 0:
                ct = cross_taxon_penalty_v2(logits, y, sp_taxon_t, margin=margin)
                loss = loss + lambda_ct * ct
                if float(ct) > 0: n_trig += 1
            loss.backward()
            opt.step()
            ep_bce += bce.item(); ep_ct += float(ct); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            ev_pred = torch.sigmoid(model(Xev_t)).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        if macro > best_auc:
            best_auc = macro; best_pred = ev_pred
        if verbose:
            print(f"  ep {ep:02d} bce {ep_bce/nb:.3f} ct {ep_ct/nb:.4f} (n_trig_batches {n_trig})  macro {macro:.4f}")

    return best_auc, best_pred


def main():
    print("=== exp119b: Cross-taxon penalty (FIXED — pure non-Aves only) ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    sp_taxon_idx = np.array([TAXA.index(t) if t in TAXA else 0 for t in sp_taxon], dtype=np.int64)

    perch_emb = load_perch_emb_labeled()

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

    X_train = np.concatenate([ta_emb[valid], perch_emb[tr_mask]], axis=0)
    Y_train = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])

    aves_mask_arr = sp_taxon == "Aves"
    has_aves = (Y_train[:, aves_mask_arr] > 0).any(axis=1)
    has_non_aves = (Y_train[:, ~aves_mask_arr] > 0).any(axis=1)
    pure_non_aves = (~has_aves) & has_non_aves
    print(f"  Train rows: pure non-Aves {pure_non_aves.sum()} / {len(Y_train)} ({100*pure_non_aves.mean():.1f}%)\n")

    variants = [
        ("V0: BCE baseline",           dict(lambda_ct=0.0)),
        ("V1: penalty λ=0.5 m=1",      dict(lambda_ct=0.5, margin=1.0)),
        ("V2: penalty λ=1.0 m=1",      dict(lambda_ct=1.0, margin=1.0)),
        ("V3: penalty λ=2.0 m=1",      dict(lambda_ct=2.0, margin=1.0)),
        ("V4: penalty λ=0.5 m=2",      dict(lambda_ct=0.5, margin=2.0)),
        ("V5: penalty λ=1.0 m=2",      dict(lambda_ct=1.0, margin=2.0)),
        ("V6: penalty λ=5.0 m=1",      dict(lambda_ct=5.0, margin=1.0)),
    ]

    results = []
    for name, kwargs in variants:
        print(f"=== {name} ===", flush=True)
        best, ev_pred = train_with_penalty(
            X_train, Y_train, src_w, perch_emb[ev_mask], Y_ss_ev,
            W_init, b_init, sp_taxon_idx, n_epochs=15, verbose=True, **kwargs
        )
        pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
        rec = {"variant": name, "macro": best,
                **{t: pt.get(t, float('nan')) for t in TAXA}}
        results.append(rec)
        print(f"  macro {best:.4f} | "
              f"Aves {pt.get('Aves', float('nan')):.4f}, "
              f"Amphib {pt.get('Amphibia', float('nan')):.4f}, "
              f"Insecta {pt.get('Insecta', float('nan')):.4f}, "
              f"Mam {pt.get('Mammalia', float('nan')):.4f}, "
              f"Rept {pt.get('Reptilia', float('nan')):.4f}\n", flush=True)

    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df.sort_values("macro", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
