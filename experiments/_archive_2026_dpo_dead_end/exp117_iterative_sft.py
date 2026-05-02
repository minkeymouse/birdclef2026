#!/usr/bin/env python3
"""exp117 — Iterative SFT (boosting-style) on misidentified rows.

Different from exp114 hard-DPO:
  - exp114: DPO on (positive, negative) pairs where model is wrong
  - exp117: SFT (BCE) with extra weight on rows where model fails

Why this might differ:
  - DPO trains relative ranking; pairs may share site shortcut
  - Re-weighted BCE forces model to fit hard rows directly
  - Iterating is like AdaBoost: each round corrects previous round's errors

Method:
  Round 0: train BCE_0 on TA + SS_train (uniform weights)
  Round k: identify rows where BCE_k-1 fails (misses positive at threshold)
            up-weight those rows by w_hard, train BCE_k from scratch
            (or from BCE_k-1 weights — try both)
  Eval: same-site 122 + LOSO

If iterative SFT improves LOSO over BCE, real signal.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        ROOT, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from exp106_pnew_hybrid import build_perch_init, PerchHybrid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_bce_with_row_weights(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                                  init_state=None, n_epochs=15, lr=1e-3, verbose=False):
    """BCE training with per-row weights (extends src_weight for misidentified rows)."""
    model = PerchHybrid(W_init, b_init).to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state)

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

    n = len(X_t); BATCH = 512
    best_auc = 0.0; best_state = None; best_pred = None
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
            best_auc = macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_pred = ev_pred
        if verbose: print(f"  ep {ep:02d}  macro {macro:.4f}")
    model.load_state_dict(best_state)
    return model, best_auc, best_pred


def identify_hard_rows(model, X_train, Y_train, mode="missed_positive"):
    """Identify rows where model is currently failing.

    mode="missed_positive": rows where model predicts <0.3 on a true positive.
    """
    model.eval()
    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    n = len(X_t); BATCH = 2048
    hard_mask = np.zeros(n, dtype=bool)
    with torch.no_grad():
        for s in range(0, n, BATCH):
            x = X_t[s:s+BATCH]
            logits = model(x).cpu()
            probs = torch.sigmoid(logits).numpy()
            y_b = Y_train[s:s+BATCH]
            for i in range(len(x)):
                pos_idx = np.where(y_b[i] > 0)[0]
                if len(pos_idx) == 0: continue
                # Hard if min prediction on positives is below 0.3
                if probs[i, pos_idx].min() < 0.3:
                    hard_mask[s + i] = True
    return hard_mask


def main():
    print("=== exp117: Iterative SFT (boosting on hard rows) ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
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
    Y_train_combined = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    base_src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])

    print(f"  Train: {len(X_train)} rows ({valid.sum()} TA + {tr_mask.sum()} SS)\n")

    # Round 0: BCE baseline
    print("=== Round 0: BCE baseline ===", flush=True)
    model, macro_0, ev_pred_0 = train_bce_with_row_weights(
        X_train, Y_train_combined, base_src_w, perch_emb[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=15
    )
    print(f"  Round 0 best macro: {macro_0:.4f}")
    pt_0 = per_taxon_macro(Y_ss_ev, ev_pred_0, sp_taxon)
    print(f"    per-taxon: " + ", ".join(f"{t} {pt_0.get(t, float('nan')):.4f}" for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]))

    results = [{"round": 0, "macro": macro_0, "n_hard": 0,
                **{t: pt_0.get(t, float('nan')) for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}}]

    # Rounds 1-3: identify hard rows, up-weight, retrain
    current_src_w = base_src_w.copy()
    current_state = {k: v.clone() for k, v in model.state_dict().items()}

    for round_k in range(1, 4):
        print(f"\n=== Round {round_k}: identify hard rows + up-weight ===", flush=True)
        hard_mask = identify_hard_rows(model, X_train, Y_train_combined)
        n_hard = hard_mask.sum()
        print(f"  Hard rows (missed positive @ <0.3): {n_hard:,} / {len(X_train):,} ({100*n_hard/len(X_train):.1f}%)")

        # Up-weight hard rows by 3x relative to baseline
        new_src_w = current_src_w.copy()
        new_src_w[hard_mask] *= 3.0

        # Train: try both fresh init AND continued from previous
        print(f"  Training (fresh init from W_init)...", flush=True)
        model_fresh, macro_fresh, ev_pred_fresh = train_bce_with_row_weights(
            X_train, Y_train_combined, new_src_w, perch_emb[ev_mask], Y_ss_ev,
            W_init, b_init, init_state=None, n_epochs=12
        )
        pt_fresh = per_taxon_macro(Y_ss_ev, ev_pred_fresh, sp_taxon)
        print(f"    fresh: macro {macro_fresh:.4f}, "
              f"Aves {pt_fresh.get('Aves', float('nan')):.4f}, "
              f"Amphib {pt_fresh.get('Amphibia', float('nan')):.4f}, "
              f"Insecta {pt_fresh.get('Insecta', float('nan')):.4f}, "
              f"Mam {pt_fresh.get('Mammalia', float('nan')):.4f}")

        print(f"  Training (continue from previous state, lower lr)...", flush=True)
        model_cont, macro_cont, ev_pred_cont = train_bce_with_row_weights(
            X_train, Y_train_combined, new_src_w, perch_emb[ev_mask], Y_ss_ev,
            W_init, b_init, init_state=current_state, n_epochs=8, lr=2e-4
        )
        pt_cont = per_taxon_macro(Y_ss_ev, ev_pred_cont, sp_taxon)
        print(f"    cont: macro {macro_cont:.4f}, "
              f"Aves {pt_cont.get('Aves', float('nan')):.4f}, "
              f"Amphib {pt_cont.get('Amphibia', float('nan')):.4f}, "
              f"Insecta {pt_cont.get('Insecta', float('nan')):.4f}, "
              f"Mam {pt_cont.get('Mammalia', float('nan')):.4f}")

        # Use better of fresh/cont as next-round start
        if macro_fresh >= macro_cont:
            model, macro_k, ev_pred_k = model_fresh, macro_fresh, ev_pred_fresh
            chose = "fresh"
        else:
            model, macro_k, ev_pred_k = model_cont, macro_cont, ev_pred_cont
            chose = "cont"
        pt_k = per_taxon_macro(Y_ss_ev, ev_pred_k, sp_taxon)

        results.append({"round": round_k, "macro": macro_k, "n_hard": int(n_hard),
                          "chose": chose,
                          **{t: pt_k.get(t, float('nan')) for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}})

        current_src_w = new_src_w
        current_state = {k: v.clone() for k, v in model.state_dict().items()}

    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
