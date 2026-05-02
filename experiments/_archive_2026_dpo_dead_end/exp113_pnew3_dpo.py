#!/usr/bin/env python3
"""exp113 — P_NEW3 with DPO-style preference learning.

DPO (Direct Preference Optimization) framing for multi-label classification:
  - Reference model π_ref: P_NEW3 trained with BCE (frozen)
  - Policy model π_θ: same architecture, init from π_ref
  - For each row x with positive species c+ and negative species c-,
    interpret as preference: c+ ≻ c- (preferred over rejected)
  - DPO loss with multi-label formulation (log-sigmoid per class):
      r_θ(x, c) = β * (log_sigmoid(logit_θ[c]) - log_sigmoid(logit_ref[c]))
      Loss = -log_sigmoid(r_θ(x, c+) - r_θ(x, c-))
  - β controls how far policy can deviate from reference (KL regularization)

Why this might differ from plain ranking (exp112 row_pair):
  - Anchored to reference: policy can't drift far from BCE solution
  - Implicit KL constraint preserves Aves performance (where BCE was already
    strong — exp112 row_pair hurt Aves AUC by 0.05)
  - Symmetric in (c+, c-) but anchored — like LoRA-style fine-tune of preferences

Compared to exp112 row_pair:
  exp112 row_pair: pure pairwise, no anchor → drifted away from good Aves solution
  exp113 DPO: anchored to BCE, only changes where preferences require → preserves Aves
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from exp106_pnew_hybrid import build_perch_init, PerchHybrid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_bce_reference(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                          n_epochs=15, lr=1e-3):
    """Standard BCE training to produce reference model π_ref."""
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
    best_auc = 0.0; best_state = None
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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_auc


def dpo_loss(logits_policy, logits_ref, targets, beta=0.1, n_pairs_per_row=8):
    """DPO loss for multi-label classification.

    For each row, sample n_pairs (c+, c-) where c+ is positive and c- is negative.
    Compute log_pi(c|x) = log_sigmoid(logit[c]) (per-class binary).
    DPO reward: r(x,c) = β * (log_pi_θ(c|x) - log_pi_ref(c|x))
    Loss: -log_sigmoid(r_θ(x,c+) - r_θ(x,c-))
    """
    B, C = logits_policy.shape
    losses = []
    for i in range(B):
        pos_mask = targets[i] > 0
        neg_mask = targets[i] == 0
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)
        neg_idx = neg_mask.nonzero(as_tuple=False).squeeze(-1)

        # Sample pairs
        K = min(n_pairs_per_row, n_pos * n_neg)
        # Sample with replacement for simplicity
        ci_plus = pos_idx[torch.randint(n_pos, (K,), device=logits_policy.device)]
        ci_minus = neg_idx[torch.randint(n_neg, (K,), device=logits_policy.device)]

        log_pi_p_plus = F.logsigmoid(logits_policy[i, ci_plus])
        log_pi_p_minus = F.logsigmoid(logits_policy[i, ci_minus])
        log_pi_r_plus = F.logsigmoid(logits_ref[i, ci_plus])
        log_pi_r_minus = F.logsigmoid(logits_ref[i, ci_minus])

        # r_θ(x, c) = β * (log π_θ(c|x) - log π_ref(c|x))
        # Loss = -log_sigmoid(r_θ(x, c+) - r_θ(x, c-))
        margin = beta * ((log_pi_p_plus - log_pi_r_plus) - (log_pi_p_minus - log_pi_r_minus))
        loss_i = -F.logsigmoid(margin).mean()
        losses.append(loss_i)
    if not losses:
        return logits_policy.sum() * 0.0
    return torch.stack(losses).mean()


def train_dpo(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                 ref_model, beta=0.1, n_epochs=10, lr=5e-4, verbose=False):
    """Train policy with DPO loss, anchored to ref_model."""
    # Initialize policy from reference
    policy = PerchHybrid(W_init, b_init).to(DEVICE)
    policy.load_state_dict(ref_model.state_dict())

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/10)

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    Y_t = torch.from_numpy(Y_train.astype(np.float32)).to(DEVICE)
    Xev_t = torch.from_numpy(X_eval.astype(np.float32)).to(DEVICE)

    n = len(X_t); BATCH = 256
    best_auc = 0.0; best_pred = None; best_ep = -1

    policy.eval()
    with torch.no_grad():
        ev_pred0 = torch.sigmoid(policy(Xev_t)).cpu().numpy()
    macro0, _ = macro_auc(Y_eval, ev_pred0)
    if verbose: print(f"  ep -1 (ref initial)  eval_macro {macro0:.4f}")

    for ep in range(n_epochs):
        policy.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_loss = 0.0; nb = 0
        for s in range(0, n, BATCH):
            idx = perm[s:s+BATCH]
            x = X_t[idx]; y = Y_t[idx]
            opt.zero_grad()
            logits_p = policy(x)
            with torch.no_grad():
                logits_r = ref_model(x)
            loss = dpo_loss(logits_p, logits_r, y, beta=beta, n_pairs_per_row=8)
            loss.backward()
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        policy.eval()
        with torch.no_grad():
            ev_pred = torch.sigmoid(policy(Xev_t)).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        if macro > best_auc:
            best_auc = macro; best_pred = ev_pred; best_ep = ep
        if verbose:
            print(f"  ep {ep:02d}  loss {ep_loss/nb:.4f}  eval_macro {macro:.4f}")
    return best_auc, best_pred, macro0, best_ep, policy


def main():
    print("=== exp113: P_NEW3 with DPO loss ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()

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
    print(f"  Train: TA {valid.sum()} + SS_train {tr_mask.sum()}")
    print(f"  Eval (122 SS): {ev_mask.sum()}\n")

    # Step 1: Train BCE reference
    print("=== Step 1: Train BCE reference (π_ref) ===", flush=True)
    ref_model, ref_macro = train_bce_reference(
        X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=15
    )
    print(f"  Reference (BCE) eval macro: {ref_macro:.4f}\n")

    pt_ref = per_taxon_macro(Y_ss_ev, torch.sigmoid(ref_model(torch.from_numpy(perch_emb_ss[ev_mask]).to(DEVICE))).detach().cpu().numpy(), sp_taxon)
    print(f"  Per-taxon: Aves {pt_ref['Aves']:.4f}, Amphib {pt_ref['Amphibia']:.4f}, "
          f"Insecta {pt_ref['Insecta']:.4f}, Mam {pt_ref['Mammalia']:.4f}, Rept {pt_ref['Reptilia']:.4f}")

    # Step 2: DPO with various beta
    print("\n=== Step 2: DPO sweep (β controls policy deviation from ref) ===\n", flush=True)
    results = []
    saved_preds = {"reference (BCE)": None}
    # Save reference predictions
    ref_model.eval()
    with torch.no_grad():
        saved_preds["reference (BCE)"] = torch.sigmoid(ref_model(torch.from_numpy(perch_emb_ss[ev_mask]).to(DEVICE))).cpu().numpy()
    pt = pt_ref
    results.append({"variant": "reference (BCE)", "macro": ref_macro, "best_ep": -1,
                     "Aves": pt["Aves"], "Amphibia": pt["Amphibia"], "Insecta": pt["Insecta"],
                     "Mammalia": pt["Mammalia"], "Reptilia": pt["Reptilia"]})

    for beta in [0.05, 0.1, 0.3, 1.0, 3.0]:
        print(f"\n--- DPO β={beta} ---", flush=True)
        best, ev_pred, init_macro, best_ep, _ = train_dpo(
            X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
            W_init, b_init, ref_model, beta=beta, n_epochs=10, verbose=True
        )
        pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
        results.append({"variant": f"DPO β={beta}", "macro": best, "best_ep": best_ep,
                         "Aves": pt["Aves"], "Amphibia": pt["Amphibia"],
                         "Insecta": pt["Insecta"], "Mammalia": pt["Mammalia"],
                         "Reptilia": pt["Reptilia"]})
        saved_preds[f"DPO β={beta}"] = ev_pred

    df = pd.DataFrame(results)
    print("\n=== Summary (122 eval, sorted by macro) ===")
    print(df.sort_values("macro", ascending=False).to_string(index=False))

    # Save best DPO predictions
    dpo_only = df[df.variant.str.startswith("DPO")]
    if len(dpo_only) > 0:
        best_dpo = dpo_only.loc[dpo_only.macro.idxmax()]
        np.savez_compressed(
            EXP80 / "p_new3_dpo_predictions.npz",
            predictions=saved_preds[best_dpo["variant"]].astype(np.float32),
            variant=best_dpo["variant"],
        )
        print(f"\nSaved best DPO predictions ({best_dpo['variant']}) → p_new3_dpo_predictions.npz")


if __name__ == "__main__":
    main()
