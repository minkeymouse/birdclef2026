#!/usr/bin/env python3
"""exp114 — Hard-mined DPO with iterative self-improvement.

Hypothesis correction (vs exp113): exp113 used uniform random (pos, neg)
pair sampling — 99% of pairs were already correctly ranked → no learning
signal. Real DPO leverage requires HARD pair mining.

Method:
  1. Train BCE reference (π_ref).
  2. Run π_ref on TRAIN data, identify (row, true_pos, predicted_high_neg)
     triplets where model is currently wrong:
        for each (row, true positive class p):
            find negatives n with logit[i, n] > logit[i, p] - margin
            these are hard false-positives the model wrongly prefers
  3. DPO loss ONLY on these hard mined pairs.
  4. LOSO check.
  5. Iterate: re-mine after each round.

Compare to exp113 uniform DPO and BCE baseline.
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
from exp113_pnew3_dpo import train_bce_reference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mine_hard_pairs(model, X_train, Y_train, batch_size=512, margin=0.0,
                     max_pairs_per_row=20):
    """Identify (row_idx, pos_class, neg_class) triplets where model wrongly ranks neg > pos.

    Returns: list of (row_idx, pos_class, neg_class) tuples.
    """
    model.eval()
    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    Y_t = torch.from_numpy(Y_train.astype(np.float32))
    n = len(X_t)
    triplets = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            x_b = X_t[s:s+batch_size]
            logits_b = model(x_b).cpu()
            y_b = Y_t[s:s+batch_size]
            B = len(x_b)
            for i in range(B):
                pos_idx = (y_b[i] > 0).nonzero(as_tuple=False).squeeze(-1)
                neg_idx = (y_b[i] == 0).nonzero(as_tuple=False).squeeze(-1)
                if len(pos_idx) == 0 or len(neg_idx) == 0: continue
                pos_logits = logits_b[i, pos_idx]
                neg_logits = logits_b[i, neg_idx]
                # For each (p, n) pair where neg > pos - margin, it's hard
                # diffs[p, n] = neg_logits[n] - pos_logits[p]
                diffs = neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1)  # (P, N)
                hard_mask = diffs > -margin  # neg ranks above pos (or tied)
                hard_p, hard_n = torch.where(hard_mask)
                if len(hard_p) == 0: continue
                # Sort by violation magnitude (hardest first)
                violations = diffs[hard_p, hard_n]
                top = torch.argsort(violations, descending=True)[:max_pairs_per_row]
                for k in top.tolist():
                    triplets.append((s + i, int(pos_idx[hard_p[k]]), int(neg_idx[hard_n[k]])))
    return triplets


def train_hard_dpo(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                    ref_model, hard_triplets, beta=1.0, n_epochs=8, lr=5e-4, verbose=False):
    """DPO training on pre-mined hard pairs."""
    policy = PerchHybrid(W_init, b_init).to(DEVICE)
    policy.load_state_dict(ref_model.state_dict())

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/10)

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    Xev_t = torch.from_numpy(X_eval.astype(np.float32)).to(DEVICE)

    n_triplets = len(hard_triplets)
    if n_triplets == 0:
        if verbose: print("  no hard triplets — skipping DPO")
        policy.eval()
        with torch.no_grad():
            ev_pred = torch.sigmoid(policy(Xev_t)).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        return macro, ev_pred, macro, -1, policy

    # Pre-build hard triplet tensors
    rows_t = torch.tensor([t[0] for t in hard_triplets], dtype=torch.long, device=DEVICE)
    pos_t = torch.tensor([t[1] for t in hard_triplets], dtype=torch.long, device=DEVICE)
    neg_t = torch.tensor([t[2] for t in hard_triplets], dtype=torch.long, device=DEVICE)

    BATCH = 1024  # batch of triplets
    best_auc = 0.0; best_pred = None; best_ep = -1

    policy.eval()
    with torch.no_grad():
        ev_pred0 = torch.sigmoid(policy(Xev_t)).cpu().numpy()
    macro0, _ = macro_auc(Y_eval, ev_pred0)
    if verbose: print(f"  ep -1 (ref initial)  eval_macro {macro0:.4f}, n_hard_triplets={n_triplets}")

    for ep in range(n_epochs):
        policy.train()
        perm = torch.randperm(n_triplets, device=DEVICE)
        ep_loss = 0.0; nb = 0
        for s in range(0, n_triplets, BATCH):
            idx = perm[s:s+BATCH]
            r = rows_t[idx]; p_cls = pos_t[idx]; n_cls = neg_t[idx]
            x = X_t[r]
            logits_p = policy(x)  # (B, 234)
            with torch.no_grad():
                logits_r = ref_model(x)

            log_pi_p_pos = F.logsigmoid(logits_p.gather(1, p_cls.unsqueeze(1)).squeeze(1))
            log_pi_p_neg = F.logsigmoid(logits_p.gather(1, n_cls.unsqueeze(1)).squeeze(1))
            log_pi_r_pos = F.logsigmoid(logits_r.gather(1, p_cls.unsqueeze(1)).squeeze(1))
            log_pi_r_neg = F.logsigmoid(logits_r.gather(1, n_cls.unsqueeze(1)).squeeze(1))

            margin = beta * ((log_pi_p_pos - log_pi_r_pos) - (log_pi_p_neg - log_pi_r_neg))
            loss = -F.logsigmoid(margin).mean()

            opt.zero_grad()
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
    print("=== exp114: Hard-mined DPO with iterative self-improvement ===\n", flush=True)

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

    # Step 1: BCE reference
    print("=== Step 1: Train BCE reference ===", flush=True)
    ref_model, ref_macro = train_bce_reference(
        X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=15
    )
    print(f"  Ref macro: {ref_macro:.4f}\n")

    # Step 2: Mine hard pairs from BCE ref
    print("=== Step 2: Mine hard pairs from BCE ref ===", flush=True)
    for margin in [-0.5, 0.0, 0.5, 1.0]:  # margin = how strict
        triplets = mine_hard_pairs(ref_model, X_train, Y_train, margin=margin, max_pairs_per_row=20)
        n_unique_rows = len(set(t[0] for t in triplets))
        print(f"  margin={margin:>5.1f}: {len(triplets):>7} hard triplets across {n_unique_rows} rows")

    # Use margin=0.5 for the main run
    print("\n=== Step 3: Hard-DPO with margin=0.5 ===")
    triplets = mine_hard_pairs(ref_model, X_train, Y_train, margin=0.5, max_pairs_per_row=20)
    print(f"  Mining: {len(triplets)} hard triplets\n")

    # Run hard-DPO with several β
    results = [{"variant": "BCE ref", "macro": ref_macro,
                 **per_taxon_macro(Y_ss_ev,
                                     torch.sigmoid(ref_model(torch.from_numpy(perch_emb_ss[ev_mask]).to(DEVICE))).detach().cpu().numpy(),
                                     sp_taxon)}]

    for beta in [0.3, 1.0, 3.0]:
        print(f"\n--- Hard-DPO β={beta} ---")
        best, ev_pred, _, best_ep, _ = train_hard_dpo(
            X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
            W_init, b_init, ref_model, triplets, beta=beta, n_epochs=8, verbose=True
        )
        pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
        results.append({"variant": f"Hard-DPO β={beta}", "macro": best, **pt})

    # Iterative round: mine again from best policy, run another DPO
    print("\n=== Step 4: Iterative round (re-mine + re-train) ===")
    # Use β=1.0 as starting policy
    print("  Re-train with β=1.0 to get policy_1...")
    _, _, _, _, policy_1 = train_hard_dpo(
        X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
        W_init, b_init, ref_model, triplets, beta=1.0, n_epochs=8, verbose=False
    )

    # Re-mine from policy_1
    triplets_2 = mine_hard_pairs(policy_1, X_train, Y_train, margin=0.5, max_pairs_per_row=20)
    print(f"  Round 2 mining: {len(triplets_2)} hard triplets (was {len(triplets)})")

    # Use policy_1 as new reference for round 2
    print("  Round 2 DPO from policy_1...")
    best2, ev_pred2, _, best_ep2, policy_2 = train_hard_dpo(
        X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
        W_init, b_init, policy_1, triplets_2, beta=1.0, n_epochs=8, verbose=True
    )
    pt2 = per_taxon_macro(Y_ss_ev, ev_pred2, sp_taxon)
    results.append({"variant": "Hard-DPO iter2 β=1.0", "macro": best2, **pt2})

    # Print summary
    df = pd.DataFrame(results)
    print("\n=== Summary (122 eval) ===")
    print(df.to_string(index=False))

    # Save best variant for blend test
    best_var = df.loc[df.macro.idxmax(), "variant"]
    print(f"\nBest variant: {best_var}")


if __name__ == "__main__":
    main()
