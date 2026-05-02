#!/usr/bin/env python3
"""exp116 — RLAIF (Reinforcement Learning from AI Feedback) on unlabeled SS.

First experiment in our pipeline that USES the 127k unlabeled SS rows for
training signal (not just BG mixing).

Method:
  1. Train BCE reference P_NEW3 on TA + labeled SS → π_ref
  2. Run π_ref + Perch on 127k unlabeled SS → ensemble = mean(Perch, P_NEW3)
  3. Mine HIGH-AGREEMENT preferences:
      - For each row, if ensemble[c] > 0.7 AND π_ref[c] < 0.5 → push toward c
      - If ensemble[c] < 0.1 AND π_ref[c] > 0.5 → push away from c
     These are (preferred, rejected) class pairs WHERE Perch and π_ref agree
     but the model needs to update.
  4. DPO training on synthetic preferences (β=1.0, anchored to π_ref)
  5. Eval on 122 labeled SS (with ground truth)

Differences from earlier experiments:
  - First time using unlabeled data as training signal
  - Soft preferences (not hard pseudo-labels) — mitigates risk noted in
    CLAUDE.md ("pseudo-labeling on unlabeled SS is risky")
  - Ensemble agreement provides built-in noise filter
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from exp106_pnew_hybrid import build_perch_init, PerchHybrid
from exp113_pnew3_dpo import train_bce_reference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_pnew3_on(model, X, batch_size=2048):
    """Run P_NEW3 inference on X, return sigmoid scores."""
    model.eval()
    out = np.zeros((len(X), N_CLS), dtype=np.float32)
    X_t = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            x_b = X_t[s:s+batch_size].to(DEVICE)
            logits = model(x_b)
            out[s:s+batch_size] = torch.sigmoid(logits).cpu().numpy()
    return out


def mine_ensemble_preferences(ensemble, ref_pred, hi=0.7, lo=0.1, ref_thresh=0.5,
                                 max_per_row=20):
    """Mine (row, push_toward_class, push_away_class) triplets from ensemble.

    For each row:
      - Class A: ensemble[A] > hi AND ref_pred[A] < ref_thresh → preferred
      - Class B: ensemble[B] < lo AND ref_pred[B] > ref_thresh → rejected
    Pair (A, B) becomes (preferred, rejected).
    """
    n_rows = len(ensemble)
    triplets = []
    for i in range(n_rows):
        e = ensemble[i]
        r = ref_pred[i]
        # Confident preferred classes (ensemble says yes, ref says no)
        pref_mask = (e > hi) & (r < ref_thresh)
        # Confident rejected classes (ensemble says no, ref says yes)
        rej_mask = (e < lo) & (r > ref_thresh)
        pref_idx = np.where(pref_mask)[0]
        rej_idx = np.where(rej_mask)[0]
        if len(pref_idx) == 0 or len(rej_idx) == 0:
            continue
        # All pairs (pref, rej)
        for p in pref_idx:
            for q in rej_idx:
                triplets.append((i, int(p), int(q)))
                if len(triplets) % 1_000_000 == 0:
                    print(f"  ... {len(triplets):,} triplets")
        # Cap per-row pairs by sub-sampling
    # Sub-sample if too many
    if len(triplets) > 5_000_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(triplets), 5_000_000, replace=False)
        triplets = [triplets[i] for i in idx]
    return triplets


def train_dpo_on_unlabeled(X_unlab, triplets, X_eval, Y_eval, W_init, b_init,
                              ref_model, beta=1.0, n_epochs=4, lr=2e-4, verbose=True):
    """DPO with unlabeled rows + synthetic preferences."""
    policy = PerchHybrid(W_init, b_init).to(DEVICE)
    policy.load_state_dict(ref_model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/10)

    X_unlab_t = torch.from_numpy(X_unlab.astype(np.float32))
    Xev_t = torch.from_numpy(X_eval.astype(np.float32)).to(DEVICE)

    rows_t = torch.tensor([t[0] for t in triplets], dtype=torch.long)
    pos_t = torch.tensor([t[1] for t in triplets], dtype=torch.long)
    neg_t = torch.tensor([t[2] for t in triplets], dtype=torch.long)

    n_triplets = len(triplets)
    BATCH = 2048
    best_auc = 0.0; best_pred = None; best_ep = -1

    policy.eval()
    with torch.no_grad():
        ev_pred0 = torch.sigmoid(policy(Xev_t)).cpu().numpy()
    macro0, _ = macro_auc(Y_eval, ev_pred0)
    best_auc = macro0; best_pred = ev_pred0; best_ep = -1
    if verbose: print(f"  ep -1 (init)  macro {macro0:.4f}, n_triplets={n_triplets}")

    for ep in range(n_epochs):
        policy.train()
        perm = torch.randperm(n_triplets)
        ep_loss = 0.0; nb = 0
        for s in range(0, n_triplets, BATCH):
            idx = perm[s:s+BATCH]
            r = rows_t[idx]
            p_cls = pos_t[idx].to(DEVICE)
            n_cls = neg_t[idx].to(DEVICE)
            x = X_unlab_t[r].to(DEVICE)
            logits_p = policy(x)
            with torch.no_grad():
                logits_r = ref_model(x)
            log_p_pos = F.logsigmoid(logits_p.gather(1, p_cls.unsqueeze(1)).squeeze(1))
            log_p_neg = F.logsigmoid(logits_p.gather(1, n_cls.unsqueeze(1)).squeeze(1))
            log_r_pos = F.logsigmoid(logits_r.gather(1, p_cls.unsqueeze(1)).squeeze(1))
            log_r_neg = F.logsigmoid(logits_r.gather(1, n_cls.unsqueeze(1)).squeeze(1))
            margin = beta * ((log_p_pos - log_r_pos) - (log_p_neg - log_r_neg))
            loss = -F.logsigmoid(margin).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        policy.eval()
        with torch.no_grad():
            ev_pred = torch.sigmoid(policy(Xev_t)).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        if macro > best_auc:
            best_auc = macro; best_pred = ev_pred; best_ep = ep
        if verbose: print(f"  ep {ep:02d}  loss {ep_loss/nb:.4f}  macro {macro:.4f}")

    return best_auc, best_pred, macro0, best_ep, policy


def main():
    print("=== exp116: RLAIF — Ensemble Feedback DPO on unlabeled SS ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_lab = load_perch_emb_labeled()
    perch_prob_lab = load_perch_scores_labeled()

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

    # Step 1: Train BCE reference
    print("=== Step 1: BCE reference ===", flush=True)
    X_train = np.concatenate([ta_emb[valid], perch_emb_lab[tr_mask]], axis=0)
    Y_train = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])
    ref_model, ref_macro = train_bce_reference(
        X_train, Y_train, src_w, perch_emb_lab[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=15
    )
    print(f"  BCE ref macro: {ref_macro:.4f}\n")

    # Step 2: Load unlabeled SS
    print("=== Step 2: Load unlabeled SS Perch features ===", flush=True)
    unlab = np.load(ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz", mmap_mode="r")
    unlab_emb = unlab["emb"]; unlab_perch_scores = unlab["scores"]
    print(f"  Unlabeled SS: {unlab_emb.shape} embeddings, {unlab_perch_scores.shape} Perch scores")

    # Step 3: Run P_NEW3 on unlabeled
    print("\n=== Step 3: Run P_NEW3 on unlabeled SS ===", flush=True)
    pnew3_unlab = run_pnew3_on(ref_model, unlab_emb, batch_size=2048)
    print(f"  P_NEW3 unlabeled scores: {pnew3_unlab.shape}, range [{pnew3_unlab.min():.4f}, {pnew3_unlab.max():.4f}]")

    # Step 4: Build ensemble = mean(Perch, P_NEW3)
    print("\n=== Step 4: Mine ensemble-feedback preferences ===", flush=True)
    ensemble = (np.asarray(unlab_perch_scores) + pnew3_unlab) / 2

    # Sweep different (hi, lo) thresholds to see triplet counts
    for hi in [0.5, 0.7]:
        for lo in [0.05, 0.1]:
            triplets_quick = []
            n_check = 5000  # quick scan first 5k rows for sanity
            for i in range(n_check):
                e = ensemble[i]; r = pnew3_unlab[i]
                pref = np.where((e > hi) & (r < 0.5))[0]
                rej = np.where((e < lo) & (r > 0.5))[0]
                triplets_quick.append(len(pref) * len(rej))
            avg_pairs = np.mean(triplets_quick)
            print(f"  hi={hi}, lo={lo}: avg {avg_pairs:.1f} pairs/row (~{int(avg_pairs * len(unlab_emb) / 1e6)}M total)")

    # Use moderate threshold
    print("\n  Mining with hi=0.5, lo=0.1...")
    triplets = []
    n_total = len(unlab_emb)
    for i in range(n_total):
        e = ensemble[i]; r = pnew3_unlab[i]
        pref = np.where((e > 0.5) & (r < 0.5))[0]
        rej = np.where((e < 0.1) & (r > 0.5))[0]
        if len(pref) == 0 or len(rej) == 0: continue
        # Sample at most 5 (pref, rej) pairs per row
        for p in pref[:5]:
            for q in rej[:3]:
                triplets.append((i, int(p), int(q)))
        if i % 20000 == 0 and i > 0:
            print(f"    {i}/{n_total} rows scanned, {len(triplets):,} triplets so far", flush=True)

    print(f"\n  Total triplets: {len(triplets):,}")
    # Cap to manageable size
    if len(triplets) > 2_000_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(triplets), 2_000_000, replace=False)
        triplets = [triplets[i] for i in idx]
        print(f"  Sub-sampled to {len(triplets):,}")

    if len(triplets) < 1000:
        print("  Too few triplets — no DPO training possible.")
        return

    # Distribution of preferred / rejected classes
    pref_classes = [t[1] for t in triplets]
    rej_classes = [t[2] for t in triplets]
    from collections import Counter
    top_pref = Counter(pref_classes).most_common(10)
    top_rej = Counter(rej_classes).most_common(10)
    print(f"\n  Top preferred classes: {[(primary[c], n) for c, n in top_pref]}")
    print(f"  Top rejected classes: {[(primary[c], n) for c, n in top_rej]}")

    # Step 5: DPO training
    print("\n=== Step 5: DPO with ensemble-feedback preferences ===", flush=True)
    for beta in [0.3, 1.0, 3.0]:
        print(f"\n--- β={beta} ---")
        best, ev_pred, _, best_ep, _ = train_dpo_on_unlabeled(
            unlab_emb, triplets, perch_emb_lab[ev_mask], Y_ss_ev,
            W_init, b_init, ref_model, beta=beta, n_epochs=4, verbose=True
        )
        pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
        print(f"  best macro {best:.4f} (Δ vs ref {best-ref_macro:+.4f})")
        print(f"    Aves {pt.get('Aves', float('nan')):.4f}, Amphib {pt.get('Amphibia', float('nan')):.4f}, "
              f"Insecta {pt.get('Insecta', float('nan')):.4f}, Mam {pt.get('Mammalia', float('nan')):.4f}")


if __name__ == "__main__":
    main()
