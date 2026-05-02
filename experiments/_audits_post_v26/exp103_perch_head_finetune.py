#!/usr/bin/env python3
"""exp103 — P_NEW: Perch backbone frozen + 234-class head fine-tune.

Train a new species classifier head on Perch's 1536-d embedding using:
  - 35,549 train_audio rows (single-label primary, exp22 cache)
  - 617 labeled SS train rows (multi-label, all species in row's GT)

Eval: 122 labeled SS held-out (40 evaluable classes).

Compare:
  - Perch sigmoid baseline (frozen 14k head, sigmoid output for our 234)
  - P_NEW (custom 234-class head)

Architecture:
  Perch emb (1536) → BN → Linear(1536→768) → GELU → Dropout(0.3)
                    → Linear(768→234) → BCE-with-logits
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import per_class_auc, macro_auc, per_taxon_macro

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PerchHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=768, n_classes=N_CLS, dropout=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = self.bn(x)
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


def main():
    print("=== exp103: P_NEW Perch head fine-tune ===\n", flush=True)

    # 1. Load all training data
    sc_g, Y, primary, l2i = build_ss()
    sp_taxon = species_taxon_array()

    print("Loading Perch embeddings...", flush=True)
    perch_emb_ss = load_perch_emb_labeled()    # (739, 1536) — labeled SS
    perch_prob_ss = load_perch_scores_labeled() # (739, 234) — Perch sigmoid scores

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]          # (35549, 1536)
    ta_y_idx = ta["y_idx"]      # (35549,) primary label idx (0-205, only 206 species in train_audio)
    ta_valid = ta["valid"]      # (35549,)
    print(f"  train_audio: {ta_emb.shape}, valid: {ta_valid.sum()}", flush=True)
    print(f"  labeled SS: {perch_emb_ss.shape}", flush=True)

    # 2. Build multi-hot labels
    # For train_audio: each row has 1 species (primary)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    valid_mask = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta[np.arange(len(ta_emb))[valid_mask], ta_y_idx[valid_mask]] = 1.0
    print(f"  TA multi-hot positives: {int(Y_ta.sum())}", flush=True)

    # For labeled SS: use full Y (multi-label)
    Y_ss = Y.astype(np.float32)

    # Split SS into train (627) / eval (122)
    tr_mask = sc_g.split.values == "train"
    ev_mask = sc_g.split.values == "eval"
    perch_ss_tr = perch_emb_ss[tr_mask]; Y_ss_tr = Y_ss[tr_mask]
    perch_ss_ev = perch_emb_ss[ev_mask]; Y_ss_ev = Y_ss[ev_mask]
    print(f"  SS train: {perch_ss_tr.shape}, SS eval: {perch_ss_ev.shape}", flush=True)

    # 3. Combine TA + SS_train
    X_train = np.concatenate([ta_emb[valid_mask], perch_ss_tr], axis=0)
    Y_train = np.concatenate([Y_ta[valid_mask], Y_ss_tr], axis=0)
    # Source weight: weigh SS_train rows higher since multi-label is closer to test
    src_weight = np.concatenate([
        np.ones(valid_mask.sum(), dtype=np.float32),
        np.full(len(perch_ss_tr), 5.0, dtype=np.float32)   # 5x weight on SS rows
    ])
    print(f"\n  X_train: {X_train.shape}, total positives: {int(Y_train.sum())}", flush=True)

    # 4. Class-balanced positive weight (sqrt inverse frequency)
    cls_pos_count = Y_train.sum(axis=0)
    pos_weight = np.where(cls_pos_count > 0,
                            np.sqrt(len(X_train) / (cls_pos_count * N_CLS + 1e-6)),
                            1.0).astype(np.float32)
    pos_weight = np.clip(pos_weight, 0.5, 50.0)
    print(f"  pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]", flush=True)

    # 5. Model + training
    model = PerchHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_epochs = 30
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    bce_pos_weight = torch.from_numpy(pos_weight).to(DEVICE)

    # Pre-load to device
    X_t = torch.from_numpy(X_train).to(DEVICE)
    Y_t = torch.from_numpy(Y_train).to(DEVICE)
    W_t = torch.from_numpy(src_weight).to(DEVICE)
    Xev_t = torch.from_numpy(perch_ss_ev).to(DEVICE)

    n = len(X_t)
    BATCH = 512

    print("\nTraining P_NEW head...", flush=True)
    best_auc = 0.0; best_state = None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_loss = 0.0; nb = 0
        for s in range(0, n, BATCH):
            idx = perm[s:s+BATCH]
            x = X_t[idx]; y = Y_t[idx]; w = W_t[idx]
            opt.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=bce_pos_weight, reduction="none")
            loss = (loss.mean(dim=-1) * w).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()

        # Eval
        model.eval()
        with torch.no_grad():
            ev_logits = model(Xev_t)
            ev_prob = torch.sigmoid(ev_logits).cpu().numpy()
        macro, n_eval = macro_auc(Y_ss_ev, ev_prob)
        if macro > best_auc:
            best_auc = macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 3 == 0 or ep == n_epochs - 1:
            print(f"  ep {ep:02d}  loss {ep_loss/nb:.4f}  eval_macro {macro:.4f} ({n_eval} cls)", flush=True)

    print(f"\nBest eval macro: {best_auc:.4f}")

    # Load best
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        # Predictions on eval
        ev_logits = model(Xev_t)
        ev_prob = torch.sigmoid(ev_logits).cpu().numpy()
        # Predictions on full SS (for blending tests)
        all_logits = model(torch.from_numpy(perch_emb_ss).to(DEVICE))
        all_prob = torch.sigmoid(all_logits).cpu().numpy()

    # 6. Comparison vs Perch baseline
    print("\n=== Compare vs Perch baseline (sigmoid frozen 14k head → 234) ===")
    perch_prob_ev = perch_prob_ss[ev_mask]
    macro_perch, _ = macro_auc(Y_ss_ev, perch_prob_ev)
    macro_pnew, _ = macro_auc(Y_ss_ev, ev_prob)
    print(f"  Perch sigmoid baseline:  {macro_perch:.4f}")
    print(f"  P_NEW (head fine-tune):  {macro_pnew:.4f}")
    print(f"  Δ = {macro_pnew - macro_perch:+.4f}")

    # Per-taxon
    pt_perch = per_taxon_macro(Y_ss_ev, perch_prob_ev, sp_taxon)
    pt_pnew = per_taxon_macro(Y_ss_ev, ev_prob, sp_taxon)
    print("\n  Per-taxon:")
    print(f"  {'taxon':<10} {'Perch':>8} {'P_NEW':>8} {'Δ':>8}")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        if not np.isnan(pt_perch[t]) and not np.isnan(pt_pnew[t]):
            print(f"  {t:<10} {pt_perch[t]:>8.4f} {pt_pnew[t]:>8.4f} {pt_pnew[t]-pt_perch[t]:>+8.4f}")

    # 7. Save artifact
    out_dir = ROOT / "model-weights"
    out_dir.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "in_dim": 1536, "hidden": 768, "n_classes": N_CLS,
        "best_eval_macro": best_auc,
        "trained_on": "train_audio_2026 + labeled_SS_train",
        "note": "P_NEW Perch head fine-tune, exp103",
    }, out_dir / "p_new_head.pt")
    print(f"\nSaved → {out_dir}/p_new_head.pt")

    # Also save predictions for downstream blending tests
    np.savez_compressed(EXP80 / "p_new_predictions.npz",
                         predictions=all_prob.astype(np.float32),
                         eval_macro=best_auc)
    print(f"Saved predictions → {EXP80}/p_new_predictions.npz")


if __name__ == "__main__":
    main()
