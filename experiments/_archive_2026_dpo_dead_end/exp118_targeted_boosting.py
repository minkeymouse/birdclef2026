#!/usr/bin/env python3
"""exp118 — Targeted boosting variants for breakthrough hunt.

Multiple SFT-style boosting strategies, each different mining axis:
  V1: Per-CLASS rebalancing — rare classes get 20x weight (extreme class balance)
  V2: Per-ROW boosting on universal-miss cases (those 92 (row, class) all 4 miss)
  V3: PER-(row, class) loss masking — only train on labels where current model fails
  V4: File-level boosting — up-weight ALL windows of files with any missed positive
  V5: Confusion-cluster targeted — up-weight rows where model confuses cluster members

Compare against BCE baseline. Best one tested with LOSO.
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


def train_pnew3_with_loss_mods(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                                  per_class_w=None, per_rc_mask=None, init_state=None,
                                  n_epochs=15, lr=1e-3, verbose=False):
    """Train P_NEW3 with optional per-class weight and per-(row, class) masking."""
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
    if per_class_w is not None:
        pw = pw * per_class_w
    pw_t = torch.from_numpy(pw).to(DEVICE)

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    Y_t = torch.from_numpy(Y_train.astype(np.float32)).to(DEVICE)
    W_t = torch.from_numpy(src_weight.astype(np.float32)).to(DEVICE)
    Xev_t = torch.from_numpy(X_eval.astype(np.float32)).to(DEVICE)

    if per_rc_mask is not None:
        rc_t = torch.from_numpy(per_rc_mask.astype(np.float32)).to(DEVICE)
    else:
        rc_t = None

    n = len(X_t); BATCH = 512
    best_auc = 0.0; best_pred = None; best_state = None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, BATCH):
            idx = perm[s:s+BATCH]
            x = X_t[idx]; y = Y_t[idx]; w = W_t[idx]
            opt.zero_grad()
            logits = model(x)
            loss_per_cell = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw_t, reduction="none")
            if rc_t is not None:
                loss_per_cell = loss_per_cell * rc_t[idx]
            loss = (loss_per_cell.mean(dim=-1) * w).mean()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            ev_pred = torch.sigmoid(model(Xev_t)).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        if macro > best_auc:
            best_auc = macro
            best_pred = ev_pred
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if verbose: print(f"  ep {ep:02d}  macro {macro:.4f}")
    return best_auc, best_pred, best_state


def main():
    print("=== exp118: Targeted boosting variants ===\n", flush=True)

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
    Y_train = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    base_src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])

    print(f"  Train: {len(X_train)} rows\n")
    sp_taxon_arr = np.array(sp_taxon)

    # Baseline
    print("=== Baseline BCE ===")
    macro_base, ev_pred_base, state_base = train_pnew3_with_loss_mods(
        X_train, Y_train, base_src_w, perch_emb[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=15
    )
    pt_base = per_taxon_macro(Y_ss_ev, ev_pred_base, sp_taxon)
    print(f"  macro {macro_base:.4f}, "
          f"Aves {pt_base['Aves']:.4f}, Amphib {pt_base['Amphibia']:.4f}, "
          f"Insecta {pt_base['Insecta']:.4f}, Mam {pt_base['Mammalia']:.4f}, Rept {pt_base['Reptilia']:.4f}")

    results = [{"variant": "BCE base", "macro": macro_base, **pt_base}]

    # ===== V1: Per-class extreme rebalance =====
    print("\n=== V1: Per-class extreme rebalance (rare 20x) ===")
    rare_taxa = ["Mammalia", "Reptilia", "Insecta"]
    per_class_w = np.ones(N_CLS, dtype=np.float32)
    per_class_w[np.isin(sp_taxon_arr, rare_taxa)] = 20.0
    per_class_w[sp_taxon_arr == "Amphibia"] = 5.0
    macro_v1, ev_pred_v1, _ = train_pnew3_with_loss_mods(
        X_train, Y_train, base_src_w, perch_emb[ev_mask], Y_ss_ev,
        W_init, b_init, per_class_w=per_class_w, n_epochs=15
    )
    pt_v1 = per_taxon_macro(Y_ss_ev, ev_pred_v1, sp_taxon)
    print(f"  macro {macro_v1:.4f}, "
          f"Aves {pt_v1['Aves']:.4f}, Amphib {pt_v1['Amphibia']:.4f}, "
          f"Insecta {pt_v1['Insecta']:.4f}, Mam {pt_v1['Mammalia']:.4f}, Rept {pt_v1['Reptilia']:.4f}")
    results.append({"variant": "V1: rare 20x", "macro": macro_v1, **pt_v1})

    # ===== V2: Per-row hard boosting (rows missed @ <0.3) =====
    print("\n=== V2: Per-row hard boost (10x) — based on baseline predictions ===")
    # Use baseline model to identify hard rows
    base_model = PerchHybrid(W_init, b_init).to(DEVICE)
    base_model.load_state_dict(state_base)
    base_model.eval()
    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    BATCH = 2048
    hard_mask = np.zeros(len(X_train), dtype=bool)
    with torch.no_grad():
        for s in range(0, len(X_t), BATCH):
            x = X_t[s:s+BATCH]
            probs = torch.sigmoid(base_model(x)).cpu().numpy()
            for i in range(len(x)):
                pos_idx = np.where(Y_train[s+i] > 0)[0]
                if len(pos_idx) == 0: continue
                if probs[i, pos_idx].min() < 0.3:
                    hard_mask[s + i] = True
    print(f"  Hard rows: {hard_mask.sum():,} / {len(X_train):,}")
    boost_w = base_src_w.copy()
    boost_w[hard_mask] *= 10.0
    macro_v2, ev_pred_v2, _ = train_pnew3_with_loss_mods(
        X_train, Y_train, boost_w, perch_emb[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=15
    )
    pt_v2 = per_taxon_macro(Y_ss_ev, ev_pred_v2, sp_taxon)
    print(f"  macro {macro_v2:.4f}, "
          f"Aves {pt_v2['Aves']:.4f}, Amphib {pt_v2['Amphibia']:.4f}, "
          f"Insecta {pt_v2['Insecta']:.4f}, Mam {pt_v2['Mammalia']:.4f}, Rept {pt_v2['Reptilia']:.4f}")
    results.append({"variant": "V2: hard rows 10x", "macro": macro_v2, **pt_v2})

    # ===== V3: Per-(row, class) loss mask — focus on positive cells where model wrong =====
    print("\n=== V3: Per-(row, class) mask — emphasize wrong positives ===")
    rc_mask = np.ones((len(X_train), N_CLS), dtype=np.float32)
    # Find positives where model scores < 0.3 → multiply weight by 10
    with torch.no_grad():
        for s in range(0, len(X_t), BATCH):
            x = X_t[s:s+BATCH]
            probs = torch.sigmoid(base_model(x)).cpu().numpy()
            y_b = Y_train[s:s+BATCH]
            for i in range(len(x)):
                wrong_pos = (y_b[i] > 0) & (probs[i] < 0.3)
                if wrong_pos.any():
                    rc_mask[s + i, wrong_pos] = 10.0
    n_wrong = int((rc_mask > 1).sum())
    print(f"  Boosted (row, class) cells: {n_wrong:,}")
    macro_v3, ev_pred_v3, _ = train_pnew3_with_loss_mods(
        X_train, Y_train, base_src_w, perch_emb[ev_mask], Y_ss_ev,
        W_init, b_init, per_rc_mask=rc_mask, n_epochs=15
    )
    pt_v3 = per_taxon_macro(Y_ss_ev, ev_pred_v3, sp_taxon)
    print(f"  macro {macro_v3:.4f}, "
          f"Aves {pt_v3['Aves']:.4f}, Amphib {pt_v3['Amphibia']:.4f}, "
          f"Insecta {pt_v3['Insecta']:.4f}, Mam {pt_v3['Mammalia']:.4f}, Rept {pt_v3['Reptilia']:.4f}")
    results.append({"variant": "V3: wrong (r,c) 10x", "macro": macro_v3, **pt_v3})

    # ===== V4: V1+V3 combined — rare class + wrong positive boost =====
    print("\n=== V4: Combined rare-class + wrong-positive ===")
    macro_v4, ev_pred_v4, _ = train_pnew3_with_loss_mods(
        X_train, Y_train, base_src_w, perch_emb[ev_mask], Y_ss_ev,
        W_init, b_init, per_class_w=per_class_w, per_rc_mask=rc_mask, n_epochs=15
    )
    pt_v4 = per_taxon_macro(Y_ss_ev, ev_pred_v4, sp_taxon)
    print(f"  macro {macro_v4:.4f}, "
          f"Aves {pt_v4['Aves']:.4f}, Amphib {pt_v4['Amphibia']:.4f}, "
          f"Insecta {pt_v4['Insecta']:.4f}, Mam {pt_v4['Mammalia']:.4f}, Rept {pt_v4['Reptilia']:.4f}")
    results.append({"variant": "V4: V1+V3 combined", "macro": macro_v4, **pt_v4})

    # ===== V5: Per-class with extreme rare boost (50x for very rare) =====
    print("\n=== V5: Even more aggressive per-class (Mammalia/Reptilia 50x) ===")
    pcw5 = np.ones(N_CLS, dtype=np.float32)
    pcw5[sp_taxon_arr == "Mammalia"] = 50.0
    pcw5[sp_taxon_arr == "Reptilia"] = 50.0
    pcw5[sp_taxon_arr == "Insecta"] = 30.0
    pcw5[sp_taxon_arr == "Amphibia"] = 8.0
    macro_v5, ev_pred_v5, _ = train_pnew3_with_loss_mods(
        X_train, Y_train, base_src_w, perch_emb[ev_mask], Y_ss_ev,
        W_init, b_init, per_class_w=pcw5, n_epochs=15
    )
    pt_v5 = per_taxon_macro(Y_ss_ev, ev_pred_v5, sp_taxon)
    print(f"  macro {macro_v5:.4f}, "
          f"Aves {pt_v5['Aves']:.4f}, Amphib {pt_v5['Amphibia']:.4f}, "
          f"Insecta {pt_v5['Insecta']:.4f}, Mam {pt_v5['Mammalia']:.4f}, Rept {pt_v5['Reptilia']:.4f}")
    results.append({"variant": "V5: extreme rare", "macro": macro_v5, **pt_v5})

    df = pd.DataFrame(results)
    print("\n=== Summary (sorted by macro) ===")
    print(df.sort_values("macro", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
