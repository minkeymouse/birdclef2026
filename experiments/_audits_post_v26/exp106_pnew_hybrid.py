#!/usr/bin/env python3
"""exp106 — P_NEW hybrid: frozen Perch-init linear + trainable correction MLP.

logit = perch_init_linear(L2norm(emb)) + correction_mlp(emb)

- Perch branch: Linear(1536, 234) initialized from Perch's ProtoPNet prototypes,
  L2-norm input, FROZEN. Always produces ~Perch baseline.
- Correction branch: BN + 2-layer MLP. Final layer initialized to ZERO.
  At init, correction = 0, model output = Perch baseline (0.622).
- Training only updates the correction MLP. Aves preserved by Perch branch;
  rare-class signal added by correction.
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA = ROOT / "data/birdclef-2026"
PERCH_DIR = ROOT / "perch_v2"
HEAD_NPZ = ROOT / "model-weights/perch_head_extracted.npz"


def build_perch_init():
    head = np.load(HEAD_NPZ)
    W_proto = head["W"]; B = head["B"]; S = head["S"]
    W_eff = (W_proto * S[None, :, :]).sum(axis=-1).astype(np.float32)

    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    species = sorted(taxonomy["primary_label"].astype(str).tolist())
    sp2idx = {s: i for i, s in enumerate(species)}
    sci2pl = dict(zip(taxonomy["scientific_name"], taxonomy["primary_label"]))
    perch_labels = open(PERCH_DIR / "assets" / "labels.csv").read().strip().split("\n")

    mapped_perch_idx = np.full(N_CLS, -1, dtype=np.int64)
    for pi, pname in enumerate(perch_labels):
        if pname in sci2pl and sci2pl[pname] in sp2idx:
            mapped_perch_idx[sp2idx[sci2pl[pname]]] = pi

    rng = np.random.RandomState(42)
    W_init = np.zeros((1536, N_CLS), dtype=np.float32)
    b_init = np.zeros(N_CLS, dtype=np.float32)
    for c in range(N_CLS):
        if mapped_perch_idx[c] >= 0:
            W_init[:, c] = W_eff[:, mapped_perch_idx[c]]
            b_init[c] = B[mapped_perch_idx[c]]
        else:
            bound = np.sqrt(6.0 / 1536)
            W_init[:, c] = rng.uniform(-bound, bound, size=1536)
    return W_init, b_init, mapped_perch_idx


class PerchHybrid(nn.Module):
    def __init__(self, W_init, b_init, hidden=768, dropout=0.3):
        super().__init__()
        # Frozen Perch-init branch (cosine-prototype style)
        self.perch_fc = nn.Linear(1536, N_CLS)
        with torch.no_grad():
            self.perch_fc.weight.copy_(torch.from_numpy(W_init.T))
            self.perch_fc.bias.copy_(torch.from_numpy(b_init))
        for p in self.perch_fc.parameters():
            p.requires_grad_(False)

        # Trainable correction
        self.bn = nn.BatchNorm1d(1536)
        self.fc1 = nn.Linear(1536, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, N_CLS)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1, eps=1e-6)
        perch_logit = self.perch_fc(x_norm)
        h = self.bn(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        delta = self.fc2(h)
        return perch_logit + delta


def train_hybrid(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                  n_epochs=20, lr=1e-3, verbose=False):
    model = PerchHybrid(W_init, b_init).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
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
    best_auc = 0.0; best_pred = None; best_ep = -1

    model.eval()
    with torch.no_grad():
        ev_pred0 = torch.sigmoid(model(Xev_t)).cpu().numpy()
    macro0, _ = macro_auc(Y_eval, ev_pred0)
    if verbose: print(f"  ep -1 (Perch only)  eval_macro {macro0:.4f}")

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
            best_auc = macro; best_pred = ev_pred; best_ep = ep
        if verbose and (ep % 3 == 0 or ep == n_epochs - 1):
            print(f"  ep {ep:02d}  eval_macro {macro:.4f}")
    return best_auc, best_pred, macro0, best_ep, model


def main():
    print("=== exp106: P_NEW hybrid (frozen Perch-init + trainable MLP) ===\n", flush=True)

    sc_g, Y, primary, l2i = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()
    perch_prob_ss = load_perch_scores_labeled()

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    valid_mask = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta[np.arange(len(ta_emb))[valid_mask], ta_y_idx[valid_mask]] = 1.0

    W_init, b_init, mapped_idx = build_perch_init()
    n_mapped = (mapped_idx >= 0).sum()
    print(f"  Perch mapping: {n_mapped}/{N_CLS} mapped\n")

    # ===== T0. Same-site eval (122) =====
    print("=== T0. Same-site eval (122 SS held-out) ===")
    tr_mask = sc_g.split.values == "train"
    ev_mask = sc_g.split.values == "eval"
    Y_ss_tr = Y[tr_mask].astype(np.float32)
    Y_ss_ev = Y[ev_mask].astype(np.float32)
    perch_ss_tr = perch_emb_ss[tr_mask]; perch_ss_ev = perch_emb_ss[ev_mask]

    X_train = np.concatenate([ta_emb[valid_mask], perch_ss_tr], axis=0)
    Y_train = np.concatenate([Y_ta[valid_mask], Y_ss_tr], axis=0)
    src_weight = np.concatenate([
        np.ones(valid_mask.sum()),
        np.full(len(perch_ss_tr), 5.0)
    ])

    macro_perch, _ = macro_auc(Y_ss_ev, perch_prob_ss[ev_mask])
    print(f"  Perch sigmoid baseline:  {macro_perch:.4f}")

    best, ev_pred, init_macro, best_ep, model = train_hybrid(
        X_train, Y_train, src_weight, perch_ss_ev, Y_ss_ev,
        W_init, b_init, n_epochs=20, verbose=True
    )
    print(f"\n  Hybrid init (Perch only):    {init_macro:.4f}")
    print(f"  Hybrid best (ep {best_ep:02d}):         {best:.4f}")

    pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
    pt_perch = per_taxon_macro(Y_ss_ev, perch_prob_ss[ev_mask], sp_taxon)
    print("\n  Per-taxon (122 eval):")
    print(f"  {'taxon':<10} {'Perch':>8} {'Hybrid':>8} {'Δ':>8}")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        if not np.isnan(pt[t]) and not np.isnan(pt_perch[t]):
            print(f"  {t:<10} {pt_perch[t]:>8.4f} {pt[t]:>8.4f} {pt[t]-pt_perch[t]:>+8.4f}")

    # ===== T1. LOSO-site CV =====
    print("\n=== T1. LOSO-site CV ===")
    print(f"  {'holdout':<8} {'n_eval':>7} {'init':>8} {'best':>8} {'Δ':>8} {'Aves':>8} {'Insecta':>8} {'Mam':>8}")
    sites_arr = sc_g.site.values
    unique_sites = sorted(set(sites_arr))
    overall_init = []; overall_best = []
    overall_per_taxon = {t: [] for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}

    for ho_site in unique_sites:
        ho_mask = sites_arr == ho_site
        if ho_mask.sum() < 5: continue
        if (Y[ho_mask].sum(axis=0) > 0).sum() < 1: continue

        keep_mask = ~ho_mask
        X_ss_train = perch_emb_ss[keep_mask]
        Y_ss_train = Y[keep_mask].astype(np.float32)
        X_ss_eval = perch_emb_ss[ho_mask]
        Y_ss_eval = Y[ho_mask].astype(np.float32)

        X_tr = np.concatenate([ta_emb[valid_mask], X_ss_train], axis=0)
        Y_tr = np.concatenate([Y_ta[valid_mask], Y_ss_train], axis=0)
        src_w = np.concatenate([np.ones(valid_mask.sum()), np.full(len(X_ss_train), 5.0)])

        best_auc, ev_pred, init_macro, _, _ = train_hybrid(
            X_tr, Y_tr, src_w, X_ss_eval, Y_ss_eval,
            W_init, b_init, n_epochs=15, verbose=False
        )

        pt = per_taxon_macro(Y_ss_eval, ev_pred, sp_taxon)
        a = pt.get("Aves", float("nan")); i = pt.get("Insecta", float("nan")); m = pt.get("Mammalia", float("nan"))
        a_str = f"{a:>8.4f}" if not np.isnan(a) else f"{'--':>8}"
        i_str = f"{i:>8.4f}" if not np.isnan(i) else f"{'--':>8}"
        m_str = f"{m:>8.4f}" if not np.isnan(m) else f"{'--':>8}"
        print(f"  {ho_site:<8} {ho_mask.sum():>7} {init_macro:>8.4f} {best_auc:>8.4f} {best_auc-init_macro:>+8.4f} {a_str} {i_str} {m_str}", flush=True)
        overall_init.append(init_macro); overall_best.append(best_auc)
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
            v = pt.get(t, float("nan"))
            if not np.isnan(v): overall_per_taxon[t].append(v)

    print(f"\n  Mean LOSO macro — init (Perch alone): {np.mean(overall_init):.4f}")
    print(f"  Mean LOSO macro — best (after train):  {np.mean(overall_best):.4f}")
    print("  Per-taxon LOSO mean (Hybrid best):")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        if overall_per_taxon[t]:
            print(f"    {t}: {np.mean(overall_per_taxon[t]):.4f} (n={len(overall_per_taxon[t])})")

    # ===== Save full SS predictions for blend tests =====
    # Re-train on TA + ALL SS (no holdout), apply to all 739 rows.
    print("\nTraining final model on TA + all SS for blend tests...")
    X_all = np.concatenate([ta_emb[valid_mask], perch_emb_ss], axis=0)
    Y_all = np.concatenate([Y_ta[valid_mask], Y.astype(np.float32)], axis=0)
    src_all = np.concatenate([np.ones(valid_mask.sum()), np.full(len(perch_emb_ss), 5.0)])

    _, _, _, _, final_model = train_hybrid(
        X_all, Y_all, src_all, perch_ss_ev, Y_ss_ev,  # monitor on eval only
        W_init, b_init, n_epochs=20, verbose=False
    )
    final_model.eval()
    with torch.no_grad():
        all_emb_t = torch.from_numpy(perch_emb_ss).to(DEVICE)
        all_logits = final_model(all_emb_t)
        all_prob = torch.sigmoid(all_logits).cpu().numpy()
    out_path = EXP80 / "p_new2_hybrid_predictions.npz"
    np.savez_compressed(out_path, predictions=all_prob.astype(np.float32))
    print(f"Saved → {out_path}  shape {all_prob.shape}")


if __name__ == "__main__":
    main()
