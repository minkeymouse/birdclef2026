#!/usr/bin/env python3
"""exp119 — Cross-taxon penalty + auxiliary taxon head.

Failure mode from exp108:
  - 74113 (Mammalia) → predicted as grfdov1, 47144, picpig2 (all Aves)
  - 326272 (Amphibia) → predicted as 22973, compau, grepot1 (all Aves)
  - All non-Aves universally-missed cases get pushed to Aves cluster

Hypothesis: Perch's bird-centric training (14k species, ~99% birds) creates
inductive bias toward Aves predictions. BCE doesn't penalize this — each
class's logit is independent.

Fix: directly penalize "Aves dominance when truth is non-Aves":
    L_ct = max(0, max_Aves_logit - max_TrueNonAves_logit + margin)

This is an explicit RL-style reward shaping: when model commits taxonomic
error, gradient pushes Aves DOWN and true-non-Aves UP simultaneously.

Combined with auxiliary taxon head:
    L_aux = BCE(taxon_logits, taxon_y)

The shared 1536-d feature must support BOTH species and taxon discrimination.

Variants tested:
  V0: BCE baseline
  V1: BCE + cross-taxon margin penalty (margin=1.0, λ=0.5)
  V2: BCE + aux taxon head only (λ=0.5)
  V3: BCE + cross-taxon + aux taxon (combined)
  V4: V3 with aggressive λ values (λ_ct=2.0, λ_taxon=1.0)
  V5: Cross-taxon with within-row pushdown (more aggressive)
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
N_TAXA = len(TAXA)


class PerchHybridWithTaxon(nn.Module):
    """P_NEW3 hybrid + auxiliary 5-class taxon head."""
    def __init__(self, W_init, b_init, hidden=768, dropout=0.3):
        super().__init__()
        self.perch_fc = nn.Linear(1536, N_CLS)
        with torch.no_grad():
            self.perch_fc.weight.copy_(torch.from_numpy(W_init.T))
            self.perch_fc.bias.copy_(torch.from_numpy(b_init))
        for p in self.perch_fc.parameters():
            p.requires_grad_(False)
        self.bn = nn.BatchNorm1d(1536)
        self.fc1 = nn.Linear(1536, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, N_CLS)
        self.taxon_head = nn.Linear(hidden, N_TAXA)
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1, eps=1e-6)
        perch_logit = self.perch_fc(x_norm)
        h = self.bn(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        species = perch_logit + self.fc2(h)
        taxon = self.taxon_head(h)
        return species, taxon


def cross_taxon_penalty(logits, y, sp_taxon_idx, margin=1.0):
    """Penalty: max_Aves_logit > max_TrueNonAves_logit + margin → penalty.

    Args:
      logits: (B, 234)
      y: (B, 234) binary multi-label
      sp_taxon_idx: (234,) int — taxon idx per species (0=Aves, 1=Amphibia, ..., 4=Reptilia)
      margin: required gap between max non-Aves true and max Aves
    """
    aves_mask = (sp_taxon_idx == 0)  # bool (234,)
    aves_logits = logits[:, aves_mask]  # (B, n_aves)
    non_aves_logits = logits[:, ~aves_mask]  # (B, n_non_aves)
    non_aves_y = y[:, ~aves_mask]  # (B, n_non_aves)

    has_non_aves_pos = non_aves_y.sum(dim=1) > 0  # (B,)
    if has_non_aves_pos.sum() == 0:
        return logits.sum() * 0.0

    rows = has_non_aves_pos.nonzero(as_tuple=False).squeeze(-1)
    max_aves = aves_logits[rows].max(dim=1).values

    # Max non-Aves logit on TRUE positives only
    masked_non_aves = non_aves_logits[rows].masked_fill(non_aves_y[rows] == 0, -1e9)
    max_non_aves_pos = masked_non_aves.max(dim=1).values

    # Penalty if max_aves > max_non_aves_pos - margin
    penalty = F.relu(max_aves - max_non_aves_pos + margin)
    return penalty.mean()


def train_with_taxon_loss(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                            sp_taxon_idx, lambda_ct=0.0, lambda_taxon=0.0, margin=1.0,
                            n_epochs=15, lr=1e-3, verbose=False):
    """Train P_NEW3 with optional cross-taxon penalty + auxiliary taxon head."""
    if lambda_taxon > 0:
        model = PerchHybridWithTaxon(W_init, b_init).to(DEVICE)
    else:
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

    # Build taxon labels per row
    taxon_y = np.zeros((len(Y_train), N_TAXA), dtype=np.float32)
    for c in range(N_CLS):
        if Y_train[:, c].any():
            t_idx = sp_taxon_idx[c]
            taxon_y[Y_train[:, c] > 0, t_idx] = 1.0

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    Y_t = torch.from_numpy(Y_train.astype(np.float32)).to(DEVICE)
    W_t = torch.from_numpy(src_weight.astype(np.float32)).to(DEVICE)
    Tax_t = torch.from_numpy(taxon_y).to(DEVICE)
    Xev_t = torch.from_numpy(X_eval.astype(np.float32)).to(DEVICE)
    sp_taxon_t = torch.from_numpy(sp_taxon_idx).to(DEVICE)

    n = len(X_t); BATCH = 512
    best_auc = 0.0; best_pred = None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_bce = 0.0; ep_ct = 0.0; ep_tax = 0.0; nb = 0
        for s in range(0, n, BATCH):
            idx = perm[s:s+BATCH]
            x = X_t[idx]; y = Y_t[idx]; w = W_t[idx]; t = Tax_t[idx]
            opt.zero_grad()
            if lambda_taxon > 0:
                species_logits, taxon_logits = model(x)
            else:
                species_logits = model(x)
                taxon_logits = None

            bce = F.binary_cross_entropy_with_logits(species_logits, y, pos_weight=pw_t, reduction="none")
            bce = (bce.mean(dim=-1) * w).mean()
            loss = bce
            ct = torch.tensor(0.0, device=DEVICE)
            tax_loss = torch.tensor(0.0, device=DEVICE)

            if lambda_ct > 0:
                ct = cross_taxon_penalty(species_logits, y, sp_taxon_t, margin=margin)
                loss = loss + lambda_ct * ct
            if lambda_taxon > 0 and taxon_logits is not None:
                tax_loss = F.binary_cross_entropy_with_logits(taxon_logits, t)
                loss = loss + lambda_taxon * tax_loss

            loss.backward()
            opt.step()
            ep_bce += bce.item(); ep_ct += float(ct); ep_tax += float(tax_loss); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            if lambda_taxon > 0:
                species_pred, _ = model(Xev_t)
            else:
                species_pred = model(Xev_t)
            ev_pred = torch.sigmoid(species_pred).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        if macro > best_auc:
            best_auc = macro; best_pred = ev_pred
        if verbose and (ep % 3 == 0 or ep == n_epochs - 1):
            print(f"  ep {ep:02d}  bce {ep_bce/nb:.3f} ct {ep_ct/nb:.3f} tax {ep_tax/nb:.3f}  macro {macro:.4f}")

    return best_auc, best_pred


def main():
    print("=== exp119: Cross-taxon penalty + aux taxon head ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    sp_taxon_idx = np.array([TAXA.index(t) if t in TAXA else 0 for t in sp_taxon], dtype=np.int64)
    print(f"  Taxon distribution: " + ", ".join(f"{t}:{int((sp_taxon == t).sum())}" for t in TAXA))

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

    n_non_aves_rows = int((Y_train[:, sp_taxon != "Aves"].sum(axis=1) > 0).sum())
    print(f"  Train rows with any non-Aves positive: {n_non_aves_rows} / {len(Y_train)}\n")

    variants = [
        ("V0: BCE baseline",         dict(lambda_ct=0.0, lambda_taxon=0.0)),
        ("V1: cross-taxon λ=0.5",    dict(lambda_ct=0.5, lambda_taxon=0.0, margin=1.0)),
        ("V1b: cross-taxon λ=2.0",   dict(lambda_ct=2.0, lambda_taxon=0.0, margin=1.0)),
        ("V1c: cross-taxon λ=0.5 m=2", dict(lambda_ct=0.5, lambda_taxon=0.0, margin=2.0)),
        ("V2: aux taxon λ=0.5",      dict(lambda_ct=0.0, lambda_taxon=0.5)),
        ("V2b: aux taxon λ=1.0",     dict(lambda_ct=0.0, lambda_taxon=1.0)),
        ("V3: combined λ=0.5",       dict(lambda_ct=0.5, lambda_taxon=0.5, margin=1.0)),
        ("V4: aggressive (λ_ct=2 λ_t=1)", dict(lambda_ct=2.0, lambda_taxon=1.0, margin=2.0)),
    ]

    results = []
    for name, kwargs in variants:
        print(f"=== {name} ===", flush=True)
        best, ev_pred = train_with_taxon_loss(
            X_train, Y_train, src_w, perch_emb[ev_mask], Y_ss_ev,
            W_init, b_init, sp_taxon_idx, n_epochs=15, verbose=True, **kwargs
        )
        pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
        rec = {"variant": name, "macro": best,
                **{t: pt.get(t, float('nan')) for t in TAXA}}
        results.append(rec)
        print(f"  best macro {best:.4f} | "
              f"Aves {pt.get('Aves', float('nan')):.4f}, "
              f"Amphib {pt.get('Amphibia', float('nan')):.4f}, "
              f"Insecta {pt.get('Insecta', float('nan')):.4f}, "
              f"Mam {pt.get('Mammalia', float('nan')):.4f}, "
              f"Rept {pt.get('Reptilia', float('nan')):.4f}\n", flush=True)

    df = pd.DataFrame(results)
    print("\n=== Summary (sorted by macro) ===")
    print(df.sort_values("macro", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
