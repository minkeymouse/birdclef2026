#!/usr/bin/env python3
"""exp119c — Joint taxon-species training with multiplicative gate.

Architecture forces taxonomic awareness inside the model:
  shared features (1536 → 768)
        ↓
   species_head: 768 → 234 (raw species logits)
   taxon_head:   768 → 5 (taxon logits)

Forward:
  taxon_log_prob = log_sigmoid(taxon_logits)  shape (B, 5)
  species_logit_gated[i, c] = species_logits[i, c] + taxon_log_prob[i, taxon_of_c]
  return species_logit_gated, taxon_logits

This makes species predictions DEPEND on taxon prediction. If taxon_head
correctly predicts "Mammalia" for a Mammalia row → its species gets full
weight, Aves get -inf. Joint training learns both decisions together.

Loss = BCE(species_gated, y_species) + λ * BCE(taxon_logits, y_taxon)
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
from exp106_pnew_hybrid import build_perch_init

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TAXA = len(TAXA)


class TaxonGatedHybrid(nn.Module):
    """Frozen Perch-init Linear (per-class) + trainable shared features that
    produce both species and taxon logits, with multiplicative taxon gate.
    """
    def __init__(self, W_init, b_init, sp_taxon_idx, hidden=768, dropout=0.3):
        super().__init__()
        # Frozen Perch branch
        self.perch_fc = nn.Linear(1536, N_CLS)
        with torch.no_grad():
            self.perch_fc.weight.copy_(torch.from_numpy(W_init.T))
            self.perch_fc.bias.copy_(torch.from_numpy(b_init))
        for p in self.perch_fc.parameters():
            p.requires_grad_(False)

        # Trainable backbone
        self.bn = nn.BatchNorm1d(1536)
        self.fc1 = nn.Linear(1536, hidden)
        self.dropout = nn.Dropout(dropout)
        self.species_head = nn.Linear(hidden, N_CLS)
        self.taxon_head = nn.Linear(hidden, N_TAXA)
        with torch.no_grad():
            self.species_head.weight.zero_()
            self.species_head.bias.zero_()

        # Save taxon idx for gating
        self.register_buffer("sp_taxon_idx", torch.from_numpy(sp_taxon_idx))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1, eps=1e-6)
        perch_logit = self.perch_fc(x_norm)

        h = self.bn(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        species_raw = self.species_head(h)
        taxon_logit = self.taxon_head(h)

        # Multiplicative gate (additive in log-space)
        # taxon_log_prob: (B, 5), select per species: (B, 234)
        taxon_log_prob = F.logsigmoid(taxon_logit)  # (B, 5)
        per_species_taxon_log_prob = taxon_log_prob[:, self.sp_taxon_idx]  # (B, 234)

        species_gated = perch_logit + species_raw + per_species_taxon_log_prob
        return species_gated, taxon_logit


def train_joint(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                  sp_taxon_idx, lambda_taxon=1.0, n_epochs=15, lr=1e-3, verbose=False):
    model = TaxonGatedHybrid(W_init, b_init, sp_taxon_idx).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/10)

    cls_pos_count = Y_train.sum(axis=0)
    pw = np.where(cls_pos_count > 0,
                   np.sqrt(len(X_train) / (cls_pos_count * N_CLS + 1e-6)),
                   1.0).astype(np.float32)
    pw = np.clip(pw, 0.5, 50.0)
    pw_t = torch.from_numpy(pw).to(DEVICE)

    # Build taxon multi-hot labels
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

    n = len(X_t); BATCH = 512
    best_auc = 0.0; best_pred = None
    for ep in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_bce = 0.0; ep_tax = 0.0; nb = 0
        for s in range(0, n, BATCH):
            idx = perm[s:s+BATCH]
            x = X_t[idx]; y = Y_t[idx]; w = W_t[idx]; t = Tax_t[idx]
            opt.zero_grad()
            species_gated, taxon_logits = model(x)
            bce = F.binary_cross_entropy_with_logits(species_gated, y, pos_weight=pw_t, reduction="none")
            bce = (bce.mean(dim=-1) * w).mean()
            tax_loss = F.binary_cross_entropy_with_logits(taxon_logits, t)
            loss = bce + lambda_taxon * tax_loss
            loss.backward()
            opt.step()
            ep_bce += bce.item(); ep_tax += tax_loss.item(); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            species_gated, _ = model(Xev_t)
            ev_pred = torch.sigmoid(species_gated).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        if macro > best_auc:
            best_auc = macro; best_pred = ev_pred
        if verbose and (ep % 3 == 0 or ep == n_epochs - 1):
            print(f"  ep {ep:02d}  bce {ep_bce/nb:.4f}  tax {ep_tax/nb:.4f}  macro {macro:.4f}")
    return best_auc, best_pred


def main():
    print("=== exp119c: Joint taxon-species with gate ===\n", flush=True)

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

    # Compare to BCE baseline P_NEW3 first
    print("=== BCE P_NEW3 baseline (no gate) ===", flush=True)
    from exp113_pnew3_dpo import train_bce_reference
    ref_model, ref_macro = train_bce_reference(
        X_train, Y_train, src_w, perch_emb[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=15
    )
    pt_ref = per_taxon_macro(Y_ss_ev,
                              torch.sigmoid(ref_model(torch.from_numpy(perch_emb[ev_mask]).to(DEVICE))).detach().cpu().numpy(),
                              sp_taxon)
    print(f"  BCE macro: {ref_macro:.4f}, "
          f"Aves {pt_ref['Aves']:.4f}, Amphib {pt_ref['Amphibia']:.4f}, "
          f"Insecta {pt_ref['Insecta']:.4f}, Mam {pt_ref['Mammalia']:.4f}, Rept {pt_ref['Reptilia']:.4f}\n")

    # Joint variants
    variants = [
        ("V1: joint λ=0.5", 0.5),
        ("V2: joint λ=1.0", 1.0),
        ("V3: joint λ=2.0", 2.0),
        ("V4: joint λ=5.0", 5.0),
    ]

    results = [{"variant": "V0: BCE baseline", "macro": ref_macro,
                 **{t: pt_ref.get(t, float('nan')) for t in TAXA}}]

    for name, lam in variants:
        print(f"=== {name} ===", flush=True)
        best, ev_pred = train_joint(
            X_train, Y_train, src_w, perch_emb[ev_mask], Y_ss_ev,
            W_init, b_init, sp_taxon_idx, lambda_taxon=lam, n_epochs=15, verbose=True
        )
        pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
        results.append({"variant": name, "macro": best,
                          **{t: pt.get(t, float('nan')) for t in TAXA}})
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
