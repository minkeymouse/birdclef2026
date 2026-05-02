#!/usr/bin/env python3
"""exp132 — Round 1.5: pseudo retrain with MASKED BCE.

Diagnosis from exp129/exp131: pseudo as multi-label hurts non-Aves AUC because
all classes not in pseudo CSV are treated as negatives. Many TRUE positives
(v33 below threshold) get learned as negatives → AUC drops.

Fix: per-cell BCE mask. For pseudo rows:
  v33[r,c] > τ_pos (0.5) → label=1, mask=1 (learn positive)
  v33[r,c] < τ_neg (0.05) → label=0, mask=1 (confident negative)
  middle (0.05–0.5)        → mask=0 (no signal, skip BCE)

For TA / hard-labeled SS rows: full mask (no skipping).

Uses _shared/ module for everything else.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))
from _shared import (DATA, ROOT, EXP50_CKPT, BG_PATH, N_CLS, BACKBONE,
                       SEED, EPOCHS, LR, WD, BATCH_SIZE, NUM_WORKERS,
                       TA_WEIGHT, SS_LABELED_WEIGHT, SS_PSEUDO_WEIGHT,
                       build_primaries, build_ta_combined, build_ss_splits,
                       build_pseudo_ss, TADataset, SSDataset, SSPseudoDataset,
                       SEDModel, aggressive_mixup, load_bg_pool,
                       get_taxon_array, train_sed_loop)

PSEUDO_CSV = DATA / "pseudo_soundscapes_labels_verified_v2.csv"
OUT = ROOT / "experiments/_data_pipelines/exp132_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"


def main():
    print("=== exp132 — Round 1.5: pseudo retrain with masked BCE ===\n", flush=True)
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    primary, l2i = build_primaries()
    n_cls = len(primary)
    taxon_array = get_taxon_array(primary)

    print("Loading data...")
    ta_train, ta_val = build_ta_combined(l2i)
    ss_train, ss_eval = build_ss_splits(l2i)
    eval_files = set(ss_eval.filename.unique())
    pseudo_ss = build_pseudo_ss(PSEUDO_CSV, l2i, eval_files=eval_files)
    score_lookup = pseudo_ss.attrs.get("score_lookup")
    print(f"  ta_train {len(ta_train)}, ta_val {len(ta_val)}")
    print(f"  ss_train {len(ss_train)}, ss_eval {len(ss_eval)}")
    print(f"  pseudo_ss (v2) {len(pseudo_ss)} unique (file, window)")
    print(f"  score_lookup: {len(score_lookup) if score_lookup else 0} entries (for masking)")

    bg_pool = load_bg_pool()
    if bg_pool is not None:
        print(f"  BG pool: {bg_pool.shape}")

    ta_ds = TADataset(ta_train, l2i, train=True)
    ta_val_ds = TADataset(ta_val, l2i, train=False)
    ss_train_ds = SSDataset(ss_train, l2i, train=True)
    ss_eval_ds = SSDataset(ss_eval, l2i, train=False)
    pseudo_ds = SSPseudoDataset(pseudo_ss, l2i, train=True,
                                  score_lookup=score_lookup,
                                  tau_pos=0.5, tau_neg=0.05)

    combined = ConcatDataset([ta_ds, ss_train_ds, pseudo_ds])
    weights = np.concatenate([
        np.full(len(ta_ds), TA_WEIGHT, dtype=np.float32),
        np.full(len(ss_train_ds), SS_LABELED_WEIGHT, dtype=np.float32),
        np.full(len(pseudo_ds), SS_PSEUDO_WEIGHT, dtype=np.float32),
    ])
    sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)
    train_loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_ta = DataLoader(ta_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_ss = DataLoader(ss_eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
    print(f"\n  Total combined: {len(combined)}")

    print("\nLoading exp50 ckpt...")
    model = SEDModel(n_cls).to(DEVICE)
    if EXP50_CKPT.exists():
        ckpt = torch.load(str(EXP50_CKPT), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded exp50 (val_SS={ckpt.get('val_SS', '?')})")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=LR/10)

    best_val_ss, best_state, history = train_sed_loop(
        model, train_loader, val_loader_ta, val_loader_ss,
        bg_pool, taxon_array, optim, scheduler, EPOCHS, DEVICE,
        OUT, mixup_fn=aggressive_mixup, use_masked_bce=True, log_per_taxon=True
    )
    print(f"\nDone. Best val_SS: {best_val_ss:.4f}  (exp50 baseline 0.838)")


if __name__ == "__main__":
    main()
