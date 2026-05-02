#!/usr/bin/env python3
"""exp136 — Retrain SED with v3 pseudo (Aves + Amphib + Insecta sonotype + Mam + Rept).

v3 has 351k entries with all-taxa coverage:
  - 198k Aves
  - 84k Amphibia
  - 68k Insecta (sonotype confusion-based)
  - 1.3k Mammalia
  - 210 Reptilia

Uses SSPseudoDatasetV3 with full-vector v33 lookup:
  - Pseudo positive (y=1): mask=1 (positive supervision)
  - v33 < 0.05 (confident negative): mask=1 (negative supervision)
  - Uncertain: mask=0 (no penalty, avoids false negative)

This handles sonotype rows correctly: sonotype label gets +supervision while
v33-confident-negatives still provide neg signal but uncertain classes don't
hurt training.

Continues from exp50. 8 epochs.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))
from _shared import (DATA, ROOT, EXP50_CKPT, N_CLS, BACKBONE,
                       SEED, EPOCHS, LR, WD, BATCH_SIZE,
                       TA_WEIGHT, SS_LABELED_WEIGHT, SS_PSEUDO_WEIGHT,
                       build_primaries, build_ta_combined, build_ss_splits,
                       build_pseudo_ss, TADataset, SSDataset,
                       SEDModel, aggressive_mixup, load_bg_pool,
                       get_taxon_array, train_sed_loop)
NUM_WORKERS = 4  # try with workers, debug hang
from _shared.data_v3 import SSPseudoDatasetV3

V3_CSV = DATA / "pseudo_soundscapes_labels_v3.csv"
V126_SCORES = ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz"
EXP125_UNLAB = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
OUT = ROOT / "experiments/_data_pipelines/exp136_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"


def build_v33_lookup():
    """Build (filename, end_sec) → np.array(234,) lookup of v33 scores."""
    print("Building v33 lookup...")
    v33_data = np.load(V126_SCORES, allow_pickle=True)
    v33 = v33_data["v33"].astype(np.float32)
    filenames = v33_data["filenames"].astype(str)
    end_secs = v33_data["end_secs"].astype(int)
    lookup = {}
    for i in range(len(v33)):
        lookup[(filenames[i], int(end_secs[i]))] = v33[i]
    print(f"  Built lookup: {len(lookup)} (filename, end_sec) keys")
    return lookup


def main():
    print("=== exp136 — v3 retrain (all-taxa pseudo, sonotype-aware mask) ===\n", flush=True)
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    primary, l2i = build_primaries()
    n_cls = len(primary)
    taxon_array = get_taxon_array(primary)

    print("Loading data...")
    ta_train, ta_val = build_ta_combined(l2i)
    ss_train, ss_eval = build_ss_splits(l2i)
    eval_files = set(ss_eval.filename.unique())
    pseudo_ss = build_pseudo_ss(V3_CSV, l2i, eval_files=eval_files)
    print(f"  ta_train {len(ta_train)}, ta_val {len(ta_val)}")
    print(f"  ss_train {len(ss_train)}, ss_eval {len(ss_eval)}")
    print(f"  pseudo_ss (v3) {len(pseudo_ss)} unique (file, window)")

    v33_lookup = build_v33_lookup()
    print("Loading BG pool...", flush=True)
    bg_pool = load_bg_pool()
    print(f"  BG pool loaded: {bg_pool.shape if bg_pool is not None else None}", flush=True)

    print("Building datasets...", flush=True)
    ta_ds = TADataset(ta_train, l2i, train=True)
    ta_val_ds = TADataset(ta_val, l2i, train=False)
    ss_train_ds = SSDataset(ss_train, l2i, train=True)
    ss_eval_ds = SSDataset(ss_eval, l2i, train=False)
    print("  TA/SS datasets built; building pseudo with mask precompute...", flush=True)
    pseudo_ds = SSPseudoDatasetV3(pseudo_ss, l2i, train=True,
                                       v33_lookup=v33_lookup, tau_neg=0.05)
    print("  All datasets ready.", flush=True)

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
    print(f"\nDone. Best val_SS: {best_val_ss:.4f}  (exp50 0.838)")


if __name__ == "__main__":
    main()
