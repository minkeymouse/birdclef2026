"""SS dataset v3 — supports BOTH mapped (v33-based) and unmapped (confusion-based) pseudo rows.

Differences from SSPseudoDataset (v2):
  - Uses full v33 vector lookup (filename, end_sec → 234-dim) instead of per-class CSV
  - Mask logic:
      y[c]=1 (label in row.lbls) → mask=1 (positive supervision)
      v33[r,c] < tau_neg → mask=1 (confident negative)
      else (uncertain) → mask=0 (no penalty)
  - This handles sonotype rows correctly:
      Sonotype label gets positive supervision (mask=1, y=1)
      v33 still tells us OTHER classes' confident negatives
      Uncertain classes don't penalize — no false negative
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import DATA, SR, WINDOW_SEC, CLIP_SAMPLES, FILE_SAMPLES
from .audio import load_audio


class SSPseudoDatasetV3(Dataset):
    """Pseudo SS with full per-row v33 lookup for proper masking.

    v33_lookup arg can be either:
      - dict {(filename, end_sec): np.ndarray(234,)} — slow with multi-worker
      - tuple (filenames_arr, end_secs_arr, v33_matrix) — pre-indexed, fast
    """
    def __init__(self, pseudo_df, l2i, train=True, v33_lookup=None,
                  tau_neg=0.05):
        self.ss = pseudo_df.reset_index(drop=True)
        self.l2i = l2i
        self.train = train
        self.n_cls = len(l2i)
        self.tau_neg = tau_neg

        # Build per-row mask vectors UPFRONT, vectorized (avoids slow pandas iloc)
        if v33_lookup is not None:
            print(f"  Pre-computing per-row v33 masks for {len(self.ss)} pseudo rows (vectorized)...")
            fnames = self.ss.filename.values  # np array
            end_secs = self.ss.end_sec.values.astype(int)
            self.row_masks = np.zeros((len(self.ss), self.n_cls), dtype=np.float32)
            n_hit = 0
            for i in range(len(self.ss)):
                key = (fnames[i], end_secs[i])
                v = v33_lookup.get(key)
                if v is not None:
                    self.row_masks[i] = (v < tau_neg).astype(np.float32)
                    n_hit += 1
            print(f"  Pre-computed mask matrix: {self.row_masks.shape}, "
                  f"hit {n_hit}/{len(self.ss)}, avg confident-neg rate: {self.row_masks.mean():.3f}")
        else:
            self.row_masks = None

    def __len__(self): return len(self.ss)

    def __getitem__(self, idx):
        row = self.ss.iloc[idx]
        p = DATA / "train_soundscapes" / row.filename
        wav = load_audio(p, FILE_SAMPLES)
        end_sec = int(row.end_sec)
        target_c = (end_sec - WINDOW_SEC / 2) * SR
        cs = int(max(0, target_c - CLIP_SAMPLES / 2))
        cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
        if self.train:
            cs = int(cs + random.randint(-SR, SR)); cs = max(0, min(cs, FILE_SAMPLES - CLIP_SAMPLES))
        clip = wav[cs:cs + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))

        # Build label vector from row.lbls
        y = np.zeros(self.n_cls, dtype=np.float32)
        positive_classes = set()
        for l in row.lbls:
            if l in self.l2i:
                y[self.l2i[l]] = 1.0
                positive_classes.add(self.l2i[l])

        # Build mask from pre-computed matrix (fast, no dict access)
        if self.row_masks is None:
            mask = np.ones(self.n_cls, dtype=np.float32)
        else:
            mask = self.row_masks[idx].copy()
            # Override: positive classes always get mask=1
            for c in positive_classes:
                mask[c] = 1.0

        return (torch.from_numpy(clip.astype(np.float32)),
                torch.from_numpy(y), torch.from_numpy(mask), -1, 0)
