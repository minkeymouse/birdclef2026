#!/usr/bin/env python3
"""
`dataloader.py` – dataset, sampler, and training loop utilities
============================================================
Shared by `train_efficientnet.py` and `train_regnety.py`.

Classes:
- `BirdClefDataset`   — loads mel-spectrogram chunks, labels, and chunk IDs from metadata.
- `create_dataloader` — builds a PyTorch DataLoader with optional `WeightedRandomSampler`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import random

__all__ = ["BirdClefDataset", "create_dataloader", "collate_fn"]

class BirdClefDataset(Dataset):
    """Dataset for BirdCLEF: loads mel-spectrograms, labels, and chunk IDs from metadata."""

    def __init__(
        self,
        label2id: Dict[str, int],
        metadata_df: pd.DataFrame,
        num_classes: int,
        *,
        mode: str = "train"
    ) -> None:
        self.df = metadata_df.reset_index(drop=True)
        if "weight" not in self.df.columns:
            raise KeyError("metadata_df must contain a 'weight' column for sampling.")
        if "chunk_id" not in self.df.columns:
            raise KeyError("metadata_df must contain a 'chunk_id' column for identification.")
        self.sample_weights = self.df["weight"].astype(float).tolist()
        self.label2id = label2id
        self.num_classes = num_classes
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        chunk_id = row["chunk_id"]

        mel = np.load(row["mel_path"])  # (H, W) or (1, H, W)
        mel = np.nan_to_num(mel, nan=0.0)
        # ensure channel dim = 1
        if mel.ndim == 2:
            mel = mel[np.newaxis, ...]

        label = np.load(row["label_path"]).astype(np.float32)
        label = np.nan_to_num(label, nan=0.0)

        if self.mode == "train":
            mel = self.apply_spec_augmentations(mel)

        return {
            "mel": torch.tensor(mel, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "weight": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
            "chunk_id": chunk_id,
        }

    def apply_spec_augmentations(self, spec: np.ndarray) -> np.ndarray:
        """Simple SpecAugment: time/freq masking + brightness/contrast."""
        # spec shape: (1, freq_bins, time_steps)
        # time mask
        if random.random() < 0.5:
            t = spec.shape[2]
            for _ in range(random.randint(1, 3)):
                width = random.randint(5, min(20, t // 2))
                start = random.randint(0, t - width)
                spec[0, :, start:start + width] = 0
        # freq mask
        if random.random() < 0.5:
            f = spec.shape[1]
            for _ in range(random.randint(1, 3)):
                height = random.randint(5, min(20, f // 2))
                start = random.randint(0, f - height)
                spec[0, start:start + height, :] = 0
        # brightness/contrast
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = np.clip(spec, 0.0, 1.0)
        return spec

    def _weights(self) -> List[float]:
        return self.sample_weights


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate examples into batch, stacking tensors and collecting chunk IDs."""
    batch = [b for b in batch if b]
    keys = batch[0].keys()
    out: Dict[str, list] = {k: [] for k in keys}
    for b in batch:
        for k, v in b.items():
            out[k].append(v)
    # stack tensors
    out["mel"] = torch.stack(out["mel"])
    out["label"] = torch.stack(out["label"])
    out["weight"] = torch.stack(out["weight"])
    # chunk_id remains as List[str]
    return out


def create_dataloader(
    dataset: BirdClefDataset,
    batch_size: int,
    num_workers: Optional[int] = None,
    pin_memory: bool = True
) -> DataLoader:
    """Build DataLoader with WeightedRandomSampler for imbalanced data."""
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 4
    sampler = WeightedRandomSampler(
        weights=dataset._weights(),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
