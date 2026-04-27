#!/usr/bin/env python3
"""exp41b — Mahalanobis distance filter for pseudo-label confidence.

Idea: unlabeled SS chunks whose Perch embedding lies far from labeled distribution
are unreliable for pseudo-labeling. Use Mahalanobis distance as a gate.

Perch embeddings:
  - labeled 59 SS files: 708 × 1536 (exp21_outputs/perch_cache)
  - unlabeled 10,592 files: **NOT AVAILABLE** locally (would need Perch inference)

Workaround: use labeled SS embedding space as reference. Compute M-distance for
each **labeled** chunk too (leave-one-out) to get threshold distribution, then
apply at inference time. For unlabeled SS, the distance can't be computed
without running Perch on them.

Alternative (what we can do today): use **SED29 embedding** as proxy. Extract
pre-head feature from SED29 on unlabeled SS chunks → compute Mahalanobis in
that space vs labeled SS distribution.

This exp41b outputs:
  - labeled_ref_mu, labeled_ref_cov  : labeled SS SED29 pre-head feature stats
  - unlabeled_distance : (10592, 12) float distances
  - confidence_weight  : (10592, 12) = exp(-dist / threshold) in [0, 1]
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP41 = ROOT / "experiments/exp41_outputs"
EXP41.mkdir(exist_ok=True)

SR = 32000
CLIP_SAMPLES = 20 * SR
N_WINDOWS = 12
N_CLASSES = 234
N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
DEVICE = "cuda"


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class FeatExtractor(nn.Module):
    """SED29 backbone + mel + freq-mean pool → (B, C, T). Returns time-mean (B, C)."""
    def __init__(self, ckpt_path):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(BACKBONE, pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        # Strip head params
        bk_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        bn_sd = {k[len("bn0."):]: v for k, v in sd.items() if k.startswith("bn0.")}
        self.backbone.load_state_dict(bk_sd)
        self.bn0.load_state_dict(bn_sd)

    def forward(self, x):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        if feat.dim() == 4: feat = feat.mean(2)  # freq pool → (B, C, T)
        feat = feat.mean(-1)  # time pool → (B, C)
        return feat


class SSChunkDataset(Dataset):
    def __init__(self, items, root):
        self.items = items  # list of (filename, chunk_idx)
        self.root = root
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        fn, ci = self.items[idx]
        try:
            y, sr = sf.read(self.root / fn, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(1)
            if sr != SR:
                y = torchaudio.functional.resample(
                    torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
        except Exception:
            y = np.zeros(60 * SR, dtype=np.float32)
        if len(y) < 60 * SR: y = np.pad(y, (0, 60 * SR - len(y)))
        clip = y[ci * CLIP_SAMPLES:(ci + 1) * CLIP_SAMPLES]
        return torch.from_numpy(clip.astype(np.float32)), idx


def extract_feats(model, items, root, batch_size=32):
    ds = SSChunkDataset(items, root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    n = len(items)
    # Determine feat dim by running 1 batch
    _x = next(iter(dl))[0].to(DEVICE)
    with torch.inference_mode():
        _f = model(_x)
    d = _f.shape[1]
    feats = np.zeros((n, d), dtype=np.float32)
    print(f"feat dim: {d}")
    with torch.inference_mode():
        for x, idxs in tqdm(dl, desc="feat"):
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                f = model(x)
            feats[idxs.numpy()] = f.float().cpu().numpy()
    return feats


def main():
    # Step 1: labeled SS chunks (59 files × 3 chunks = 177)
    labeled_files = sorted(pd.read_csv(DATA / "train_soundscapes_labels.csv")
                           .drop_duplicates().filename.unique())
    print(f"Labeled files: {len(labeled_files)}")
    labeled_items = [(fn, ci) for fn in labeled_files for ci in range(3)]

    model = FeatExtractor(EXP29 / "best_ckpt.pt").to(DEVICE).eval()
    print("Feature extractor ready")

    t0 = time.time()
    lab_feats = extract_feats(model, labeled_items, DATA / "train_soundscapes")
    print(f"Labeled feats: {lab_feats.shape}  ({time.time()-t0:.0f}s)")

    # Compute mean + shrinkage covariance
    mu = lab_feats.mean(0)
    X = lab_feats - mu
    cov = (X.T @ X) / (len(X) - 1)
    # Shrinkage toward diagonal (Ledoit-Wolf-lite)
    diag_mean = np.diag(cov).mean()
    alpha = 0.1
    cov_s = (1 - alpha) * cov + alpha * diag_mean * np.eye(cov.shape[0])
    cov_inv = np.linalg.pinv(cov_s.astype(np.float64)).astype(np.float32)
    print(f"Reference stats saved. cond(cov_s) ≈ {np.linalg.cond(cov_s):.2e}")

    # Step 2: labeled distance distribution (for threshold calibration)
    lab_dists = np.array([(X[i] @ cov_inv @ X[i].T) for i in range(len(X))])
    print(f"Labeled Mahalanobis: median={np.median(lab_dists):.1f}  "
          f"p50={np.percentile(lab_dists, 50):.1f}  p90={np.percentile(lab_dists, 90):.1f}  "
          f"p99={np.percentile(lab_dists, 99):.1f}")

    # Step 3: unlabeled SS chunks (10592 × 3 = 31776)
    all_files = sorted([f.name for f in (DATA / "train_soundscapes").iterdir() if f.suffix == ".ogg"])
    unlabeled = [f for f in all_files if f not in set(labeled_files)]
    print(f"Unlabeled files: {len(unlabeled)}")

    unl_items = [(fn, ci) for fn in unlabeled for ci in range(3)]
    t0 = time.time()
    unl_feats = extract_feats(model, unl_items, DATA / "train_soundscapes")
    print(f"Unlabeled feats: {unl_feats.shape}  ({time.time()-t0:.0f}s)")

    # Mahalanobis distances
    U = unl_feats - mu
    # Vectorized: dist = sum(U @ cov_inv * U, axis=1)
    unl_dists = np.einsum('ij,jk,ik->i', U, cov_inv, U).astype(np.float32)
    print(f"Unlabeled Mahalanobis: median={np.median(unl_dists):.1f}  "
          f"p90={np.percentile(unl_dists, 90):.1f}  p99={np.percentile(unl_dists, 99):.1f}")

    # Reshape to (n_files, 3 chunks) → expand to (n_files, 12 windows)
    unl_dists_rs = unl_dists.reshape(len(unlabeled), 3)
    unl_dists_12 = np.repeat(unl_dists_rs, 4, axis=1)  # (n, 12)

    # Confidence weight: softly decay with distance. threshold = labeled p90
    thresh = float(np.percentile(lab_dists, 90))
    # Shape: exp(-max(0, d-t)/t)
    w = np.exp(-np.clip(unl_dists_12 - thresh, 0, None) / thresh).astype(np.float16)
    print(f"Unlabeled weight: median={np.median(w):.3f}  p10={np.percentile(w, 10):.3f}  "
          f"p90={np.percentile(w, 90):.3f}  pct>0.5={(w>0.5).mean()*100:.1f}%")

    np.savez_compressed(EXP41 / "mahal_weights.npz",
                        mu=mu, cov_inv=cov_inv,
                        labeled_dists=lab_dists,
                        unlabeled_dists=unl_dists_12.astype(np.float16),
                        confidence_weights=w,
                        filenames=np.array(unlabeled),
                        labeled_p90=thresh)
    print(f"Saved: {EXP41}/mahal_weights.npz")


if __name__ == "__main__":
    main()
