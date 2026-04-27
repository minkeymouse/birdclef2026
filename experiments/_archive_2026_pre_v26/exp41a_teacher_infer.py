#!/usr/bin/env python3
"""exp41a — Teacher inference on unlabeled train_soundscapes.

Teacher = SED29 (HGNetV2-B0, train_audio only, Val-A 0.7374).
Output: per-file (12, 234) sigmoid probabilities → exp41_outputs/pseudo_probs.npz.
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
OUT = ROOT / "experiments/exp41_outputs"
OUT.mkdir(exist_ok=True)

SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
N_WINDOWS = 12
N_CLASSES = 234
N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
BATCH_FILES = 16
DEVICE = "cuda"


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class SEDHead(nn.Module):
    def __init__(self, d, nc):
        super().__init__()
        self.att = nn.Conv1d(d, nc, 1); self.cla = nn.Conv1d(d, nc, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values


class SEDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = MelExtractor(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(BACKBONE, pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(f.shape[1], N_CLASSES)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        if feat.dim() == 4: feat = feat.mean(2)
        return self.head(feat)


class SSFullFileDataset(Dataset):
    """Read 60s file → emit 3×20s chunks per file (covers full 60s)."""
    def __init__(self, filenames, root=DATA / "train_soundscapes"):
        self.files = filenames
        self.root = root
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fn = self.files[idx]
        try:
            y, sr = sf.read(self.root / fn, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(1)
            if sr != SR:
                y = torchaudio.functional.resample(
                    torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
        except Exception:
            y = np.zeros(60 * SR, dtype=np.float32)
        if len(y) < 60 * SR:
            y = np.pad(y, (0, 60 * SR - len(y)))
        # Extract 3 × 20s chunks
        chunks = np.stack([y[c * CLIP_SAMPLES:(c + 1) * CLIP_SAMPLES] for c in range(3)])
        return torch.from_numpy(chunks.astype(np.float32)), idx


def main():
    # Load file list (unlabeled only)
    all_files = sorted([f for f in (DATA / "train_soundscapes").iterdir() if f.suffix == ".ogg"])
    labeled = set(pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().filename.unique())
    unlabeled = [f.name for f in all_files if f.name not in labeled]
    print(f"Total SS: {len(all_files)}  labeled: {len(labeled)}  unlabeled: {len(unlabeled)}")

    ds = SSFullFileDataset(unlabeled)
    dl = DataLoader(ds, batch_size=BATCH_FILES, shuffle=False, num_workers=8,
                    pin_memory=True, persistent_workers=True)

    model = SEDModel().to(DEVICE)
    ckpt = torch.load(EXP29 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded SED29 teacher (val_auc={ckpt.get('val_auc', '?')})")

    # Output: (n_files, N_WINDOWS, N_CLASSES). 12 windows = 3 chunks × 4 windows each.
    n = len(unlabeled)
    probs = np.zeros((n, N_WINDOWS, N_CLASSES), dtype=np.float16)  # half to save memory

    t0 = time.time()
    with torch.inference_mode():
        for chunks, idxs in tqdm(dl, desc="teacher"):
            # chunks: (B, 3, CLIP_SAMPLES)
            B = chunks.shape[0]
            x = chunks.view(B * 3, CLIP_SAMPLES).to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip, _ = model(x)
            p = torch.sigmoid(clip).float().cpu().numpy()
            p = p.reshape(B, 3, N_CLASSES)  # (B, 3 chunks, 234)
            # Expand 3 chunks → 12 windows (each chunk covers 4 windows of 5s)
            for bi, idx in enumerate(idxs.tolist()):
                for ci in range(3):
                    probs[idx, ci * 4:(ci + 1) * 4] = p[bi, ci].astype(np.float16)

    dt = time.time() - t0
    print(f"\nInference: {dt:.0f}s ({n*N_WINDOWS/dt:.0f} windows/s)")
    np.savez_compressed(OUT / "pseudo_probs.npz",
                        probs=probs,
                        filenames=np.array(unlabeled))
    print(f"Saved: {OUT}/pseudo_probs.npz  shape={probs.shape}")

    # Quick stats
    max_prob = probs.max(axis=2)  # (n, 12)
    print(f"\nWindow max-prob distribution (sigmoid):")
    for thr in [0.1, 0.2, 0.3, 0.5, 0.7]:
        frac = (max_prob > thr).mean()
        print(f"  p>{thr}: {frac*100:.1f}% of windows ({int(frac*n*N_WINDOWS)} windows)")


if __name__ == "__main__":
    main()
