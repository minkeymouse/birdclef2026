#!/usr/bin/env python3
"""exp41g — Iterative round 2 teacher inference.

Teacher = exp41f (pseudo-trained ensemble student, Val-A_v2 0.9024).
Generate pseudo on unlabeled SS → save → student exp41h training.
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
EXP41F = ROOT / "experiments/exp41f_outputs"
OUT = ROOT / "experiments/exp41_outputs"

SR = 32000
CLIP_SAMPLES = 20 * SR
N_WINDOWS = 12; N_CLASSES = 234
N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
DEVICE = "cuda"


class Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)

class Head(nn.Module):
    def __init__(self, d, nc):
        super().__init__()
        self.att = nn.Conv1d(d, nc, 1); self.cla = nn.Conv1d(d, nc, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values

class SEDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = Mel(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(BACKBONE, pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = Head(f.shape[1], N_CLASSES)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        if feat.dim() == 4: feat = feat.mean(2)
        return self.head(feat)


class SSFullDS(Dataset):
    def __init__(self, filenames, root=DATA / "train_soundscapes"):
        self.files = filenames; self.root = root
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fn = self.files[idx]
        try:
            y, sr = sf.read(self.root / fn, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(1)
            if sr != SR:
                y = torchaudio.functional.resample(torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
        except Exception:
            y = np.zeros(60 * SR, dtype=np.float32)
        if len(y) < 60 * SR: y = np.pad(y, (0, 60 * SR - len(y)))
        chunks = np.stack([y[c * CLIP_SAMPLES:(c + 1) * CLIP_SAMPLES] for c in range(3)])
        return torch.from_numpy(chunks.astype(np.float32)), idx


def main():
    all_files = sorted([f.name for f in (DATA / "train_soundscapes").iterdir() if f.suffix == ".ogg"])
    labeled = set(pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().filename.unique())
    unl = [f for f in all_files if f not in labeled]
    print(f"Unlabeled: {len(unl)}")

    model = SEDM().to(DEVICE).eval()
    ck = torch.load(EXP41F / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    print(f"Teacher exp41f ep={ck.get('epoch')} val_auc={ck.get('val_auc'):.4f}")

    dl = DataLoader(SSFullDS(unl), batch_size=16, shuffle=False, num_workers=6, pin_memory=True)
    n = len(unl)
    probs = np.zeros((n, N_WINDOWS, N_CLASSES), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for chunks, idxs in tqdm(dl, desc="R2 teacher"):
            B = chunks.shape[0]
            x = chunks.view(B * 3, CLIP_SAMPLES).to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip, _ = model(x)
            p = torch.sigmoid(clip).float().cpu().numpy().reshape(B, 3, N_CLASSES)
            for bi, idx in enumerate(idxs.tolist()):
                for ci in range(3):
                    probs[idx, ci * 4:(ci + 1) * 4] = p[bi, ci].astype(np.float16)
    print(f"Inference: {time.time()-t0:.0f}s")

    np.savez_compressed(OUT / "pseudo_probs_r2.npz", probs=probs, filenames=np.array(unl))
    mx = probs.max(-1)
    for thr in [0.1, 0.3, 0.5, 0.7]:
        print(f"  max>{thr}: {(mx>thr).mean()*100:.1f}%")
    print(f"Saved: {OUT}/pseudo_probs_r2.npz")


if __name__ == "__main__":
    main()
