#!/usr/bin/env python3
"""exp44 — Multi-teacher pseudo-label generation.

Uses ensemble of SED29 + SED38b + SED39 to generate more robust pseudo labels.
Teacher = mean(sigmoid(SED29_logit), sigmoid(SED38b_logit), sigmoid(SED39_logit))

Rationale: 2025 3rd prize used 10 labelers for pseudo. Ensemble reduces noise.
Our 3 teachers have Pearson 0.430 (SED29-SED39) → meaningful diversity.
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
WEIGHTS = ROOT / "model-weights"
OUT = ROOT / "experiments/exp41_outputs"
OUT.mkdir(exist_ok=True)

SR = 32000
CLIP_SAMPLES = 20 * SR
N_WINDOWS = 12; N_CLASSES = 234
N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000

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
    def __init__(self, backbone):
        super().__init__()
        self.mel = Mel(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(backbone, pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = Head(f.shape[1], N_CLASSES)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        if feat.dim() == 4: feat = feat.mean(2)
        return self.head(feat)


def load(ckpt_path):
    st = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    bb = st.get("backbone", "hgnetv2_b0.ssld_stage2_ft_in1k")
    m = SEDM(bb).eval()
    m.load_state_dict(st["state_dict"])
    return m, bb


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


def run_teacher(model, files, desc):
    dl = DataLoader(SSFullDS(files), batch_size=16, shuffle=False, num_workers=6, pin_memory=True)
    n = len(files)
    probs = np.zeros((n, N_WINDOWS, N_CLASSES), dtype=np.float32)
    model = model.to(DEVICE)
    t0 = time.time()
    with torch.inference_mode():
        for chunks, idxs in tqdm(dl, desc=desc):
            B = chunks.shape[0]
            x = chunks.view(B * 3, CLIP_SAMPLES).to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip, _ = model(x)
            p = torch.sigmoid(clip).float().cpu().numpy().reshape(B, 3, N_CLASSES)
            for bi, idx in enumerate(idxs.tolist()):
                for ci in range(3):
                    probs[idx, ci * 4:(ci + 1) * 4] = p[bi, ci]
    print(f"{desc}: {time.time()-t0:.0f}s")
    return probs


def main():
    all_files = sorted([f.name for f in (DATA / "train_soundscapes").iterdir() if f.suffix == ".ogg"])
    labeled = set(pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().filename.unique())
    unl = [f for f in all_files if f not in labeled]
    print(f"Unlabeled: {len(unl)}")

    teachers = [
        ("SED29", WEIGHTS / "exp29_hgnet_sed.pt"),
        ("SED38b", WEIGHTS / "exp38b_hgnet_sed.pt"),
        ("SED39", WEIGHTS / "exp39_effnet_sed.pt"),
    ]
    all_probs = []
    for name, p in teachers:
        if not p.exists():
            print(f"Missing {name} at {p}")
            continue
        m, bb = load(p)
        print(f"\n{name} ({bb})")
        probs = run_teacher(m, unl, name)
        all_probs.append((name, probs))
        del m
        torch.cuda.empty_cache()

    # Average (sigmoid space)
    ens = np.stack([p for _, p in all_probs]).mean(0).astype(np.float16)
    np.savez_compressed(OUT / "pseudo_probs_ensemble.npz",
                        probs=ens, filenames=np.array(unl),
                        teacher_names=[n for n, _ in all_probs])
    print(f"\nSaved ensemble: {OUT}/pseudo_probs_ensemble.npz  shape={ens.shape}")

    # Stats
    mx = ens.max(-1)
    for thr in [0.1, 0.3, 0.5, 0.7]:
        f = (mx > thr).mean()
        print(f"  max>{thr}: {f*100:.1f}%")

    # Teacher agreement: per chunk, count how many teachers exceed 0.3 per class
    agr = np.zeros_like(ens, dtype=np.uint8)
    for _, p in all_probs:
        agr = agr + (p > 0.3).astype(np.uint8)
    agree_2p = (agr >= 2).sum(-1)  # how many classes have ≥2 teachers confident
    print(f"\nChunks with ≥2-teacher agreement on ≥1 class: {(agree_2p > 0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
