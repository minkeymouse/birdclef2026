#!/usr/bin/env python3
"""exp38b — full 66 labeled SS retrain (no held-out) for submission.

Based on exp38 (55/11 split) which converged at ep9 (Val-A_v2 0.8107).
Train exactly same recipe on all 66 files, save ckpt every epoch from ep7-ep12.
Use ep9 ckpt for submission (matches holdout's best).
"""
from __future__ import annotations
import json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
OUT = ROOT / "experiments" / "exp38b_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5

N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000

BATCH_SIZE = 32; EPOCHS = 12
LR = 1e-3; WD = 1e-2
NUM_WORKERS = 8
MIXUP_ALPHA = 0.5; MIXUP_P = 0.5
SS_SAMPLE_SHARE = 0.25
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
N_CLASSES = 234


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_ss():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sample_sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)}))
          .reset_index(name="label_list"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    return sc, l2i


def load_train_audio(l2i):
    t = pd.read_csv(DATA / "train.csv")
    t["primary_label"] = t["primary_label"].astype(str)

    def parse_sec(x):
        if pd.isna(x) or x == "[]": return []
        return [s.strip().strip("'\"") for s in str(x).strip("[]").split(",") if s.strip()]

    t["sec_labels"] = t["secondary_labels"].apply(parse_sec)
    return t


class AudioDataset(Dataset):
    def __init__(self, df, l2i, root=DATA / "train_audio"):
        self.df = df.reset_index(drop=True)
        self.l2i = l2i; self.nc = len(l2i); self.root = root

    def __len__(self): return len(self.df)

    def _load(self, fn):
        try:
            y, sr = sf.read(self.root / fn, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(1)
            if sr != SR:
                y = torchaudio.functional.resample(torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
        except Exception:
            y = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        return y.astype(np.float32)

    def _crop(self, y):
        if len(y) < CLIP_SAMPLES:
            reps = (CLIP_SAMPLES + len(y) - 1) // len(y)
            y = np.tile(y, reps)[:CLIP_SAMPLES]
        elif len(y) > CLIP_SAMPLES:
            s = np.random.randint(0, len(y) - CLIP_SAMPLES)
            y = y[s:s + CLIP_SAMPLES]
        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self._crop(self._load(row["filename"]))
        t = np.zeros(self.nc, dtype=np.float32)
        if row["primary_label"] in self.l2i: t[self.l2i[row["primary_label"]]] = 1.0
        for l in row["sec_labels"]:
            if l in self.l2i: t[self.l2i[l]] = 0.5
        return torch.from_numpy(y), torch.from_numpy(t)


class SSDataset(Dataset):
    def __init__(self, sc, l2i, root=DATA / "train_soundscapes"):
        self.df = sc.reset_index(drop=True); self.l2i = l2i; self.nc = len(l2i); self.root = root

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        end = int(row["end_sec"]); start = end - WINDOW_SEC
        try:
            y, sr = sf.read(self.root / row["filename"], dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(1)
            if sr != SR:
                y = torchaudio.functional.resample(torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
        except Exception:
            y = np.zeros(SR * 60, dtype=np.float32)
        c = ((start + end) // 2) * SR
        half = CLIP_SAMPLES // 2
        s = max(0, c - half); e = s + CLIP_SAMPLES
        if e > len(y): e = len(y); s = max(0, e - CLIP_SAMPLES)
        clip = y[s:e]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        t = np.zeros(self.nc, dtype=np.float32)
        for l in row["label_list"]:
            if l in self.l2i: t[self.l2i[l]] = 1.0
        return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(t)


class MixDataset(Dataset):
    def __init__(self, a, b): self.a = a; self.b = b
    def __len__(self): return len(self.a) + len(self.b)
    def __getitem__(self, i):
        return self.a[i] if i < len(self.a) else self.b[i - len(self.a)]


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
        self.backbone = timm.create_model(BACKBONE, pretrained=True, in_chans=1, num_classes=0, global_pool="")
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(f.shape[1], N_CLASSES)

    def forward(self, x):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        if feat.dim() == 4: feat = feat.mean(2)
        return self.head(feat)


def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], torch.maximum(lam * y, (1 - lam) * y[idx])


def main():
    t0 = time.time()
    set_seed(SEED)
    print(f"[exp38b] full 66-file retrain. backbone={BACKBONE}")

    sc, l2i = build_ss()
    print(f"Labeled SS: {sc.filename.nunique()} files ({len(sc)} segs)")

    tdf = load_train_audio(l2i)
    print(f"train_audio: {len(tdf)} rows")

    a_ds = AudioDataset(tdf, l2i)
    s_ds = SSDataset(sc, l2i)
    mix = MixDataset(a_ds, s_ds)

    wa = (1 - SS_SAMPLE_SHARE) / len(a_ds)
    ws = SS_SAMPLE_SHARE / len(s_ds)
    weights = np.concatenate([np.full(len(a_ds), wa, dtype=np.float64),
                              np.full(len(s_ds), ws, dtype=np.float64)])
    sampler = WeightedRandomSampler(weights, num_samples=len(a_ds), replacement=True)
    loader = DataLoader(mix, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=True, drop_last=True)

    model = SEDModel().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    history = []
    for ep in range(1, EPOCHS + 1):
        ep_t = time.time()
        model.train()
        total = 0.0; n = 0
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            if random.random() < MIXUP_P: x, y = mixup(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip, fmax = model(x)
                loss = F.binary_cross_entropy_with_logits(clip, y) + \
                       F.binary_cross_entropy_with_logits(fmax, y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        dt = time.time() - ep_t
        avg = total / n
        print(f"Ep {ep:02d}/{EPOCHS}  loss {avg:.4f}  ({dt:.0f}s)")
        history.append({"epoch": ep, "loss": avg, "time_s": dt})
        # Save all epochs ep7-ep12 for post-hoc pick
        if 7 <= ep <= 12:
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "backbone": BACKBONE},
                       OUT / f"ckpt_ep{ep:02d}.pt")
        (OUT / "results.json").write_text(json.dumps(
            {"backbone": BACKBONE, "clip_sec": CLIP_SEC,
             "ss_files": int(sc.filename.nunique()),
             "history": history, "elapsed_s": time.time() - t0}, indent=2))

    # Default submission ckpt: ep9 (matches exp38 holdout best)
    import shutil
    shutil.copy(OUT / "ckpt_ep09.pt", OUT / "submit_ckpt.pt")
    print(f"\nDone. {(time.time()-t0)/60:.1f} min. Submission ckpt: ep9 → submit_ckpt.pt")


if __name__ == "__main__":
    main()
