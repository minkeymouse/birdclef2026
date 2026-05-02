#!/usr/bin/env python3
"""exp125 — Run exp50 SED on full 10,658 unlabeled SS files.

Output: experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz
shape (n_files * 12, 234) = (~127,896, 234)

Then we have everything needed to compute v33 on unlabeled:
  - Perch embeddings: experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz
  - Perch scores: same file
  - exp50 scores: this output
  - V9 gate + file-max applied as in v33 pipeline
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
import timm
import soundfile as sf

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
EXP50_CKPT = ROOT / "experiments/_data_pipelines/exp50_outputs/best_ckpt.pt"
OUT = ROOT / "experiments/_data_pipelines/exp125_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
SR = 32000
N_WINDOWS = 12
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = SR * 60
N_FFT = 2048; HOP = 512; N_MELS = 128
F_MIN, F_MAX = 50, 14000
SED_CHUNK_SEC = 20
SED_CHUNK_SAMPLES = SR * SED_CHUNK_SEC

N_CLS = 234


class _MelExt(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=F_MIN, f_max=F_MAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class _SEDHead(nn.Module):
    def __init__(self, fd, nc):
        super().__init__()
        self.att = nn.Conv1d(fd, nc, 1); self.cla = nn.Conv1d(fd, nc, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values


class _SED(nn.Module):
    def __init__(self, bb="hgnetv2_b0.ssld_stage2_ft_in1k"):
        super().__init__()
        self.mel = _MelExt(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(bb, pretrained=False, in_chans=1, num_classes=0, global_pool='')
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = _SEDHead(f.shape[1], N_CLS)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
        clip, _ = self.head(f); return clip


def main():
    print("=== exp125: exp50 inference on unlabeled SS ===\n", flush=True)

    files = sorted((DATA / "train_soundscapes").glob("*.ogg"))
    print(f"Found {len(files)} unlabeled SS files")

    # Load model
    st = torch.load(str(EXP50_CKPT), map_location=DEVICE, weights_only=False)
    bb = st.get("config", {}).get("backbone", "hgnetv2_b0.ssld_stage2_ft_in1k")
    m = _SED(bb).to(DEVICE).eval()
    m.load_state_dict(st["state_dict"])
    print(f"Loaded exp50 ({bb}, val_SS={st.get('val_SS', '?')})")

    n_files = len(files)
    n_rows = n_files * N_WINDOWS
    out = np.zeros((n_rows, N_CLS), dtype=np.float32)
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)

    BATCH_F = 8
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, n_files, BATCH_F):
            batch = files[s:s+BATCH_F]
            chunks = []
            for fn in batch:
                wav, _ = sf.read(str(fn), dtype="float32", always_2d=False)
                if wav.ndim == 2: wav = wav.mean(axis=1)
                if len(wav) < FILE_SAMPLES: wav = np.pad(wav, (0, FILE_SAMPLES - len(wav)))
                wav = wav[:FILE_SAMPLES]
                # 3 chunks of 20s
                for ci in range(3):
                    st_idx = ci * SED_CHUNK_SAMPLES
                    chunks.append(wav[st_idx:st_idx + SED_CHUNK_SAMPLES])

            x = torch.from_numpy(np.stack(chunks).astype(np.float32)).to(DEVICE)
            logits = m(x)
            probs = torch.sigmoid(logits).cpu().numpy()  # (3*BATCH_F, 234)

            # Each file has 3 chunks, each chunk represents 4 windows
            for bi, fn in enumerate(batch):
                for ci in range(3):
                    chunk_prob = probs[bi*3 + ci]
                    # Windows for this chunk: 4 windows of 5sec each in 20sec chunk
                    # End times: ci*20+5, ci*20+10, ci*20+15, ci*20+20
                    for win_in_chunk in range(4):
                        end_sec = ci * 20 + (win_in_chunk + 1) * 5
                        # Global window index: (file_idx * N_WINDOWS) + (end_sec/5 - 1)
                        file_idx = s + bi
                        win_idx = (end_sec // 5) - 1
                        global_idx = file_idx * N_WINDOWS + win_idx
                        out[global_idx] = chunk_prob
                        row_ids[global_idx] = f"{fn.stem}_{end_sec}"
                        filenames[global_idx] = fn.name

            if s % 80 == 0:
                elapsed = time.time() - t0
                rate = (s + BATCH_F) / max(elapsed, 0.1)
                eta = (n_files - s - BATCH_F) / max(rate, 0.1)
                print(f"  {s + BATCH_F}/{n_files} files, "
                      f"{elapsed:.0f}s elapsed, ETA {eta/60:.1f} min", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")

    out_path = OUT / "exp50_unlabeled_scores.npz"
    np.savez_compressed(out_path,
        scores=out, row_ids=row_ids, filenames=filenames)
    print(f"Saved → {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"  shape {out.shape}, range [{out.min():.5f}, {out.max():.5f}]")


if __name__ == "__main__":
    main()
