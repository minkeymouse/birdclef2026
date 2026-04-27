#!/usr/bin/env python3
"""exp49a — Extract quiet 5-sec windows from 2025 soundscapes as BG mix source.

From 9726 2025 soundscape files, sample quiet (low RMS) 5-sec windows and
save as npy for fast loading during exp50/51 training.

Criteria for 'quiet':
  - RMS < 20th percentile of the file
  - Not all silence (RMS > 1e-4)
  - Not clipping (abs max < 0.95)

Output: experiments/exp49_outputs/bg_quiet_2025.npz
  windows: (N, 32000*5) float32, with N ~ 20000 samples
  file_ids: source file indicator
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import librosa
import random

ROOT = Path("/data/birdclef2026")
DATA25 = ROOT / "data/birdclef-2025"
OUT = ROOT / "experiments/exp49_outputs"
OUT.mkdir(exist_ok=True, parents=True)
SR = 32000
WIN = SR * 5
N_PER_FILE = 3  # 3 windows per soundscape file
MAX_FILES = 4000  # cap for speed (9726 total)
SEED = 42


def main():
    random.seed(SEED); np.random.seed(SEED)
    files = sorted((DATA25 / "train_soundscapes").glob("*.ogg"))
    print(f"Found {len(files)} 2025 soundscape files")
    random.shuffle(files)
    files = files[:MAX_FILES]

    all_wins = []
    file_ids = []
    for fidx, f in enumerate(files):
        try:
            wav, sr = sf.read(str(f), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            if len(wav) < WIN * 4: continue  # skip very short files

            # Split into non-overlapping 5-sec windows
            n_w = len(wav) // WIN
            if n_w < 3: continue
            wins = wav[:n_w * WIN].reshape(n_w, WIN)
            rms = np.sqrt((wins ** 2).mean(axis=1) + 1e-10)
            # Filter: not silent, not clipping, RMS in bottom 40%
            max_abs = np.abs(wins).max(axis=1)
            valid = (rms > 1e-4) & (max_abs < 0.95)
            if valid.sum() == 0: continue
            rms_valid = rms[valid]
            threshold = np.quantile(rms_valid, 0.4) if len(rms_valid) > 3 else rms_valid.max()
            quiet_mask = valid & (rms <= threshold)
            q_idx = np.where(quiet_mask)[0]
            if len(q_idx) == 0: continue
            # Sample N_PER_FILE
            pick = np.random.choice(q_idx, size=min(N_PER_FILE, len(q_idx)), replace=False)
            for pi in pick:
                all_wins.append(wins[pi].astype(np.float32))
                file_ids.append(fidx)
            if (fidx + 1) % 200 == 0:
                print(f"  progress: {fidx+1}/{len(files)}  collected={len(all_wins)}")
        except Exception as e:
            continue

    arr = np.stack(all_wins)
    fids = np.array(file_ids, dtype=np.int32)
    print(f"\nTotal windows: {arr.shape}, fids {fids.shape}")
    np.savez_compressed(OUT / "bg_quiet_2025.npz", windows=arr, file_ids=fids)
    print(f"Saved → {OUT / 'bg_quiet_2025.npz'}")


if __name__ == "__main__":
    main()
