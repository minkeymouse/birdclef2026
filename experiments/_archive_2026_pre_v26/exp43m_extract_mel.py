#!/usr/bin/env python3
"""exp43m_extract_mel — log-mel spectrogram for all 10,658 SS files.

Output: (127896, T=50, F=64) fp16 npz (~820 MB)
Each 5-sec window → 50 time frames (100ms each), 64 mel bins.
GPU-accelerated via torchaudio.
"""
from __future__ import annotations
import os, re, time
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm.auto import tqdm

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
OUT = ROOT / "experiments/exp43m_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
BATCH_FILES = 16
FILE_SAMPLES = SR * 60
T_OUT = 50          # target frames per window
F_OUT = 64          # mel bins

# Hop adjusted to land T=50 frames on a 5-sec window
HOP = WINDOW_SAMPLES // T_OUT   # = 3200 samples (100ms)
N_FFT = 2048
FMIN, FMAX = 50, 14000

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_meta(name):
    m = FNAME_RE.match(name)
    if not m: return None, -1
    _, site, _, hms = m.groups()
    return site, int(hms[:2])


def read_60s(path):
    wav, _ = sf.read(str(path), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if len(wav) < FILE_SAMPLES:
        wav = np.pad(wav, (0, FILE_SAMPLES - len(wav)))
    return wav[:FILE_SAMPLES]


def main():
    paths = sorted((DATA / "train_soundscapes").glob("*.ogg"))
    print(f"Found {len(paths)} SS files")

    out_npz = OUT / "mel_ss_all.npz"
    if out_npz.exists():
        print("Already exists."); return

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
        f_min=FMIN, f_max=FMAX, n_mels=F_OUT, power=2.0, center=False,
    ).to(DEVICE)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(DEVICE)

    n_files = len(paths)
    n_rows = n_files * N_WINDOWS
    mel_all = np.zeros((n_rows, T_OUT, F_OUT), dtype=np.float16)
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    write = 0; t_start = time.time()
    for start in tqdm(range(0, n_files, BATCH_FILES), desc="Mel-GPU"):
        batch = paths[start:start + BATCH_FILES]
        bn = len(batch)
        x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        bstart = write
        for bi, p in enumerate(batch):
            try: audio = read_60s(p)
            except Exception: audio = np.zeros(FILE_SAMPLES, dtype=np.float32)
            x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
            site, hour = parse_meta(p.name)
            for t_idx, t_end in enumerate(range(5, 65, 5)):
                wi = write + bi * N_WINDOWS + t_idx
                row_ids[wi] = f"{p.stem}_{t_end}"; filenames[wi] = p.name
                sites[wi] = site; hours[wi] = hour
        write += bn * N_WINDOWS

        xb = torch.from_numpy(x).to(DEVICE)              # (bn*12, 160000)
        mb = mel(xb)                                      # (bn*12, 64, T)
        mb = to_db(mb)                                    # log-mel in dB
        if mb.shape[-1] != T_OUT:                         # trim if center=False rounding
            mb = mb[..., :T_OUT]
        mb = mb.transpose(1, 2).contiguous()              # (B, T, F)
        mel_all[bstart:write] = mb.half().cpu().numpy()

        if (start // BATCH_FILES) % 50 == 0:
            elapsed = time.time() - t_start
            pct = (start + bn) / n_files
            eta = elapsed / max(pct, 1e-6) - elapsed
            print(f"  [{start+bn}/{n_files}] {elapsed:.0f}s, ETA {eta/60:.1f}m")

    print(f"Done. mel {mel_all.shape} {mel_all.dtype} ({mel_all.nbytes/1e9:.1f} GB). Wall {(time.time()-t_start)/60:.1f}m")

    meta = pd.DataFrame({"row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours})
    meta.to_parquet(OUT / "mel_ss_all_meta.parquet", index=False)
    # Use uncompressed savez to avoid multi-minute zlib bottleneck on 800 MB fp16
    np.savez(out_npz, mel=mel_all)
    print(f"Saved: {out_npz} ({out_npz.stat().st_size/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
