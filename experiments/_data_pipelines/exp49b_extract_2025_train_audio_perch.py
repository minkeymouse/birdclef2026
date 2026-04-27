#!/usr/bin/env python3
"""exp49b — Perch v2 embedding for 2025 train_audio clips (ONNX GPU).

For each 2025 train_audio clip, take center 5-sec window and extract
Perch embedding. Output: (N, 1536) float32 + primary_label array.

Used to retrain V9 taxon gate with 2025+2026 pool (exp49c).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import onnxruntime as ort

ROOT = Path("/data/birdclef2026")
DATA25 = ROOT / "data/birdclef-2025"
OUT = ROOT / "experiments/exp49_outputs"
OUT.mkdir(exist_ok=True, parents=True)
ONNX_PATH = Path("/tmp/perch_v2.onnx")
SR = 32000; WIN = SR * 5
BATCH = 128


def load_center(path):
    try:
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        if len(wav) < WIN:
            wav = np.tile(wav, WIN // max(len(wav), 1) + 1)[:WIN]
        else:
            s = (len(wav) - WIN) // 2
            wav = wav[s:s + WIN]
        return wav.astype(np.float32)
    except Exception:
        return np.zeros(WIN, dtype=np.float32)


def main():
    df = pd.read_csv(DATA25 / "train.csv")
    print(f"2025 train.csv: {len(df)} clips")
    paths = [DATA25 / "train_audio" / f for f in df.filename.values]
    # Quick sanity
    exists_mask = np.array([p.exists() for p in paths])
    print(f"  files exist: {exists_mask.sum()} / {len(paths)}")
    if exists_mask.sum() < len(paths):
        df = df[exists_mask].reset_index(drop=True)
        paths = [DATA25 / "train_audio" / f for f in df.filename.values]

    print("Loading Perch ONNX on GPU...")
    sess = ort.InferenceSession(str(ONNX_PATH),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"  providers={sess.get_providers()}")
    iname = sess.get_inputs()[0].name

    N = len(df)
    embs = np.zeros((N, 1536), dtype=np.float32)
    t0 = time.time()
    for i in range(0, N, BATCH):
        j = min(N, i + BATCH)
        wavs = np.stack([load_center(p) for p in paths[i:j]])
        out = sess.run(["embedding", "label"], {iname: wavs})
        embs[i:j] = out[0]
        if (i // BATCH) % 20 == 0:
            print(f"  {i}/{N}  elapsed {time.time()-t0:.0f}s")
    print(f"Done. Elapsed {time.time()-t0:.0f}s")

    np.savez_compressed(OUT / "train_audio_2025_perch.npz",
                         embs=embs,
                         primary_label=df.primary_label.astype(str).values,
                         filename=df.filename.values)
    print(f"Saved → {OUT / 'train_audio_2025_perch.npz'}")


if __name__ == "__main__":
    main()
