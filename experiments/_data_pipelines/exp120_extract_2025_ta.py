#!/usr/bin/env python3
"""exp120 — Extract Perch embeddings for 2025 train_audio (41 overlap species).

ONNX-GPU. Reads first 5 seconds of each clip (matches exp22 protocol).
Output drops into experiments/_data_pipelines/exp120_outputs/.
"""
from __future__ import annotations
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
from tqdm.auto import tqdm

ROOT = Path("/data/birdclef2026")
DATA25 = ROOT / "data/birdclef-2025"
DATA26 = ROOT / "data/birdclef-2026"
ONNX = Path("/tmp/perch_v2.onnx")
OUT = ROOT / "experiments/_data_pipelines/exp120_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
BATCH = 32
N_CLS = 234


def read_5s(path):
    try:
        wav, _ = sf.read(str(path), dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        if len(wav) >= WINDOW_SAMPLES:
            return wav[:WINDOW_SAMPLES]
        else:
            return np.pad(wav, (0, WINDOW_SAMPLES - len(wav)))
    except Exception as e:
        return None


def main():
    print("=== exp120: 2025 train_audio Perch extraction ===\n", flush=True)

    # 1. Build index of clips for overlap species
    tax25 = pd.read_csv(DATA25 / "taxonomy.csv")
    tax26 = pd.read_csv(DATA26 / "taxonomy.csv")
    overlap = sorted(set(tax25.primary_label) & set(tax26.primary_label))
    print(f"  Overlap species: {len(overlap)}")

    # 2026 sample_submission column order = canonical primary order
    sample_sub = pd.read_csv(DATA26 / "sample_submission.csv")
    primary_2026 = sample_sub.columns[1:].tolist()
    sp2idx = {s: i for i, s in enumerate(primary_2026)}

    train25 = pd.read_csv(DATA25 / "train.csv")
    train25_overlap = train25[train25.primary_label.isin(overlap)].reset_index(drop=True)
    print(f"  2025 clips for overlap species: {len(train25_overlap)}")
    print(f"  by collection: {train25_overlap.collection.value_counts().to_dict()}")

    # 2. Load Perch ONNX with GPU
    print("\n  Loading Perch ONNX (GPU)...")
    sess = ort.InferenceSession(str(ONNX), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"  Providers: {sess.get_providers()[:1]}")

    # 3. Iterate
    n = len(train25_overlap)
    emb = np.zeros((n, 1536), dtype=np.float32)
    y_idx = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=np.bool_)

    audio_buf = np.zeros((BATCH, WINDOW_SAMPLES), dtype=np.float32)
    file_paths = []
    queued_indices = []

    t0 = time.time()
    for i, row in tqdm(train25_overlap.iterrows(), total=n, desc="2025 TA"):
        path = DATA25 / "train_audio" / row.filename
        if not path.exists():
            continue
        wav = read_5s(path)
        if wav is None:
            continue

        audio_buf[len(queued_indices)] = wav
        queued_indices.append(i)
        file_paths.append(str(path))

        if len(queued_indices) >= BATCH:
            try:
                out = sess.run(["embedding"], {"inputs": audio_buf[:len(queued_indices)]})[0]
                for k, idx in enumerate(queued_indices):
                    emb[idx] = out[k]
                    y_idx[idx] = sp2idx.get(train25_overlap.iloc[idx].primary_label, -1)
                    valid[idx] = (y_idx[idx] >= 0)
            except Exception as e:
                print(f"\n  Batch failed: {e}")
            queued_indices = []
            file_paths = []

    # Flush
    if queued_indices:
        out = sess.run(["embedding"], {"inputs": audio_buf[:len(queued_indices)]})[0]
        for k, idx in enumerate(queued_indices):
            emb[idx] = out[k]
            y_idx[idx] = sp2idx.get(train25_overlap.iloc[idx].primary_label, -1)
            valid[idx] = (y_idx[idx] >= 0)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min, valid {valid.sum()}/{n}")

    # 4. Save
    out_path = OUT / "ta25_perch.npz"
    np.savez_compressed(
        out_path,
        emb=emb, y_idx=y_idx, valid=valid,
        primary_label=train25_overlap.primary_label.values.astype("U16"),
        filename=train25_overlap.filename.values.astype("U200"),
    )
    print(f"  Saved → {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    # Summary
    print("\n  Per-species clip counts (top 20):")
    cnt = pd.Series(train25_overlap[valid].primary_label.values).value_counts().head(20)
    print(cnt.to_string())


if __name__ == "__main__":
    main()
