#!/usr/bin/env python3
"""
exp4_extract_perch.py — Extract Perch v2 soft labels for all train_audio.
Run this BEFORE exp4_perch_kd.py. CPU-only TensorFlow.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
PERCH_DIR = ROOT / "perch_v2"
OUT_DIR = ROOT / "experiments" / "exp4_outputs"
PERCH_CACHE = OUT_DIR / "perch_cache"
PERCH_CACHE.mkdir(parents=True, exist_ok=True)

SR = 32000
CHUNK_LEN = 160000  # 5s @ 32kHz
TEMPERATURE = 3.0
NUM_CLASSES = 234
BATCH_SIZE = 8  # batch inference for speed

# Build Perch -> BirdCLEF mapping
taxonomy_df = pd.read_csv(DATA / "taxonomy.csv")
SPECIES_LIST = sorted(taxonomy_df["primary_label"].astype(str).tolist())
SPECIES2IDX = {sp: i for i, sp in enumerate(SPECIES_LIST)}
sci_to_pl = dict(zip(taxonomy_df["scientific_name"], taxonomy_df["primary_label"]))
perch_labels = open(PERCH_DIR / "assets" / "labels.csv").read().strip().split("\n")

PERCH_TO_BC = {}
for pi, pname in enumerate(perch_labels):
    if pname in sci_to_pl and sci_to_pl[pname] in SPECIES2IDX:
        PERCH_TO_BC[pi] = SPECIES2IDX[sci_to_pl[pname]]

print(f"Perch->BirdCLEF mapping: {len(PERCH_TO_BC)} species")

# Load model
print("Loading Perch v2...")
model = tf.saved_model.load(str(PERCH_DIR))
infer = model.signatures["serving_default"]
print("Perch v2 loaded.")

# Process with batched inference
train_df = pd.read_csv(DATA / "train.csv")
start_time = time.time()
skipped = 0
already_cached = 0

# Collect work items (idx, filepath) that need processing
work_items = []
for idx, row in train_df.iterrows():
    cache_path = PERCH_CACHE / f"{idx}.npy"
    if cache_path.exists():
        already_cached += 1
        continue
    filepath = DATA / "train_audio" / row["filename"]
    if not filepath.exists():
        skipped += 1
        continue
    work_items.append((idx, filepath))

print(f"Already cached: {already_cached}, To process: {len(work_items)}, Skipped: {skipped}")

# Perch index array for fast mapping
perch_indices = np.array(list(PERCH_TO_BC.keys()), dtype=np.int64)
bc_indices = np.array(list(PERCH_TO_BC.values()), dtype=np.int64)


def process_batch(items):
    """Process a batch of (idx, filepath) pairs."""
    # Load audio and extract first 5s chunk for each
    batch_chunks = []
    valid_items = []
    for idx, filepath in items:
        try:
            wav, _ = librosa.load(filepath, sr=SR)
            if len(wav) == 0:
                continue
        except Exception:
            continue

        # Take first 5s chunk (Perch input)
        chunk = wav[:CHUNK_LEN]
        if len(chunk) < CHUNK_LEN:
            chunk = np.pad(chunk, (0, CHUNK_LEN - len(chunk)))
        batch_chunks.append(chunk[:CHUNK_LEN])
        valid_items.append((idx, wav))

    if not batch_chunks:
        return

    # Batch Perch inference
    batch_arr = np.stack(batch_chunks, axis=0)
    out = infer(tf.constant(batch_arr, dtype=tf.float32))
    batch_logits = out["label"].numpy()  # (B, 14795)

    # For files longer than 5s, also process additional chunks
    for i, (idx, wav) in enumerate(valid_items):
        n_chunks = max(1, len(wav) // CHUNK_LEN)
        n_chunks = min(n_chunks, 4)

        if n_chunks > 1:
            extra_chunks = []
            for c in range(1, n_chunks):
                start = c * CHUNK_LEN
                chunk = wav[start:start + CHUNK_LEN]
                if len(chunk) < CHUNK_LEN:
                    chunk = np.pad(chunk, (0, CHUNK_LEN - len(chunk)))
                extra_chunks.append(chunk[:CHUNK_LEN])

            if extra_chunks:
                extra_arr = np.stack(extra_chunks, axis=0)
                extra_out = infer(tf.constant(extra_arr, dtype=tf.float32))
                extra_logits = extra_out["label"].numpy()
                avg_logits = np.mean(
                    np.concatenate([batch_logits[i:i+1], extra_logits], axis=0),
                    axis=0
                )
            else:
                avg_logits = batch_logits[i]
        else:
            avg_logits = batch_logits[i]

        # Map to BirdCLEF classes
        bc_logits = np.full(NUM_CLASSES, -20.0, dtype=np.float32)
        bc_logits[bc_indices] = avg_logits[perch_indices]

        # Temperature-scaled soft probabilities
        bc_soft = 1.0 / (1.0 + np.exp(-bc_logits / TEMPERATURE))
        np.save(PERCH_CACHE / f"{idx}.npy", bc_soft)


# Process in batches
for i in tqdm(range(0, len(work_items), BATCH_SIZE), desc="Perch extraction"):
    batch = work_items[i:i + BATCH_SIZE]
    process_batch(batch)

elapsed = (time.time() - start_time) / 60
print(f"\nDone. Cached: {already_cached}, New: {len(work_items)}, Skipped: {skipped}")
print(f"Time: {elapsed:.1f} min")

# Write done flag
(PERCH_CACHE / "_done.flag").touch()
