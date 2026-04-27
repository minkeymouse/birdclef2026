#!/usr/bin/env python3
"""Quick GPU test for Perch v2 inference speed."""
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# NO CUDA_VISIBLE_DEVICES restriction — let TF use GPU

from pathlib import Path
import numpy as np
import tensorflow as tf

print("Physical devices:", tf.config.list_physical_devices())

PERCH_DIR = Path("/data/birdclef2026/perch_v2")
BATCH = 96  # 8 files × 12 windows
WINDOW_SAMPLES = 5 * 32000

t0 = time.time()
print("Loading Perch v2 SavedModel on GPU...")
model = tf.saved_model.load(str(PERCH_DIR))
infer = model.signatures["serving_default"]
print(f"Model loaded in {time.time()-t0:.1f}s")

# Dummy batch to force JIT
x = np.random.randn(BATCH, WINDOW_SAMPLES).astype(np.float32)
print("First inference call (JIT compile may take 30+ min)...")
t0 = time.time()
out = infer(inputs=tf.constant(x))
emb = out["embedding"].numpy()
print(f"FIRST call: {time.time()-t0:.1f}s  emb.shape={emb.shape}")

# Second call should be fast
print("Second call (post-JIT)...")
t0 = time.time()
out = infer(inputs=tf.constant(x))
emb = out["embedding"].numpy()
print(f"SECOND call: {time.time()-t0:.1f}s")

# Third call timing
t0 = time.time()
for _ in range(5):
    out = infer(inputs=tf.constant(x))
    emb = out["embedding"].numpy()
print(f"5 more calls: {time.time()-t0:.1f}s → {(time.time()-t0)/5:.2f}s/batch")
print("CPU was 8s/batch. If GPU < 2s/batch → worth switching.")
