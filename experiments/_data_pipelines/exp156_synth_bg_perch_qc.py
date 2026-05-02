"""exp156 (DEFERRED — needs Perch ONNX): Perch QC filter on the multi-region
BG candidates from exp155.

Reject any window where Perch top-1 species prob exceeds QC_TAU. Goal:
purge windows that contain audible target-species calls (would pollute
BG mixing). Soltero et al. 2025 emphasize strict BG cleanliness.

Prereq: download Perch ONNX to /tmp/perch_v2.onnx via
  uv run kaggle datasets download rishikeshjani/perch-onnx-for-birdclef-2026 -p /tmp --unzip

Cost: ~10 min on RTX 5090 GPU.

Output:
  exp156_outputs/bg_multiregion_clean.npz   filtered raw audio
  exp156_outputs/bg_multiregion_clean_meta.parquet
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort

ROOT = Path("/data/birdclef2026")
EXP155 = ROOT / "experiments/_data_pipelines/exp155_outputs"
OUT = ROOT / "experiments/_data_pipelines/exp156_outputs"
OUT.mkdir(exist_ok=True)

PERCH_ONNX = Path("/tmp/perch_v2.onnx")  # CLAUDE.md: ONNX + CUDAExecutionProvider
SR = 32000
WIN_SAMPLES = SR * 5
QC_TAU = 0.05  # reject windows with any species prob > this
BATCH = 64


def main():
    if not PERCH_ONNX.exists():
        print(f"[exp156] ERROR: Perch ONNX not at {PERCH_ONNX}. Copy from Kaggle dataset first.")
        sys.exit(1)
    arr = np.load(EXP155 / "bg_multiregion_raw.npz")["windows"]
    meta = pd.read_parquet(EXP155 / "bg_multiregion_meta.parquet")
    print(f"[exp156] input: {arr.shape}, meta {len(meta)}")

    sess = ort.InferenceSession(str(PERCH_ONNX), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    print(f"[exp156] Perch ONNX: input={in_name}, providers={sess.get_providers()}")

    keep = np.zeros(len(arr), dtype=bool)
    max_prob = np.zeros(len(arr), dtype=np.float32)

    t0 = time.time()
    for i in range(0, len(arr), BATCH):
        batch = arr[i:i+BATCH]
        # Perch v2 expects (B, 160000) float32 mono, 32kHz
        out = sess.run(None, {in_name: batch})[0]  # (B, n_classes)
        # softmax → max prob per window
        m = out.max(axis=1)
        max_prob[i:i+len(batch)] = m
        keep[i:i+len(batch)] = m < QC_TAU
        if (i // BATCH) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+len(batch)}/{len(arr)}] keep_rate={keep[:i+len(batch)].mean():.3f} t={elapsed:.0f}s")

    print(f"[exp156] kept {keep.sum()}/{len(arr)} ({100*keep.mean():.1f}%) at QC_TAU={QC_TAU}")

    arr_clean = arr[keep]
    meta_clean = meta[keep].reset_index(drop=True)
    meta_clean["perch_max_prob"] = max_prob[keep]

    np.savez(OUT / "bg_multiregion_clean.npz", windows=arr_clean)
    meta_clean.to_parquet(OUT / "bg_multiregion_clean_meta.parquet")
    print(f"[exp156] saved → {OUT}")
    print(f"  geographic spread: {meta_clean[['latitude','longitude']].describe()}")


if __name__ == "__main__":
    main()
