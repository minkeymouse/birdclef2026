"""exp155 (DEFERRED — awaiting go-ahead): Full multi-region BG mining from
train_audio.

Builds on exp154 diagnostic (+38% PSD diversity confirmed). Extracts bottom-K
lowest-RMS 5-sec windows per train_audio clip across ALL 35k+ files. Saves
raw audio + metadata.

Cost: ~15 min CPU (8 workers), ~24 GB disk for ~43k windows × 5s × 32kHz × f32.

Two-stage filter (this is stage 1):
  Stage 1 (this script): energy-based (bottom-K RMS per file)
  Stage 2 (exp156): Perch QC — drop any window with top-1 species prob > QC_TAU

Output:
  exp155_outputs/bg_multiregion_raw.npz   shape (N, 160000) float32
  exp155_outputs/bg_multiregion_meta.parquet   per-window metadata
"""
import sys, os, time, json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
OUT = ROOT / "experiments/_data_pipelines/exp155_outputs"
OUT.mkdir(exist_ok=True)

SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC
BOTTOM_K = 2     # take 2 lowest-RMS windows per file
N_WORKERS = 8


def file_quiet_windows(arg):
    """Return (windows, rms_list, fname) for one file, k lowest RMS."""
    idx, fname = arg
    path = DATA / "train_audio" / fname
    try:
        x, sr = sf.read(str(path), dtype="float32")
    except Exception:
        return idx, fname, [], []
    if sr != SR or len(x) < WIN_SAMPLES:
        return idx, fname, [], []
    if x.ndim > 1:
        x = x.mean(axis=1)
    n_win = len(x) // WIN_SAMPLES
    if n_win < 2:
        return idx, fname, [], []
    rms = np.array([np.sqrt(np.mean(x[i*WIN_SAMPLES:(i+1)*WIN_SAMPLES]**2))
                    for i in range(n_win)])
    order = np.argsort(rms)[:BOTTOM_K]
    wins = [x[i*WIN_SAMPLES:(i+1)*WIN_SAMPLES].copy().astype(np.float32) for i in order]
    rms_sel = rms[order].astype(np.float32).tolist()
    return idx, fname, wins, rms_sel


def main():
    train = pd.read_csv(DATA / "train.csv")
    args = list(enumerate(train.filename.tolist()))
    print(f"[exp155] mining {len(args)} train_audio files, K={BOTTOM_K}, workers={N_WORKERS}")

    t0 = time.time()
    all_wins = []
    meta_rows = []
    with Pool(N_WORKERS) as pool:
        for cnt, (idx, fname, wins, rms_sel) in enumerate(pool.imap_unordered(file_quiet_windows, args, chunksize=8)):
            for w, r in zip(wins, rms_sel):
                all_wins.append(w)
                row = train.iloc[idx]
                meta_rows.append({
                    "src_idx": idx,
                    "filename": fname,
                    "rms": r,
                    "primary_label": row.primary_label,
                    "class_name": row.class_name,
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "collection": row.collection,
                })
            if (cnt + 1) % 1000 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (len(args) / (cnt + 1) - 1)
                print(f"  [{cnt+1}/{len(args)}] wins={len(all_wins)} t={elapsed:.0f}s ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[exp155] done: {len(all_wins)} windows from {len(args)} files in {elapsed:.0f}s")

    arr = np.stack(all_wins).astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    print(f"  arr.shape={arr.shape}, ~{arr.nbytes/1e9:.1f} GB")
    print(f"  unique src classes: {meta.class_name.value_counts().to_dict()}")

    np.savez(OUT / "bg_multiregion_raw.npz", windows=arr)
    meta.to_parquet(OUT / "bg_multiregion_meta.parquet")
    print(f"[exp155] saved → {OUT}")


if __name__ == "__main__":
    main()
