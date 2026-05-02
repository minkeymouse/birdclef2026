"""exp155 — Multi-region BG mining from train_audio (sensible, ~12k pool).

Builds on exp154 diagnostic (+38% PSD diversity confirmed). Sample ~12k
train_audio files stratified by lat/lon bucket, take 1 quietest 5-sec
window per file. Pool size matches existing `bg_quiet_2025.npz` (11.9k
Pantanal-only) — comparable scale, but multi-region.

**Memory note (2026-05-03):** Earlier version used K=2 × all 35k files =
55k windows × 640 KB = 33 GB single array, which OOM'd a 64 GiB machine
during np.stack (peak ~66 GB). This version is bounded to ~12k × 640 KB
= 7.7 GB peak — comfortable for a 64 GiB box.

Output (exp50/exp159 compatible — same key as bg_quiet_2025.npz):
  exp155_outputs/bg_multiregion_raw.npz   `windows`: (N, 160000) float32
  exp155_outputs/bg_multiregion_meta.parquet   per-window metadata
"""
import sys, os, time, json, random
from pathlib import Path
from multiprocessing import Pool
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
TARGET_N = 12000      # match existing 11.9k Pantanal pool
N_WORKERS = 8
SEED = 42

random.seed(SEED); np.random.seed(SEED)


def lat_lon_bucket(lat, lon, size=10):
    if pd.isna(lat) or pd.isna(lon):
        return "NA"
    return f"{int(np.floor(lat/size)*size):+d}_{int(np.floor(lon/size)*size):+d}"


def file_quiet_window(arg):
    """Return (idx, fname, win, rms) — single quietest 5-sec window from file."""
    idx, fname = arg
    path = DATA / "train_audio" / fname
    try:
        x, sr = sf.read(str(path), dtype="float32")
    except Exception:
        return idx, fname, None, None
    if sr != SR or len(x) < WIN_SAMPLES * 2:
        return idx, fname, None, None
    if x.ndim > 1:
        x = x.mean(axis=1)
    n_win = len(x) // WIN_SAMPLES
    rms = np.array([
        np.sqrt(np.mean(x[i*WIN_SAMPLES:(i+1)*WIN_SAMPLES]**2))
        for i in range(n_win)
    ])
    i_q = int(np.argmin(rms))
    win = x[i_q*WIN_SAMPLES:(i_q+1)*WIN_SAMPLES].astype(np.float32).copy()
    return idx, fname, win, float(rms[i_q])


def stratified_sample(train, n_target):
    """Return n_target file rows, stratified by lat/lon bucket (10° grid)."""
    train = train.copy()
    train["bucket"] = [lat_lon_bucket(la, lo)
                        for la, lo in zip(train.latitude, train.longitude)]
    counts = train.bucket.value_counts()
    n_buckets = len(counts)
    quota = max(1, n_target // n_buckets)
    pieces = []
    for b in counts.index:
        sub = train[train.bucket == b]
        take = min(len(sub), quota)
        pieces.append(sub.sample(take, random_state=SEED))
    sampled = pd.concat(pieces).reset_index(drop=True)
    if len(sampled) > n_target:
        sampled = sampled.sample(n_target, random_state=SEED).reset_index(drop=True)
    elif len(sampled) < n_target:
        # top up from largest bucket
        rest = train[~train.filename.isin(sampled.filename)]
        topup = rest.sample(min(n_target - len(sampled), len(rest)), random_state=SEED)
        sampled = pd.concat([sampled, topup]).reset_index(drop=True)
    return sampled


def main():
    train = pd.read_csv(DATA / "train.csv")
    sampled = stratified_sample(train, TARGET_N)
    print(f"[exp155] sampled {len(sampled)} files across {sampled['bucket'].nunique()} buckets")
    print(f"  expected output: {len(sampled) * WIN_SAMPLES * 4 / 1e9:.1f} GB")
    bucket_dist = sampled.bucket.value_counts().head(8).to_dict()
    print(f"  top buckets: {bucket_dist}")

    args = list(zip(sampled.index.tolist(), sampled.filename.tolist()))
    t0 = time.time()
    wins = [None] * len(args)
    rms = [None] * len(args)

    with Pool(N_WORKERS) as pool:
        for cnt, (idx, fname, win, r) in enumerate(
            pool.imap_unordered(file_quiet_window, args, chunksize=8)
        ):
            if win is not None:
                wins[idx] = win
                rms[idx] = r
            if (cnt + 1) % 1000 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (len(args) / (cnt + 1) - 1)
                ok = sum(1 for w in wins if w is not None)
                print(f"  [{cnt+1}/{len(args)}] ok={ok} t={elapsed:.0f}s ETA={eta:.0f}s",
                      flush=True)

    keep = [i for i, w in enumerate(wins) if w is not None]
    n_kept = len(keep)
    elapsed = time.time() - t0
    print(f"[exp155] mining done: {n_kept}/{len(args)} files yielded windows in {elapsed:.0f}s")

    # Single np.stack — small enough (~7.7 GB) to fit comfortably
    arr = np.stack([wins[i] for i in keep])
    print(f"  arr.shape={arr.shape}, ~{arr.nbytes/1e9:.1f} GB")

    sampled_kept = sampled.iloc[keep].copy().reset_index(drop=True)
    sampled_kept["rms"] = [rms[i] for i in keep]
    print(f"  unique src classes: {sampled_kept.class_name.value_counts().to_dict()}")

    np.savez(OUT / "bg_multiregion_raw.npz", windows=arr)
    sampled_kept.to_parquet(OUT / "bg_multiregion_meta.parquet")
    print(f"[exp155] saved → {OUT}")


if __name__ == "__main__":
    main()
