"""exp154 (RAN 2026-05-02): Diagnostic — can we mine a diverse multi-region BG
pool from train_audio for synthetic-data domain randomization?

Inspired by DRASDIC (Hoffman 2025) + Soltero (2025): site decoupling via
multi-region BG diversification. Current `bg_quiet_2025.npz` is single-site
(Pantanal). Train_audio (xeno-canto, ~35k clips, lat/lon known) is globally
distributed → silent portions form a free multi-region BG pool.

Steps:
  1. Sample N train_audio clips stratified by geographic bucket.
  2. Per clip, extract the K lowest-energy 5-sec windows (energy = RMS).
  3. For each extracted window, compute mean log-power-spectrum (PSD).
  4. Compare PSD distribution vs current Pantanal-only pool (bg_quiet_2025.npz).
  5. Save extracted candidate windows + diagnostic.

Result (360 files, K=2, 12 buckets):
  - 280/360 files yielded windows (78%)
  - 560 windows total; extrapolated ~43k mineable from full 35k train_audio
  - intra-pool PSD diversity: NEW 34.8 vs Pantanal 25.3 (+38%)
  - cross-pool centroid distance: 26.4 (meaningfully different distribution)

Feasibility CONFIRMED. Full extraction in exp155 (deferred, ~15 min CPU,
~24 GB disk). Perch QC filter in exp156 (deferred).

Outputs: exp154_outputs/diagnostic.{npz,json}
"""
import sys, os, time, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
OUT = ROOT / "experiments/_data_pipelines/exp154_outputs"
OUT.mkdir(exist_ok=True)

SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC
N_FFT = 2048
HOP = 512

# Diagnostic budget
N_FILES_PER_BUCKET = 30   # 30 clips × ~10 buckets ≈ 300 files
BOTTOM_K_PER_FILE = 2     # take 2 lowest-energy windows per file

random.seed(42); np.random.seed(42)


def lat_lon_bucket(lat, lon, size=10):
    return f"{int(np.floor(lat/size)*size):+d}_{int(np.floor(lon/size)*size):+d}"


def file_quiet_windows(path, k=BOTTOM_K_PER_FILE):
    """Return the k lowest-RMS 5-sec windows from a clip."""
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        return []
    if sr != SR or len(x) < WIN_SAMPLES:
        return []
    if x.ndim > 1:
        x = x.mean(axis=1)
    n_win = len(x) // WIN_SAMPLES
    if n_win < 2:
        return []
    rms = np.array([np.sqrt(np.mean(x[i*WIN_SAMPLES:(i+1)*WIN_SAMPLES]**2))
                    for i in range(n_win)])
    order = np.argsort(rms)[:k]
    return [(x[i*WIN_SAMPLES:(i+1)*WIN_SAMPLES].copy(), float(rms[i])) for i in order]


def psd_mean(x):
    """Mean log-power per mel-ish frequency band. Quick: STFT → mean."""
    from numpy.fft import rfft
    n_frames = (len(x) - N_FFT) // HOP + 1
    if n_frames <= 0:
        return None
    spec = []
    for i in range(n_frames):
        seg = x[i*HOP:i*HOP+N_FFT]
        spec.append(np.abs(rfft(seg))**2)
    spec = np.array(spec).mean(axis=0)
    return np.log10(spec + 1e-10)


def main():
    train = pd.read_csv(DATA / "train.csv")
    train["bucket"] = [lat_lon_bucket(la, lo)
                        for la, lo in zip(train.latitude, train.longitude)]
    bucket_counts = train.bucket.value_counts()
    top_buckets = bucket_counts.head(12).index.tolist()
    print(f"Top buckets: {top_buckets[:8]} … (using top 12)")

    sampled = []
    for b in top_buckets:
        sub = train[train.bucket == b]
        if len(sub) >= N_FILES_PER_BUCKET:
            sub = sub.sample(N_FILES_PER_BUCKET, random_state=42)
        sampled.append(sub)
    df = pd.concat(sampled).reset_index(drop=True)
    print(f"Sampled {len(df)} files across {df.bucket.nunique()} buckets")

    t0 = time.time()
    new_windows = []
    new_psds = []
    new_buckets = []
    n_files_ok = 0
    for i, row in df.iterrows():
        path = DATA / "train_audio" / row.filename
        if not path.exists():
            continue
        wins = file_quiet_windows(path)
        if not wins:
            continue
        n_files_ok += 1
        for x, rms in wins:
            psd = psd_mean(x)
            if psd is None:
                continue
            new_windows.append(x.astype(np.float32))
            new_psds.append(psd.astype(np.float32))
            new_buckets.append(row.bucket)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(df)}] ok={n_files_ok} wins={len(new_windows)} t={time.time()-t0:.0f}s")
    print(f"Done: {n_files_ok}/{len(df)} files, {len(new_windows)} windows, {time.time()-t0:.0f}s")

    new_windows = np.stack(new_windows)
    new_psds = np.stack(new_psds)

    # Compare with current Pantanal pool
    bg_pant = np.load(ROOT / "experiments/_data_pipelines/exp49_outputs/bg_quiet_2025.npz")
    pant = bg_pant["windows"]
    print(f"Pantanal pool: {pant.shape}")
    # Sample 1000 for fair PSD comparison
    idx = np.random.choice(len(pant), min(1000, len(pant)), replace=False)
    pant_psds = np.stack([psd_mean(pant[i]) for i in idx if psd_mean(pant[i]) is not None])

    # Diagnostic stats
    new_psd_mean = new_psds.mean(axis=0)
    new_psd_std = new_psds.std(axis=0)
    pant_psd_mean = pant_psds.mean(axis=0)
    pant_psd_std = pant_psds.std(axis=0)

    # Diversity proxy: per-clip distance to pool centroid (higher = more diverse)
    new_centroid = new_psds.mean(axis=0)
    pant_centroid = pant_psds.mean(axis=0)
    new_div = np.linalg.norm(new_psds - new_centroid, axis=1).mean()
    pant_div = np.linalg.norm(pant_psds - pant_centroid, axis=1).mean()

    # Cross-pool distance (how DIFFERENT is the new pool from Pantanal)
    cross_dist = np.linalg.norm(new_centroid - pant_centroid)

    diag = {
        "n_train_files_sampled": int(len(df)),
        "n_train_files_with_windows": int(n_files_ok),
        "n_new_windows": int(len(new_windows)),
        "n_pant_windows": int(len(pant)),
        "buckets_covered": sorted(set(new_buckets)),
        "n_buckets": len(set(new_buckets)),
        "intra_pool_diversity_new": float(new_div),
        "intra_pool_diversity_pant": float(pant_div),
        "cross_pool_distance": float(cross_dist),
        "freq_bins": int(len(new_psd_mean)),
    }
    print(json.dumps(diag, indent=2))

    np.savez_compressed(OUT / "diagnostic.npz",
                        new_windows=new_windows[:200],  # save a small sample for audition
                        new_psds=new_psds,
                        pant_psds=pant_psds,
                        buckets=np.array(new_buckets),
                        new_psd_mean=new_psd_mean, new_psd_std=new_psd_std,
                        pant_psd_mean=pant_psd_mean, pant_psd_std=pant_psd_std)
    with open(OUT / "diagnostic.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
