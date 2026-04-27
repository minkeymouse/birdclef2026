#!/usr/bin/env python3
"""exp43j — Perch spatial_embedding extraction for iVDFM input.

ONNX Perch outputs spatial_embedding with shape (B, 16, 4, 1536):
  - 16 = temporal patches per 5-sec window (~313 ms each)
  - 4 = frequency patches
  - 1536 = channel dim

For iVDFM we want a compact time-series representation per window. We average
over the 4 frequency patches → (B, 16, 1536). This preserves the key temporal
structure for dynamic factor modeling while halving disk size.

Output:
  exp43j_outputs/spatial_ss_all.npz     — (127896, 16, 1536) float16 (~6 GB → 3 GB compressed)
  exp43j_outputs/spatial_ss_all_meta.parquet  — same format as exp43a meta
"""
from __future__ import annotations
import gc, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
from tqdm.auto import tqdm

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
ONNX_PATH = Path("/tmp/perch_v2.onnx")
OUT = ROOT / "experiments/exp43j_outputs"
OUT.mkdir(exist_ok=True)

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
BATCH_FILES = 16
FILE_SAMPLES = SR * 60

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

    out_npz = OUT / "spatial_ss_all.npz"
    if out_npz.exists():
        print(f"Already exists: {out_npz}"); return

    print("Loading Perch ONNX GPU...")
    sess = ort.InferenceSession(str(ONNX_PATH),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    # Warmup
    print("Warmup...")
    _ = sess.run(["spatial_embedding"], {iname: np.zeros((BATCH_FILES * N_WINDOWS, WINDOW_SAMPLES), np.float32)})

    n_files = len(paths)
    n_rows = n_files * N_WINDOWS
    # spatial: (16, 1536) per window after freq-avg — use float16 to save disk
    spat = np.zeros((n_rows, 16, 1536), dtype=np.float16)
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)

    write = 0
    t0 = time.time()
    for start in tqdm(range(0, n_files, BATCH_FILES), desc="Perch-spatial"):
        batch = paths[start:start + BATCH_FILES]
        bn = len(batch)
        x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        bstart = write
        for bi, p in enumerate(batch):
            try:
                audio = read_60s(p)
            except Exception:
                audio = np.zeros(FILE_SAMPLES, dtype=np.float32)
            x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
            site, hour = parse_meta(p.name)
            for t_idx, t_end in enumerate(range(5, 65, 5)):
                wi = write + bi * N_WINDOWS + t_idx
                row_ids[wi] = f"{p.stem}_{t_end}"
                filenames[wi] = p.name
                sites[wi] = site
                hours[wi] = hour
        write += bn * N_WINDOWS

        spatial_b = sess.run(["spatial_embedding"], {iname: x})[0]  # (B*12, 16, 4, 1536)
        spat[bstart:write] = spatial_b.mean(axis=2).astype(np.float16)  # avg 4 freq → (B*12, 16, 1536)

        if (start // BATCH_FILES) % 50 == 0:
            elapsed = time.time() - t0
            pct = (start + bn) / n_files
            eta = elapsed / max(pct, 1e-6) - elapsed
            print(f"  [{start+bn}/{n_files}] {elapsed:.0f}s, ETA {eta/60:.1f}m")

    print(f"Done. spatial {spat.shape} {spat.dtype} ({spat.nbytes/1e9:.1f} GB). Wall {(time.time()-t0)/60:.1f}m")

    meta = pd.DataFrame({
        "row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours,
    })
    meta.to_parquet(OUT / "spatial_ss_all_meta.parquet", index=False)
    np.savez_compressed(out_npz, spatial=spat)
    print(f"Saved: {out_npz} ({out_npz.stat().st_size/1e9:.1f} GB)")
    print(f"       {OUT}/spatial_ss_all_meta.parquet")


if __name__ == "__main__":
    main()
