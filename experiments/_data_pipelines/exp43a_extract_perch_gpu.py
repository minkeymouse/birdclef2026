#!/usr/bin/env python3
"""exp43a2 — Perch v2 embedding extraction on GPU via ONNX.

Replaces exp43a (CPU-only, 8s/batch, ETA 3h) with ONNX + CUDAExecutionProvider
(0.81s/batch benchmarked, ETA ~20 min).

ONNX model: /tmp/perch_v2.onnx from Kaggle dataset rishikeshjani/perch-onnx-for-birdclef-2026
Matches exp21 output convention: same emb (1536-d) + scores (234) per window.
"""
from __future__ import annotations
import gc, os, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
from tqdm.auto import tqdm

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
PERCH_DIR = ROOT / "perch_v2"
ONNX_PATH = Path("/tmp/perch_v2.onnx")
OUT = ROOT / "experiments/exp43a_outputs"  # reuse same output dir as CPU attempt
OUT.mkdir(exist_ok=True)

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
BATCH_FILES = 16  # 16 files × 12 windows = 192 per batch. GPU has headroom.
FILE_SECS = 60
FILE_SAMPLES = SR * FILE_SECS

# Perch → BirdCLEF mapping
taxonomy = pd.read_csv(DATA / "taxonomy.csv")
SPECIES = sorted(taxonomy["primary_label"].astype(str).tolist())
SP2IDX = {s: i for i, s in enumerate(SPECIES)}
sci2pl = dict(zip(taxonomy["scientific_name"], taxonomy["primary_label"]))
perch_labels = open(PERCH_DIR / "assets" / "labels.csv").read().strip().split("\n")
MAPPED_PERCH, MAPPED_BC = [], []
for pi, pname in enumerate(perch_labels):
    if pname in sci2pl and sci2pl[pname] in SP2IDX:
        MAPPED_PERCH.append(pi); MAPPED_BC.append(SP2IDX[sci2pl[pname]])
MAPPED_PERCH = np.array(MAPPED_PERCH, dtype=np.int64)
MAPPED_BC = np.array(MAPPED_BC, dtype=np.int64)

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

    print("Loading Perch ONNX on GPU...")
    t0 = time.time()
    sess = ort.InferenceSession(str(ONNX_PATH),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"  loaded in {time.time()-t0:.1f}s  providers={sess.get_providers()}")
    iname = sess.get_inputs()[0].name
    # warmup
    t0 = time.time()
    _ = sess.run(["embedding", "label"], {iname: np.zeros((BATCH_FILES * N_WINDOWS, WINDOW_SAMPLES), np.float32)})
    print(f"  warmup: {time.time()-t0:.1f}s")

    n_files = len(paths)
    n_rows = n_files * N_WINDOWS
    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, 234), dtype=np.float32)
    emb = np.zeros((n_rows, 1536), dtype=np.float32)

    write = 0
    t_start = time.time()
    for start in tqdm(range(0, n_files, BATCH_FILES), desc="Perch-GPU"):
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

        emb_b, logits = sess.run(["embedding", "label"], {iname: x})
        scores[bstart:write, MAPPED_BC] = logits[:, MAPPED_PERCH]
        emb[bstart:write] = emb_b

        if (start // BATCH_FILES) % 50 == 0:
            elapsed = time.time() - t_start
            pct = (start + bn) / n_files
            eta = elapsed / max(pct, 1e-6) - elapsed
            print(f"  [{start+bn}/{n_files}] {elapsed:.0f}s, ETA {eta/60:.1f}m")

    emb = emb[:write]; scores = scores[:write]
    row_ids = row_ids[:write]; filenames = filenames[:write]
    sites = sites[:write]; hours = hours[:write]
    print(f"Done. emb {emb.shape}  wall {(time.time()-t_start)/60:.1f}m")

    meta = pd.DataFrame({
        "row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours,
    })
    meta.to_parquet(OUT / "perch_ss_all_meta.parquet", index=False)
    np.savez_compressed(OUT / "perch_ss_all.npz", emb=emb, scores=scores)
    print(f"Saved: {OUT}/perch_ss_all.npz  {OUT}/perch_ss_all_meta.parquet")


if __name__ == "__main__":
    main()
