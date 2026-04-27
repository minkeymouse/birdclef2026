#!/usr/bin/env python3
"""exp43a — Perch v2 embedding extraction on all train_soundscapes (10,658 files).

For iVAE training we need Perch embeddings on the full SS corpus, not just the
708 labeled windows in exp21 cache. This produces:
  experiments/exp43a_outputs/perch_ss_all.npz  (emb, scores, filenames)
  experiments/exp43a_outputs/perch_ss_all_meta.parquet  (row_id/filename/site/hour)

Matches exp21_oof_ablation.perch_infer_files signature exactly, so resulting
embeddings are drop-in compatible with the existing exp21 labeled cache.
"""
from __future__ import annotations
import gc, os, re, time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only: matches exp21, avoids RTX 5090 JIT

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from tqdm.auto import tqdm

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
PERCH_DIR = ROOT / "perch_v2"
OUT = ROOT / "experiments/exp43a_outputs"
OUT.mkdir(exist_ok=True)

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
BATCH_FILES = 8
FILE_SECS = 60
FILE_SAMPLES = SR * FILE_SECS

# Build Perch → BirdCLEF mapping (subset of 234 classes)
taxonomy = pd.read_csv(DATA / "taxonomy.csv")
SPECIES = sorted(taxonomy["primary_label"].astype(str).tolist())
SP2IDX = {s: i for i, s in enumerate(SPECIES)}
sci2pl = dict(zip(taxonomy["scientific_name"], taxonomy["primary_label"]))
perch_labels = open(PERCH_DIR / "assets" / "labels.csv").read().strip().split("\n")
MAPPED_PERCH = []
MAPPED_BC = []
for pi, pname in enumerate(perch_labels):
    if pname in sci2pl and sci2pl[pname] in SP2IDX:
        MAPPED_PERCH.append(pi)
        MAPPED_BC.append(SP2IDX[sci2pl[pname]])
MAPPED_PERCH = np.array(MAPPED_PERCH, dtype=np.int64)
MAPPED_BC = np.array(MAPPED_BC, dtype=np.int64)
print(f"Perch→BC mapping: {len(MAPPED_BC)} species")

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

    out_npz = OUT / "perch_ss_all.npz"
    out_meta = OUT / "perch_ss_all_meta.parquet"
    if out_npz.exists() and out_meta.exists():
        print("Cache exists, exiting.")
        return

    print("Loading Perch v2 SavedModel...")
    t0 = time.time()
    model = tf.saved_model.load(str(PERCH_DIR))
    infer = model.signatures["serving_default"]
    # warmup with one batch
    _ = infer(inputs=tf.zeros((1, WINDOW_SAMPLES), dtype=tf.float32))
    print(f"Loaded + warmup in {time.time()-t0:.1f}s")

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
    for start in tqdm(range(0, n_files, BATCH_FILES), desc="Perch"):
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
        out = infer(inputs=tf.constant(x, dtype=tf.float32))
        logits = out["label"].numpy().astype(np.float32)
        emb_b = out["embedding"].numpy().astype(np.float32)
        scores[bstart:write, MAPPED_BC] = logits[:, MAPPED_PERCH]
        emb[bstart:write] = emb_b
        del x, out, logits, emb_b
        if (start // BATCH_FILES) % 50 == 0:
            elapsed = time.time() - t_start
            pct = (start + bn) / n_files
            eta = elapsed / max(pct, 1e-6) - elapsed
            print(f"  [{start+bn}/{n_files}]  {elapsed:.0f}s elapsed, ETA {eta/60:.1f}min")
            gc.collect()

    # trim to actually-written rows (all files processed)
    emb = emb[:write]
    scores = scores[:write]
    row_ids = row_ids[:write]
    filenames = filenames[:write]
    sites = sites[:write]
    hours = hours[:write]

    print(f"Done. emb {emb.shape}  elapsed {(time.time()-t_start)/60:.1f}min")

    meta = pd.DataFrame({
        "row_id": row_ids, "filename": filenames, "site": sites, "hour_utc": hours,
    })
    meta.to_parquet(out_meta, index=False)
    np.savez_compressed(out_npz, emb=emb, scores=scores)
    print(f"Saved: {out_npz} ({out_npz.stat().st_size/1e6:.0f} MB), {out_meta}")


if __name__ == "__main__":
    main()
