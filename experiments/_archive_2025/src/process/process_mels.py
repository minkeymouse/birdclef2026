#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import logging
import sys
from pathlib import Path
from typing import List, Set
import ast
import math
import cv2
import yaml
import librosa
import numpy as np
import pandas as pd
import tqdm

# Project setup
project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "process.yaml"
sys.path.insert(0, str(project_root))
from src.utils import utils

# ─── Load configs ──────────────────────────────────────────
with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
paths_cfg = CFG["paths"]
audio_cfg = CFG["audio"]
dedup_cfg = CFG.get("deduplication", {})
selection_cfg = CFG.get("selection", {})

# Convert string paths to Path objects
for key in ["DATA_ROOT", "audio_dir", "mel_dir", "label_dir", "meta_data"]:
    paths_cfg[key] = Path(paths_cfg[key])

# Ensure output directories exist
paths_cfg["mel_dir"].mkdir(parents=True, exist_ok=True)
paths_cfg["label_dir"].mkdir(parents=True, exist_ok=True)
paths_cfg["meta_data"].parent.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("process_mels")

# ─── Load metadata ─────────────────────────────────────────
log.info("Loading taxonomy and training metadata…")
taxonomy_df = pd.read_csv(paths_cfg["DATA_ROOT"] / "taxonomy.csv")
train_df = pd.read_csv(paths_cfg["DATA_ROOT"] / "train.csv")

# Build label2id mapping
label_list = sorted(train_df["primary_label"].unique().tolist())
label2id = {lab: i for i, lab in enumerate(label_list)}
num_classes = len(label_list)
log.info(f"Found {num_classes} unique species labels")

# Prepare DataFrame of examples
working_df = train_df[["primary_label", "secondary_labels", "rating", "filename"]].copy()
working_df["filepath"] = paths_cfg["audio_dir"] / working_df["filename"]

# Filter by minimum rating if configured
min_rating = selection_cfg.get("minimum_rating")
if min_rating is not None:
    before = len(working_df)
    working_df = working_df[working_df["rating"] >= min_rating]
    log.info(f"Filtered out {before - len(working_df)} files with rating < {min_rating}")

# Audio parameters
TARGET_H, TARGET_W = audio_cfg["target_shape"]
db_thresh = audio_cfg.get("silence_thresh_db", -50.0)
dur_samples = int(audio_cfg["train_duration"] * audio_cfg["sample_rate"])
hop_samples = int(audio_cfg["train_chunk_hop"] * audio_cfg["sample_rate"])

# Deduplication state
seen_hashes: Set[str] = set()

def parse_secondary(s) -> List[str]:
    if pd.isna(s) or s in ["", "[]", "['']"]:
        return []
    return ast.literal_eval(s)


def audio2melspec(wav: np.ndarray) -> np.ndarray:
    # Handle NaNs
    if np.isnan(wav).any():
        wav = np.nan_to_num(wav, nan=np.nanmean(wav))
    # Compute mel-spectrogram
    m = librosa.feature.melspectrogram(
        y=wav,
        sr=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        fmin=audio_cfg["fmin"],
        fmax=audio_cfg["fmax"],
        power=2.0,
    )
    m_db = librosa.power_to_db(m, ref=np.max)
    return (m_db - m_db.min()) / (m_db.max() - m_db.min() + 1e-8)

# Main processing loop
log.info(f"Processing {len(working_df)} audio files → chunks...")
meta_rows: List[dict] = []
errors: List[tuple] = []

for _, row in tqdm.tqdm(working_df.iterrows(), total=len(working_df), desc="Audio chunks"):
    path = row.filepath
    fname = row.filename
    sec_labels = parse_secondary(row.secondary_labels)
    file_weight = float(row.rating) / 5.0

    if not path.exists():
        log.warning(f"Missing file: {path}")
        continue

    try:
        wav, _ = librosa.load(path, sr=audio_cfg["sample_rate"])
        wav, _ = librosa.effects.trim(wav, top_db=audio_cfg["trim_top_db"])
        if wav.size == 0:
            continue

        # Optional deduplication
        if dedup_cfg.get("enabled", False):
            h = hashlib.md5(wav.tobytes()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

        # Pad or tile
        if wav.size < dur_samples:
            reps = math.ceil(dur_samples / wav.size)
            wav = np.tile(wav, reps)[:dur_samples]

        ptr = 0
        while ptr + dur_samples <= wav.size:
            chunk = wav[ptr : ptr + dur_samples]
            ptr += hop_samples

            # Skip silent or too-quiet segments
            if utils.is_silent(chunk, db_thresh=db_thresh):
                continue
            if utils.contains_voice(chunk, audio_cfg["sample_rate"]):
                continue

            # Create mel-spectrogram
            m = audio2melspec(chunk)
            if m.shape != (TARGET_H, TARGET_W):
                m = cv2.resize(m, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            m = m.astype(np.float32)

            chunk_id = utils.hash_chunk_id(fname, ptr / audio_cfg["sample_rate"])
            m_path = paths_cfg["mel_dir"] / f"{chunk_id}.npy"
            label_path = paths_cfg["label_dir"] / f"{chunk_id}.npy"

            # Save data
            np.save(m_path, m)
            # Build soft-label vector
            onehot = np.zeros(num_classes, dtype=np.float32)
            onehot[label2id[row.primary_label]] = CFG["labeling"]["primary_label_weight"]
            if sec_labels:
                sec_w = CFG["labeling"]["secondary_label_weight"] / len(sec_labels)
                for sl in sec_labels:
                    if sl in label2id:
                        onehot[label2id[sl]] = sec_w
            np.save(label_path, onehot)

            meta_rows.append({
                "chunk_id": str(chunk_id),
                "filename": fname,
                "end_sec": round(ptr / audio_cfg["sample_rate"], 3),
                "mel_path": str(m_path),
                "label_path": str(label_path),
                "weight": file_weight,
                "source": "train_audio",
            })

    except Exception as e:
        errors.append((str(path), str(e)))

# Write metadata CSV
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(paths_cfg["meta_data"], index=False)

log.info(f"Saved {len(meta_rows)} chunks (errors: {len(errors)})")
if errors:
    log.info(f"Sample errors: {errors[:5]}")
