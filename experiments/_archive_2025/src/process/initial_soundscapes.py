#!/usr/bin/env python3
"""
initialize_soundscape.py — preprocess unlabeled soundscape .ogg files into 10s mel chunks
and append to metadata CSV with zero labels for future Kalman filtering.
"""
import logging
import math
from pathlib import Path
from typing import List, Set

import librosa
import numpy as np
import pandas as pd
import yaml
import cv2
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import utils

# Project root and config
project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "process.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
paths_cfg      = CFG["paths"]
audio_cfg      = CFG["audio"]
selection_cfg  = CFG.get("selection", {})

# Convert string paths to Path
for key in ["DATA_ROOT", "mel_dir", "label_dir", "meta_data", "train_soundscapes"]:
    paths_cfg[key] = Path(paths_cfg[key])

# Ensure output directories exist
paths_cfg["mel_dir"].mkdir(parents=True, exist_ok=True)
paths_cfg["label_dir"].mkdir(parents=True, exist_ok=True)
paths_cfg["meta_data"].parent.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("init_soundscape")

# Audio parameters
sr            = audio_cfg["sample_rate"]
dur_samples   = int(audio_cfg["train_duration"] * sr)
hop_samples   = int(audio_cfg["train_chunk_hop"] * sr)
TARGET_H, TARGET_W = audio_cfg["target_shape"]
db_thresh     = audio_cfg.get("silence_thresh_db", -50.0)

# Label dimensions
train_csv     = paths_cfg["DATA_ROOT"] / "train.csv"
train_df      = pd.read_csv(train_csv)
label_list    = sorted(train_df["primary_label"].unique().tolist())
num_classes   = len(label_list)

# Existing metadata
meta_csv      = paths_cfg["meta_data"]
if meta_csv.exists():
    existing_meta = pd.read_csv(meta_csv)
else:
    existing_meta = pd.DataFrame(columns=["chunk_id","filename","end_sec","mel_path","label_path","weight","source"])

# Dedup set for chunk_ids
seen_ids: Set[str] = set(existing_meta["chunk_id"].astype(str).tolist())

# Helper: waveform → normalized melspectrogram
def audio2melspec(wav: np.ndarray) -> np.ndarray:
    if np.isnan(wav).any():
        wav = np.nan_to_num(wav, nan=np.nanmean(wav))
    m = librosa.feature.melspectrogram(
        y=wav, sr=sr,
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
        n_mels=audio_cfg["n_mels"],
        fmin=audio_cfg["fmin"],
        fmax=audio_cfg["fmax"],
        power=2.0,
    )
    m_db = librosa.power_to_db(m, ref=np.max)
    return (m_db - m_db.min()) / (m_db.max() - m_db.min() + 1e-8)

# Process soundscapes
sound_dir = paths_cfg["train_soundscapes"]
ogg_files = list(sound_dir.rglob("*.ogg"))
log.info(f"Found {len(ogg_files)} soundscape files to process.")

meta_rows: List[dict] = []
for ogg_path in ogg_files:
    rel_fname = ogg_path.relative_to(paths_cfg["DATA_ROOT"]) if paths_cfg["DATA_ROOT"] in ogg_path.parents else ogg_path.name
    try:
        wav, _ = librosa.load(str(ogg_path), sr=sr)
        wav, _ = librosa.effects.trim(wav, top_db=audio_cfg["trim_top_db"])
        if wav.size == 0:
            continue
        # Pad or tile
        if wav.size < dur_samples:
            reps = math.ceil(dur_samples / wav.size)
            wav = np.tile(wav, reps)[:dur_samples]

        ptr = 0
        while ptr + dur_samples <= wav.size:
            chunk = wav[ptr:ptr + dur_samples]
            ptr += hop_samples

            if utils.is_silent(chunk, db_thresh=db_thresh):
                continue
            if utils.contains_voice(chunk, sr):
                continue

            # Compute mel and save
            m = audio2melspec(chunk)
            if m.shape != (TARGET_H, TARGET_W):
                m = cv2.resize(m, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            m = m.astype(np.float32)

            chunk_id = utils.hash_chunk_id(ogg_path.name, ptr / sr)
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            m_path = paths_cfg["mel_dir"] / f"{chunk_id}.npy"
            label_path = paths_cfg["label_dir"] / f"{chunk_id}.npy"

            np.save(m_path, m)
            zeros = np.zeros(num_classes, dtype=np.float32)
            np.save(label_path, zeros)

            weight = CFG["labeling"]["pseudo_label_weight"]
            meta_rows.append({
                "chunk_id": str(chunk_id),
                "filename": str(rel_fname),
                "end_sec": round(ptr / sr, 3),
                "mel_path": str(m_path),
                "label_path": str(label_path),
                "weight": weight,
                "source": "train_soundscape",
            })
    except Exception as e:
        log.error(f"Error processing {ogg_path}: {e}")

# Append and save
new_meta = pd.DataFrame(meta_rows)
combined = pd.concat([existing_meta, new_meta], ignore_index=True)
combined.to_csv(meta_csv, index=False)
log.info(f"Appended {len(new_meta)} chunks; total metadata rows = {len(combined)}.")
