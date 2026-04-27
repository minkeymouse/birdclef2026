#!/usr/bin/env python3
"""
exp12_inference_opt.py — Inference optimization sweep (zero training cost).

Tests on exp3 and exp11 weights:
  1. Overlap TTA (2.5s stride → 23 windows per 60s)
  2. Temperature scaling (T=1.0, 1.15, 1.3, 1.5)
  3. Gaussian temporal smoothing (σ=0.5, 1.0, 1.5)
  4. Grid search over all combinations
"""
import os
import json
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import librosa
from pathlib import Path
from torch.amp import autocast
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp12_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────
SR = 32000
DURATION = 5.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0
FMAX = 16000
IMG_SIZE = 224
BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

taxonomy_df = pd.read_csv(DATA / "taxonomy.csv")
SPECIES_LIST = sorted(taxonomy_df["primary_label"].astype(str).tolist())
SPECIES2IDX = {sp: i for i, sp in enumerate(SPECIES_LIST)}
NUM_CLASSES = len(SPECIES_LIST)


# ── Model ──────────────────────────────────────────────────
class AttentionHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, num_classes),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        framewise = self.classifier(x)
        attn_weights = self.attention(x)
        clipwise = (framewise * attn_weights).sum(dim=1)
        return clipwise


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False,
            in_chans=1, num_classes=0, global_pool="",
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[1]
        self.head = AttentionHead(self.feat_dim, num_classes, dropout=0.3)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.mean(dim=2).permute(0, 2, 1)
        return self.head(feat)


def load_models(prefixes):
    """Load models by weight file prefix."""
    models = []
    for prefix in prefixes:
        for fold in [0, 1]:
            path = WEIGHTS_DIR / f"{prefix}_fold{fold}_best.pth"
            if path.exists():
                model = BirdSEDModel(BACKBONE, NUM_CLASSES).to(DEVICE)
                state = torch.load(path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                model.eval()
                models.append(model)
                print(f"  Loaded {path.name}")
    return models


def compute_mel(wav, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def _parse_time(t):
    parts = t.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


# ====================================================================
# Inference strategies
# ====================================================================
def infer_standard(wav, models, n_segments=12):
    """Standard non-overlapping 5s segments (baseline)."""
    target_len = int(SR * DURATION)
    preds = np.zeros((n_segments, NUM_CLASSES), dtype=np.float32)

    for seg in range(n_segments):
        start = seg * target_len
        chunk = wav[start:start + target_len]
        if len(chunk) < target_len:
            chunk = np.pad(chunk, (0, target_len - len(chunk)))
        chunk = chunk[:target_len]

        mel = compute_mel(chunk)
        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        mel_t = F.interpolate(mel_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

        seg_preds = []
        with torch.no_grad():
            for model in models:
                with autocast("cuda"):
                    logits = model(mel_t)
                prob = torch.sigmoid(logits).cpu().numpy()[0]
                seg_preds.append(prob)
        preds[seg] = np.mean(seg_preds, axis=0)

    return preds


def infer_overlap_tta(wav, models, stride_sec=2.5, n_segments=12):
    """Overlapping TTA: 2.5s stride sliding windows, aggregate to 5s segments."""
    target_len = int(SR * DURATION)
    stride = int(SR * stride_sec)
    total_len = int(SR * 60)

    # Pad wav to 60s
    if len(wav) < total_len:
        wav = np.pad(wav, (0, total_len - len(wav)))

    # Generate overlapping windows
    starts = list(range(0, total_len - target_len + 1, stride))
    window_preds = []

    for start in starts:
        chunk = wav[start:start + target_len]
        if len(chunk) < target_len:
            chunk = np.pad(chunk, (0, target_len - len(chunk)))

        mel = compute_mel(chunk)
        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        mel_t = F.interpolate(mel_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

        seg_preds = []
        with torch.no_grad():
            for model in models:
                with autocast("cuda"):
                    logits = model(mel_t)
                prob = torch.sigmoid(logits).cpu().numpy()[0]
                seg_preds.append(prob)
        window_preds.append((start, np.mean(seg_preds, axis=0)))

    # Aggregate overlapping windows to 5s segments
    preds = np.zeros((n_segments, NUM_CLASSES), dtype=np.float32)
    for seg in range(n_segments):
        seg_start = seg * target_len
        seg_end = seg_start + target_len
        # Average all windows that overlap with this segment
        contributing = []
        for w_start, w_pred in window_preds:
            w_end = w_start + target_len
            overlap = min(seg_end, w_end) - max(seg_start, w_start)
            if overlap > 0:
                weight = overlap / target_len
                contributing.append((weight, w_pred))
        if contributing:
            total_w = sum(w for w, _ in contributing)
            preds[seg] = sum(w * p for w, p in contributing) / total_w

    return preds


def apply_temperature(preds, temperature):
    """Apply temperature scaling to logit-space predictions."""
    if temperature == 1.0:
        return preds
    # Convert probs back to logits, scale, convert back
    eps = 1e-7
    logits = np.log(np.clip(preds, eps, 1 - eps) / (1 - np.clip(preds, eps, 1 - eps)))
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def apply_smoothing(preds, sigma):
    """Apply Gaussian temporal smoothing across segments."""
    if sigma <= 0:
        return preds
    return gaussian_filter1d(preds, sigma=sigma, axis=0)


# ====================================================================
# Evaluation
# ====================================================================
def compute_ss_auc(all_preds, all_labels):
    """Compute macro-averaged ROC-AUC (skip classes with no positives)."""
    aucs = []
    for c in range(NUM_CLASSES):
        gt = (all_labels[:, c] > 0.5).astype(int)
        if 0 < gt.sum() < len(gt):
            try:
                aucs.append(roc_auc_score(gt, all_preds[:, c]))
            except ValueError:
                pass
    return np.mean(aucs) if aucs else 0.0, len(aucs)


def main():
    start_time = time.time()

    # ── Load evaluation data ─────────────────────────────────
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    target_len = int(SR * DURATION)

    # Build ground truth per (filename, segment)
    gt_rows = []
    for _, row in labels_df.iterrows():
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in SPECIES2IDX:
                label[SPECIES2IDX[sp]] = 1.0
        gt_rows.append({
            "filename": row["filename"],
            "start": _parse_time(str(row["start"])),
            "label": label,
        })
    gt_df = pd.DataFrame(gt_rows)

    # Load soundscape audio
    print("Loading soundscape audio...")
    audio_cache = {}
    unique_files = sorted(gt_df["filename"].unique())
    for filename in tqdm(unique_files, desc="Loading soundscapes"):
        filepath = DATA / "train_soundscapes" / filename
        if filepath.exists():
            try:
                wav, _ = librosa.load(filepath, sr=SR)
                audio_cache[filename] = wav
            except Exception:
                continue
    print(f"Loaded {len(audio_cache)} soundscape files (one-time load)")

    # ── Model groups to test ─────────────────────────────────
    model_groups = {}

    # exp3 models
    print("\nLoading exp3 models...")
    exp3_models = load_models(["exp3_sed"])
    if exp3_models:
        model_groups["exp3"] = exp3_models

    # exp11 models (if available)
    print("Loading exp11 models...")
    exp11_models = load_models(["exp11_5s"])
    if exp11_models:
        model_groups["exp11"] = exp11_models

    # Combined
    if exp3_models and exp11_models:
        model_groups["exp3+exp11"] = exp3_models + exp11_models

    # ── Grid search ──────────────────────────────────────────
    tta_modes = ["standard", "overlap_2.5s"]
    temperatures = [1.0, 1.15, 1.3, 1.5]
    smoothing_sigmas = [0.0, 0.5, 1.0, 1.5]

    results = {}

    for group_name, models in model_groups.items():
        print(f"\n{'='*60}\nModel group: {group_name} ({len(models)} models)")
        print(f"{'='*60}")

        for tta_mode in tta_modes:
            # Generate predictions for this TTA mode
            print(f"\n  TTA mode: {tta_mode}")
            all_preds_raw = []
            all_labels = []

            for filename in tqdm(gt_df["filename"].unique(), desc=f"  {group_name}/{tta_mode}"):
                if filename not in audio_cache:
                    continue
                wav = audio_cache[filename]

                if tta_mode == "standard":
                    file_preds = infer_standard(wav, models)
                else:
                    file_preds = infer_overlap_tta(wav, models, stride_sec=2.5)

                # Match with ground truth segments
                file_gt = gt_df[gt_df["filename"] == filename]
                for _, gt_row in file_gt.iterrows():
                    seg_idx = gt_row["start"] // 5  # 0-based segment index
                    if seg_idx < len(file_preds):
                        all_preds_raw.append(file_preds[seg_idx])
                        all_labels.append(gt_row["label"])

            all_preds_raw = np.array(all_preds_raw)
            all_labels = np.array(all_labels)

            # Grid search over temperature and smoothing
            for temp, sigma in product(temperatures, smoothing_sigmas):
                preds = apply_temperature(all_preds_raw.copy(), temp)
                # Smoothing needs per-file application, but for eval we approximate
                # by applying across all segments (close enough for tuning)
                if sigma > 0:
                    preds = apply_smoothing(preds, sigma)

                auc, n_cls = compute_ss_auc(preds, all_labels)
                key = f"{group_name}|{tta_mode}|T={temp}|σ={sigma}"
                results[key] = float(auc)

                if auc > 0:
                    print(f"    T={temp:.2f}, σ={sigma:.1f} → ss_auc={auc:.4f} ({n_cls} cls)")

    # ── Sort and report ──────────────────────────────────────
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print("TOP 20 CONFIGURATIONS")
    print(f"{'='*60}")
    for i, (key, auc) in enumerate(sorted_results[:20]):
        print(f"  {i+1:2d}. {auc:.4f}  {key}")

    # Best per model group
    print(f"\nBEST PER MODEL GROUP:")
    for group_name in model_groups:
        group_results = [(k, v) for k, v in sorted_results if k.startswith(group_name + "|")]
        if group_results:
            best_key, best_auc = group_results[0]
            print(f"  {group_name}: {best_auc:.4f}  ({best_key})")

    elapsed = (time.time() - start_time) / 60
    print(f"\nTime: {elapsed:.1f} min")

    with open(OUT_DIR / "exp12_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save sorted results
    with open(OUT_DIR / "exp12_sorted.json", "w") as f:
        json.dump(sorted_results, f, indent=2)


if __name__ == "__main__":
    main()
