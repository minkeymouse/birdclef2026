#!/usr/bin/env python3
"""
exp20_combined.py — Evaluate post-processing tricks on exp14 weights.

Tests the impact of:
  1. Overlapping windows (2.5s stride) vs standard 5s windows
  2. Time-shift TTA (1.25s)
  3. Temporal smoothing (confidence-sharpened)
  4. Global soundscape prior (file-max leakage)
  5. Gaussian smoothing (sigma=1.0)
  6. Ensemble: exp14 + exp15

Goal: Measure how much post-processing improves ss_auc.
"""
import os
import sys
import gc
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import timm
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp20_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = {
    "sr": 32000,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 0,
    "fmax": 16000,
    "img_size": 224,
    "infer_duration": 5.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

taxonomy_df = pd.read_csv(DATA / "taxonomy.csv")
SPECIES_LIST = sorted(taxonomy_df["primary_label"].astype(str).tolist())
SPECIES2IDX = {sp: i for i, sp in enumerate(SPECIES_LIST)}
NUM_CLASSES = len(SPECIES_LIST)

import librosa


# ====================================================================
# Models (must match training architectures)
# ====================================================================
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
    def __init__(self, backbone_name, num_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=1, num_classes=0, global_pool="",
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, CFG["img_size"], CFG["img_size"])
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[1]
        self.head = AttentionHead(self.feat_dim, num_classes, dropout=0.0)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.mean(dim=2).permute(0, 2, 1)
        clipwise = self.head(feat)
        return clipwise


# ====================================================================
# Inference helpers
# ====================================================================
def compute_mel(wav, sr):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr,
        n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
        n_mels=CFG["n_mels"], fmin=CFG["fmin"], fmax=CFG["fmax"],
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def predict_chunk(chunk, sr, models):
    """Predict on a single 5s chunk with model ensemble."""
    mel = compute_mel(chunk, sr)
    mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mel_t = F.interpolate(mel_t, size=(CFG["img_size"], CFG["img_size"]),
                          mode="bilinear", align_corners=False)
    mel_t = mel_t.to(DEVICE)

    preds = []
    with torch.no_grad(), autocast("cuda"):
        for model in models:
            logits = model(mel_t)
            pred = torch.sigmoid(logits).cpu().numpy()[0]
            preds.append(pred)
    return np.mean(preds, axis=0)


def predict_soundscape_advanced(models, wav, sr):
    """Predict with overlap + TTA, return raw window predictions and starts."""
    target_len = int(sr * CFG["infer_duration"])
    stride = target_len // 2  # 2.5s overlap
    tta_shift = target_len // 4  # 1.25s shift

    # Pad to ensure we have enough audio
    padded_len = max(len(wav), 12 * target_len)
    wav = np.pad(wav, (0, max(0, padded_len - len(wav))))[:padded_len]

    starts = list(range(0, padded_len - target_len + 1, stride))

    all_probs = []
    for s in starts:
        chunk = wav[s:s + target_len]
        # Original prediction
        pred_orig = predict_chunk(chunk, sr, models)
        # TTA: shifted
        chunk_shift = np.roll(chunk, tta_shift)
        pred_shift = predict_chunk(chunk_shift, sr, models)
        # Average
        all_probs.append((pred_orig + pred_shift) / 2.0)

    return np.array(all_probs), starts


def merge_windows(probs_windows, starts, n_windows=12):
    """Merge overlapping windows back to 12 standard 5s segments."""
    target_len = int(CFG["sr"] * CFG["infer_duration"])
    n_classes = probs_windows.shape[1]
    merged = np.zeros((n_windows, n_classes), dtype=np.float64)
    counts = np.zeros((n_windows, 1), dtype=np.float64)

    for j, s in enumerate(starts):
        i_lo = max(0, s // target_len)
        i_hi = min(n_windows - 1, (s + target_len - 1) // target_len)
        for i in range(i_lo, i_hi + 1):
            merged[i] += probs_windows[j]
            counts[i] += 1

    return (merged / np.maximum(counts, 1)).astype(np.float32)


def apply_temporal_smoothing(probs):
    """Confidence-sharpened temporal smoothing."""
    n = probs.shape[0]
    if n > 4:
        sharpen_power = 1.5
        probs_sharp = probs ** sharpen_power
        smooth_w = np.array([0.05, 0.15, 0.60, 0.15, 0.05])
        p_pad = np.pad(probs_sharp, ((2, 2), (0, 0)), mode="edge")
        smoothed = (smooth_w[0] * p_pad[:-4] +
                    smooth_w[1] * p_pad[1:-3] +
                    smooth_w[2] * p_pad[2:-2] +
                    smooth_w[3] * p_pad[3:-1] +
                    smooth_w[4] * p_pad[4:])
        probs = smoothed ** (1.0 / sharpen_power)
    elif n > 2:
        smooth_w = np.array([0.20, 0.60, 0.20])
        p_pad = np.pad(probs, ((1, 1), (0, 0)), mode="edge")
        probs = (smooth_w[0] * p_pad[:-2] +
                 smooth_w[1] * p_pad[1:-1] +
                 smooth_w[2] * p_pad[2:])
    return probs


def apply_global_prior(probs, weight=0.05):
    """Global file-max leakage."""
    file_max = np.max(probs, axis=0, keepdims=True)
    return probs + weight * file_max


def load_model(weight_path):
    """Load model from checkpoint."""
    backbone = "tf_efficientnet_b0.ns_jft_in1k"
    model = BirdSEDModel(backbone, NUM_CLASSES, pretrained=False).to(DEVICE)
    state = torch.load(weight_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def evaluate_config(models, config_name, use_overlap=False, use_tta=False,
                    use_smoothing=False, use_global_prior=False, use_gaussian=False, sigma=1.0):
    """Evaluate a configuration on soundscape eval set."""
    sr = CFG["sr"]
    target_len = int(sr * CFG["infer_duration"])
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()

    file_groups = labels_df.groupby("filename")
    all_preds = []
    all_labels = []

    audio_cache = {}
    for filename in labels_df["filename"].unique():
        filepath = DATA / "train_soundscapes" / filename
        if filepath.exists():
            try:
                wav, _ = librosa.load(filepath, sr=sr)
                audio_cache[filename] = wav
            except Exception:
                continue

    for filename, group in tqdm(file_groups, desc=f"Eval {config_name}"):
        if filename not in audio_cache:
            continue
        wav = audio_cache[filename]

        if use_overlap:
            # Advanced prediction with overlap + TTA
            probs_windows, starts = predict_soundscape_advanced(models, wav, sr)
            preds = merge_windows(probs_windows, starts)
        else:
            # Standard non-overlapping prediction
            n_windows = 12
            preds = np.zeros((n_windows, NUM_CLASSES))
            for seg in range(n_windows):
                start = seg * target_len
                chunk = wav[start:start + target_len]
                if len(chunk) < target_len:
                    chunk = np.pad(chunk, (0, target_len - len(chunk)))

                if use_tta:
                    tta_shift = target_len // 4
                    pred_orig = predict_chunk(chunk, sr, models)
                    pred_shift = predict_chunk(np.roll(chunk, tta_shift), sr, models)
                    preds[seg] = (pred_orig + pred_shift) / 2.0
                else:
                    preds[seg] = predict_chunk(chunk, sr, models)

        # Post-processing
        if use_smoothing:
            preds = apply_temporal_smoothing(preds)
        if use_global_prior:
            preds = apply_global_prior(preds)
        if use_gaussian and sigma > 0:
            preds = gaussian_filter1d(preds, sigma=sigma, axis=0)

        for _, row in group.iterrows():
            def _parse_time(t):
                parts = t.strip().split(":")
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            start_sec = _parse_time(str(row["start"]))
            win_idx = min(int(start_sec / 5.0), 11)

            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            for sp in str(row["primary_label"]).split(";"):
                sp = sp.strip()
                if sp in SPECIES2IDX:
                    label[SPECIES2IDX[sp]] = 1.0
            all_preds.append(preds[win_idx])
            all_labels.append(label)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

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

    # Load models
    weight_configs = []
    for fold in [0, 1]:
        p = WEIGHTS_DIR / f"exp14_aug_fold{fold}_best.pth"
        if p.exists():
            weight_configs.append(("exp14", fold, str(p)))
    for fold in [0, 1]:
        p = WEIGHTS_DIR / f"exp15_st_fold{fold}_best.pth"
        if p.exists():
            weight_configs.append(("exp15", fold, str(p)))

    print(f"Available models: {len(weight_configs)}")
    for name, fold, path in weight_configs:
        print(f"  {name} fold{fold}: {Path(path).name}")

    if not weight_configs:
        print("ERROR: No model weights found.")
        sys.exit(1)

    results = {}

    # Test 1: exp14 2-fold baseline (no post-processing)
    print(f"\n{'='*60}\nTest 1: exp14 2-fold (baseline)\n{'='*60}")
    exp14_models = []
    for name, fold, path in weight_configs:
        if name == "exp14":
            exp14_models.append(load_model(path))

    auc, n = evaluate_config(exp14_models, "baseline")
    results["exp14_baseline"] = auc
    print(f"  ss_auc={auc:.4f} ({n} cls)")

    # Test 2: + TTA only
    print(f"\n{'='*60}\nTest 2: exp14 + TTA\n{'='*60}")
    auc, n = evaluate_config(exp14_models, "tta", use_tta=True)
    results["exp14_tta"] = auc
    print(f"  ss_auc={auc:.4f} ({n} cls)")

    # Test 3: + Overlap windows + TTA (combined)
    print(f"\n{'='*60}\nTest 3: exp14 + Overlap + TTA\n{'='*60}")
    auc, n = evaluate_config(exp14_models, "overlap_tta", use_overlap=True)
    results["exp14_overlap_tta"] = auc
    print(f"  ss_auc={auc:.4f} ({n} cls)")

    # Test 4: + Temporal smoothing
    print(f"\n{'='*60}\nTest 4: exp14 + Overlap + TTA + Smoothing\n{'='*60}")
    auc, n = evaluate_config(exp14_models, "overlap_tta_smooth",
                              use_overlap=True, use_smoothing=True)
    results["exp14_overlap_tta_smooth"] = auc
    print(f"  ss_auc={auc:.4f} ({n} cls)")

    # Test 5: + Global prior
    print(f"\n{'='*60}\nTest 5: exp14 + Overlap + TTA + Smoothing + Global Prior\n{'='*60}")
    auc, n = evaluate_config(exp14_models, "full_pipeline",
                              use_overlap=True, use_smoothing=True, use_global_prior=True)
    results["exp14_full_pipeline"] = auc
    print(f"  ss_auc={auc:.4f} ({n} cls)")

    # Test 6: + Gaussian smoothing instead
    print(f"\n{'='*60}\nTest 6: exp14 + Overlap + TTA + Gaussian σ=1.0\n{'='*60}")
    auc, n = evaluate_config(exp14_models, "gauss",
                              use_overlap=True, use_gaussian=True, sigma=1.0)
    results["exp14_overlap_tta_gauss"] = auc
    print(f"  ss_auc={auc:.4f} ({n} cls)")

    # Test 7: exp14 + exp15 full ensemble
    print(f"\n{'='*60}\nTest 7: exp14+exp15 ensemble + full pipeline\n{'='*60}")
    all_models = []
    for name, fold, path in weight_configs:
        all_models.append(load_model(path))

    if len(all_models) > len(exp14_models):
        auc, n = evaluate_config(all_models, "ensemble_full",
                                  use_overlap=True, use_smoothing=True, use_global_prior=True)
        results["exp14_exp15_full"] = auc
        print(f"  ss_auc={auc:.4f} ({n} cls)")

    # Cleanup
    del exp14_models, all_models
    torch.cuda.empty_cache()

    # Sort and report
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nRESULTS (exp20: Post-Processing Evaluation)")
    print(f"{'='*60}")
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    for key, auc in sorted_results:
        print(f"  {key}: {auc:.4f}")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp20_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)


if __name__ == "__main__":
    main()
