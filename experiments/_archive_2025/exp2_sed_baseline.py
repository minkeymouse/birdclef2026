#!/usr/bin/env python3
"""
exp2_sed_baseline.py — SED Baseline: EfficientNet-B0 trained on train_audio + labeled soundscapes.

Pipeline:
  Phase 1: Precompute mel spectrograms → cache as .npy
  Phase 2: Train 1-fold EfficientNet-B0 with mixup + SpecAugment
  Phase 3: Validate and report macro-AUC
"""
import os
import sys
import gc
import json
import time
import random
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import timm
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp2_outputs"
CACHE_DIR = OUT_DIR / "mel_cache"

for d in [WEIGHTS_DIR, OUT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────
CFG = {
    "seed": 42,
    "sr": 32000,
    "duration": 5.0,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 0,
    "fmax": 16000,
    "img_size": 224,
    "backbone": "tf_efficientnet_b0_ns",
    "pretrained": True,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "epochs": 15,
    "batch_size": 64,
    "num_workers": 4,
    "n_folds": 5,
    "train_folds": [0],
    "mixup_alpha": 0.4,
    "label_smoothing": 0.01,
}

# ── Reproducibility ────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Taxonomy ───────────────────────────────────────────────
taxonomy_df = pd.read_csv(DATA / "taxonomy.csv")
SPECIES_LIST = sorted(taxonomy_df["primary_label"].astype(str).tolist())
SPECIES2IDX = {sp: i for i, sp in enumerate(SPECIES_LIST)}
NUM_CLASSES = len(SPECIES_LIST)
print(f"Number of classes: {NUM_CLASSES}")


# ====================================================================
# Phase 1: Precompute mel spectrograms
# ====================================================================
def compute_mel(wav: np.ndarray, sr: int) -> np.ndarray:
    """Compute normalized log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr,
        n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
        n_mels=CFG["n_mels"], fmin=CFG["fmin"], fmax=CFG["fmax"],
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def pad_or_crop(wav: np.ndarray, target_len: int, random_crop: bool = False) -> np.ndarray:
    """Pad (tile) or crop audio to target_len samples."""
    if len(wav) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(wav) < target_len:
        reps = int(np.ceil(target_len / len(wav)))
        wav = np.tile(wav, reps)
    if len(wav) > target_len:
        if random_crop:
            start = random.randint(0, len(wav) - target_len)
        else:
            start = 0
        wav = wav[start:start + target_len]
    return wav[:target_len]


def precompute_train_audio(train_df: pd.DataFrame) -> pd.DataFrame:
    """Precompute mel specs for train_audio. Returns metadata DataFrame."""
    sr = CFG["sr"]
    target_len = int(sr * CFG["duration"])
    rows = []
    cache_sub = CACHE_DIR / "train_audio"
    cache_sub.mkdir(exist_ok=True)

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Precompute train_audio"):
        filepath = DATA / "train_audio" / row["filename"]
        if not filepath.exists():
            continue
        try:
            wav, _ = librosa.load(filepath, sr=sr)
            if len(wav) == 0:
                continue
        except Exception:
            continue

        # Extract chunks: if audio > duration, extract multiple non-overlapping chunks
        n_chunks = max(1, len(wav) // target_len)
        for c in range(min(n_chunks, 3)):  # cap at 3 chunks per file
            start = c * target_len
            chunk = pad_or_crop(wav[start:], target_len)
            mel = compute_mel(chunk, sr)

            mel_id = f"ta_{idx}_{c}"
            mel_path = cache_sub / f"{mel_id}.npy"
            np.save(mel_path, mel)

            # Build label
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            pl = str(row["primary_label"])
            if pl in SPECIES2IDX:
                label[SPECIES2IDX[pl]] = 1.0
            # Secondary labels
            sec = row.get("secondary_labels", "[]")
            if isinstance(sec, str) and sec not in ("[]", "", "nan"):
                try:
                    import ast
                    sec_list = ast.literal_eval(sec)
                    if isinstance(sec_list, list):
                        for s in sec_list:
                            s = str(s).strip()
                            if s in SPECIES2IDX:
                                label[SPECIES2IDX[s]] = 0.3
                except Exception:
                    pass

            rows.append({
                "mel_path": str(mel_path),
                "primary_label": pl,
                "source": "train_audio",
                "label": label,
            })

    return pd.DataFrame(rows)


def parse_time_str(t: str) -> float:
    """Parse HH:MM:SS to seconds."""
    parts = t.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def precompute_soundscape_labels() -> pd.DataFrame:
    """Precompute mel specs for labeled soundscape segments."""
    sr = CFG["sr"]
    target_len = int(sr * CFG["duration"])
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    rows = []
    cache_sub = CACHE_DIR / "soundscapes"
    cache_sub.mkdir(exist_ok=True)

    # Group by filename to load each file once
    audio_cache = {}
    for filename in tqdm(labels_df["filename"].unique(), desc="Loading soundscapes"):
        filepath = DATA / "train_soundscapes" / filename
        if not filepath.exists():
            continue
        try:
            wav, _ = librosa.load(filepath, sr=sr)
            audio_cache[filename] = wav
        except Exception:
            continue

    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Precompute soundscapes"):
        filename = row["filename"]
        if filename not in audio_cache:
            continue
        wav = audio_cache[filename]
        start_sec = parse_time_str(str(row["start"]))
        start_sample = int(start_sec * sr)
        end_sample = start_sample + target_len

        chunk = wav[start_sample:end_sample]
        if len(chunk) < target_len:
            chunk = pad_or_crop(chunk, target_len)
        mel = compute_mel(chunk, sr)

        mel_id = f"sc_{idx}"
        mel_path = cache_sub / f"{mel_id}.npy"
        np.save(mel_path, mel)

        # Multi-label
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        species = str(row["primary_label"]).split(";")
        for sp in species:
            sp = sp.strip()
            if sp in SPECIES2IDX:
                label[SPECIES2IDX[sp]] = 1.0

        rows.append({
            "mel_path": str(mel_path),
            "primary_label": species[0] if species else "unknown",
            "source": "soundscape",
            "label": label,
        })

    return pd.DataFrame(rows)


# ====================================================================
# Phase 2: Dataset & Model
# ====================================================================
class MelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.img_size = CFG["img_size"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"])  # (n_mels, time)
        label = row["label"]
        if isinstance(label, str):
            label = np.array(json.loads(label), dtype=np.float32)

        # Resize to (img_size, img_size)
        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)  # (1, H, W)

        if self.is_train:
            mel_t = self._spec_augment(mel_t)

        return mel_t, torch.tensor(label, dtype=torch.float32)

    def _spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        """SpecAugment: time and frequency masking."""
        _, h, w = spec.shape
        # Freq mask
        if random.random() < 0.5:
            for _ in range(random.randint(1, 2)):
                mask_h = random.randint(5, min(20, h // 4))
                start = random.randint(0, h - mask_h)
                spec[:, start:start + mask_h, :] = 0
        # Time mask
        if random.random() < 0.5:
            for _ in range(random.randint(1, 2)):
                mask_w = random.randint(5, min(20, w // 4))
                start = random.randint(0, w - mask_w)
                spec[:, :, start:start + mask_w] = 0
        return spec


class BirdModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=1, num_classes=0, global_pool="avg"
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ====================================================================
# Phase 3: Training
# ====================================================================
def mixup(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Train Epoch {epoch+1}")
    for mels, labels in pbar:
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # Mixup
        mels, labels = mixup(mels, labels, CFG["mixup_alpha"])

        # Label smoothing
        ls = CFG["label_smoothing"]
        labels = labels * (1 - ls) + ls / NUM_CLASSES

        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(mels)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")

    return np.mean(losses)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    losses = []

    for mels, labels in tqdm(loader, desc="Validation"):
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast("cuda"):
            logits = model(mels)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        losses.append(loss.item())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    val_loss = np.mean(losses)

    # Macro AUC (skip classes with no positives)
    aucs = []
    scored_classes = []
    for c in range(NUM_CLASSES):
        gt = (all_labels[:, c] > 0.5).astype(int)
        if gt.sum() > 0 and gt.sum() < len(gt):
            try:
                auc = roc_auc_score(gt, all_preds[:, c])
                aucs.append(auc)
                scored_classes.append(SPECIES_LIST[c])
            except ValueError:
                continue

    macro_auc = np.mean(aucs) if aucs else 0.0
    return val_loss, macro_auc, len(aucs)


def run_training(train_meta: pd.DataFrame):
    """Run training with stratified k-fold."""
    # Use primary_label for stratification
    train_meta["strat_label"] = train_meta["primary_label"].astype(str)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["strat_label"])):
        if fold not in CFG["train_folds"]:
            continue

        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")

        train_df = train_meta.iloc[train_idx].reset_index(drop=True)
        val_df = train_meta.iloc[val_idx].reset_index(drop=True)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")

        train_ds = MelDataset(train_df, is_train=True)
        val_ds = MelDataset(val_df, is_train=False)

        train_loader = DataLoader(
            train_ds, batch_size=CFG["batch_size"], shuffle=True,
            num_workers=CFG["num_workers"], pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
            num_workers=CFG["num_workers"], pin_memory=True,
        )

        model = BirdModel(CFG["backbone"], NUM_CLASSES, pretrained=CFG["pretrained"]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        total_steps = len(train_loader) * CFG["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
        scaler = GradScaler("cuda")

        best_auc = 0.0
        best_epoch = -1
        history = []

        for epoch in range(CFG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
            val_loss, val_auc, n_scored = validate(model, val_loader)

            print(f"Epoch {epoch+1}/{CFG['epochs']} — "
                  f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                  f"val_auc: {val_auc:.4f} ({n_scored} classes scored)")

            history.append({
                "epoch": epoch + 1, "train_loss": train_loss,
                "val_loss": val_loss, "val_auc": val_auc, "n_scored": n_scored,
            })

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                ckpt_path = WEIGHTS_DIR / f"exp2_effb0_fold{fold}_best.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_auc": val_auc,
                    "config": CFG,
                }, ckpt_path)
                print(f"  → Saved best model (AUC={val_auc:.4f}) to {ckpt_path}")

        results[fold] = {
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "history": history,
        }

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ====================================================================
# Main
# ====================================================================
def main():
    start_time = time.time()

    # ── Phase 1: Precompute ────────────────────────────────
    train_audio_cache = CACHE_DIR / "train_audio"
    soundscape_cache = CACHE_DIR / "soundscapes"
    meta_path = OUT_DIR / "all_meta.parquet"

    if meta_path.exists():
        print("Loading cached metadata...")
        all_meta = pd.read_parquet(meta_path)
        # Reconstruct label arrays from stored lists
        all_meta["label"] = all_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    else:
        print("Phase 1: Precomputing mel spectrograms...")
        train_df = pd.read_csv(DATA / "train.csv")
        ta_meta = precompute_train_audio(train_df)
        sc_meta = precompute_soundscape_labels()

        all_meta = pd.concat([ta_meta, sc_meta], ignore_index=True)
        print(f"\nTotal samples: {len(all_meta)} "
              f"(train_audio: {len(ta_meta)}, soundscapes: {len(sc_meta)})")

        # Save metadata (convert label arrays to lists for parquet)
        save_meta = all_meta.copy()
        save_meta["label"] = save_meta["label"].apply(lambda x: x.tolist())
        save_meta.to_parquet(meta_path)
        print(f"Saved metadata to {meta_path}")

    # ── Phase 2: Training ─────────────────────────────────
    print(f"\nPhase 2: Training on {DEVICE}...")
    results = run_training(all_meta)

    # ── Phase 3: Report ───────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for fold, res in results.items():
        print(f"Fold {fold}: best_auc={res['best_auc']:.4f} @ epoch {res['best_epoch']}")
    print(f"Total time: {elapsed:.1f} min")

    # Save results
    with open(OUT_DIR / "exp2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

    # Save config
    with open(OUT_DIR / "exp2_config.json", "w") as f:
        json.dump(CFG, f, indent=2)


if __name__ == "__main__":
    main()
