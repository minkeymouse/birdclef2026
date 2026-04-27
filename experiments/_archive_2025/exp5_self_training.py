#!/usr/bin/env python3
"""
exp5_self_training.py — Iterative self-training on unlabeled soundscapes.

Pipeline:
  Phase 1: Generate pseudo-labels for unlabeled soundscapes using exp4 KD model
  Phase 2: Retrain with train_audio + pseudo-labeled soundscapes (noise injection)
  Phase 3: (Optional) Iterate: re-pseudolabel with improved model

Key techniques from winning solutions:
  - High-confidence threshold for pseudo-labels
  - Power transform on pseudo-label probabilities to reduce noise
  - Mixup + SpecAugment + stochastic depth as noise injection
  - Labeled soundscapes held out for honest evaluation
"""
import os
import sys
import gc
import json
import time
import random
import ast
from pathlib import Path

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
OUT_DIR = ROOT / "experiments" / "exp5_outputs"
CACHE_DIR = OUT_DIR / "mel_cache"
PSEUDO_DIR = OUT_DIR / "pseudo_labels"

for d in [WEIGHTS_DIR, OUT_DIR, CACHE_DIR, PSEUDO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────
CFG = {
    "seed": 42,
    "sr": 32000,
    "train_duration": 10.0,
    "infer_duration": 5.0,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 0,
    "fmax": 16000,
    "img_size": 224,
    "backbone": "tf_efficientnet_b0.ns_jft_in1k",
    "pretrained": True,
    "lr": 5e-4,             # lower LR for self-training (fine-tuning)
    "weight_decay": 1e-2,
    "epochs": 15,
    "batch_size": 64,
    "num_workers": 4,
    "n_folds": 5,
    "train_folds": [0, 1],
    "mixup_alpha": 0.5,     # stronger mixup for noise injection
    # Self-training params
    "pseudo_threshold": 0.5,     # confidence threshold for pseudo-labels
    "pseudo_power": 0.7,         # power transform to sharpen/soften pseudo-labels
    "pseudo_weight": 0.5,        # weight for pseudo-labeled samples in loss
    "teacher_weights": "exp3",   # exp3 has better ss_auc (0.73 vs exp4's 0.60)
    "self_training_rounds": 1,   # number of self-training iterations
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

taxonomy_df = pd.read_csv(DATA / "taxonomy.csv")
SPECIES_LIST = sorted(taxonomy_df["primary_label"].astype(str).tolist())
SPECIES2IDX = {sp: i for i, sp in enumerate(SPECIES_LIST)}
NUM_CLASSES = len(SPECIES_LIST)
print(f"Classes: {NUM_CLASSES}, Device: {DEVICE}")


# ====================================================================
# Model (same architecture as exp3/exp4)
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
        framewise_max = framewise.max(dim=1).values
        return clipwise, framewise_max


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=1, num_classes=0, global_pool="",
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, CFG["img_size"], CFG["img_size"])
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[1]
        self.head = AttentionHead(self.feat_dim, num_classes, dropout=0.3)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.mean(dim=2).permute(0, 2, 1)
        clipwise, framewise_max = self.head(feat)
        return clipwise, framewise_max


# ====================================================================
# Audio & Mel utilities
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


def pad_or_crop(wav, target_len, random_crop=False):
    if len(wav) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(wav) < target_len:
        reps = int(np.ceil(target_len / len(wav)))
        wav = np.tile(wav, reps)
    if len(wav) > target_len:
        start = random.randint(0, len(wav) - target_len) if random_crop else 0
        wav = wav[start:start + target_len]
    return wav[:target_len]


def _parse_time(t):
    parts = t.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


# ====================================================================
# Phase 1: Generate pseudo-labels for unlabeled soundscapes
# ====================================================================
def load_teacher_models(prefix="exp4"):
    """Load teacher model(s) for pseudo-labeling."""
    models = []
    pattern = f"{prefix}_*_best.pth"
    for ckpt_path in sorted(WEIGHTS_DIR.glob(pattern)):
        print(f"Loading teacher: {ckpt_path.name}")
        model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=False)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval().to(DEVICE)
        models.append(model)
    return models


def generate_pseudo_labels(teacher_models, round_num=0):
    """Generate pseudo-labels for all unlabeled soundscapes using teacher ensemble."""
    sr = CFG["sr"]
    target_len = int(sr * CFG["infer_duration"])  # 5s chunks for inference
    train_target = int(sr * CFG["train_duration"])  # 10s for training mels

    # Find unlabeled soundscapes
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    labeled_files = set(labels_df["filename"].unique())
    all_ss = sorted(DATA / "train_soundscapes" / f
                    for f in os.listdir(DATA / "train_soundscapes")
                    if f.endswith(".ogg"))
    unlabeled = [f for f in all_ss if f.name not in labeled_files]
    print(f"Unlabeled soundscapes: {len(unlabeled)}")

    cache_sub = CACHE_DIR / f"pseudo_r{round_num}"
    cache_sub.mkdir(exist_ok=True)
    pseudo_rows = []
    stats = {"total_chunks": 0, "above_threshold": 0}

    for fpath in tqdm(unlabeled, desc=f"Pseudo-labeling (round {round_num})"):
        try:
            wav, _ = librosa.load(fpath, sr=sr)
        except Exception:
            continue

        if len(wav) == 0:
            continue

        # Process in 5s chunks, predict with teacher ensemble
        n_5s = max(1, len(wav) // target_len)
        file_preds = []

        for seg in range(n_5s):
            chunk_5s = wav[seg * target_len:(seg + 1) * target_len]
            if len(chunk_5s) < target_len:
                chunk_5s = np.pad(chunk_5s, (0, target_len - len(chunk_5s)))

            mel = compute_mel(chunk_5s, sr)
            mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mel_t = F.interpolate(mel_t, size=(CFG["img_size"], CFG["img_size"]),
                                  mode="bilinear", align_corners=False).to(DEVICE)

            seg_preds = []
            with torch.no_grad():
                for model in teacher_models:
                    with autocast("cuda"):
                        clipwise, _ = model(mel_t)
                    prob = torch.sigmoid(clipwise).cpu().numpy()[0]
                    seg_preds.append(prob)

            avg_pred = np.mean(seg_preds, axis=0)
            file_preds.append(avg_pred)

        # Average over all 5s segments -> file-level pseudo-label
        file_pred = np.mean(file_preds, axis=0)

        # Apply confidence threshold + power transform
        max_conf = file_pred.max()
        stats["total_chunks"] += 1

        if max_conf < CFG["pseudo_threshold"]:
            continue

        stats["above_threshold"] += 1

        # Power transform to sharpen confident predictions
        pseudo_label = np.power(file_pred, CFG["pseudo_power"])
        # Zero out low-confidence species
        pseudo_label[pseudo_label < 0.1] = 0.0

        # Create 10s training chunks from this soundscape
        for c in range(max(1, len(wav) // train_target)):
            mel_id = f"ps_r{round_num}_{fpath.stem}_{c}"
            mel_path = cache_sub / f"{mel_id}.npy"

            if not mel_path.exists():
                start = c * train_target
                chunk = pad_or_crop(wav[start:], train_target)
                mel = compute_mel(chunk, sr)
                np.save(mel_path, mel)

            # Determine pseudo primary label (for stratification)
            pseudo_primary = SPECIES_LIST[np.argmax(pseudo_label)]

            pseudo_rows.append({
                "mel_path": str(mel_path),
                "primary_label": pseudo_primary,
                "source": "pseudo",
                "label": pseudo_label.copy(),
                "weight": CFG["pseudo_weight"],
            })

    print(f"Pseudo-label stats: {stats['above_threshold']}/{stats['total_chunks']} "
          f"above threshold ({100*stats['above_threshold']/(stats['total_chunks']+1e-8):.1f}%)")
    print(f"Pseudo training samples: {len(pseudo_rows)}")

    return pd.DataFrame(pseudo_rows)


# ====================================================================
# Data preparation
# ====================================================================
def prepare_train_data():
    """Prepare original train_audio mel cache (reuse from exp3/exp4 if available)."""
    sr = CFG["sr"]
    target_len = int(sr * CFG["train_duration"])

    # Try to reuse exp3/exp4 mel cache
    for prev_exp in ["exp4_outputs", "exp3_outputs"]:
        prev_cache = ROOT / "experiments" / prev_exp / "mel_cache" / "train_audio"
        prev_meta = ROOT / "experiments" / prev_exp / "train_meta.parquet"
        if prev_meta.exists() and prev_cache.exists():
            print(f"Reusing mel cache from {prev_exp}")
            meta = pd.read_parquet(prev_meta)
            meta["label"] = meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
            # Add weight column
            meta["weight"] = 1.0
            # Filter to only entries where mel files exist
            meta = meta[meta["mel_path"].apply(os.path.exists)].reset_index(drop=True)
            if len(meta) > 0:
                return meta

    # Fallback: precompute from scratch
    print("Precomputing train_audio mels from scratch...")
    train_df = pd.read_csv(DATA / "train.csv")
    cache_sub = CACHE_DIR / "train_audio"
    cache_sub.mkdir(exist_ok=True)
    rows = []

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

        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        pl = str(row["primary_label"])
        if pl in SPECIES2IDX:
            label[SPECIES2IDX[pl]] = 1.0
        sec = row.get("secondary_labels", "[]")
        if isinstance(sec, str) and sec not in ("[]", "", "nan"):
            try:
                sec_list = ast.literal_eval(sec)
                if isinstance(sec_list, list):
                    for s in sec_list:
                        s = str(s).strip()
                        if s in SPECIES2IDX:
                            label[SPECIES2IDX[s]] = 0.3
            except Exception:
                pass

        n_chunks = max(1, len(wav) // target_len)
        for c in range(min(n_chunks, 2)):
            mel_id = f"ta_{idx}_{c}"
            mel_path = cache_sub / f"{mel_id}.npy"
            if not mel_path.exists():
                start = c * target_len
                chunk = pad_or_crop(wav[start:], target_len)
                mel = compute_mel(chunk, sr)
                np.save(mel_path, mel)

            rows.append({
                "mel_path": str(mel_path),
                "primary_label": pl,
                "source": "train_audio",
                "label": label,
                "weight": 1.0,
            })

    return pd.DataFrame(rows)


def prepare_eval_data():
    """Prepare soundscape eval data (reuse from exp3/exp4 if available)."""
    for prev_exp in ["exp4_outputs", "exp3_outputs"]:
        prev_meta = ROOT / "experiments" / prev_exp / "eval_meta.parquet"
        if prev_meta.exists():
            print(f"Reusing eval data from {prev_exp}")
            meta = pd.read_parquet(prev_meta)
            meta["label"] = meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
            meta["weight"] = 1.0
            meta = meta[meta["mel_path"].apply(os.path.exists)].reset_index(drop=True)
            if len(meta) > 0:
                return meta

    # Fallback: precompute
    sr = CFG["sr"]
    target_len = int(sr * CFG["infer_duration"])
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    cache_sub = CACHE_DIR / "soundscapes"
    cache_sub.mkdir(exist_ok=True)
    rows = []

    audio_cache = {}
    for filename in tqdm(labels_df["filename"].unique(), desc="Loading soundscapes"):
        filepath = DATA / "train_soundscapes" / filename
        if filepath.exists():
            try:
                wav, _ = librosa.load(filepath, sr=sr)
                audio_cache[filename] = wav
            except Exception:
                continue

    for idx, row in labels_df.iterrows():
        filename = row["filename"]
        if filename not in audio_cache:
            continue
        wav = audio_cache[filename]
        start_sec = _parse_time(str(row["start"]))
        start_sample = int(start_sec * sr)
        chunk = wav[start_sample:start_sample + target_len]
        if len(chunk) < target_len:
            chunk = pad_or_crop(chunk, target_len)
        mel = compute_mel(chunk, sr)

        mel_id = f"sc_{idx}"
        mel_path = cache_sub / f"{mel_id}.npy"
        if not mel_path.exists():
            np.save(mel_path, mel)

        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in SPECIES2IDX:
                label[SPECIES2IDX[sp]] = 1.0

        rows.append({
            "mel_path": str(mel_path),
            "primary_label": sp,
            "source": "soundscape",
            "label": label,
            "weight": 1.0,
        })

    return pd.DataFrame(rows)


# ====================================================================
# Dataset
# ====================================================================
class STDataset(Dataset):
    """Dataset with sample weights for mixing real + pseudo data."""
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.img_size = CFG["img_size"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"])
        label = row["label"]
        if isinstance(label, str):
            label = np.array(json.loads(label), dtype=np.float32)
        weight = float(row.get("weight", 1.0))

        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)

        if self.is_train:
            mel_t = self._spec_augment(mel_t)

        return mel_t, torch.tensor(label, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)

    def _spec_augment(self, spec):
        _, h, w = spec.shape
        if random.random() < 0.5:
            for _ in range(random.randint(1, 2)):
                mask_h = random.randint(5, min(25, h // 4))
                start = random.randint(0, h - mask_h)
                spec[:, start:start + mask_h, :] = 0
        if random.random() < 0.5:
            for _ in range(random.randint(1, 2)):
                mask_w = random.randint(5, min(25, w // 4))
                start = random.randint(0, w - mask_w)
                spec[:, :, start:start + mask_w] = 0
        return spec


# ====================================================================
# Training
# ====================================================================
def mixup(x, y, w, alpha=0.4):
    if alpha <= 0:
        return x, y, w
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx], lam * w + (1 - lam) * w[idx]


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Train E{epoch+1}")

    for mels, labels, weights in pbar:
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        weights = weights.to(DEVICE, non_blocking=True)

        mels, labels, weights = mixup(mels, labels, weights, CFG["mixup_alpha"])

        optimizer.zero_grad()
        with autocast("cuda"):
            clipwise, framewise_max = model(mels)
            loss_clip = F.binary_cross_entropy_with_logits(clipwise, labels, reduction="none")
            loss_frame = F.binary_cross_entropy_with_logits(framewise_max, labels, reduction="none")
            # Per-sample weighting
            loss_clip = (loss_clip.mean(dim=1) * weights).mean()
            loss_frame = (loss_frame.mean(dim=1) * weights).mean()
            loss = 0.5 * loss_clip + 0.5 * loss_frame

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
    all_preds, all_labels, losses = [], [], []

    for batch in loader:
        mels = batch[0].to(DEVICE, non_blocking=True)
        labels = batch[1].to(DEVICE, non_blocking=True)

        with autocast("cuda"):
            clipwise, _ = model(mels)
            loss = F.binary_cross_entropy_with_logits(clipwise, labels)

        all_preds.append(torch.sigmoid(clipwise).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        losses.append(loss.item())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    aucs = []
    for c in range(NUM_CLASSES):
        gt = (all_labels[:, c] > 0.5).astype(int)
        if 0 < gt.sum() < len(gt):
            try:
                aucs.append(roc_auc_score(gt, all_preds[:, c]))
            except ValueError:
                pass

    return np.mean(losses), np.mean(aucs) if aucs else 0.0, len(aucs)


# ====================================================================
# Main
# ====================================================================
def main():
    start_time = time.time()

    # ── Prepare base data ─────────────────────────────────────
    print("Preparing train data...")
    train_meta = prepare_train_data()
    print(f"Train audio samples: {len(train_meta)}")

    print("Preparing eval data...")
    eval_meta = prepare_eval_data()
    print(f"Eval samples: {len(eval_meta)}")

    eval_ds = STDataset(eval_meta, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False,
                             num_workers=CFG["num_workers"], pin_memory=True)

    # ── Self-training rounds ──────────────────────────────────
    teacher_prefix = CFG["teacher_weights"]

    for round_num in range(CFG["self_training_rounds"]):
        print(f"\n{'#'*60}")
        print(f"# SELF-TRAINING ROUND {round_num}")
        print(f"{'#'*60}")

        # Phase 1: Pseudo-label with teacher
        print("\nPhase 1: Generating pseudo-labels...")
        teacher_models = load_teacher_models(teacher_prefix)
        if not teacher_models:
            print(f"ERROR: No teacher weights found for prefix '{teacher_prefix}'")
            return

        pseudo_meta = generate_pseudo_labels(teacher_models, round_num)

        # Free teacher models
        del teacher_models
        torch.cuda.empty_cache()
        gc.collect()

        if len(pseudo_meta) == 0:
            print("No pseudo-labels generated. Stopping.")
            return

        # Phase 2: Combine and train
        combined = pd.concat([train_meta, pseudo_meta], ignore_index=True)
        n_real = len(train_meta)
        n_pseudo = len(pseudo_meta)
        print(f"\nPhase 2: Training on {n_real} real + {n_pseudo} pseudo = {len(combined)} total")

        # Train
        skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
        results = {}

        for fold, (train_idx, val_idx) in enumerate(skf.split(combined, combined["primary_label"])):
            if fold not in CFG["train_folds"]:
                continue

            print(f"\n{'='*60}\nROUND {round_num} — FOLD {fold}\n{'='*60}")

            train_df = combined.iloc[train_idx].reset_index(drop=True)
            val_df = combined.iloc[val_idx].reset_index(drop=True)

            # Count real vs pseudo in train split
            n_train_real = (train_df["source"] != "pseudo").sum()
            n_train_pseudo = (train_df["source"] == "pseudo").sum()
            print(f"Train: {len(train_df)} ({n_train_real} real + {n_train_pseudo} pseudo), Val: {len(val_df)}")

            train_ds = STDataset(train_df, is_train=True)
            val_ds = STDataset(val_df, is_train=False)

            train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                      num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                                    num_workers=CFG["num_workers"], pin_memory=True)

            # Initialize from teacher weights for faster convergence
            model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=False).to(DEVICE)
            teacher_ckpt = WEIGHTS_DIR / f"{teacher_prefix}_kd_fold{fold}_best.pth"
            if not teacher_ckpt.exists():
                # Try other naming patterns
                teacher_ckpt = WEIGHTS_DIR / f"{teacher_prefix}_sed_fold{fold}_best.pth"
            if teacher_ckpt.exists():
                print(f"Initializing from {teacher_ckpt.name}")
                state = torch.load(teacher_ckpt, map_location=DEVICE, weights_only=False)
                model.load_state_dict(state["model_state_dict"])
            else:
                print(f"No teacher checkpoint found, training from pretrained backbone")
                model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=True).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
            total_steps = len(train_loader) * CFG["epochs"]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
            scaler = GradScaler("cuda")

            best_auc = 0.0
            best_epoch = -1
            weight_name = f"exp5_st_r{round_num}_fold{fold}_best.pth"

            for epoch in range(CFG["epochs"]):
                train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
                val_loss, val_auc, n_scored = validate(model, val_loader)

                ss_str = ""
                if (epoch + 1) % 5 == 0 or epoch == CFG["epochs"] - 1:
                    _, ss_auc, ss_n = validate(model, eval_loader)
                    ss_str = f", ss_auc: {ss_auc:.4f}({ss_n})"

                print(f"E{epoch+1}/{CFG['epochs']} — loss: {train_loss:.4f}, "
                      f"val_auc: {val_auc:.4f}({n_scored}){ss_str}")

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_epoch = epoch + 1
                    ckpt = WEIGHTS_DIR / weight_name
                    torch.save({"model_state_dict": model.state_dict(),
                                "epoch": epoch + 1, "val_auc": val_auc,
                                "round": round_num, "config": CFG}, ckpt)
                    print(f"  -> Saved {ckpt.name} (AUC={val_auc:.4f})")

            # Final soundscape eval
            state = torch.load(WEIGHTS_DIR / weight_name, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            _, final_ss, ss_n = validate(model, eval_loader)
            print(f"Round {round_num} Fold {fold} best -> ss_auc: {final_ss:.4f} ({ss_n} cls)")

            results[fold] = {"best_auc": best_auc, "best_epoch": best_epoch, "ss_auc": final_ss}

            del model, optimizer, scheduler, scaler
            torch.cuda.empty_cache()
            gc.collect()

        # Update teacher prefix for next round
        teacher_prefix = f"exp5_st_r{round_num}"

        # ── Report round ──────────────────────────────────────
        print(f"\n{'='*60}\nROUND {round_num} RESULTS\n{'='*60}")
        for fold, res in results.items():
            print(f"Fold {fold}: val_auc={res['best_auc']:.4f}@E{res['best_epoch']}, ss_auc={res['ss_auc']:.4f}")

    # ── Final report ──────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    print(f"\nTotal time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp5_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    with open(OUT_DIR / "exp5_config.json", "w") as f:
        json.dump(CFG, f, indent=2)


if __name__ == "__main__":
    main()
