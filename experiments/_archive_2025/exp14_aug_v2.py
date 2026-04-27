#!/usr/bin/env python3
"""
exp14_aug_v2.py — Spectrogram-level augmentation + background mixing.

CPU-friendly redesign: all augmentation at mel-spectrogram level using cached mels.
NO on-the-fly librosa loading during training.

Augmentations (all spectrogram-level):
  1. Background mixing: precomputed 2025 soundscape mel spectrograms
  2. Random gain: multiplicative scaling on mel
  3. Gaussian noise: additive on mel
  4. Enhanced SpecAugment: wider + more masks

Reuses exp11 mel cache for train data. Precomputes 2025 background mels once.
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
DATA_2025 = ROOT / "data" / "birdclef-2025"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp14_outputs"
BG_CACHE_DIR = OUT_DIR / "bg_mel_cache"

for d in [WEIGHTS_DIR, OUT_DIR, BG_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CFG = {
    "seed": 42,
    "sr": 32000,
    "train_duration": 5.0,
    "infer_duration": 5.0,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 0,
    "fmax": 16000,
    "img_size": 224,
    "backbone": "tf_efficientnet_b0.ns_jft_in1k",
    "pretrained": True,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "epochs": 15,
    "batch_size": 64,
    "num_workers": 2,
    "n_folds": 5,
    "train_folds": [0, 1],
    "mixup_alpha": 0.4,
    # Augmentation params
    "bg_mix_prob": 0.5,
    "bg_mix_alpha_range": (0.1, 0.4),  # blend weight for background mel
    "mel_gain_range": (0.8, 1.2),       # multiplicative gain on mel
    "mel_noise_prob": 0.3,
    "mel_noise_std": 0.02,
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


# ====================================================================
# Precompute background noise mels from 2025 soundscapes (one-time)
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


def precompute_background_mels(max_files=300, chunks_per_file=4):
    """Precompute 5s mel spectrograms from 2025 soundscapes as background noise."""
    bg_paths = []
    existing = list(BG_CACHE_DIR.glob("bg_*.npy"))
    if len(existing) > 100:
        print(f"Reusing {len(existing)} cached background mels")
        return [str(p) for p in existing]

    ss_dir = DATA_2025 / "train_soundscapes"
    if not ss_dir.exists():
        print(f"WARNING: 2025 soundscapes not found at {ss_dir}")
        return []

    sr = CFG["sr"]
    target_len = int(sr * CFG["train_duration"])
    all_files = sorted(ss_dir.glob("*.ogg"))
    rng = np.random.RandomState(CFG["seed"])
    rng.shuffle(all_files)

    for fpath in tqdm(all_files[:max_files], desc="Precomputing background mels"):
        try:
            wav, _ = librosa.load(fpath, sr=sr)
            if len(wav) < target_len:
                continue
            for c in range(min(chunks_per_file, len(wav) // target_len)):
                start = rng.randint(0, max(1, len(wav) - target_len))
                chunk = wav[start:start + target_len]
                mel = compute_mel(chunk, sr)
                mel_path = BG_CACHE_DIR / f"bg_{fpath.stem}_{c}.npy"
                np.save(mel_path, mel)
                bg_paths.append(str(mel_path))
        except Exception:
            continue

    print(f"Background mel pool: {len(bg_paths)} mels")
    return bg_paths


def _parse_time(t):
    parts = t.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


# ====================================================================
# Dataset with mel-level augmentation (CPU-friendly, no librosa at train time)
# ====================================================================
class AugMelDataset(Dataset):
    """Loads cached mels, applies augmentation at spectrogram level only."""

    def __init__(self, df, bg_mel_paths=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.bg_mel_paths = bg_mel_paths if bg_mel_paths else []
        self.is_train = is_train
        self.img_size = CFG["img_size"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"]).copy()  # copy for in-place augmentation
        label = row["label"]
        if isinstance(label, str):
            label = np.array(json.loads(label), dtype=np.float32)

        if self.is_train:
            mel = self._augment_mel(mel)

        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)

        if self.is_train:
            mel_t = self._spec_augment(mel_t)

        return mel_t, torch.tensor(label, dtype=torch.float32)

    def _augment_mel(self, mel):
        """Apply augmentation at mel spectrogram level."""
        # 1. Background mixing (mel-level blend)
        if self.bg_mel_paths and random.random() < CFG["bg_mix_prob"]:
            bg_path = random.choice(self.bg_mel_paths)
            bg_mel = np.load(bg_path)
            # Match shapes: crop/pad to match target mel
            if bg_mel.shape[1] != mel.shape[1]:
                min_w = min(bg_mel.shape[1], mel.shape[1])
                bg_mel = bg_mel[:, :min_w]
                mel_cropped = mel[:, :min_w]
            else:
                mel_cropped = mel
                bg_mel = bg_mel
            lo, hi = CFG["bg_mix_alpha_range"]
            alpha = random.uniform(lo, hi)
            mel[:, :bg_mel.shape[1]] = (1 - alpha) * mel_cropped + alpha * bg_mel

        # 2. Random gain (multiplicative on mel)
        lo, hi = CFG["mel_gain_range"]
        gain = random.uniform(lo, hi)
        mel = np.clip(mel * gain, 0, 1)

        # 3. Gaussian noise on mel
        if random.random() < CFG["mel_noise_prob"]:
            noise = np.random.randn(*mel.shape).astype(np.float32) * CFG["mel_noise_std"]
            mel = np.clip(mel + noise, 0, 1)

        return mel

    def _spec_augment(self, spec):
        """Enhanced SpecAugment: wider masks, more masks."""
        _, h, w = spec.shape
        # Freq mask (up to 3 masks, wider)
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                mask_h = random.randint(5, min(30, h // 4))
                start = random.randint(0, h - mask_h)
                spec[:, start:start + mask_h, :] = 0
        # Time mask (up to 3 masks, wider)
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                mask_w = random.randint(5, min(30, w // 4))
                start = random.randint(0, w - mask_w)
                spec[:, :, start:start + mask_w] = 0
        return spec


class MelDataset(Dataset):
    """Simple eval dataset."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.img_size = CFG["img_size"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"])
        label = row["label"]
        if isinstance(label, str):
            label = np.array(json.loads(label), dtype=np.float32)
        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)
        return mel_t, torch.tensor(label, dtype=torch.float32)


# ====================================================================
# Model (same as exp11)
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
# Training
# ====================================================================
def mixup(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Train E{epoch+1}")
    for mels, labels in pbar:
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        mels, labels = mixup(mels, labels, CFG["mixup_alpha"])

        optimizer.zero_grad()
        with autocast("cuda"):
            clipwise, framewise_max = model(mels)
            loss_clip = F.binary_cross_entropy_with_logits(clipwise, labels)
            loss_frame = F.binary_cross_entropy_with_logits(framewise_max, labels)
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
    for mels, labels in loader:
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
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


def main():
    start_time = time.time()

    # ── Precompute background mels (one-time CPU cost) ───────
    bg_mel_paths = precompute_background_mels(max_files=300, chunks_per_file=4)

    # ── Load training data from exp11 cache ──────────────────
    exp11_dir = ROOT / "experiments" / "exp11_outputs"
    meta_path = exp11_dir / "train_meta.parquet"
    eval_meta_path = exp11_dir / "eval_meta.parquet"

    if not meta_path.exists() or not eval_meta_path.exists():
        print("ERROR: exp11 cache not found. Run exp11 first.")
        sys.exit(1)

    print("Loading exp11 mel cache (no librosa during training)...")
    train_meta = pd.read_parquet(meta_path)
    train_meta["label"] = train_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    eval_meta = pd.read_parquet(eval_meta_path)
    eval_meta["label"] = eval_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))

    print(f"Train: {len(train_meta)}, Eval: {len(eval_meta)}, Background mels: {len(bg_mel_paths)}")

    # ── Training ─────────────────────────────────────────────
    eval_ds = MelDataset(eval_meta)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False,
                             num_workers=CFG["num_workers"], pin_memory=True)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["primary_label"])):
        if fold not in CFG["train_folds"]:
            continue

        print(f"\n{'='*60}\nFOLD {fold} (mel-level aug + bg mixing)\n{'='*60}")
        fold_train = train_meta.iloc[train_idx].reset_index(drop=True)
        fold_val = train_meta.iloc[val_idx].reset_index(drop=True)

        train_ds = AugMelDataset(fold_train, bg_mel_paths=bg_mel_paths, is_train=True)
        val_ds = MelDataset(fold_val)

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                                num_workers=CFG["num_workers"], pin_memory=True)

        model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=CFG["pretrained"]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        total_steps = len(train_loader) * CFG["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
        scaler = GradScaler("cuda")

        best_auc = 0.0
        best_epoch = -1

        for epoch in range(CFG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
            val_loss, val_auc, n_scored = validate(model, val_loader)

            ss_str = ""
            if (epoch + 1) % 3 == 0 or epoch == CFG["epochs"] - 1:
                _, ss_auc, ss_n = validate(model, eval_loader)
                ss_str = f", ss_auc: {ss_auc:.4f}({ss_n})"

            print(f"E{epoch+1}/{CFG['epochs']} — loss: {train_loss:.4f}, "
                  f"val_auc: {val_auc:.4f}({n_scored}){ss_str}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                ckpt = WEIGHTS_DIR / f"exp14_aug_fold{fold}_best.pth"
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch + 1, "val_auc": val_auc, "config": CFG}, ckpt)
                print(f"  -> Saved {ckpt.name} (AUC={val_auc:.4f})")

        state = torch.load(WEIGHTS_DIR / f"exp14_aug_fold{fold}_best.pth",
                           map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        _, final_ss, ss_n = validate(model, eval_loader)
        print(f"Fold {fold} best -> ss_auc: {final_ss:.4f} ({ss_n} cls)")
        results[fold] = {"best_auc": best_auc, "best_epoch": best_epoch, "ss_auc": final_ss}

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nRESULTS (exp14: mel-level aug + bg mixing)")
    print(f"{'='*60}")
    for fold, res in results.items():
        print(f"Fold {fold}: val_auc={res['best_auc']:.4f}@E{res['best_epoch']}, ss_auc={res['ss_auc']:.4f}")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp14_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    with open(OUT_DIR / "exp14_config.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, tuple) else v for k, v in CFG.items()}, f, indent=2)


if __name__ == "__main__":
    main()
