#!/usr/bin/env python3
"""
exp18_sampling_warmrestart.py — Equal species sampling + Cosine warm restarts.

1st place: "equal sampling weight for each species" + "cosine annealing warm
restarts with warm restart every 5 epochs, total epochs 15"

Key changes from exp14 baseline:
  1. WeightedRandomSampler: equal probability per species (class-balanced)
  2. CosineAnnealingWarmRestarts: T_0=5 epochs, T_mult=1 (restart every 5 epochs)
  3. Label smoothing (2nd place: alpha=0.05)
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp18_outputs"

for d in [WEIGHTS_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CFG = {
    "seed": 42,
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
    # Augmentation (same as exp14)
    "bg_mix_prob": 0.5,
    "bg_mix_alpha_range": (0.1, 0.4),
    "mel_gain_range": (0.8, 1.2),
    "mel_noise_prob": 0.3,
    "mel_noise_std": 0.02,
    # New: warm restarts
    "warm_restart_T0": 5,  # restart every 5 epochs
    # New: label smoothing
    "label_smoothing": 0.05,
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
# Equal Species Sampling
# ====================================================================
def make_species_balanced_sampler(df):
    """Create WeightedRandomSampler with equal probability per species.
    1st place: 'equal sampling weight for each species'."""
    species_counts = df["primary_label"].value_counts()
    weights = df["primary_label"].map(lambda s: 1.0 / species_counts.get(s, 1)).values
    weights = weights / weights.sum()  # normalize
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(df),
        replacement=True,
    )
    return sampler


# ====================================================================
# Dataset (same as exp14)
# ====================================================================
class AugMelDataset(Dataset):
    def __init__(self, df, bg_mel_paths=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.bg_mel_paths = bg_mel_paths if bg_mel_paths else []
        self.is_train = is_train
        self.img_size = CFG["img_size"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"]).copy()
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
        if self.bg_mel_paths and random.random() < CFG["bg_mix_prob"]:
            bg_path = random.choice(self.bg_mel_paths)
            bg_mel = np.load(bg_path)
            if bg_mel.shape[1] != mel.shape[1]:
                min_w = min(bg_mel.shape[1], mel.shape[1])
                bg_mel = bg_mel[:, :min_w]
                mel_out = mel[:, :min_w].copy()
            else:
                mel_out = mel.copy()
            lo, hi = CFG["bg_mix_alpha_range"]
            alpha = random.uniform(lo, hi)
            mel[:, :bg_mel.shape[1]] = (1 - alpha) * mel_out + alpha * bg_mel

        lo, hi = CFG["mel_gain_range"]
        mel = np.clip(mel * random.uniform(lo, hi), 0, 1)

        if random.random() < CFG["mel_noise_prob"]:
            noise = np.random.randn(*mel.shape).astype(np.float32) * CFG["mel_noise_std"]
            mel = np.clip(mel + noise, 0, 1)
        return mel

    def _spec_augment(self, spec):
        _, h, w = spec.shape
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                mask_h = random.randint(5, min(30, h // 4))
                start = random.randint(0, h - mask_h)
                spec[:, start:start + mask_h, :] = 0
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                mask_w = random.randint(5, min(30, w // 4))
                start = random.randint(0, w - mask_w)
                spec[:, :, start:start + mask_w] = 0
        return spec


class MelDataset(Dataset):
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
# Model (same as exp14)
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
# Training with label smoothing
# ====================================================================
def mixup(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def smooth_labels(labels, epsilon=0.05):
    """Label smoothing: y' = y * (1 - eps) + 0.5 * eps"""
    return labels * (1 - epsilon) + 0.5 * epsilon


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Train E{epoch+1}")
    for mels, labels in pbar:
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        mels, labels = mixup(mels, labels, CFG["mixup_alpha"])

        # Apply label smoothing
        labels = smooth_labels(labels, CFG["label_smoothing"])

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
    all_preds, all_labels = [], []
    for mels, labels in loader:
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast("cuda"):
            clipwise, _ = model(mels)
        all_preds.append(torch.sigmoid(clipwise).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
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
    return np.mean(aucs) if aucs else 0.0, len(aucs)


def main():
    start_time = time.time()

    exp11_dir = ROOT / "experiments" / "exp11_outputs"
    exp14_dir = ROOT / "experiments" / "exp14_outputs"

    train_meta = pd.read_parquet(exp11_dir / "train_meta.parquet")
    train_meta["label"] = train_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    eval_meta = pd.read_parquet(exp11_dir / "eval_meta.parquet")
    eval_meta["label"] = eval_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))

    bg_mel_paths = [str(p) for p in (exp14_dir / "bg_mel_cache").glob("bg_*.npy")]
    print(f"Train: {len(train_meta)}, Eval: {len(eval_meta)}, BG mels: {len(bg_mel_paths)}")

    eval_ds = MelDataset(eval_meta)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False,
                             num_workers=CFG["num_workers"], pin_memory=True)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["primary_label"])):
        if fold not in CFG["train_folds"]:
            continue

        print(f"\n{'='*60}\nFOLD {fold} (Equal Sampling + Warm Restarts + Label Smoothing)\n{'='*60}")
        fold_train = train_meta.iloc[train_idx].reset_index(drop=True)
        fold_val = train_meta.iloc[val_idx].reset_index(drop=True)

        # Species-balanced sampler
        sampler = make_species_balanced_sampler(fold_train)

        train_ds = AugMelDataset(fold_train, bg_mel_paths=bg_mel_paths, is_train=True)
        val_ds = MelDataset(fold_val)

        # Note: shuffle=False because sampler handles ordering
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                                num_workers=CFG["num_workers"], pin_memory=True)

        model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=CFG["pretrained"]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

        # Cosine warm restarts (1st place: restart every 5 epochs)
        steps_per_epoch = len(train_loader)
        T_0_steps = CFG["warm_restart_T0"] * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0_steps, T_mult=1, eta_min=1e-6
        )
        scaler = GradScaler("cuda")

        best_ss = 0.0
        best_epoch = -1

        for epoch in range(CFG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
            val_auc, n_scored = validate(model, val_loader)
            ss_auc, ss_n = validate(model, eval_loader)

            print(f"E{epoch+1}/{CFG['epochs']} — loss: {train_loss:.4f}, "
                  f"val_auc: {val_auc:.4f}({n_scored}), ss_auc: {ss_auc:.4f}({ss_n})")

            if ss_auc > best_ss:
                best_ss = ss_auc
                best_epoch = epoch + 1
                ckpt = WEIGHTS_DIR / f"exp18_sampler_fold{fold}_best.pth"
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch + 1, "ss_auc": ss_auc, "val_auc": val_auc}, ckpt)
                print(f"  -> Saved {ckpt.name} (ss_auc={ss_auc:.4f})")

        print(f"Fold {fold} best -> ss_auc: {best_ss:.4f} @ E{best_epoch}")
        results[fold] = {"val_auc": val_auc, "best_epoch": best_epoch, "ss_auc": best_ss}

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nRESULTS (exp18: Equal Sampling + Warm Restarts)")
    print(f"{'='*60}")
    exp14_ss = {0: 0.7820, 1: 0.7681}
    for fold, res in results.items():
        delta = res['ss_auc'] - exp14_ss.get(fold, 0)
        print(f"Fold {fold}: ss_auc={res['ss_auc']:.4f}@E{res['best_epoch']}, Δ vs exp14: {delta:+.4f}")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp18_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)


if __name__ == "__main__":
    main()
