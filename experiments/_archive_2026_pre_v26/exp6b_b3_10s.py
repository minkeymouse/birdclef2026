#!/usr/bin/env python3
"""
exp6b_b3_10s.py — EfficientNet-B3 with 10s chunks (reuse exp3 mel cache).

Finding from exp6: 20s train → 5s eval creates temporal mismatch that hurts ss_auc.
Fix: keep 10s (matches exp3), only upgrade backbone B0 → B3.
Reuses exp3 mel cache to skip precompute.
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
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import timm
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp6_outputs"
# Reuse exp3 mel cache
EXP3_DIR = ROOT / "experiments" / "exp3_outputs"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "img_size": 224,              # keep 224 (B3 handles it fine)
    "backbone": "tf_efficientnet_b3.ns_jft_in1k",
    "pretrained": True,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "epochs": 15,
    "batch_size": 48,             # B3 at 224 fits larger batch than B3 at 300
    "num_workers": 4,
    "n_folds": 5,
    "train_folds": [0, 1],
    "mixup_alpha": 0.4,
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
print(f"Config: backbone={CFG['backbone']}, duration={CFG['train_duration']}s, "
      f"img_size={CFG['img_size']}, batch_size={CFG['batch_size']}")


# ====================================================================
# Dataset & Model (same architecture)
# ====================================================================
class MelDataset(Dataset):
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

        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)

        if self.is_train:
            mel_t = self._spec_augment(mel_t)

        return mel_t, torch.tensor(label, dtype=torch.float32)

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

    # Load exp3 cached metadata
    meta_path = EXP3_DIR / "train_meta.parquet"
    eval_meta_path = EXP3_DIR / "eval_meta.parquet"

    if not meta_path.exists():
        print("ERROR: exp3 mel cache not found. Run exp3 first.")
        return

    print("Loading exp3 cached metadata (10s mels)...")
    train_meta = pd.read_parquet(meta_path)
    train_meta["label"] = train_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    eval_meta = pd.read_parquet(eval_meta_path)
    eval_meta["label"] = eval_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))

    print(f"Train: {len(train_meta)}, Eval: {len(eval_meta)}")

    eval_ds = MelDataset(eval_meta, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=96, shuffle=False,
                             num_workers=CFG["num_workers"], pin_memory=True)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["primary_label"])):
        if fold not in CFG["train_folds"]:
            continue

        print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")

        train_df = train_meta.iloc[train_idx].reset_index(drop=True)
        val_df = train_meta.iloc[val_idx].reset_index(drop=True)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")

        train_ds = MelDataset(train_df, is_train=True)
        val_ds = MelDataset(val_df, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                                num_workers=CFG["num_workers"], pin_memory=True)

        model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=CFG["pretrained"]).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model params: {n_params:.1f}M")

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
                ckpt = WEIGHTS_DIR / f"exp6_b3_fold{fold}_best.pth"
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch + 1, "val_auc": val_auc, "config": CFG}, ckpt)
                print(f"  -> Saved {ckpt.name} (AUC={val_auc:.4f})")

        state = torch.load(WEIGHTS_DIR / f"exp6_b3_fold{fold}_best.pth",
                           map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        _, final_ss, ss_n = validate(model, eval_loader)
        print(f"Fold {fold} best -> ss_auc: {final_ss:.4f} ({ss_n} cls)")

        results[fold] = {"best_auc": best_auc, "best_epoch": best_epoch, "ss_auc": final_ss}

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    for fold, res in results.items():
        print(f"Fold {fold}: val_auc={res['best_auc']:.4f}@E{res['best_epoch']}, ss_auc={res['ss_auc']:.4f}")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp6b_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    with open(OUT_DIR / "exp6b_config.json", "w") as f:
        json.dump(CFG, f, indent=2)


if __name__ == "__main__":
    main()
