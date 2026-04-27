#!/usr/bin/env python3
"""
exp13_multi_backbone.py — Multi-backbone training with 5s chunks.

2025 winner used 6 different backbones. All our models are EfficientNet-B0.
Backbone diversity is critical for ensemble robustness and surviving shakeup.

Trains 3 backbones with exp11 recipe (5s chunks):
  1. EfficientNet-B2 (7.7M) — same family, more capacity
  2. ConvNeXtV2 nano (15M) — entirely different architecture family
  3. ECA-NFNet-L0 (21.8M) — used by 2025 winner

Each backbone: 2-fold, same training recipe as exp11.
Then tests cross-backbone ensemble.

Usage: python exp13_multi_backbone.py [--backbone b2|convnext|nfnet|all]
"""
import os
import sys
import gc
import json
import time
import random
import ast
import argparse
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
OUT_DIR = ROOT / "experiments" / "exp13_outputs"
CACHE_DIR = ROOT / "experiments" / "exp11_outputs" / "mel_cache"  # Reuse exp11 cache!

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Backbone configs ───────────────────────────────────────
BACKBONE_CFGS = {
    "b2": {
        "name": "tf_efficientnet_b2.ns_jft_in1k",
        "prefix": "exp13_b2",
        "batch_size": 64,
        "img_size": 224,
    },
    "convnext": {
        "name": "convnextv2_nano.fcmae_ft_in22k_in1k",
        "prefix": "exp13_cnv2",
        "batch_size": 48,  # Larger model, smaller batch
        "img_size": 224,
    },
    "nfnet": {
        "name": "eca_nfnet_l0.ra2_in1k",
        "prefix": "exp13_nfnet",
        "batch_size": 32,  # Largest model
        "img_size": 224,
    },
}

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
    "pretrained": True,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "epochs": 15,
    "num_workers": 2,
    "n_folds": 5,
    "train_folds": [0, 1],
    "mixup_alpha": 0.4,
    "max_chunks": 4,
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
# Data loading (reuse exp11 mel cache)
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


def load_or_create_metadata():
    """Load exp11 metadata or create if not available."""
    exp11_dir = ROOT / "experiments" / "exp11_outputs"
    meta_path = exp11_dir / "train_meta.parquet"
    eval_meta_path = exp11_dir / "eval_meta.parquet"

    if meta_path.exists() and eval_meta_path.exists():
        print("Reusing exp11 mel cache...")
        train_meta = pd.read_parquet(meta_path)
        train_meta["label"] = train_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
        eval_meta = pd.read_parquet(eval_meta_path)
        eval_meta["label"] = eval_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
        return train_meta, eval_meta

    # Fallback: create own cache (same logic as exp11)
    print("exp11 cache not found, creating own...")
    own_cache = OUT_DIR / "mel_cache"
    own_cache.mkdir(parents=True, exist_ok=True)

    sr = CFG["sr"]
    target_len = int(sr * CFG["train_duration"])

    # Train audio
    train_df = pd.read_csv(DATA / "train.csv")
    cache_sub = own_cache / "train_audio"
    cache_sub.mkdir(exist_ok=True)
    rows = []

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Precompute train_audio (5s)"):
        filepath = DATA / "train_audio" / row["filename"]
        if not filepath.exists():
            continue
        try:
            wav, _ = librosa.load(filepath, sr=sr)
            if len(wav) == 0:
                continue
        except Exception:
            continue

        n_chunks = max(1, len(wav) // target_len)
        for c in range(min(n_chunks, CFG["max_chunks"])):
            mel_id = f"ta_{idx}_{c}"
            mel_path = cache_sub / f"{mel_id}.npy"
            if not mel_path.exists():
                start = c * target_len
                chunk = pad_or_crop(wav[start:], target_len)
                mel = compute_mel(chunk, sr)
                np.save(mel_path, mel)

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
            rows.append({"mel_path": str(mel_path), "primary_label": pl, "source": "train_audio", "label": label})

    train_meta = pd.DataFrame(rows)

    # Soundscape eval
    eval_target = int(sr * CFG["infer_duration"])
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    cache_sub2 = own_cache / "soundscapes"
    cache_sub2.mkdir(exist_ok=True)
    eval_rows = []
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
        chunk = wav[start_sample:start_sample + eval_target]
        if len(chunk) < eval_target:
            chunk = pad_or_crop(chunk, eval_target)
        mel = compute_mel(chunk, sr)
        mel_id = f"sc_{idx}"
        mel_path = cache_sub2 / f"{mel_id}.npy"
        if not mel_path.exists():
            np.save(mel_path, mel)
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in SPECIES2IDX:
                label[SPECIES2IDX[sp]] = 1.0
        eval_rows.append({"mel_path": str(mel_path), "primary_label": sp, "source": "soundscape", "label": label})

    eval_meta = pd.DataFrame(eval_rows)
    return train_meta, eval_meta


# ====================================================================
# Dataset & Model
# ====================================================================
class MelDataset(Dataset):
    def __init__(self, df, img_size=224, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.img_size = img_size

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
    def __init__(self, backbone_name, num_classes, pretrained=True, img_size=224):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=1, num_classes=0, global_pool="",
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, img_size, img_size)
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


def train_backbone(backbone_key, train_meta, eval_meta):
    """Train a specific backbone."""
    bcfg = BACKBONE_CFGS[backbone_key]
    backbone_name = bcfg["name"]
    prefix = bcfg["prefix"]
    batch_size = bcfg["batch_size"]
    img_size = bcfg["img_size"]

    print(f"\n{'#'*60}")
    print(f"# BACKBONE: {backbone_key} ({backbone_name})")
    print(f"# Batch size: {batch_size}, Image size: {img_size}")
    print(f"{'#'*60}")

    eval_ds = MelDataset(eval_meta, img_size=img_size, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False,
                             num_workers=CFG["num_workers"], pin_memory=True)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["primary_label"])):
        if fold not in CFG["train_folds"]:
            continue

        print(f"\n{'='*60}\n{backbone_key} FOLD {fold}\n{'='*60}")
        train_df = train_meta.iloc[train_idx].reset_index(drop=True)
        val_df = train_meta.iloc[val_idx].reset_index(drop=True)

        train_ds = MelDataset(train_df, img_size=img_size, is_train=True)
        val_ds = MelDataset(val_df, img_size=img_size, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                                num_workers=CFG["num_workers"], pin_memory=True)

        model = BirdSEDModel(backbone_name, NUM_CLASSES, pretrained=True, img_size=img_size).to(DEVICE)
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
                ckpt = WEIGHTS_DIR / f"{prefix}_fold{fold}_best.pth"
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch + 1, "val_auc": val_auc,
                            "backbone": backbone_name, "config": CFG}, ckpt)
                print(f"  -> Saved {ckpt.name} (AUC={val_auc:.4f})")

        state = torch.load(WEIGHTS_DIR / f"{prefix}_fold{fold}_best.pth",
                           map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        _, final_ss, ss_n = validate(model, eval_loader)
        print(f"Fold {fold} best -> ss_auc: {final_ss:.4f} ({ss_n} cls)")
        results[f"{backbone_key}_f{fold}"] = {"best_auc": best_auc, "best_epoch": best_epoch, "ss_auc": final_ss}

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    return results


def ensemble_eval(model_configs, eval_meta):
    """Evaluate ensemble of multiple backbone models."""
    print(f"\n{'='*60}\nENSEMBLE EVALUATION\n{'='*60}")

    all_models = []
    for backbone_key, prefix in model_configs:
        bcfg = BACKBONE_CFGS[backbone_key]
        for fold in [0, 1]:
            path = WEIGHTS_DIR / f"{prefix}_fold{fold}_best.pth"
            if path.exists():
                model = BirdSEDModel(bcfg["name"], NUM_CLASSES, pretrained=False,
                                     img_size=bcfg["img_size"]).to(DEVICE)
                state = torch.load(path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                model.eval()
                all_models.append((backbone_key, fold, model))
                print(f"  Loaded {path.name}")

    if len(all_models) < 2:
        print("Not enough models for ensemble")
        return {}

    # Evaluate
    eval_ds = MelDataset(eval_meta, img_size=224, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False,
                             num_workers=CFG["num_workers"], pin_memory=True)

    all_preds_per_model = {}
    all_labels = None

    for bk, fold, model in all_models:
        key = f"{bk}_f{fold}"
        preds_list, labels_list = [], []
        for mels, labels in eval_loader:
            mels = mels.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                with autocast("cuda"):
                    clipwise, _ = model(mels)
            preds_list.append(torch.sigmoid(clipwise).cpu().numpy())
            if all_labels is None:
                labels_list.append(labels.numpy())
        all_preds_per_model[key] = np.concatenate(preds_list)
        if all_labels is None:
            all_labels = np.concatenate(labels_list)

    # Test ensemble combinations
    from itertools import combinations
    results = {}
    model_keys = list(all_preds_per_model.keys())

    for r in range(2, len(model_keys) + 1):
        for combo in combinations(model_keys, r):
            ensemble_pred = np.mean([all_preds_per_model[k] for k in combo], axis=0)
            aucs = []
            for c in range(NUM_CLASSES):
                gt = (all_labels[:, c] > 0.5).astype(int)
                if 0 < gt.sum() < len(gt):
                    try:
                        aucs.append(roc_auc_score(gt, ensemble_pred[:, c]))
                    except ValueError:
                        pass
            auc = np.mean(aucs) if aucs else 0.0
            key = "+".join(combo)
            results[key] = float(auc)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 ensemble combinations:")
    for i, (key, auc) in enumerate(sorted_results[:10]):
        print(f"  {i+1}. {auc:.4f}  {key}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="all", choices=["b2", "convnext", "nfnet", "all"])
    args = parser.parse_args()

    start_time = time.time()

    train_meta, eval_meta = load_or_create_metadata()
    print(f"Train: {len(train_meta)}, Eval: {len(eval_meta)}")

    all_results = {}
    backbones_to_train = list(BACKBONE_CFGS.keys()) if args.backbone == "all" else [args.backbone]

    for bk in backbones_to_train:
        results = train_backbone(bk, train_meta, eval_meta)
        all_results.update(results)

    # Ensemble evaluation
    model_configs = [(bk, BACKBONE_CFGS[bk]["prefix"]) for bk in backbones_to_train]
    # Also include exp11 B0 if available
    exp11_path = WEIGHTS_DIR / "exp11_5s_fold0_best.pth"
    if exp11_path.exists():
        # Add a pseudo entry for B0
        BACKBONE_CFGS["b0"] = {
            "name": "tf_efficientnet_b0.ns_jft_in1k",
            "prefix": "exp11_5s",
            "batch_size": 64,
            "img_size": 224,
        }
        model_configs.append(("b0", "exp11_5s"))

    ensemble_results = ensemble_eval(model_configs, eval_meta)
    all_results["ensemble"] = ensemble_results

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nALL RESULTS\n{'='*60}")
    for k, v in all_results.items():
        if k != "ensemble":
            print(f"  {k}: ss_auc={v['ss_auc']:.4f}")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp13_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)


if __name__ == "__main__":
    main()
