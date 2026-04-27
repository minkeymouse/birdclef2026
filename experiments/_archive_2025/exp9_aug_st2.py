#!/usr/bin/env python3
"""
exp9_aug_st2.py — Enhanced augmentation + 2nd round self-training.

Hypothesis: Stronger augmentation (wave-level background mixing, gaussian noise,
stronger SpecAugment) acts as noise injection for self-training, improving
generalization. Using exp8 (best model) as teacher for 2nd round pseudo-labels
should provide higher-quality supervision than exp3 teacher.

Key changes from exp8:
1. Teacher: exp8 ensemble (not exp3) for pseudo-labels → better quality
2. Augmentation: stronger SpecAugment, random gain, cutmix on spectrograms
3. Pseudo threshold: 0.3 → 0.2 (exp8 teacher is more reliable)
4. Init from exp8 weights (not exp3)
5. Focal loss retained from exp8
"""
import os
import sys
import gc
import json
import time
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import timm
import librosa
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp9_outputs"
CACHE_DIR = OUT_DIR / "mel_cache"
EXP3_DIR = ROOT / "experiments" / "exp3_outputs"
EXP7_DIR = ROOT / "experiments" / "exp7_outputs"

for d in [WEIGHTS_DIR, OUT_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    "pretrained": False,
    "lr": 5e-6,              # even lower LR for 2nd round fine-tuning
    "weight_decay": 1e-2,
    "epochs": 5,
    "batch_size": 64,
    "num_workers": 4,
    "n_folds": 5,
    "train_folds": [0, 1],
    "mixup_alpha": 0.5,      # slightly stronger mixup
    # Loss params (from exp8)
    "focal_gamma": 2.0,
    "label_smoothing": 0.05,
    # Self-training round 2
    "pseudo_threshold": 0.2,  # lower threshold (exp8 teacher is better)
    "pseudo_power": 0.7,
    "pseudo_weight": 0.3,
    "ss_train_ratio": 0.5,
    # Enhanced augmentation
    "spec_mask_freq_num": 3,      # number of freq masks (was 1-2)
    "spec_mask_freq_width": 30,   # max freq mask width (was 25)
    "spec_mask_time_num": 3,      # number of time masks (was 1-2)
    "spec_mask_time_width": 40,   # max time mask width (was 25)
    "random_gain_range": 6.0,     # dB range for random gain
    "cutmix_prob": 0.3,           # probability of cutmix augmentation
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
# Focal Loss (from exp8)
# ====================================================================
def focal_bce_loss(logits, targets, gamma=2.0, label_smoothing=0.0, reduction="none"):
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    loss = focal_weight * bce
    if reduction == "mean":
        return loss.mean()
    return loss


# ====================================================================
# Model (same architecture)
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
# Mel & Data utils
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


def pad_or_crop(wav, target_len):
    if len(wav) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(wav) < target_len:
        wav = np.tile(wav, int(np.ceil(target_len / len(wav))))
    return wav[:target_len]


def _parse_time(t):
    parts = t.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def prepare_labeled_soundscapes():
    sr = CFG["sr"]
    target_len = int(sr * CFG["infer_duration"])
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    cache_sub = CACHE_DIR / "soundscapes"
    cache_sub.mkdir(exist_ok=True)

    audio_cache = {}
    for filename in labels_df["filename"].unique():
        filepath = DATA / "train_soundscapes" / filename
        if filepath.exists():
            try:
                wav, _ = librosa.load(filepath, sr=sr)
                audio_cache[filename] = wav
            except Exception:
                continue

    rows = []
    for idx, row in labels_df.iterrows():
        filename = row["filename"]
        if filename not in audio_cache:
            continue
        wav = audio_cache[filename]
        start_sec = _parse_time(str(row["start"]))
        chunk = wav[int(start_sec * sr):int(start_sec * sr) + target_len]
        if len(chunk) < target_len:
            chunk = pad_or_crop(chunk, target_len)

        mel_path = cache_sub / f"sc_{idx}.npy"
        if not mel_path.exists():
            np.save(mel_path, compute_mel(chunk, sr))

        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in SPECIES2IDX:
                label[SPECIES2IDX[sp]] = 1.0

        rows.append({"mel_path": str(mel_path), "primary_label": sp,
                      "source": "soundscape", "label": label, "weight": 1.0, "filename": filename})

    ss_df = pd.DataFrame(rows)
    unique_files = sorted(ss_df["filename"].unique())
    np.random.seed(CFG["seed"])
    np.random.shuffle(unique_files)
    split_idx = int(len(unique_files) * CFG["ss_train_ratio"])
    train_files = set(unique_files[:split_idx])

    ss_train = ss_df[ss_df["filename"].isin(train_files)].drop(columns=["filename"]).reset_index(drop=True)
    ss_eval = ss_df[~ss_df["filename"].isin(train_files)].reset_index(drop=True)
    print(f"Labeled SS: {len(ss_train)} train, {len(ss_eval)} eval")
    return ss_train, ss_eval


def generate_pseudo_labels_from_cache(teacher_models):
    """Generate pseudo-labels using exp8 teacher on exp7 cached mel files."""
    # Reuse exp7 pseudo mel cache
    exp7_pseudo = EXP7_DIR / "mel_cache" / "pseudo"
    cache_sub = CACHE_DIR / "pseudo"
    if exp7_pseudo.exists() and not cache_sub.exists():
        cache_sub.symlink_to(exp7_pseudo.resolve())
        print(f"Symlinked pseudo cache from exp7")

    mel_files = sorted(cache_sub.glob("ps_*.npy"))
    print(f"Cached pseudo mels: {len(mel_files)}")
    if len(mel_files) == 0:
        return pd.DataFrame()

    file_groups = defaultdict(list)
    for mp in mel_files:
        parts = mp.stem.split("_")
        src_name = "_".join(parts[1:-1])
        file_groups[src_name].append(mp)

    rows = []
    batch_size = 128
    stats = {"total": 0, "above": 0}
    all_mels_flat = []
    all_src_names = []
    all_paths_flat = []

    for src_name, paths in file_groups.items():
        first_path = sorted(paths)[0]
        all_mels_flat.append(first_path)
        all_src_names.append(src_name)
        all_paths_flat.append(paths)

    print(f"Source files: {len(all_src_names)}, batch inference with exp8 teacher...")
    all_preds = []
    for i in tqdm(range(0, len(all_mels_flat), batch_size), desc="Pseudo batch inference"):
        batch_paths = all_mels_flat[i:i + batch_size]
        mels = []
        for mp in batch_paths:
            mel = np.load(mp)
            mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
            mel_t = F.interpolate(mel_t.unsqueeze(0), size=(CFG["img_size"], CFG["img_size"]),
                                  mode="bilinear", align_corners=False).squeeze(0)
            mels.append(mel_t)
        batch = torch.stack(mels).to(DEVICE)
        preds_batch = []
        with torch.no_grad():
            for m in teacher_models:
                with autocast("cuda"):
                    c, _ = m(batch)
                preds_batch.append(torch.sigmoid(c).cpu().numpy())
        avg = np.mean(preds_batch, axis=0)
        all_preds.append(avg)

    all_preds = np.concatenate(all_preds, axis=0)

    for idx, (src_name, paths) in enumerate(zip(all_src_names, all_paths_flat)):
        avg = all_preds[idx]
        stats["total"] += 1
        if avg.max() < CFG["pseudo_threshold"]:
            continue
        stats["above"] += 1
        pl = np.power(avg, CFG["pseudo_power"])
        pl[pl < 0.05] = 0.0
        for mp in paths:
            rows.append({"mel_path": str(mp), "primary_label": SPECIES_LIST[np.argmax(pl)],
                          "source": "pseudo", "label": pl.copy(), "weight": CFG["pseudo_weight"]})

    print(f"Pseudo: {stats['above']}/{stats['total']} ({100*stats['above']/(stats['total']+1e-8):.1f}%)")
    return pd.DataFrame(rows)


# ====================================================================
# Enhanced Dataset with stronger augmentation
# ====================================================================
class AugDataset(Dataset):
    """Dataset with enhanced augmentation for self-training."""
    def __init__(self, df, is_train=True, all_mels=None):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        # Store all mel paths for cutmix source sampling
        self.all_mel_paths = df["mel_path"].tolist() if all_mels is None else all_mels

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
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=(CFG["img_size"], CFG["img_size"]),
                              mode="bilinear", align_corners=False).squeeze(0)

        if self.is_train:
            mel_t = self._enhanced_augment(mel_t)

        return mel_t, torch.tensor(label, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)

    def _enhanced_augment(self, spec):
        _, h, w = spec.shape

        # 1. Random gain (simulate volume variation)
        gain_db = random.uniform(-CFG["random_gain_range"], CFG["random_gain_range"])
        gain_factor = 10 ** (gain_db / 20)
        spec = spec * gain_factor
        spec = spec.clamp(0, 1)

        # 2. Strong SpecAugment - frequency masking
        for _ in range(random.randint(1, CFG["spec_mask_freq_num"])):
            if random.random() < 0.7:
                mask_h = random.randint(5, min(CFG["spec_mask_freq_width"], h // 3))
                start = random.randint(0, h - mask_h)
                spec[:, start:start + mask_h, :] = 0

        # 3. Strong SpecAugment - time masking
        for _ in range(random.randint(1, CFG["spec_mask_time_num"])):
            if random.random() < 0.7:
                mask_w = random.randint(5, min(CFG["spec_mask_time_width"], w // 3))
                start = random.randint(0, w - mask_w)
                spec[:, :, start:start + mask_w] = 0

        # 4. Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(spec) * random.uniform(0.005, 0.02)
            spec = (spec + noise).clamp(0, 1)

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
            loss_clip = focal_bce_loss(clipwise, labels, gamma=CFG["focal_gamma"],
                                        label_smoothing=CFG["label_smoothing"])
            loss_frame = focal_bce_loss(framewise_max, labels, gamma=CFG["focal_gamma"],
                                         label_smoothing=CFG["label_smoothing"])
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
    all_preds, all_labels = [], []
    for batch in loader:
        mels = batch[0].to(DEVICE, non_blocking=True)
        labels = batch[1].to(DEVICE, non_blocking=True)
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

    # Load train_audio cache from exp3
    print("Loading exp3 train cache...")
    train_meta = pd.read_parquet(EXP3_DIR / "train_meta.parquet")
    train_meta["label"] = train_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    train_meta["weight"] = 1.0

    # Labeled soundscapes
    ss_train, ss_eval = prepare_labeled_soundscapes()

    # Use exp8 as teacher (better than exp3)
    print("Loading exp8 teacher models...")
    teacher_models = []
    for fold in CFG["train_folds"]:
        ckpt = WEIGHTS_DIR / f"exp8_focal_fold{fold}_best.pth"
        if ckpt.exists():
            m = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=False)
            state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            m.load_state_dict(state["model_state_dict"])
            m.eval().to(DEVICE)
            teacher_models.append(m)
            print(f"  Loaded {ckpt.name}")

    if not teacher_models:
        print("ERROR: No exp8 weights found. Run exp8 first.")
        return

    pseudo_meta = generate_pseudo_labels_from_cache(teacher_models)
    del teacher_models
    torch.cuda.empty_cache()
    gc.collect()

    combined = pd.concat([train_meta, ss_train, pseudo_meta], ignore_index=True)
    print(f"Combined: {len(train_meta)} audio + {len(ss_train)} ss + {len(pseudo_meta)} pseudo = {len(combined)}")

    # Eval loaders
    full_eval = pd.read_parquet(EXP3_DIR / "eval_meta.parquet")
    full_eval["label"] = full_eval["label"].apply(lambda x: np.array(x, dtype=np.float32))
    full_eval["weight"] = 1.0
    full_eval_loader = DataLoader(AugDataset(full_eval, is_train=False), batch_size=128,
                                   shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)
    eval_loader = DataLoader(AugDataset(ss_eval, is_train=False), batch_size=128,
                              shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

    results = {}
    for fold in CFG["train_folds"]:
        print(f"\n{'='*60}")
        print(f"FOLD {fold} (2nd round ST, enhanced aug)")
        print(f"{'='*60}")

        train_ds = AugDataset(combined, is_train=True)
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)

        model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=False).to(DEVICE)

        # Init from exp8 (best model so far)
        ckpt = WEIGHTS_DIR / f"exp8_focal_fold{fold}_best.pth"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            print(f"Init from {ckpt.name}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        total_steps = len(train_loader) * CFG["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
        scaler = GradScaler("cuda")

        best_ss = 0.0
        best_epoch = -1

        for epoch in range(CFG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
            ss_auc, ss_n = validate(model, eval_loader)
            full_ss, full_n = validate(model, full_eval_loader)

            print(f"E{epoch+1}/{CFG['epochs']} — loss: {train_loss:.4f}, "
                  f"ss_auc: {ss_auc:.4f}({ss_n}), full_ss: {full_ss:.4f}({full_n})")

            if ss_auc > best_ss:
                best_ss = ss_auc
                best_epoch = epoch + 1
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch + 1, "ss_auc": ss_auc, "config": CFG},
                           WEIGHTS_DIR / f"exp9_aug_fold{fold}_best.pth")
                print(f"  -> Saved exp9_aug_fold{fold}_best.pth (ss_auc={ss_auc:.4f})")

        # Final eval with best checkpoint
        state = torch.load(WEIGHTS_DIR / f"exp9_aug_fold{fold}_best.pth",
                           map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        final_ss, _ = validate(model, eval_loader)
        final_full, _ = validate(model, full_eval_loader)
        print(f"Fold {fold} best -> ss_auc: {final_ss:.4f}, full_ss: {final_full:.4f}")

        results[fold] = {"best_ss": best_ss, "best_epoch": best_epoch,
                         "final_ss": final_ss, "final_full": final_full}

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    for fold, res in results.items():
        print(f"Fold {fold}: ss_auc={res['best_ss']:.4f}@E{res['best_epoch']}, full_ss={res['final_full']:.4f}")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp9_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    with open(OUT_DIR / "exp9_config.json", "w") as f:
        json.dump(CFG, f, indent=2)


if __name__ == "__main__":
    main()
