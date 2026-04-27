#!/usr/bin/env python3
"""
exp7_st_v2.py — Self-training v2: Domain adaptation via soundscape inclusion.

Key insight from exp3-6: domain shift (train_audio → soundscapes) is the bottleneck,
not model capacity. Attack the domain gap directly.

Changes from exp5 (self-training v1):
  1. Include labeled soundscapes IN training (50/50 split for train/eval)
  2. Lower pseudo-label threshold: 0.5 → 0.3
  3. Much lower LR: 5e-4 → 1e-5 (fine-tuning, not retraining)
  4. Fewer epochs: 15 → 5
  5. Pseudo sample weight 0.5 → 0.3 (less trust in noisy labels)
  6. Train from exp3 checkpoint (best ss_auc)
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
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp7_outputs"
CACHE_DIR = OUT_DIR / "mel_cache"
EXP3_DIR = ROOT / "experiments" / "exp3_outputs"

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
    "pretrained": False,  # init from exp3 checkpoint
    "lr": 1e-5,
    "weight_decay": 1e-2,
    "epochs": 5,
    "batch_size": 64,
    "num_workers": 4,
    "n_folds": 5,
    "train_folds": [0, 1],
    "mixup_alpha": 0.4,
    # Self-training v2 params
    "pseudo_threshold": 0.3,
    "pseudo_power": 0.7,
    "pseudo_weight": 0.3,
    "ss_train_ratio": 0.5,  # fraction of labeled soundscapes for training
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
# Model
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
# Mel & Audio utils
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
# Data preparation
# ====================================================================
def prepare_labeled_soundscapes():
    """Prepare labeled soundscape mels, split into train/eval."""
    sr = CFG["sr"]
    target_len = int(sr * CFG["infer_duration"])
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    cache_sub = CACHE_DIR / "soundscapes"
    cache_sub.mkdir(exist_ok=True)

    audio_cache = {}
    for filename in tqdm(labels_df["filename"].unique(), desc="Loading soundscapes"):
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
        start_sample = int(start_sec * sr)
        chunk = wav[start_sample:start_sample + target_len]
        if len(chunk) < target_len:
            chunk = pad_or_crop(chunk, target_len)

        mel_id = f"sc_{idx}"
        mel_path = cache_sub / f"{mel_id}.npy"
        if not mel_path.exists():
            mel = compute_mel(chunk, sr)
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
            "filename": filename,
        })

    ss_df = pd.DataFrame(rows)

    # Split by filename (keep segments from same file together)
    unique_files = sorted(ss_df["filename"].unique())
    np.random.seed(CFG["seed"])
    np.random.shuffle(unique_files)
    split_idx = int(len(unique_files) * CFG["ss_train_ratio"])
    train_files = set(unique_files[:split_idx])
    eval_files = set(unique_files[split_idx:])

    ss_train = ss_df[ss_df["filename"].isin(train_files)].reset_index(drop=True)
    ss_eval = ss_df[ss_df["filename"].isin(eval_files)].reset_index(drop=True)

    print(f"Labeled soundscapes: {len(ss_train)} train ({len(train_files)} files), "
          f"{len(ss_eval)} eval ({len(eval_files)} files)")

    return ss_train, ss_eval


def generate_pseudo_labels(teacher_models):
    """Generate pseudo-labels for unlabeled soundscapes."""
    sr = CFG["sr"]
    target_len = int(sr * CFG["infer_duration"])
    train_target = int(sr * CFG["train_duration"])

    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    labeled_files = set(labels_df["filename"].unique())
    all_ss = sorted(DATA / "train_soundscapes" / f
                    for f in os.listdir(DATA / "train_soundscapes")
                    if f.endswith(".ogg"))
    unlabeled = [f for f in all_ss if f.name not in labeled_files]
    print(f"Unlabeled soundscapes: {len(unlabeled)}")

    cache_sub = CACHE_DIR / "pseudo"
    cache_sub.mkdir(exist_ok=True)
    pseudo_rows = []
    stats = {"total": 0, "above_threshold": 0}

    for fpath in tqdm(unlabeled, desc="Pseudo-labeling"):
        try:
            wav, _ = librosa.load(fpath, sr=sr)
        except Exception:
            continue
        if len(wav) == 0:
            continue

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

            file_preds.append(np.mean(seg_preds, axis=0))

        file_pred = np.mean(file_preds, axis=0)
        stats["total"] += 1

        if file_pred.max() < CFG["pseudo_threshold"]:
            continue

        stats["above_threshold"] += 1
        pseudo_label = np.power(file_pred, CFG["pseudo_power"])
        pseudo_label[pseudo_label < 0.05] = 0.0

        # Create 10s chunks
        for c in range(max(1, len(wav) // train_target)):
            mel_id = f"ps_{fpath.stem}_{c}"
            mel_path = cache_sub / f"{mel_id}.npy"

            if not mel_path.exists():
                start = c * train_target
                chunk = pad_or_crop(wav[start:], train_target)
                mel = compute_mel(chunk, sr)
                np.save(mel_path, mel)

            pseudo_rows.append({
                "mel_path": str(mel_path),
                "primary_label": SPECIES_LIST[np.argmax(pseudo_label)],
                "source": "pseudo",
                "label": pseudo_label.copy(),
                "weight": CFG["pseudo_weight"],
            })

    pct = 100 * stats["above_threshold"] / (stats["total"] + 1e-8)
    print(f"Pseudo-label stats: {stats['above_threshold']}/{stats['total']} "
          f"above threshold ({pct:.1f}%)")
    print(f"Pseudo training samples: {len(pseudo_rows)}")

    return pd.DataFrame(pseudo_rows)


# ====================================================================
# Dataset
# ====================================================================
class STDataset(Dataset):
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

    # ── Load train_audio data (reuse exp3 cache) ──────────────
    meta_path = EXP3_DIR / "train_meta.parquet"
    if not meta_path.exists():
        print("ERROR: exp3 mel cache not found.")
        return

    print("Loading exp3 train_audio cache...")
    train_meta = pd.read_parquet(meta_path)
    train_meta["label"] = train_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    train_meta["weight"] = 1.0
    print(f"Train audio: {len(train_meta)}")

    # ── Prepare labeled soundscapes (train/eval split) ────────
    print("\nPreparing labeled soundscapes...")
    ss_train, ss_eval = prepare_labeled_soundscapes()

    # ── Load teacher and generate pseudo-labels ───────────────
    print("\nLoading exp3 teacher models...")
    teacher_models = []
    for fold in CFG["train_folds"]:
        ckpt = WEIGHTS_DIR / f"exp3_sed_fold{fold}_best.pth"
        if ckpt.exists():
            model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=False)
            state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            model.eval().to(DEVICE)
            teacher_models.append(model)
            print(f"  Loaded {ckpt.name}")

    print(f"\nGenerating pseudo-labels (threshold={CFG['pseudo_threshold']})...")
    pseudo_meta = generate_pseudo_labels(teacher_models)

    del teacher_models
    torch.cuda.empty_cache()
    gc.collect()

    # ── Combine: train_audio + labeled_ss_train + pseudo ──────
    # Remove 'filename' column from ss_train if present
    ss_train_clean = ss_train.drop(columns=["filename"], errors="ignore")
    combined = pd.concat([train_meta, ss_train_clean, pseudo_meta], ignore_index=True)

    n_audio = len(train_meta)
    n_ss = len(ss_train_clean)
    n_pseudo = len(pseudo_meta)
    print(f"\nCombined training: {n_audio} audio + {n_ss} soundscape + {n_pseudo} pseudo = {len(combined)}")

    # ── Eval loader ───────────────────────────────────────────
    eval_ds = STDataset(ss_eval, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False,
                             num_workers=CFG["num_workers"], pin_memory=True)

    # Also keep full soundscape eval (from exp3) for comparison
    exp3_eval_path = EXP3_DIR / "eval_meta.parquet"
    if exp3_eval_path.exists():
        full_eval_meta = pd.read_parquet(exp3_eval_path)
        full_eval_meta["label"] = full_eval_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
        full_eval_meta["weight"] = 1.0
        full_eval_ds = STDataset(full_eval_meta, is_train=False)
        full_eval_loader = DataLoader(full_eval_ds, batch_size=128, shuffle=False,
                                      num_workers=CFG["num_workers"], pin_memory=True)
    else:
        full_eval_loader = None

    # ── Training ──────────────────────────────────────────────
    results = {}

    for fold in CFG["train_folds"]:
        print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")

        # For this experiment, we use ALL combined data for training
        # (no K-fold split on combined — fold only determines init weights)
        train_df = combined.copy()
        print(f"Train: {len(train_df)} (audio={n_audio}, ss={n_ss}, pseudo={n_pseudo})")

        train_ds = STDataset(train_df, is_train=True)
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)

        # Init from exp3
        model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=False).to(DEVICE)
        ckpt = WEIGHTS_DIR / f"exp3_sed_fold{fold}_best.pth"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            print(f"Initialized from {ckpt.name}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        total_steps = len(train_loader) * CFG["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
        scaler = GradScaler("cuda")

        best_ss = 0.0
        best_epoch = -1

        for epoch in range(CFG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)

            # Eval on held-out soundscapes
            _, ss_auc, ss_n = validate(model, eval_loader)

            # Also eval on full soundscape set (for comparison with exp3)
            full_str = ""
            if full_eval_loader:
                _, full_ss, full_n = validate(model, full_eval_loader)
                full_str = f", full_ss: {full_ss:.4f}({full_n})"

            print(f"E{epoch+1}/{CFG['epochs']} — loss: {train_loss:.4f}, "
                  f"ss_auc: {ss_auc:.4f}({ss_n}){full_str}")

            if ss_auc > best_ss:
                best_ss = ss_auc
                best_epoch = epoch + 1
                ckpt_path = WEIGHTS_DIR / f"exp7_st2_fold{fold}_best.pth"
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch + 1, "ss_auc": ss_auc, "config": CFG}, ckpt_path)
                print(f"  -> Saved {ckpt_path.name} (ss_auc={ss_auc:.4f})")

        # Final eval
        state = torch.load(WEIGHTS_DIR / f"exp7_st2_fold{fold}_best.pth",
                           map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        _, final_ss, ss_n = validate(model, eval_loader)
        full_final = ""
        if full_eval_loader:
            _, ff_ss, ff_n = validate(model, full_eval_loader)
            full_final = f", full_ss={ff_ss:.4f}"
        print(f"Fold {fold} best -> ss_auc: {final_ss:.4f} ({ss_n} cls){full_final}")

        results[fold] = {"best_ss_auc": best_ss, "best_epoch": best_epoch, "final_ss": final_ss}

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    for fold, res in results.items():
        print(f"Fold {fold}: ss_auc={res['best_ss_auc']:.4f}@E{res['best_epoch']}")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp7_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    with open(OUT_DIR / "exp7_config.json", "w") as f:
        json.dump(CFG, f, indent=2)


if __name__ == "__main__":
    main()
