#!/usr/bin/env python3
"""
exp15_proper_st.py — Proper self-training following 2025 winner recipe.

2025 1st place self-training recipe (we previously did it wrong):
  1. mixup_ratio=1 — ALWAYS mix train_audio with pseudo-labeled soundscape
  2. Stochastic depth (drop_path=0.15) as model-level noise
  3. Power transform on pseudo-labels (reduce noise)
  4. Sampling weights = sum of pseudo-labels per chunk
  5. Random padding (shorter audio placed at random position)
  6. Single round only (no iterative without LB validation)

Uses exp11 (5s baseline) as teacher. Trains from exp11 weights.
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import timm
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp15_outputs"
CACHE_DIR = OUT_DIR / "mel_cache"

for d in [WEIGHTS_DIR, OUT_DIR, CACHE_DIR]:
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
    "pretrained": False,  # Init from exp11 checkpoint
    "lr": 1e-5,           # Fine-tuning LR
    "weight_decay": 1e-2,
    "epochs": 5,           # Conservative
    "batch_size": 64,
    "num_workers": 2,
    "n_folds": 5,
    "train_folds": [0, 1],
    "mixup_alpha": 0.4,
    "max_chunks": 4,
    # Self-training params (2025 winner recipe)
    "pseudo_threshold": 0.3,
    "pseudo_power": 0.7,       # Power transform on pseudo-labels
    "mixup_ratio": 1.0,        # ALWAYS mix with pseudo (key difference!)
    "drop_path_rate": 0.15,    # Stochastic depth
    "teacher_prefix": "exp11_5s",  # Use exp11 as teacher
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
# Audio utils
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


def pad_or_crop(wav, target_len, random_position=False):
    """Pad or crop audio. If random_position, place shorter audio at random position."""
    if len(wav) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(wav) < target_len:
        if random_position:
            # Random padding: place audio at random position within target length
            pad_total = target_len - len(wav)
            pad_left = random.randint(0, pad_total)
            result = np.zeros(target_len, dtype=np.float32)
            result[pad_left:pad_left + len(wav)] = wav
            return result
        else:
            reps = int(np.ceil(target_len / len(wav)))
            wav = np.tile(wav, reps)
    if len(wav) > target_len:
        start = random.randint(0, len(wav) - target_len)
        wav = wav[start:start + target_len]
    return wav[:target_len]


def _parse_time(t):
    parts = t.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


# ====================================================================
# Model with stochastic depth
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
    def __init__(self, backbone_name, num_classes, pretrained=True, drop_path_rate=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=1, num_classes=0, global_pool="",
            drop_path_rate=drop_path_rate,  # Stochastic depth!
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
# Pseudo-label generation
# ====================================================================
def generate_pseudo_labels():
    """Generate pseudo-labels using exp11 teacher ensemble."""
    print("\n=== Generating pseudo-labels with exp11 teacher ===")

    # Load teacher models
    teacher_models = []
    for fold in [0, 1]:
        path = WEIGHTS_DIR / f"{CFG['teacher_prefix']}_fold{fold}_best.pth"
        if not path.exists():
            print(f"WARNING: Teacher {path.name} not found!")
            continue
        model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=False, drop_path_rate=0.0).to(DEVICE)
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        teacher_models.append(model)
        print(f"  Loaded teacher: {path.name}")

    if not teacher_models:
        print("ERROR: No teacher models found!")
        return pd.DataFrame()

    sr = CFG["sr"]
    target_len = int(sr * CFG["infer_duration"])

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

        # Per-file average prediction
        file_pred = np.mean(file_preds, axis=0)
        stats["total"] += 1

        if file_pred.max() < CFG["pseudo_threshold"]:
            continue

        stats["above_threshold"] += 1

        # Apply power transform to reduce noise
        power = CFG["pseudo_power"]

        # Create pseudo-labeled training samples per 5s segment
        for seg in range(n_5s):
            seg_pred = file_preds[seg]

            # Power transform on pseudo-labels
            pseudo_label = np.power(np.clip(seg_pred, 0, 1), power)

            # Zero out very low predictions (noise floor)
            pseudo_label[pseudo_label < 0.1] = 0.0

            # Save mel
            chunk_5s = wav[seg * target_len:(seg + 1) * target_len]
            if len(chunk_5s) < target_len:
                chunk_5s = np.pad(chunk_5s, (0, target_len - len(chunk_5s)))

            mel = compute_mel(chunk_5s, sr)
            mel_id = f"pseudo_{fpath.stem}_{seg}"
            mel_path = cache_sub / f"{mel_id}.npy"
            if not mel_path.exists():
                np.save(mel_path, mel)

            # Sampling weight = sum of pseudo-labels (2025 winner recipe)
            sample_weight = float(pseudo_label.sum())

            pseudo_rows.append({
                "mel_path": str(mel_path),
                "primary_label": "pseudo",
                "source": "pseudo",
                "label": pseudo_label.astype(np.float32),
                "weight": sample_weight,
            })

    pass_rate = stats["above_threshold"] / max(1, stats["total"])
    print(f"Pseudo-labeling: {stats['above_threshold']}/{stats['total']} "
          f"({pass_rate*100:.1f}%) above threshold {CFG['pseudo_threshold']}")
    print(f"Pseudo training samples: {len(pseudo_rows)}")

    # Clean up teacher models
    for m in teacher_models:
        del m
    torch.cuda.empty_cache()

    return pd.DataFrame(pseudo_rows)


# ====================================================================
# Dataset with forced mixup (2025 winner recipe)
# ====================================================================
class MixupSTDataset(Dataset):
    """
    Self-training dataset with forced mixup.
    Each training sample is ALWAYS mixed with a pseudo-labeled sample.
    This is the key difference from our previous self-training.
    """

    def __init__(self, train_df, pseudo_df, is_train=True):
        self.train_df = train_df.reset_index(drop=True)
        self.pseudo_df = pseudo_df.reset_index(drop=True) if len(pseudo_df) > 0 else None
        self.is_train = is_train
        self.img_size = CFG["img_size"]

        # Weighted sampling indices for pseudo data
        if self.pseudo_df is not None and len(self.pseudo_df) > 0:
            weights = self.pseudo_df["weight"].values
            weights = weights / weights.sum()
            self.pseudo_weights = weights
        else:
            self.pseudo_weights = None

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]
        mel = np.load(row["mel_path"])
        label = row["label"]
        if isinstance(label, str):
            label = np.array(json.loads(label), dtype=np.float32)

        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)

        # FORCED MIXUP with pseudo-labeled sample (mixup_ratio=1.0)
        if self.is_train and self.pseudo_df is not None and random.random() < CFG["mixup_ratio"]:
            # Sample pseudo with probability proportional to label sum
            pseudo_idx = np.random.choice(len(self.pseudo_df), p=self.pseudo_weights)
            pseudo_row = self.pseudo_df.iloc[pseudo_idx]
            pseudo_mel = np.load(pseudo_row["mel_path"])
            pseudo_label = pseudo_row["label"]
            if isinstance(pseudo_label, str):
                pseudo_label = np.array(json.loads(pseudo_label), dtype=np.float32)

            pseudo_mel_t = torch.tensor(pseudo_mel, dtype=torch.float32).unsqueeze(0)
            pseudo_mel_t = F.interpolate(pseudo_mel_t.unsqueeze(0), size=(self.img_size, self.img_size),
                                         mode="bilinear", align_corners=False).squeeze(0)

            # Mixup: λ from Beta distribution
            lam = np.random.beta(CFG["mixup_alpha"], CFG["mixup_alpha"])
            mel_t = lam * mel_t + (1 - lam) * pseudo_mel_t
            # Labels: clip(train + pseudo, 0, 1) — following 2025 2nd place recipe
            label = np.clip(label + pseudo_label, 0, 1)

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


class MelDataset(Dataset):
    def __init__(self, df, is_train=False):
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
# Training
# ====================================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Train E{epoch+1}")
    for mels, labels in pbar:
        mels = mels.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

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

    # ── Check for exp11 weights ──────────────────────────────
    teacher_found = False
    for fold in [0, 1]:
        if (WEIGHTS_DIR / f"{CFG['teacher_prefix']}_fold{fold}_best.pth").exists():
            teacher_found = True
    if not teacher_found:
        print("ERROR: exp11 teacher weights not found. Run exp11 first.")
        sys.exit(1)

    # ── Load or create training data ─────────────────────────
    exp11_meta = ROOT / "experiments" / "exp11_outputs" / "train_meta.parquet"
    exp11_eval = ROOT / "experiments" / "exp11_outputs" / "eval_meta.parquet"

    if exp11_meta.exists():
        print("Loading exp11 training metadata...")
        train_meta = pd.read_parquet(exp11_meta)
        train_meta["label"] = train_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    else:
        print("ERROR: exp11 train_meta.parquet not found. Run exp11 first.")
        sys.exit(1)

    if exp11_eval.exists():
        eval_meta = pd.read_parquet(exp11_eval)
        eval_meta["label"] = eval_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    else:
        print("ERROR: exp11 eval_meta.parquet not found.")
        sys.exit(1)

    # ── Generate pseudo-labels ───────────────────────────────
    pseudo_cache = OUT_DIR / "pseudo_meta.parquet"
    if pseudo_cache.exists():
        print("Loading cached pseudo-labels...")
        pseudo_meta = pd.read_parquet(pseudo_cache)
        pseudo_meta["label"] = pseudo_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    else:
        pseudo_meta = generate_pseudo_labels()
        if len(pseudo_meta) > 0:
            save_df = pseudo_meta.copy()
            save_df["label"] = save_df["label"].apply(lambda x: x.tolist())
            save_df.to_parquet(pseudo_cache)

    print(f"Train: {len(train_meta)}, Pseudo: {len(pseudo_meta)}, Eval: {len(eval_meta)}")

    if len(pseudo_meta) == 0:
        print("No pseudo-labels generated. Aborting.")
        sys.exit(1)

    # ── Training ─────────────────────────────────────────────
    eval_ds = MelDataset(eval_meta)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False,
                             num_workers=CFG["num_workers"], pin_memory=True)

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["primary_label"])):
        if fold not in CFG["train_folds"]:
            continue

        print(f"\n{'='*60}\nFOLD {fold} (proper ST: mixup_ratio={CFG['mixup_ratio']}, "
              f"drop_path={CFG['drop_path_rate']})")
        print(f"{'='*60}")

        fold_train = train_meta.iloc[train_idx].reset_index(drop=True)

        # Dataset with forced pseudo mixup
        train_ds = MixupSTDataset(fold_train, pseudo_meta, is_train=True)
        val_ds = MelDataset(train_meta.iloc[val_idx].reset_index(drop=True))

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2, shuffle=False,
                                num_workers=CFG["num_workers"], pin_memory=True)

        # Init from exp11 weights with stochastic depth
        model = BirdSEDModel(CFG["backbone"], NUM_CLASSES, pretrained=False,
                             drop_path_rate=CFG["drop_path_rate"]).to(DEVICE)
        init_path = WEIGHTS_DIR / f"{CFG['teacher_prefix']}_fold{fold}_best.pth"
        if init_path.exists():
            state = torch.load(init_path, map_location=DEVICE, weights_only=False)
            # Load with strict=False to handle drop_path parameter differences
            model.load_state_dict(state["model_state_dict"], strict=False)
            print(f"  Initialized from {init_path.name}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        total_steps = len(train_loader) * CFG["epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
        scaler = GradScaler("cuda")

        best_ss = 0.0
        best_epoch = -1

        for epoch in range(CFG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch)
            val_loss, val_auc, n_scored = validate(model, val_loader)
            _, ss_auc, ss_n = validate(model, eval_loader)

            print(f"E{epoch+1}/{CFG['epochs']} — loss: {train_loss:.4f}, "
                  f"val_auc: {val_auc:.4f}({n_scored}), ss_auc: {ss_auc:.4f}({ss_n})")

            # Select by ss_auc (self-training target is soundscape performance)
            if ss_auc > best_ss:
                best_ss = ss_auc
                best_epoch = epoch + 1
                ckpt = WEIGHTS_DIR / f"exp15_st_fold{fold}_best.pth"
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch + 1, "ss_auc": ss_auc,
                            "val_auc": val_auc, "config": CFG}, ckpt)
                print(f"  -> Saved {ckpt.name} (ss_auc={ss_auc:.4f})")

        # Final eval
        state = torch.load(WEIGHTS_DIR / f"exp15_st_fold{fold}_best.pth",
                           map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        _, final_ss, ss_n = validate(model, eval_loader)
        print(f"Fold {fold} best -> ss_auc: {final_ss:.4f} ({ss_n} cls)")

        results[fold] = {"best_ss": float(best_ss), "best_epoch": best_epoch, "final_ss": float(final_ss)}

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    # ── Report ───────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nRESULTS (exp15: proper self-training)")
    print(f"{'='*60}")
    for fold, res in results.items():
        print(f"Fold {fold}: ss_auc={res['final_ss']:.4f}@E{res['best_epoch']}")

    # Compare with exp11
    exp11_results_path = ROOT / "experiments" / "exp11_outputs" / "exp11_results.json"
    if exp11_results_path.exists():
        with open(exp11_results_path) as f:
            exp11_results = json.load(f)
        print("\nΔ vs exp11:")
        for fold, res in results.items():
            exp11_ss = exp11_results.get(str(fold), {}).get("ss_auc", 0)
            delta = res["final_ss"] - exp11_ss
            print(f"  Fold {fold}: {delta:+.4f}")

    print(f"\nWARNING: Self-training improved local metric in exp7-9 but DEGRADED LB.")
    print(f"This result should be validated on LB before trusting.")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp15_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    with open(OUT_DIR / "exp15_config.json", "w") as f:
        json.dump(CFG, f, indent=2)


if __name__ == "__main__":
    main()
