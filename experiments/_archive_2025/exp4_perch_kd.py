#!/usr/bin/env python3
"""
exp4_perch_kd.py — Knowledge Distillation from Perch v2 to EfficientNet-B0 SED.

Pipeline:
  Phase 1: Extract Perch v2 soft labels for all train_audio (CPU TF, cached)
  Phase 2: Train EfficientNet student with KD loss:
           loss = α * BCE(pred, hard_label) + (1-α) * BCE(pred, perch_soft_label)
           where perch_soft = sigmoid(perch_logits / temperature)
  Phase 3: Evaluate on held-out labeled soundscapes

Key differences from exp3:
  - Perch v2 soft labels as additional supervision (203/234 species)
  - Non-bird species (31) use hard labels only
  - Temperature-scaled soft targets for smoother knowledge transfer
"""
import os
import sys
import gc
import json
import time
import random
import ast
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
PERCH_DIR = ROOT / "perch_v2"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp4_outputs"
CACHE_DIR = OUT_DIR / "mel_cache"
PERCH_CACHE = OUT_DIR / "perch_cache"

for d in [WEIGHTS_DIR, OUT_DIR, CACHE_DIR, PERCH_CACHE]:
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
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "epochs": 20,
    "batch_size": 64,
    "num_workers": 4,
    "n_folds": 5,
    "train_folds": [0, 1],
    "mixup_alpha": 0.4,
    # KD params
    "kd_alpha": 0.5,          # weight for hard label loss (1-alpha for soft)
    "kd_temperature": 3.0,    # temperature for soft targets
    "perch_sr": 32000,
    "perch_chunk": 160000,    # 5s @ 32kHz for Perch
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

# Species covered by Perch (203 birds) vs not (31 non-birds)
sci_to_pl = dict(zip(taxonomy_df["scientific_name"], taxonomy_df["primary_label"]))
perch_labels = open(PERCH_DIR / "assets" / "labels.csv").read().strip().split("\n")
PERCH_TO_BC = {}  # perch_idx -> birdclef_idx
for pi, pname in enumerate(perch_labels):
    if pname in sci_to_pl and sci_to_pl[pname] in SPECIES2IDX:
        PERCH_TO_BC[pi] = SPECIES2IDX[sci_to_pl[pname]]

PERCH_COVERED = set(PERCH_TO_BC.values())  # birdclef indices covered by Perch
PERCH_MASK = np.zeros(NUM_CLASSES, dtype=np.float32)
for idx in PERCH_COVERED:
    PERCH_MASK[idx] = 1.0

print(f"Classes: {NUM_CLASSES}, Perch-covered: {len(PERCH_COVERED)}, Device: {DEVICE}")


# ====================================================================
# Phase 1: Extract Perch v2 soft labels
# ====================================================================
def extract_perch_labels():
    """Extract Perch v2 logits for all train_audio files. CPU-only TF."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf

    print("Loading Perch v2 model...")
    perch_model = tf.saved_model.load(str(PERCH_DIR))
    infer = perch_model.signatures["serving_default"]

    train_df = pd.read_csv(DATA / "train.csv")
    sr = CFG["perch_sr"]
    chunk_len = CFG["perch_chunk"]
    T = CFG["kd_temperature"]

    skipped = 0
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Perch extraction"):
        cache_path = PERCH_CACHE / f"{idx}.npy"
        if cache_path.exists():
            continue

        filepath = DATA / "train_audio" / row["filename"]
        if not filepath.exists():
            skipped += 1
            continue

        try:
            wav, _ = librosa.load(filepath, sr=sr)
            if len(wav) == 0:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        # Extract Perch logits from multiple 5s chunks, average
        n_chunks = max(1, len(wav) // chunk_len)
        n_chunks = min(n_chunks, 4)  # cap at 4 chunks = 20s

        all_logits = []
        for c in range(n_chunks):
            start = c * chunk_len
            chunk = wav[start:start + chunk_len]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
            chunk = chunk[:chunk_len]

            inp = tf.constant(chunk[None], dtype=tf.float32)
            out = infer(inp)
            logits = out["label"].numpy()[0]  # (14795,)
            all_logits.append(logits)

        avg_logits = np.mean(all_logits, axis=0)

        # Map to BirdCLEF 234 classes
        bc_logits = np.full(NUM_CLASSES, -20.0, dtype=np.float32)
        for pi, bi in PERCH_TO_BC.items():
            bc_logits[bi] = avg_logits[pi]

        # Convert to soft probabilities with temperature
        bc_soft = 1.0 / (1.0 + np.exp(-bc_logits / T))

        np.save(cache_path, bc_soft)

    print(f"Perch extraction done. Skipped: {skipped}")

    # Free TF memory
    del perch_model, infer
    gc.collect()


# ====================================================================
# Phase 2: Mel precompute (reuse from exp3 with minor changes)
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


def precompute_train_audio(train_df):
    """Precompute 10s mel specs + attach Perch soft labels."""
    sr = CFG["sr"]
    target_len = int(sr * CFG["train_duration"])
    cache_sub = CACHE_DIR / "train_audio"
    cache_sub.mkdir(exist_ok=True)
    rows = []

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Precompute train_audio"):
        filepath = DATA / "train_audio" / row["filename"]
        if not filepath.exists():
            continue

        # Check Perch soft labels exist
        perch_path = PERCH_CACHE / f"{idx}.npy"
        has_perch = perch_path.exists()

        try:
            wav, _ = librosa.load(filepath, sr=sr)
            if len(wav) == 0:
                continue
        except Exception:
            continue

        # Hard label
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
                "perch_path": str(perch_path) if has_perch else "",
            })

    return pd.DataFrame(rows)


def _parse_time(t):
    parts = t.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def precompute_soundscape_eval():
    """Precompute 5s mel specs for labeled soundscapes (eval only, deduped)."""
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
            "perch_path": "",
        })

    return pd.DataFrame(rows)


# ====================================================================
# Dataset & Model
# ====================================================================
class KDDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.img_size = CFG["img_size"]
        self.perch_mask = torch.tensor(PERCH_MASK, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"])
        label = row["label"]
        if isinstance(label, str):
            label = np.array(json.loads(label), dtype=np.float32)

        # Load Perch soft label if available
        perch_path = row.get("perch_path", "")
        if perch_path and os.path.exists(perch_path):
            perch_soft = np.load(perch_path)
            has_perch = True
        else:
            perch_soft = np.zeros(NUM_CLASSES, dtype=np.float32)
            has_perch = False

        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)

        if self.is_train:
            mel_t = self._spec_augment(mel_t)

        return (mel_t,
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(perch_soft, dtype=torch.float32),
                torch.tensor(has_perch, dtype=torch.float32))

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
# Training with KD loss
# ====================================================================
def mixup(x, y_hard, y_soft, has_perch, alpha=0.4):
    if alpha <= 0:
        return x, y_hard, y_soft, has_perch
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return (lam * x + (1 - lam) * x[idx],
            lam * y_hard + (1 - lam) * y_hard[idx],
            lam * y_soft + (1 - lam) * y_soft[idx],
            has_perch * has_perch[idx])  # both must have perch


def kd_loss(logits, hard_labels, perch_soft, has_perch, perch_mask, alpha, temperature):
    """
    Combined hard + soft knowledge distillation loss.

    For species covered by Perch: blend BCE(hard) + BCE(soft)
    For species NOT covered by Perch: BCE(hard) only
    """
    # Hard label loss (all species)
    loss_hard = F.binary_cross_entropy_with_logits(logits, hard_labels)

    # Soft label loss (only Perch-covered species, only samples with Perch labels)
    if has_perch.sum() > 0:
        mask = has_perch.bool()  # (B,)
        logits_masked = logits[mask]  # (B', C)
        soft_masked = perch_soft[mask]  # (B', C)

        # Apply Perch species mask: only compute soft loss for covered species
        pm = perch_mask.to(logits.device)  # (C,)

        # Soft BCE loss on Perch-covered species
        loss_soft = F.binary_cross_entropy_with_logits(
            logits_masked * pm,
            soft_masked * pm,
            reduction="none"
        )
        # Weight by Perch mask (zero out non-covered species)
        loss_soft = (loss_soft * pm).sum(dim=1) / pm.sum()
        loss_soft = loss_soft.mean()

        loss = alpha * loss_hard + (1 - alpha) * loss_soft
    else:
        loss = loss_hard

    return loss


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, perch_mask_t):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Train E{epoch+1}")

    for mels, hard_labels, perch_soft, has_perch in pbar:
        mels = mels.to(DEVICE, non_blocking=True)
        hard_labels = hard_labels.to(DEVICE, non_blocking=True)
        perch_soft = perch_soft.to(DEVICE, non_blocking=True)
        has_perch = has_perch.to(DEVICE, non_blocking=True)

        mels, hard_labels, perch_soft, has_perch = mixup(
            mels, hard_labels, perch_soft, has_perch, CFG["mixup_alpha"])

        optimizer.zero_grad()
        with autocast("cuda"):
            clipwise, framewise_max = model(mels)

            loss_clip = kd_loss(clipwise, hard_labels, perch_soft, has_perch,
                                perch_mask_t, CFG["kd_alpha"], CFG["kd_temperature"])
            loss_frame = kd_loss(framewise_max, hard_labels, perch_soft, has_perch,
                                 perch_mask_t, CFG["kd_alpha"], CFG["kd_temperature"])
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

    # ── Phase 1: Perch extraction ─────────────────────────────
    perch_done_flag = PERCH_CACHE / "_done.flag"
    if not perch_done_flag.exists():
        print("=" * 60)
        print("Phase 1: Extracting Perch v2 soft labels...")
        print("=" * 60)
        extract_perch_labels()
        perch_done_flag.touch()
    else:
        print("Perch soft labels already cached.")

    perch_elapsed = (time.time() - start_time) / 60
    print(f"Perch extraction: {perch_elapsed:.1f} min")

    # ── Phase 2: Precompute mels ──────────────────────────────
    meta_path = OUT_DIR / "train_meta.parquet"
    eval_meta_path = OUT_DIR / "eval_meta.parquet"

    if meta_path.exists() and eval_meta_path.exists():
        print("Loading cached metadata...")
        train_meta = pd.read_parquet(meta_path)
        train_meta["label"] = train_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
        eval_meta = pd.read_parquet(eval_meta_path)
        eval_meta["label"] = eval_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    else:
        print("Phase 2: Precomputing mel spectrograms...")
        train_df = pd.read_csv(DATA / "train.csv")
        train_meta = precompute_train_audio(train_df)

        print("Precomputing soundscape eval mels...")
        eval_meta = precompute_soundscape_eval()

        for df, path in [(train_meta, meta_path), (eval_meta, eval_meta_path)]:
            save_df = df.copy()
            save_df["label"] = save_df["label"].apply(lambda x: x.tolist())
            save_df.to_parquet(path)

    print(f"Train: {len(train_meta)}, Eval: {len(eval_meta)}")

    # Count how many have Perch labels
    n_perch = (train_meta["perch_path"] != "").sum()
    print(f"Samples with Perch soft labels: {n_perch}/{len(train_meta)}")

    # ── Phase 3: Training ────────────────────────────────────
    perch_mask_t = torch.tensor(PERCH_MASK, dtype=torch.float32).to(DEVICE)

    eval_ds = KDDataset(eval_meta, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False,
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

        train_ds = KDDataset(train_df, is_train=True)
        val_ds = KDDataset(val_df, is_train=False)

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
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                         scaler, epoch, perch_mask_t)
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
                ckpt = WEIGHTS_DIR / f"exp4_kd_fold{fold}_best.pth"
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": epoch + 1, "val_auc": val_auc, "config": CFG}, ckpt)
                print(f"  -> Saved {ckpt.name} (AUC={val_auc:.4f})")

        # Final soundscape eval
        state = torch.load(WEIGHTS_DIR / f"exp4_kd_fold{fold}_best.pth",
                           map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        _, final_ss, ss_n = validate(model, eval_loader)
        print(f"Fold {fold} best -> ss_auc: {final_ss:.4f} ({ss_n} cls)")

        results[fold] = {"best_auc": best_auc, "best_epoch": best_epoch, "ss_auc": final_ss}

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    # ── Report ───────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    for fold, res in results.items():
        print(f"Fold {fold}: val_auc={res['best_auc']:.4f}@E{res['best_epoch']}, ss_auc={res['ss_auc']:.4f}")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    with open(OUT_DIR / "exp4_config.json", "w") as f:
        json.dump(CFG, f, indent=2)


if __name__ == "__main__":
    main()
