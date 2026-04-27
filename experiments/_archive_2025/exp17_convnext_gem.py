#!/usr/bin/env python3
"""
exp17_convnext_gem.py — ConvNeXt-V2 Tiny + GEM Pooling + 3-channel mel.

Replicates the architecture from the top public notebook:
  - ConvNeXt-V2 Tiny (fcmae_ft_in22k_in1k_384 pretrained)
  - 3-channel mel spectrogram (expand mono to RGB)
  - Per-sample mean/std normalization (not min-max)
  - GEM frequency pooling + attention SED head
  - torchaudio mel on GPU (no librosa during training)

The public notebook only trained 3 epochs (val_auc=0.5755).
We train properly: 15 epochs, 2-fold, mel augmentation, SpecAugment.

Key insight: our exp13 ConvNeXt-V2 failure was due to in_chans=1.
This notebook uses 3-channel expansion which preserves LayerNorm compatibility.
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
import torchaudio
import torchaudio.transforms as T
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp17_outputs"

for d in [WEIGHTS_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CFG = {
    "seed": 42,
    "sr": 32000,
    "duration": 5.0,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "fmin": 20,
    "fmax": 16000,
    "backbone": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    "pretrained": True,
    "lr": 5e-5,  # Match public notebook
    "weight_decay": 1e-2,
    "epochs": 10,
    "batch_size": 32,  # Smaller due to larger model
    "num_workers": 2,
    "n_folds": 5,
    "train_folds": [0],
    "mixup_alpha": 0.4,
    "gem_p_init": 3.0,
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
# GPU-based Audio Transform (from public notebook)
# ====================================================================
class AudioTransform(nn.Module):
    """GPU-based mel spectrogram + normalization + 3-channel expansion."""
    def __init__(self, is_train=False):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=CFG["sr"], n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
            n_mels=CFG["n_mels"], f_min=CFG["fmin"], f_max=CFG["fmax"],
            norm='slaney', mel_scale='htk'
        )
        self.amp_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        self.is_train = is_train
        self.freq_mask = T.FrequencyMasking(freq_mask_param=int(CFG["n_mels"] * 0.2))
        self.time_mask = T.TimeMasking(
            time_mask_param=int((CFG["duration"] * CFG["sr"] / CFG["hop_length"]) * 0.2)
        )

    @torch.no_grad()
    def forward(self, x):
        x = self.mel_spec(x)
        x = self.amp_to_db(x)

        if self.is_train:
            x = self.freq_mask(x)
            x = self.time_mask(x)

        x = x.to(torch.float32)
        # Per-sample standardization (key difference from our min-max approach)
        x = (x - x.mean(dim=[-2, -1], keepdim=True)) / (x.std(dim=[-2, -1], keepdim=True) + 1e-4)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Expand mono to 3 channels (preserves LayerNorm compatibility)
        x = x.expand(-1, 3, -1, -1)
        return x


# ====================================================================
# Dataset — on-the-fly waveform loading (GPU mel transform)
# ====================================================================
class WaveformDataset(Dataset):
    """Loads raw waveforms; mel transform happens on GPU in the training loop."""
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.chunk_samples = int(CFG["sr"] * CFG["duration"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = row["label"]
        if isinstance(label, str):
            label = np.array(json.loads(label), dtype=np.float32)

        try:
            waveform, sr = torchaudio.load(filepath)
            waveform = torch.nan_to_num(waveform, nan=0.0)
            if sr != CFG["sr"]:
                waveform = torchaudio.functional.resample(waveform, sr, CFG["sr"])
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        except Exception:
            waveform = torch.zeros(1, self.chunk_samples)

        audio_len = waveform.shape[1]
        if audio_len > self.chunk_samples:
            if self.is_train:
                start = random.randint(0, audio_len - self.chunk_samples)
            else:
                start = 0
            waveform = waveform[:, start:start + self.chunk_samples]
        elif audio_len < self.chunk_samples:
            waveform = F.pad(waveform, (0, self.chunk_samples - audio_len))

        return waveform, torch.tensor(label, dtype=torch.float32)


# ====================================================================
# Model — ConvNeXt-V2 + GEM + Attention SED (from public notebook)
# ====================================================================
class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-5):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        p = self.p.float().clamp(min=1.0, max=10.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)  # Pool over frequency
        return x.clamp(min=self.eps).pow(1.0 / p)  # Standard GEM


class BirdConvNeXtModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(CFG["backbone"], pretrained=pretrained)
        self.backbone.reset_classifier(0, global_pool='')
        in_features = self.backbone.num_features

        self.pool = GEMFreqPool(p_init=CFG["gem_p_init"])
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.att_conv = nn.Conv1d(in_features, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(in_features, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)  # (B, C, F, T) — includes final LayerNorm2d
        x = self.pool(x)  # (B, C, T)

        x = x.transpose(1, 2)  # (B, T, C)
        x = self.fc(x)
        x = x.transpose(1, 2)  # (B, C, T)

        x = x.float()
        att = torch.softmax(self.att_conv(x), dim=-1)
        cls = self.cls_conv(x)
        clipwise_logits = (att * cls).sum(dim=-1)

        return clipwise_logits


# ====================================================================
# Training
# ====================================================================
def mixup(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, mel_transform):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Train E{epoch+1}")
    for waveforms, labels in pbar:
        waveforms = waveforms.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            mels = mel_transform(waveforms)

        # Mixup on mel spectrograms
        mels, labels = mixup(mels, labels, CFG["mixup_alpha"])

        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(mels)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
    return np.mean(losses) if losses else float('nan')


@torch.no_grad()
def validate(model, loader, mel_transform):
    model.eval()
    all_preds, all_labels = [], []
    for waveforms, labels in loader:
        waveforms = waveforms.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        mels = mel_transform(waveforms)
        with autocast("cuda"):
            logits = model(mels)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
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


def prepare_metadata():
    """Prepare train dataframe with filepaths and labels."""
    import ast
    train_df = pd.read_csv(DATA / "train.csv")
    rows = []
    for _, row in train_df.iterrows():
        filepath = DATA / "train_audio" / row["filename"]
        if not filepath.exists():
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
        rows.append({
            "filepath": str(filepath),
            "primary_label": pl,
            "label": label,
        })
    return pd.DataFrame(rows)


def prepare_eval_metadata():
    """Prepare soundscape eval dataframe."""
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    rows = []
    for _, row in labels_df.iterrows():
        filename = row["filename"]
        filepath = DATA / "train_soundscapes" / filename
        if not filepath.exists():
            continue

        def _parse_time(t):
            parts = t.strip().split(":")
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        start_sec = _parse_time(str(row["start"]))
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in SPECIES2IDX:
                label[SPECIES2IDX[sp]] = 1.0
        rows.append({
            "filepath": str(filepath),
            "start_sec": start_sec,
            "primary_label": str(row["primary_label"]).split(";")[0],
            "label": label,
        })
    return pd.DataFrame(rows)


class SoundscapeEvalDataset(Dataset):
    """Loads specific 5s segments from soundscape files for evaluation."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.chunk_samples = int(CFG["sr"] * CFG["duration"])
        self._audio_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        start_sec = row["start_sec"]
        label = row["label"]
        if isinstance(label, str):
            label = np.array(json.loads(label), dtype=np.float32)

        if filepath not in self._audio_cache:
            try:
                waveform, sr = torchaudio.load(filepath)
                if sr != CFG["sr"]:
                    waveform = torchaudio.functional.resample(waveform, sr, CFG["sr"])
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                self._audio_cache[filepath] = waveform
            except Exception:
                self._audio_cache[filepath] = torch.zeros(1, CFG["sr"] * 60)

        waveform = self._audio_cache[filepath]
        start_sample = int(start_sec * CFG["sr"])
        chunk = waveform[:, start_sample:start_sample + self.chunk_samples]
        if chunk.shape[1] < self.chunk_samples:
            chunk = F.pad(chunk, (0, self.chunk_samples - chunk.shape[1]))

        return chunk, torch.tensor(label, dtype=torch.float32)


def main():
    start_time = time.time()

    # Prepare data
    meta_path = OUT_DIR / "train_meta.parquet"
    if meta_path.exists():
        print("Loading cached metadata...")
        train_meta = pd.read_parquet(meta_path)
        train_meta["label"] = train_meta["label"].apply(lambda x: np.array(x, dtype=np.float32))
    else:
        print("Preparing metadata...")
        train_meta = prepare_metadata()
        save_df = train_meta.copy()
        save_df["label"] = save_df["label"].apply(lambda x: x.tolist())
        save_df.to_parquet(meta_path)

    eval_meta = prepare_eval_metadata()
    print(f"Train: {len(train_meta)}, Eval (soundscapes): {len(eval_meta)}")

    # Mel transforms (on GPU)
    mel_train = AudioTransform(is_train=True).to(DEVICE)
    mel_eval = AudioTransform(is_train=False).to(DEVICE)

    # Eval loader
    eval_ds = SoundscapeEvalDataset(eval_meta)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False,
                             num_workers=0, pin_memory=True)  # num_workers=0 due to cache

    skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
    results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_meta, train_meta["primary_label"])):
        if fold not in CFG["train_folds"]:
            continue

        print(f"\n{'='*60}\nFOLD {fold} (ConvNeXt-V2 + GEM + 3ch)\n{'='*60}")
        fold_train = train_meta.iloc[train_idx].reset_index(drop=True)
        fold_val = train_meta.iloc[val_idx].reset_index(drop=True)

        train_ds = WaveformDataset(fold_train, is_train=True)
        val_ds = WaveformDataset(fold_val, is_train=False)

        # WeightedRandomSampler to combat all-zeros collapse
        from torch.utils.data import WeightedRandomSampler
        label_counts = fold_train["primary_label"].value_counts()
        sample_weights = fold_train["primary_label"].map(lambda x: 1.0 / max(label_counts.get(x, 1), 1)).values
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                                  num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                                num_workers=CFG["num_workers"], pin_memory=True)

        model = BirdConvNeXtModel(NUM_CLASSES, pretrained=CFG["pretrained"]).to(DEVICE)

        # Train from ImageNet pretrained backbone (no public competition weights)
        print(f"  Training from ImageNet-pretrained ConvNeXt-V2 Tiny")
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-6)
        scaler = GradScaler("cuda")

        best_ss = 0.0
        best_epoch = -1

        for epoch in range(CFG["epochs"]):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, mel_train)
            scheduler.step()
            val_auc, n_val = validate(model, val_loader, mel_eval)
            ss_auc, ss_n = validate(model, eval_loader, mel_eval)

            print(f"E{epoch+1}/{CFG['epochs']} — loss: {train_loss:.4f}, "
                  f"val_auc: {val_auc:.4f}({n_val}), ss_auc: {ss_auc:.4f}({ss_n})")

            if ss_auc > best_ss:
                best_ss = ss_auc
                best_epoch = epoch + 1
                ckpt = WEIGHTS_DIR / f"exp17_convnext_fold{fold}_best.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "ss_auc": ss_auc,
                    "val_auc": val_auc,
                    "config": CFG,
                }, ckpt)
                print(f"  -> Saved {ckpt.name} (ss_auc={ss_auc:.4f})")

        print(f"Fold {fold} best -> ss_auc: {best_ss:.4f} @ E{best_epoch}")
        results[fold] = {"val_auc": val_auc, "best_epoch": best_epoch, "ss_auc": best_ss}

        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}\nRESULTS (exp17: ConvNeXt-V2 + GEM + 3ch)")
    print(f"{'='*60}")
    exp14_ss = {0: 0.7820, 1: 0.7681}
    for fold, res in results.items():
        delta = res['ss_auc'] - exp14_ss.get(fold, 0)
        print(f"Fold {fold}: ss_auc={res['ss_auc']:.4f}@E{res['best_epoch']}, Δ vs exp14: {delta:+.4f}")
    print(f"Time: {elapsed:.1f} min")

    with open(OUT_DIR / "exp17_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)


if __name__ == "__main__":
    main()
