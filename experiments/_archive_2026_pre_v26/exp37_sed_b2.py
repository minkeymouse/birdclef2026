#!/usr/bin/env python3
"""
exp37 — HGNetV2-B2 SED, Salman 정석 재현.

exp29 (alone Val-A 0.737) 실패 분석:
  - epoch 3에서 val_auc 0.7374 피크, 이후 loss 0.048→0.012로 내려가지만 val_auc 계속 하락
  - 진짜 원인: train_audio only → labeled soundscape 도메인 미스매치 과적합
  - early stop 없어서 ep20 최종 checkpoint는 오버핏 상태

exp37 주요 변경 (vs exp29):
  1. Backbone HGNetV2-B0 → **B2** (11M params, 32GB의 ~15%)
  2. Input 20s → **10s** (Salman 명시 "best single model 10s")
  3. Training data: train_audio + **labeled SS 7 files × 12 × 50 upsample** 추가
  4. **Early stopping patience 5** (exp29 ep3 피크 후 놓친 교훈)
  5. Mild SpecAugment (exp36 overkill 방지)
  6. LR 8e-4 (exp29 1e-3에서 살짝 내림)

목표 alone Val-A ≥ 0.82.

Output:
  experiments/exp37_outputs/best_ckpt.pt
  experiments/exp37_outputs/val_scores.npz  (708, 234) Val-A logits (sigmoid)
  experiments/exp37_outputs/results.json
"""
from __future__ import annotations
import json, math, random, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"
OUT = ROOT / "experiments" / "exp37_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Config ──────────────────────────────────────────────────────────────
SR = 32000
CLIP_SEC = 10
CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12

N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 50
FMAX = 14000

BATCH_SIZE = 48
EPOCHS = 25
LR = 8e-4
WD = 1e-2
NUM_WORKERS = 10
MIXUP_ALPHA = 0.5
MIXUP_P = 0.7
SPEC_AUG_P = 0.5
SPEC_F = 15
SPEC_T = 40
LABELED_SS_UPSAMPLE = 50
EARLY_STOP_PATIENCE = 5
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "hgnetv2_b2.ssld_stage2_ft_in1k"
N_CLASSES = 234


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ─── Data ────────────────────────────────────────────────────────────────

def load_labels():
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary)}
    train = pd.read_csv(DATA / "train.csv")
    train["primary_label"] = train["primary_label"].astype(str)

    def parse_sec(x):
        if pd.isna(x) or x == "[]": return []
        return [t.strip().strip("'\"") for t in str(x).strip("[]").split(",") if t.strip()]

    train["sec_labels"] = train["secondary_labels"].apply(parse_sec)
    return primary, label_to_idx, train


def build_val_truth():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary)}

    def parse_lbls(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({lbl for x in s for lbl in parse_lbls(x)}))
          .reset_index(name="label_list"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = (sc["filename"].str.replace(".ogg", "", regex=False)
                    + "_" + sc["end_sec"].astype(str))

    meta_full = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    val_files = set(meta_full.filename.unique())

    # Val-A truth
    sc_idx = sc.set_index("row_id")
    Y_SC = np.zeros((len(sc), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc["label_list"]):
        for lbl in labs:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1
    Y_FULL = np.stack([Y_SC[sc_idx.index.get_loc(rid)] for rid in meta_full["row_id"]])

    # Labeled SS for training: all files NOT in val_files
    train_ss = sc[~sc.filename.isin(val_files)].copy()
    print(f"Labeled SS for training: {train_ss.filename.nunique()} files, {len(train_ss)} segments")
    return meta_full, Y_FULL, train_ss, label_to_idx


class AudioDataset(Dataset):
    """train_audio clips. Random CLIP_SAMPLES crop. Multi-hot from primary + secondary."""
    def __init__(self, df, label_to_idx, train=True, audio_root=DATA / "train_audio"):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.n_classes = len(label_to_idx)
        self.train = train
        self.audio_root = audio_root

    def __len__(self): return len(self.df)

    def _load(self, fn):
        path = self.audio_root / fn
        try:
            y, sr = sf.read(path, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            if sr != SR:
                y = torchaudio.functional.resample(
                    torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
        except Exception:
            y = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        return y.astype(np.float32)

    def _crop(self, y):
        if len(y) < CLIP_SAMPLES:
            reps = (CLIP_SAMPLES + len(y) - 1) // len(y)
            y = np.tile(y, reps)[:CLIP_SAMPLES]
        elif len(y) > CLIP_SAMPLES:
            s = np.random.randint(0, len(y) - CLIP_SAMPLES) if self.train else (len(y) - CLIP_SAMPLES) // 2
            y = y[s:s + CLIP_SAMPLES]
        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self._crop(self._load(row["filename"]))
        target = np.zeros(self.n_classes, dtype=np.float32)
        if row["primary_label"] in self.label_to_idx:
            target[self.label_to_idx[row["primary_label"]]] = 1.0
        for lbl in row["sec_labels"]:
            if lbl in self.label_to_idx:
                target[self.label_to_idx[lbl]] = 0.5
        return torch.from_numpy(y), torch.from_numpy(target)


class LabeledSSDataset(Dataset):
    """Labeled soundscape training samples. Each row = one 5s segment; we extract CLIP_SAMPLES."""
    def __init__(self, df_segs, label_to_idx, ss_root=DATA / "train_soundscapes"):
        self.df = df_segs.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.n_classes = len(label_to_idx)
        self.ss_root = ss_root

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = row["filename"]
        end_sec = int(row["end_sec"])
        start_sec = end_sec - WINDOW_SEC

        path = self.ss_root / fn
        try:
            y, sr = sf.read(path, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            if sr != SR:
                y = torchaudio.functional.resample(
                    torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
        except Exception:
            y = np.zeros(SR * 60, dtype=np.float32)

        center = ((start_sec + end_sec) // 2) * SR
        half = CLIP_SAMPLES // 2
        s = max(0, center - half); e = s + CLIP_SAMPLES
        if e > len(y):
            e = len(y); s = max(0, e - CLIP_SAMPLES)
        clip = y[s:e]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))

        target = np.zeros(self.n_classes, dtype=np.float32)
        for lbl in row["label_list"]:
            if lbl in self.label_to_idx:
                target[self.label_to_idx[lbl]] = 1.0
        return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(target)


class CombinedDataset(Dataset):
    """Concatenation wrapper with sample-level weights via ConcatDataset."""
    def __init__(self, audio_ds, ss_ds, ss_upsample):
        self.audio_ds = audio_ds
        self.ss_ds = ss_ds
        self.ss_upsample = ss_upsample
        self.n_audio = len(audio_ds)
        self.n_ss = len(ss_ds)

    def __len__(self): return self.n_audio + self.n_ss * self.ss_upsample

    def __getitem__(self, idx):
        if idx < self.n_audio:
            return self.audio_ds[idx]
        ss_idx = (idx - self.n_audio) % self.n_ss
        return self.ss_ds[ss_idx]


class ValSSDataset(Dataset):
    """Val-A: 59 labeled SS files × 12 windows. Center-crop CLIP_SAMPLES."""
    def __init__(self, meta_full, ss_root=DATA / "train_soundscapes"):
        self.meta = meta_full.reset_index(drop=True)
        self.ss_root = ss_root

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        fn = row["filename"]
        end_sec = int(row["row_id"].rsplit("_", 1)[1])
        start_sec = end_sec - WINDOW_SEC

        path = self.ss_root / fn
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(axis=1)
        assert sr == SR

        center = ((start_sec + end_sec) // 2) * SR
        half = CLIP_SAMPLES // 2
        s = max(0, center - half); e = s + CLIP_SAMPLES
        if e > len(y):
            e = len(y); s = max(0, e - CLIP_SAMPLES)
        clip = y[s:e]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        return torch.from_numpy(clip.astype(np.float32)), idx


# ─── Model ───────────────────────────────────────────────────────────────

class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True,
        )
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        # Mild SpecAugment
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=SPEC_F)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=SPEC_T)
        self.spec_aug_p = SPEC_AUG_P

    def forward(self, x, training=False):
        m = self.adb(self.mel(x))
        if training and torch.rand(1).item() < self.spec_aug_p:
            m = self.time_mask(self.freq_mask(m))
        return m.unsqueeze(1)


class SEDHead(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.att = nn.Conv1d(feat_dim, n_classes, 1)
        self.cla = nn.Conv1d(feat_dim, n_classes, 1)

    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        clip = (w * c).sum(-1)
        fmax = c.max(-1).values
        return clip, fmax


class SEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, n_classes=N_CLASSES):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=1,
            num_classes=0, global_pool="",
        )
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        feat_dim = feat.shape[1]
        self.head = SEDHead(feat_dim, n_classes)

    def forward(self, x, training=False):
        m = self.mel(x, training=training)
        m = m.transpose(1, 2)
        m = self.bn0(m)
        m = m.transpose(1, 2)
        feat = self.backbone(m)
        feat = feat.mean(dim=2) if feat.dim() == 4 else feat
        return self.head(feat)


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mx = lam * x + (1 - lam) * x[idx]
    my = torch.maximum(lam * y, (1 - lam) * y[idx])
    return mx, my


def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def evaluate(model, val_loader, device, n_classes):
    model.eval()
    preds = np.zeros((len(val_loader.dataset), n_classes), dtype=np.float32)
    with torch.no_grad():
        for x, idxs in val_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip, _ = model(x, training=False)
            p = torch.sigmoid(clip).float().cpu().numpy()
            for i, j in zip(idxs.tolist(), range(len(p))):
                preds[i] = p[j]
    return preds


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0; n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if random.random() < MIXUP_P:
            x, y = mixup_data(x, y)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            clip, fmax = model(x, training=True)
            loss = F.binary_cross_entropy_with_logits(clip, y) + \
                   F.binary_cross_entropy_with_logits(fmax, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0); n += x.size(0)
    return total_loss / n


def main():
    t0 = time.time()
    set_seed(SEED)
    print(f"[exp37] backbone={BACKBONE}  clip_sec={CLIP_SEC}  device={DEVICE}")

    primary, label_to_idx, train_df = load_labels()
    print(f"train_audio rows: {len(train_df)}")

    meta_full, Y_FULL, train_ss_segs, _ = build_val_truth()

    audio_ds = AudioDataset(train_df, label_to_idx, train=True)
    ss_ds = LabeledSSDataset(train_ss_segs, label_to_idx)
    combined = CombinedDataset(audio_ds, ss_ds, LABELED_SS_UPSAMPLE)
    print(f"Combined train size: {len(combined)}  (audio={len(audio_ds)} + ss={len(ss_ds)}×{LABELED_SS_UPSAMPLE})")

    train_loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, drop_last=True)

    val_ds = ValSSDataset(meta_full)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Val-A rows: {len(val_ds)}  active classes: {(Y_FULL.sum(0)>0).sum()}")

    model = SEDModel(BACKBONE, N_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    history = []; best_auc = -1.0; best_epoch = -1; best_preds = None
    no_improve = 0

    for ep in range(1, EPOCHS + 1):
        ep_t = time.time()
        loss = train_one_epoch(model, train_loader, opt, DEVICE)
        sched.step()
        preds = evaluate(model, val_loader, DEVICE, N_CLASSES)
        auc = macro_auc(Y_FULL, preds)
        dt = time.time() - ep_t
        print(f"Ep {ep:02d}/{EPOCHS}  loss {loss:.4f}  val_auc {auc:.4f}  ({dt:.0f}s)")
        history.append({"epoch": ep, "loss": loss, "val_auc": auc, "time_s": dt})
        if auc > best_auc:
            best_auc = auc; best_epoch = ep; best_preds = preds
            no_improve = 0
            torch.save({
                "epoch": ep, "state_dict": model.state_dict(),
                "backbone": BACKBONE, "val_auc": auc,
            }, OUT / "best_ckpt.pt")
        else:
            no_improve += 1
        (OUT / "results.json").write_text(json.dumps({
            "backbone": BACKBONE, "clip_sec": CLIP_SEC,
            "history": history, "best_epoch": best_epoch,
            "best_val_auc": best_auc, "elapsed_s": time.time() - t0,
        }, indent=2))
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop at ep {ep} (no improve for {EARLY_STOP_PATIENCE} epochs)")
            break

    if best_preds is not None:
        np.savez_compressed(OUT / "val_scores.npz", preds=best_preds)
    print(f"\nDone. Best Val-A AUC = {best_auc:.4f} at ep {best_epoch}. Total {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
