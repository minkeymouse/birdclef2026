#!/usr/bin/env python3
"""
exp38 — SED 학습에 labeled SS 포함 (핵심 교정).

진단: 지금까지 SED29는 train_audio only로 학습됨. Salman recipe의 "labeled SS 포함"이
  빠져 있었음. 이건 Val-A 지키려다 submission SED가 labeled SS를 영원히 못 본 실수.

변경:
  - labeled SS 66 files를 **55 train / 11 eval** split (seed 42, 파일 단위)
  - 학습 data = train_audio (35549) + 55 labeled SS (~660 segs, WeightedSampler 20%)
  - Eval = 11 held-out SS files × 12 segs = 132 rows (exp29 Val-A와 호환 불가, 새 metric)
  - 나머지 exp29 동일 (B0, 20s, BCE, mixup, LR 1e-3, 20 epochs)

목표: Val-A_v2 ≥ 0.78 (labeled SS inclusion 효과 확인). LB 투입 전 local 검증.
"""
from __future__ import annotations
import json, random, time
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
OUT = ROOT / "experiments" / "exp38_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5

N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000

BATCH_SIZE = 32; EPOCHS = 20
LR = 1e-3; WD = 1e-2
NUM_WORKERS = 8
MIXUP_ALPHA = 0.5; MIXUP_P = 0.5
SS_SAMPLE_SHARE = 0.25
EARLY_STOP_PATIENCE = 5
EVAL_N_FILES = 11
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
N_CLASSES = 234


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_ss_segments():
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

    # Split files
    rng = np.random.RandomState(SEED)
    files = sorted(sc.filename.unique())
    rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    train_files = set(files[EVAL_N_FILES:])

    sc_train = sc[sc.filename.isin(train_files)].reset_index(drop=True)
    sc_eval = sc[sc.filename.isin(eval_files)].reset_index(drop=True)

    # Eval Y matrix
    Y_eval = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["label_list"]):
        for lbl in labs:
            if lbl in label_to_idx:
                Y_eval[i, label_to_idx[lbl]] = 1

    return sc_train, sc_eval, Y_eval, label_to_idx, primary


def load_train_audio(label_to_idx):
    train = pd.read_csv(DATA / "train.csv")
    train["primary_label"] = train["primary_label"].astype(str)

    def parse_sec(x):
        if pd.isna(x) or x == "[]": return []
        return [t.strip().strip("'\"") for t in str(x).strip("[]").split(",") if t.strip()]

    train["sec_labels"] = train["secondary_labels"].apply(parse_sec)
    return train


class AudioDataset(Dataset):
    def __init__(self, df, label_to_idx, audio_root=DATA / "train_audio"):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.n_classes = len(label_to_idx)
        self.audio_root = audio_root

    def __len__(self): return len(self.df)

    def _load(self, fn):
        try:
            y, sr = sf.read(self.audio_root / fn, dtype="float32", always_2d=False)
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
            s = np.random.randint(0, len(y) - CLIP_SAMPLES)
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
    def __init__(self, sc_df, label_to_idx, ss_root=DATA / "train_soundscapes", train=True):
        self.df = sc_df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.n_classes = len(label_to_idx)
        self.ss_root = ss_root
        self.train = train

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
        if self.train:
            return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(target)
        return torch.from_numpy(clip.astype(np.float32)), idx


class MixDataset(Dataset):
    def __init__(self, audio_ds, ss_ds):
        self.audio_ds = audio_ds
        self.ss_ds = ss_ds

    def __len__(self): return len(self.audio_ds) + len(self.ss_ds)

    def __getitem__(self, idx):
        if idx < len(self.audio_ds):
            return self.audio_ds[idx]
        return self.ss_ds[idx - len(self.audio_ds)]


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True,
        )
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x):
        return self.adb(self.mel(x)).unsqueeze(1)


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

    def forward(self, x):
        m = self.mel(x)
        m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        feat = feat.mean(dim=2) if feat.dim() == 4 else feat
        return self.head(feat)


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], torch.maximum(lam * y, (1 - lam) * y[idx])


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
                clip, _ = model(x)
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
            clip, fmax = model(x)
            loss = F.binary_cross_entropy_with_logits(clip, y) + \
                   F.binary_cross_entropy_with_logits(fmax, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0); n += x.size(0)
    return total_loss / n


def main():
    t0 = time.time()
    set_seed(SEED)
    print(f"[exp38] backbone={BACKBONE}  clip_sec={CLIP_SEC}  device={DEVICE}")

    sc_train, sc_eval, Y_eval, label_to_idx, primary = build_ss_segments()
    print(f"SS split: train {sc_train.filename.nunique()} files ({len(sc_train)} segs)  "
          f"eval {sc_eval.filename.nunique()} files ({len(sc_eval)} segs)")
    print(f"Eval active classes: {(Y_eval.sum(0)>0).sum()}")

    train_df = load_train_audio(label_to_idx)
    print(f"train_audio rows: {len(train_df)}")

    audio_ds = AudioDataset(train_df, label_to_idx)
    ss_train_ds = LabeledSSDataset(sc_train, label_to_idx, train=True)
    mix_ds = MixDataset(audio_ds, ss_train_ds)

    w_audio = (1 - SS_SAMPLE_SHARE) / len(audio_ds)
    w_ss = SS_SAMPLE_SHARE / len(ss_train_ds)
    weights = np.concatenate([
        np.full(len(audio_ds), w_audio, dtype=np.float64),
        np.full(len(ss_train_ds), w_ss, dtype=np.float64),
    ])
    sampler = WeightedRandomSampler(weights, num_samples=len(audio_ds), replacement=True)
    train_loader = DataLoader(mix_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, drop_last=True)

    eval_ds = LabeledSSDataset(sc_eval, label_to_idx, train=False)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"Per-epoch samples: {len(audio_ds)}  (SS share {SS_SAMPLE_SHARE:.0%})")

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
        preds = evaluate(model, eval_loader, DEVICE, N_CLASSES)
        auc = macro_auc(Y_eval, preds)
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
            "ss_train_files": int(sc_train.filename.nunique()),
            "ss_eval_files": int(sc_eval.filename.nunique()),
            "history": history, "best_epoch": best_epoch,
            "best_val_auc": best_auc, "elapsed_s": time.time() - t0,
        }, indent=2))
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop at ep {ep}")
            break

    if best_preds is not None:
        np.savez_compressed(OUT / "eval_scores.npz", preds=best_preds, Y_eval=Y_eval)
    print(f"\nDone. Best Val-A_v2 = {best_auc:.4f} at ep {best_epoch}. Total {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
