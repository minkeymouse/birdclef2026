#!/usr/bin/env python3
"""exp47 — Boredom-style SED faithful recipe.

Goal: single-model common-Aves strong SED. LB 0.94+ target (Boredom 0.947 claim).

Recipe (consolidated 2025 top-scorer insight):
  Backbone   : HGNetV2-B0 (fp32 mandatory on RTX 5090 compute cap 12.0a)
  Input      : 20s raw audio → GPU mel (n_mels=128, n_fft=2048, hop=512)
  InputNorm  : BN2d(N_MELS) (critical — prevents NaN)
  Augment    : raw-waveform mixup (alpha=0.5, p=0.6)
               + background mixing (labeled SS quiet windows)
               + mild SpecAugment (F=16, T=40)
  Loss       : CE-softmax(clipwise)+primary target for train_audio clips
               BCE(clipwise, multihot)+BCE(framewise_max) for labeled SS windows
               secondary label weight 0.3
  Schedule   : AdamW 1e-3 WD 1e-2, cosine annealing over 30 ep, warmup 3 ep
  Early stop : patience 8 on val_TA macro AUC
  Data       : train_audio 80% + labeled SS 55 training files
  Val        : train_audio 20% held-out (stratified by primary_label)
               + 11 labeled SS files (existing Val-A_v2)

Two evaluation signals reported per epoch:
  val_TA  = macro AUC on held-out train_audio (clean, common-Aves dominated — LB proxy)
  val_SS  = macro AUC on 11 labeled SS files (rare-heavy, our existing local eval)

Output:
  experiments/exp47_outputs/
    best_ckpt.pt       — best by val_TA
    history.json       — training curve
    val_scores.npz     — final predictions on both eval sets
"""
from __future__ import annotations
import json, random, re, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
OUT = ROOT / "experiments" / "exp47_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = SR * 60
N_WINDOWS_FILE = 12

N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WD = 1e-2
NUM_WORKERS = 4
MIXUP_ALPHA = 0.5
MIXUP_P = 0.6
BG_MIX_P = 0.4
SECONDARY_WEIGHT = 0.3
WARMUP_EPOCHS = 3
EARLY_STOP_PATIENCE = 8
SPEC_FREQ_MASK = 16
SPEC_TIME_MASK = 40
EVAL_SS_N_FILES = 11
SEED = 42

BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
DEVICE = "cuda"

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ─── Data prep ──────────────────────────────────────────────────────────
def build_primaries():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}
    return primary, l2i


def build_train_audio_splits(l2i, val_frac=0.20, seed=SEED):
    """Stratified 80/20 split of train_audio by primary_label."""
    df = pd.read_csv(DATA / "train.csv")
    df = df[df["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
    df["primary_idx"] = df["primary_label"].astype(str).map(l2i)
    # Stratified split
    rng = np.random.RandomState(seed)
    val_idx = []
    train_idx = []
    for lbl, g in df.groupby("primary_label"):
        g_idx = g.index.tolist()
        rng.shuffle(g_idx)
        n_val = max(1, int(len(g_idx) * val_frac)) if len(g_idx) >= 5 else 0
        val_idx.extend(g_idx[:n_val])
        train_idx.extend(g_idx[n_val:])
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    print(f"train_audio split: train {len(train_df)}, val {len(val_df)}")
    # parse secondary labels
    def parse_sec(x):
        if pd.isna(x) or x in ("[]", ""): return []
        try:
            return [s.strip("'\" ") for s in x.strip("[]").split(",") if s.strip("'\" ")]
        except Exception: return []
    train_df["secondary_list"] = train_df["secondary_labels"].apply(parse_sec)
    val_df["secondary_list"] = val_df["secondary_labels"].apply(parse_sec)
    return train_df, val_df


def build_ss_splits(l2i):
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc_raw.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    # same seed-42 11/55 split as exp38
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_SS_N_FILES])
    train_files = set(files[EVAL_SS_N_FILES:])
    ss_train = sc_g[sc_g.filename.isin(train_files)].reset_index(drop=True)
    ss_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    return ss_train, ss_eval


# ─── Audio I/O ──────────────────────────────────────────────────────────
def load_audio(path, target_samples):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != SR:
            # resample via torchaudio if needed (rare)
            import torchaudio.functional as TF
            wav = TF.resample(torch.from_numpy(wav), sr, SR).numpy()
        if len(wav) == 0: return np.zeros(target_samples, dtype=np.float32)
        # loop-pad short
        if len(wav) < target_samples:
            reps = target_samples // len(wav) + 1
            wav = np.tile(wav, reps)[:target_samples]
        return wav.astype(np.float32)
    except Exception:
        return np.zeros(target_samples, dtype=np.float32)


def random_crop(wav, target_samples):
    if len(wav) <= target_samples:
        if len(wav) < target_samples:
            wav = np.pad(wav, (0, target_samples - len(wav)))
        return wav[:target_samples]
    start = random.randint(0, len(wav) - target_samples)
    return wav[start:start + target_samples]


def center_crop(wav, target_samples):
    if len(wav) <= target_samples:
        if len(wav) < target_samples:
            wav = np.pad(wav, (0, target_samples - len(wav)))
        return wav[:target_samples]
    start = (len(wav) - target_samples) // 2
    return wav[start:start + target_samples]


# ─── Datasets ───────────────────────────────────────────────────────────
class TrainAudioDataset(Dataset):
    """Each item: 20s raw audio clip + primary-target idx + secondary multihot."""
    def __init__(self, df, l2i, train=True):
        self.df = df.reset_index(drop=True)
        self.l2i = l2i
        self.train = train
        self.n_cls = len(l2i)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = DATA / "train_audio" / row.filename
        wav = load_audio(path, CLIP_SAMPLES * 2)  # load up to 40s then crop
        wav = random_crop(wav, CLIP_SAMPLES) if self.train else center_crop(wav, CLIP_SAMPLES)
        y = np.zeros(self.n_cls, dtype=np.float32)
        y[row.primary_idx] = 1.0
        for sl in row.secondary_list:
            if sl in self.l2i:
                y[self.l2i[sl]] = SECONDARY_WEIGHT
        return torch.from_numpy(wav), torch.from_numpy(y), int(row.primary_idx), 1  # is_train_audio=1


class LabeledSSDataset(Dataset):
    """Each item: 20s clip around a labeled SS window + multihot multi-label."""
    def __init__(self, ss_df, l2i, train=True):
        self.ss = ss_df.reset_index(drop=True)
        self.l2i = l2i
        self.train = train
        self.n_cls = len(l2i)

    def __len__(self): return len(self.ss)

    def __getitem__(self, idx):
        row = self.ss.iloc[idx]
        p = DATA / "train_soundscapes" / row.filename
        wav = load_audio(p, FILE_SAMPLES)
        end_sec = int(row.end_sec)
        target_center = (end_sec - WINDOW_SEC / 2) * SR
        clip_start = int(max(0, target_center - CLIP_SAMPLES / 2))
        clip_start = min(clip_start, FILE_SAMPLES - CLIP_SAMPLES)
        if self.train:
            # small jitter
            clip_start = int(clip_start + random.randint(-SR, SR))
            clip_start = max(0, min(clip_start, FILE_SAMPLES - CLIP_SAMPLES))
        clip = wav[clip_start:clip_start + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        y = np.zeros(self.n_cls, dtype=np.float32)
        for l in row.lbls:
            if l in self.l2i: y[self.l2i[l]] = 1.0
        return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(y), -1, 0  # primary_idx=-1 means multi-label


# ─── Mel + Model ────────────────────────────────────────────────────────
class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x):
        return self.adb(self.mel(x)).unsqueeze(1)  # (B, 1, n_mels, T)


class SpecAug(nn.Module):
    def __init__(self, f=SPEC_FREQ_MASK, t=SPEC_TIME_MASK):
        super().__init__()
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=f)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=t)
    def forward(self, x):
        return self.tm(self.fm(x))


class SEDHead(nn.Module):
    def __init__(self, feat_dim, n_cls):
        super().__init__()
        self.att = nn.Conv1d(feat_dim, n_cls, 1)
        self.cla = nn.Conv1d(feat_dim, n_cls, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values   # clipwise, framewise_max


class SEDModel(nn.Module):
    def __init__(self, backbone=BACKBONE, n_cls=234):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.spec_aug = SpecAug()
        self.backbone = timm.create_model(
            backbone, pretrained=True, in_chans=1,
            drop_rate=0.1, drop_path_rate=0.1,
            num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        C = feat.shape[1]
        self.head = SEDHead(C, n_cls)
    def forward(self, x, training_aug=False):
        m = self.mel(x)
        m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        if training_aug and self.training:
            m = self.spec_aug(m)
        feat = self.backbone(m)
        f = feat.mean(dim=2) if feat.dim() == 4 else feat
        clip, fmax = self.head(f)
        return clip, fmax


# ─── Mixup ──────────────────────────────────────────────────────────────
def raw_wave_mixup(x, y, primary_idx, is_ta, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = torch.maximum(lam * y, (1 - lam) * y[idx])
    mixed_primary = primary_idx.clone()  # not meaningful after mix
    mixed_isTA = torch.minimum(is_ta, is_ta[idx])  # if either isn't TA, treat as multi-label
    return mixed_x, mixed_y, mixed_primary, mixed_isTA, lam


# ─── Loss ────────────────────────────────────────────────────────────────
def hybrid_loss(clipwise, fmax, y, primary_idx, is_ta):
    """
    For train_audio samples (is_ta==1, single primary label): CE (softmax) on clipwise.
      primary one-hot CE encourages sharp prediction.
    For labeled SS (is_ta==0, multi-label): BCE on clipwise + BCE on framewise_max.
    Mixup produces smooth labels → use BCE anyway (CE with soft target is also BCE-like).
    """
    ta_mask = is_ta == 1
    ml_mask = ~ta_mask

    losses = []
    if ta_mask.any():
        # Primary idx might be -1 if mix contaminated; use soft-CE via KL
        # Simpler: use BCE here too but stronger primary target weight via y
        losses.append(F.binary_cross_entropy_with_logits(clipwise[ta_mask], y[ta_mask]))
        losses.append(F.binary_cross_entropy_with_logits(fmax[ta_mask], y[ta_mask]))
    if ml_mask.any():
        losses.append(F.binary_cross_entropy_with_logits(clipwise[ml_mask], y[ml_mask]))
        losses.append(F.binary_cross_entropy_with_logits(fmax[ml_mask], y[ml_mask]))
    return sum(losses) / max(1, len(losses))


# ─── Training ────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, dev, use_mixup=True, use_spec_aug=True):
    model.train()
    tot, n, nan_s = 0.0, 0, 0
    for x, y, primary, is_ta in loader:
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        primary = primary.to(dev, non_blocking=True)
        is_ta = is_ta.to(dev, non_blocking=True)

        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, 0, 1, -1)

        if use_mixup and random.random() < MIXUP_P:
            x, y, primary, is_ta, _ = raw_wave_mixup(x, y, primary, is_ta)

        clip, fmax = model(x, training_aug=use_spec_aug)
        if not (torch.isfinite(clip).all() and torch.isfinite(fmax).all()):
            nan_s += 1; opt.zero_grad(set_to_none=True); continue

        loss = hybrid_loss(clip, fmax, y, primary, is_ta)
        if not torch.isfinite(loss):
            nan_s += 1; opt.zero_grad(set_to_none=True); continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            nan_s += 1; continue
        opt.step()
        tot += loss.item() * x.size(0); n += x.size(0)
    return (tot / max(n, 1)), nan_s


@torch.no_grad()
def evaluate_train_audio(model, val_df, l2i, dev, batch=16, max_samples=3000):
    """Evaluate on held-out train_audio 20%. Single-label → macro AUC of top-1 on primary.
       Subsample for speed (max 3000)."""
    model.eval()
    n_cls = len(l2i)
    # Subsample
    if len(val_df) > max_samples:
        idx = np.random.RandomState(123).choice(len(val_df), max_samples, replace=False)
        val_df = val_df.iloc[idx].reset_index(drop=True)
    n = len(val_df)
    preds = np.zeros((n, n_cls), dtype=np.float32)
    Y = np.zeros((n, n_cls), dtype=np.uint8)
    for i in range(0, n, batch):
        j = min(n, i + batch)
        batch_wavs = []
        for k in range(i, j):
            row = val_df.iloc[k]
            wav = load_audio(DATA / "train_audio" / row.filename, CLIP_SAMPLES * 2)
            wav = center_crop(wav, CLIP_SAMPLES)
            batch_wavs.append(wav)
            Y[k, int(row.primary_idx)] = 1
        x = torch.from_numpy(np.stack(batch_wavs)).to(dev)
        clip, _ = model(x)
        p = torch.sigmoid(clip).cpu().numpy().astype(np.float32)
        if not np.isfinite(p).all():
            p = np.nan_to_num(p, 0.5)
        preds[i:j] = p
    # macro AUC
    aucs = []
    for c in range(n_cls):
        if Y[:, c].sum() == 0 or Y[:, c].sum() == n: continue
        try: aucs.append(roc_auc_score(Y[:, c], preds[:, c]))
        except Exception: pass
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs), preds, Y


@torch.no_grad()
def evaluate_ss(model, ss_eval, l2i, dev, batch=4):
    model.eval()
    n_cls = len(l2i)
    n = len(ss_eval)
    preds = np.zeros((n, n_cls), dtype=np.float32)
    Y = np.zeros((n, n_cls), dtype=np.uint8)
    for i in range(0, n, batch):
        j = min(n, i + batch)
        batch_wavs = []
        for k in range(i, j):
            row = ss_eval.iloc[k]
            wav = load_audio(DATA / "train_soundscapes" / row.filename, FILE_SAMPLES)
            end_sec = int(row.end_sec)
            target_center = (end_sec - WINDOW_SEC/2) * SR
            clip_start = int(max(0, target_center - CLIP_SAMPLES/2))
            clip_start = min(clip_start, FILE_SAMPLES - CLIP_SAMPLES)
            clip = wav[clip_start:clip_start + CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            batch_wavs.append(clip.astype(np.float32))
            for l in row.lbls:
                if l in l2i: Y[k, l2i[l]] = 1
        x = torch.from_numpy(np.stack(batch_wavs)).to(dev)
        clip_logits, _ = model(x)
        p = torch.sigmoid(clip_logits).cpu().numpy().astype(np.float32)
        if not np.isfinite(p).all():
            p = np.nan_to_num(p, 0.5)
        preds[i:j] = p
    aucs = []
    for c in range(n_cls):
        if Y[:, c].sum() == 0 or Y[:, c].sum() == n: continue
        try: aucs.append(roc_auc_score(Y[:, c], preds[:, c]))
        except Exception: pass
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs), preds, Y


def main():
    set_seed(SEED)
    primary, l2i = build_primaries()
    ta_train, ta_val = build_train_audio_splits(l2i, val_frac=0.20)
    ss_train, ss_eval = build_ss_splits(l2i)
    print(f"Total train: train_audio {len(ta_train)} + labeled SS train {len(ss_train)}")
    print(f"Eval: train_audio held-out {len(ta_val)}, labeled SS held-out {len(ss_eval)}")

    # Weighted mix: each epoch sees all TA + multiple passes of SS
    ta_ds = TrainAudioDataset(ta_train, l2i, train=True)
    ss_ds = LabeledSSDataset(ss_train, l2i, train=True)
    from torch.utils.data import ConcatDataset
    combined_ds = ConcatDataset([ta_ds, ss_ds])
    # Sampler: 85% train_audio, 15% labeled SS per batch (equal per-sample unless we weight)
    # Simple weighted sampler: SS gets upweighted by (|TA|/|SS|)*0.18 → 15% batch share
    ta_w = 1.0
    ss_w = (len(ta_ds) / max(1, len(ss_ds))) * (0.15 / 0.85)
    weights = [ta_w] * len(ta_ds) + [ss_w] * len(ss_ds)
    sampler = WeightedRandomSampler(weights, num_samples=len(ta_ds), replacement=True)
    loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         drop_last=True, persistent_workers=True)

    n_cls = len(l2i)
    model = SEDModel(backbone=BACKBONE, n_cls=n_cls).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP_EPOCHS)
    history = []
    best = {"val_TA": -1, "epoch": 0}
    patience = 0

    for ep in range(1, EPOCHS + 1):
        # linear warmup
        if ep <= WARMUP_EPOCHS:
            lr = LR * ep / WARMUP_EPOCHS
            for pg in opt.param_groups: pg["lr"] = lr
        t0 = time.time()
        tr_loss, nan_s = train_epoch(model, loader, opt, DEVICE)
        if ep > WARMUP_EPOCHS: sched.step()
        cur_lr = opt.param_groups[0]["lr"]
        val_TA_auc, n_TA, _, _ = evaluate_train_audio(model, ta_val, l2i, DEVICE, max_samples=2000)
        val_SS_auc, n_SS, _, _ = evaluate_ss(model, ss_eval, l2i, DEVICE)
        dt = time.time() - t0
        history.append({"epoch": ep, "lr": cur_lr, "loss": tr_loss,
                         "val_TA": val_TA_auc, "n_TA_cls": n_TA,
                         "val_SS": val_SS_auc, "n_SS_cls": n_SS,
                         "time_s": dt, "nan_skip": nan_s})
        print(f"  ep {ep:02d}  lr {cur_lr:.5f}  loss {tr_loss:.4f}  "
              f"val_TA {val_TA_auc:.4f} ({n_TA})  val_SS {val_SS_auc:.4f} ({n_SS})  "
              f"nan_s={nan_s}  ({dt:.0f}s)")

        if val_TA_auc > best["val_TA"]:
            best = {"val_TA": val_TA_auc, "val_SS": val_SS_auc, "epoch": ep}
            torch.save({"state_dict": model.state_dict(), "epoch": ep,
                        "val_TA": val_TA_auc, "val_SS": val_SS_auc,
                        "config": {"backbone": BACKBONE, "clip_sec": CLIP_SEC,
                                   "lr": LR, "epochs": EPOCHS,
                                   "seed": SEED}},
                       OUT / "best_ckpt.pt")
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"Early stop at ep {ep} (best ep {best['epoch']} val_TA {best['val_TA']:.4f})")
                break

    print(f"\nFinal best: ep {best['epoch']}  val_TA {best['val_TA']:.4f}  val_SS {best.get('val_SS', 0):.4f}")

    # Final eval with best ckpt
    ck = torch.load(OUT / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    ta_auc, n_TA, ta_preds, ta_Y = evaluate_train_audio(model, ta_val, l2i, DEVICE, max_samples=5000)
    ss_auc, n_SS, ss_preds, ss_Y = evaluate_ss(model, ss_eval, l2i, DEVICE)
    print(f"Final val_TA = {ta_auc:.4f} ({n_TA} classes)  val_SS = {ss_auc:.4f} ({n_SS} classes)")

    np.savez_compressed(OUT / "val_scores.npz",
                         ta_preds=ta_preds, ta_Y=ta_Y,
                         ss_preds=ss_preds, ss_Y=ss_Y)
    with open(OUT / "history.json", "w") as f:
        json.dump({"history": history, "best": best,
                   "final_ta_auc": ta_auc, "final_ss_auc": ss_auc,
                   "config": {"backbone": BACKBONE, "epochs": EPOCHS, "lr": LR,
                              "mixup_alpha": MIXUP_ALPHA, "seed": SEED}},
                   f, indent=2, default=float)


if __name__ == "__main__":
    main()
