"""
exp31 — HGNetV2-B0 SED + background mixing (Salman recipe v2).

Delta vs exp29 (which peaked at epoch 3, Val-A 0.737):
  1. Background mixing: random unlabeled SS clip added at SNR 5-15 dB, p=0.8
     → attack the clean→field domain shift
  2. SpecAugment on log-mel (2 freq masks + 2 time masks)
  3. Cap per-class train samples at 300 (prevent dominant Aves classes saturating mixup)
  4. 25 epochs (vs 20) since the original 20-epoch run had already overfit by epoch 3

Eval: same 59 labeled SS × 12 windows (708 rows) Val-A, direct comparison to exp29.

Pre-loading: to keep CPU I/O sustainable, pre-cache 2048 random 20s BG clips to RAM (~5 GB at float32).
"""
from __future__ import annotations
import json
import random
import time
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
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"
OUT = ROOT / "experiments" / "exp31_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5
N_WINDOWS = 12

N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 50
FMAX = 14000

BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
WD = 1e-2
NUM_WORKERS = 8
MIXUP_ALPHA = 0.5
BG_MIX_P = 0.8           # prob of adding background
BG_SNR_RANGE = (5.0, 15.0)
CAP_PER_CLASS = 300
BG_POOL_SIZE = 2048       # pre-cached unlabeled SS clips in RAM
SPEC_MASK_P = 0.5
FREQ_MASK_F = 20
TIME_MASK_T = 40
N_FREQ_MASKS = 2
N_TIME_MASKS = 2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
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

    # Cap per-class
    pieces = []
    for _, g in train.groupby("primary_label", sort=False):
        pieces.append(g.sample(min(len(g), CAP_PER_CLASS), random_state=SEED))
    capped = pd.concat(pieces, ignore_index=True)
    return primary, label_to_idx, capped


def get_unlabeled_ss_files():
    """All SS files minus the 66 labeled ones."""
    labeled = set(pd.read_csv(DATA / "train_soundscapes_labels.csv")["filename"].unique())
    all_ss = set(p.name for p in (DATA / "train_soundscapes").glob("*.ogg"))
    return sorted(all_ss - labeled)


def precache_backgrounds(files, n_clips):
    """Load n_clips random 20s segments from random unlabeled SS files into a numpy array."""
    rng = np.random.default_rng(SEED)
    picks = rng.choice(files, size=n_clips, replace=(len(files) < n_clips))
    arr = np.zeros((n_clips, CLIP_SAMPLES), dtype=np.float32)
    for i, fn in enumerate(tqdm(picks, desc="Precaching BG")):
        try:
            y, sr = sf.read(DATA / "train_soundscapes" / fn, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            if len(y) >= CLIP_SAMPLES:
                s = rng.integers(0, len(y) - CLIP_SAMPLES + 1)
                arr[i] = y[s:s + CLIP_SAMPLES]
            else:
                reps = (CLIP_SAMPLES + len(y) - 1) // len(y)
                arr[i] = np.tile(y, reps)[:CLIP_SAMPLES]
        except Exception:
            pass
    return arr


class AudioDataset(Dataset):
    def __init__(self, df, label_to_idx, bg_arr=None, train=True):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.n_classes = len(label_to_idx)
        self.train = train
        self.bg_arr = bg_arr  # shared (read-only) numpy array, workers inherit via fork
        self.audio_root = DATA / "train_audio"

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

    def _clip(self, y, train):
        if len(y) < CLIP_SAMPLES:
            reps = (CLIP_SAMPLES + len(y) - 1) // len(y)
            y = np.tile(y, reps)[:CLIP_SAMPLES]
        elif len(y) > CLIP_SAMPLES:
            s = np.random.randint(0, len(y) - CLIP_SAMPLES) if train else (len(y) - CLIP_SAMPLES) // 2
            y = y[s:s + CLIP_SAMPLES]
        return y

    def _add_bg(self, y):
        if self.bg_arr is None or np.random.rand() > BG_MIX_P:
            return y
        bg = self.bg_arr[np.random.randint(len(self.bg_arr))]
        # SNR mixing: scale bg so that snr(y, bg) = target
        snr = np.random.uniform(*BG_SNR_RANGE)
        p_y = (y ** 2).mean() + 1e-10
        p_b = (bg ** 2).mean() + 1e-10
        scale = np.sqrt(p_y / p_b / (10 ** (snr / 10)))
        return (y + scale * bg).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self._load(row["filename"])
        y = self._clip(y, self.train)
        if self.train:
            y = self._add_bg(y)

        target = np.zeros(self.n_classes, dtype=np.float32)
        if row["primary_label"] in self.label_to_idx:
            target[self.label_to_idx[row["primary_label"]]] = 1.0
        for lbl in row["sec_labels"]:
            if lbl in self.label_to_idx:
                target[self.label_to_idx[lbl]] = 0.5
        return torch.from_numpy(y), torch.from_numpy(target)


class ValSSDataset(Dataset):
    def __init__(self, meta_full):
        self.meta = meta_full.reset_index(drop=True)
        self.root = DATA / "train_soundscapes"

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        fn = row["filename"]
        end_sec = int(row["row_id"].rsplit("_", 1)[1])
        start_sec = end_sec - WINDOW_SEC
        y, sr = sf.read(self.root / fn, dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(axis=1)
        center_sample = ((start_sec + end_sec) // 2) * SR
        half = CLIP_SAMPLES // 2
        s = max(0, center_sample - half)
        e = s + CLIP_SAMPLES
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
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x):
        return self.adb(self.mel(x)).unsqueeze(1)


def spec_augment(m, p=SPEC_MASK_P):
    """m: (B, 1, n_mels, frames). In-place masking."""
    if not m.requires_grad and torch.rand(1).item() > p:
        return m
    B, _, F_, T = m.shape
    for _ in range(N_FREQ_MASKS):
        f = np.random.randint(0, FREQ_MASK_F)
        f0 = np.random.randint(0, max(1, F_ - f))
        m[:, :, f0:f0+f, :] = m.mean()
    for _ in range(N_TIME_MASKS):
        t = np.random.randint(0, TIME_MASK_T)
        t0 = np.random.randint(0, max(1, T - t))
        m[:, :, :, t0:t0+t] = m.mean()
    return m


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
    def __init__(self):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(BACKBONE, pretrained=True, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(feat.shape[1], N_CLASSES)

    def forward(self, x, training=True):
        m = self.mel(x)
        m = m.transpose(1, 2)
        m = self.bn0(m)
        m = m.transpose(1, 2)
        if training and self.training:
            m = spec_augment(m)
        feat = self.backbone(m)
        feat = feat.mean(dim=2) if feat.dim() == 4 else feat
        return self.head(feat)


# ─── Mixup ───────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], torch.maximum(lam * y, (1 - lam) * y[idx])


# ─── Training/Eval ───────────────────────────────────────────────────────

def macro_auc(y_true, y_score):
    keep = y_true.sum(0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def build_val_truth():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sample_sub.columns[1:].tolist()

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    meta = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    lab2idx = {c: i for i, c in enumerate(primary)}
    Y = np.zeros((len(meta), len(primary)), dtype=np.uint8)
    by_rowid = sc.set_index("row_id")
    for i, rid in enumerate(meta["row_id"]):
        if rid in by_rowid.index:
            for l in by_rowid.loc[rid, "lbls"]:
                if l in lab2idx:
                    Y[i, lab2idx[l]] = 1
    return meta, Y


def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train"):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        x, y = mixup_data(x, y)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            clip, fmax = model(x)
            loss = F.binary_cross_entropy_with_logits(clip, y) + \
                   F.binary_cross_entropy_with_logits(fmax, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    preds = np.zeros((len(loader.dataset), n_classes), dtype=np.float32)
    for x, idx in tqdm(loader, desc="eval"):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            clip, _ = model(x)
        p = torch.sigmoid(clip).float().cpu().numpy()
        preds[idx.numpy()] = p
    return preds


def main():
    t0 = time.time()
    set_seed(SEED)

    primary, label_to_idx, train_df = load_labels()
    print(f"train_audio rows (capped): {len(train_df)}, classes: {train_df['primary_label'].nunique()}")

    bg_files = get_unlabeled_ss_files()
    print(f"Unlabeled SS pool: {len(bg_files)} → precaching {BG_POOL_SIZE}…")
    bg_arr = precache_backgrounds(bg_files, BG_POOL_SIZE)
    print(f"  BG array shape: {bg_arr.shape}, size: {bg_arr.nbytes/1e9:.2f} GB")

    train_ds = AudioDataset(train_df, label_to_idx, bg_arr=bg_arr, train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, drop_last=True)

    meta_full, Y_FULL = build_val_truth()
    val_ds = ValSSDataset(meta_full)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=False)
    print(f"val rows: {len(val_ds)}  active classes: {(Y_FULL.sum(0)>0).sum()}")

    model = SEDModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    history = []
    best_auc = -1.0
    best_epoch = -1
    best_preds = None

    for ep in range(1, EPOCHS + 1):
        ep_t = time.time()
        loss = train_epoch(model, train_loader, opt, DEVICE)
        sched.step()
        preds = evaluate(model, val_loader, DEVICE, N_CLASSES)
        auc = macro_auc(Y_FULL, preds)
        dt = time.time() - ep_t
        print(f"Ep {ep:02d}/{EPOCHS}  loss {loss:.4f}  val_auc {auc:.4f}  ({dt:.0f}s)")
        history.append({"epoch": ep, "loss": loss, "val_auc": auc, "time_s": dt})
        if auc > best_auc:
            best_auc = auc; best_epoch = ep; best_preds = preds
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "backbone": BACKBONE, "val_auc": auc}, OUT / "best_ckpt.pt")
        (OUT / "results.json").write_text(json.dumps({
            "backbone": BACKBONE, "bg_mix_p": BG_MIX_P, "snr_range": BG_SNR_RANGE,
            "spec_augment": True, "cap_per_class": CAP_PER_CLASS,
            "history": history, "best_epoch": best_epoch, "best_val_auc": best_auc,
            "elapsed_s": time.time() - t0}, indent=2))

    if best_preds is not None:
        np.savez_compressed(OUT / "val_scores.npz", preds=best_preds)
    print(f"\nDone. Best Val-A AUC = {best_auc:.4f} at epoch {best_epoch}. Total {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
