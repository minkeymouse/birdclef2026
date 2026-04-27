#!/usr/bin/env python3
"""
exp36 — SED regularization + backbone sweep.

Problem diagnosis from exp29 history:
  - Best Val-A 0.737 at epoch 3, then drifts 0.66-0.71 for ep4-20
  - Train loss 0.012 at ep20 → train_audio memorization, domain gap to SS field
  - Salman claims 20+ epoch works; our copy overfits by ep3 → missing regularization

Hypothesis: SpecAugment + higher WD + label smoothing + early stop will stabilize
training and push Val-A higher. If HGNet-B0 (same backbone as exp29) improves, the
lever is regularization, not architecture.

Configs (sequential, share regularization recipe):
  A  HGNetV2-B0   (reproduces SED29 backbone, isolates regularization effect)
  B  HGNetV2-B1   (scale-up test)
  C  EfficientNet-B0  (family diversity; Boredom 1st vs Salman claim)

For each config:
  - Train up to 25 epochs with early stop (patience 5)
  - Eval: Val-A alone, 2-way blend with SED29, 3-way with Perch+SED29
  - Save best_ckpt.pt, val_scores.npz

Final output: experiments/exp36_outputs/results.json with all configs' metrics.

CLAUDE.md LB discipline: DO NOT push to Kaggle until user reviews results.
"""
from __future__ import annotations
import json
import random
import re
import time
import warnings
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"
EXP28 = ROOT / "experiments" / "exp28_outputs"
EXP29 = ROOT / "experiments" / "exp29_outputs"
OUT = ROOT / "experiments" / "exp36_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Fixed hyperparameters ───────────────────────────────────────────────
SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12

N_FFT = 2048
HOP = 512
N_MELS = 128
FMIN = 50
FMAX = 14000

BATCH_SIZE = 32
MAX_EPOCHS = 25
EARLY_STOP_PATIENCE = 5
LR = 1e-3
WD = 5e-2                # exp29 was 1e-2; higher WD to fight overfitting
NUM_WORKERS = 8
MIXUP_ALPHA = 0.5
MIXUP_P = 0.5
LABEL_SMOOTH = 0.1       # new: target smoothing

# SpecAugment (new in exp36)
SPEC_FREQ_MASK = 24      # mask up to 24 mel bins
SPEC_TIME_MASK = 80      # mask up to 80 frames
SPEC_N_FREQ = 2
SPEC_N_TIME = 2
SPEC_P = 0.7             # apply with prob 0.7

SEED = 42
N_CLASSES = 234
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = [
    {"name": "A_hgnet_b0",   "backbone": "hgnetv2_b0.ssld_stage2_ft_in1k"},
    {"name": "B_hgnet_b1",   "backbone": "hgnetv2_b1.ssld_stage2_ft_in1k"},
    {"name": "C_effnet_b0",  "backbone": "tf_efficientnet_b0.ns_jft_in1k"},
]


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


class AudioDataset(Dataset):
    def __init__(self, df, label_to_idx, train=True, audio_root=DATA / "train_audio"):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.n_classes = len(label_to_idx)
        self.train = train
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

    def _crop(self, y, train):
        if len(y) < CLIP_SAMPLES:
            reps = (CLIP_SAMPLES + len(y) - 1) // len(y)
            y = np.tile(y, reps)[:CLIP_SAMPLES]
        elif len(y) > CLIP_SAMPLES:
            s = np.random.randint(0, len(y) - CLIP_SAMPLES) if train else (len(y) - CLIP_SAMPLES) // 2
            y = y[s:s + CLIP_SAMPLES]
        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self._load(row["filename"])
        y = self._crop(y, self.train)
        target = np.zeros(self.n_classes, dtype=np.float32)
        if row["primary_label"] in self.label_to_idx:
            target[self.label_to_idx[row["primary_label"]]] = 1.0
        for lbl in row["sec_labels"]:
            if lbl in self.label_to_idx:
                target[self.label_to_idx[lbl]] = 0.5
        return torch.from_numpy(y), torch.from_numpy(target)


class ValSSDataset(Dataset):
    def __init__(self, meta_full, ss_root=DATA / "train_soundscapes"):
        self.meta = meta_full.reset_index(drop=True)
        self.ss_root = ss_root

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        fn = row["filename"]
        end_sec = int(row["row_id"].rsplit("_", 1)[1])
        start_sec = end_sec - WINDOW_SEC
        y, sr = sf.read(self.ss_root / fn, dtype="float32", always_2d=False)
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
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x):
        m = self.mel(x); m = self.adb(m); return m.unsqueeze(1)


class SpecAugment(nn.Module):
    """Train-only freq+time masking on log-mel spectrograms."""
    def __init__(self):
        super().__init__()
        self.freq = nn.ModuleList([torchaudio.transforms.FrequencyMasking(SPEC_FREQ_MASK)
                                    for _ in range(SPEC_N_FREQ)])
        self.time = nn.ModuleList([torchaudio.transforms.TimeMasking(SPEC_TIME_MASK)
                                    for _ in range(SPEC_N_TIME)])

    def forward(self, m):
        # m: (B, 1, n_mels, frames)
        if not self.training or random.random() > SPEC_P:
            return m
        for mask in self.freq: m = mask(m)
        for mask in self.time: m = mask(m)
        return m


class SEDHead(nn.Module):
    def __init__(self, feat_dim, n_classes):
        super().__init__()
        self.att = nn.Conv1d(feat_dim, n_classes, 1)
        self.cla = nn.Conv1d(feat_dim, n_classes, 1)

    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values


class SEDModel(nn.Module):
    def __init__(self, backbone_name, n_classes=N_CLASSES):
        super().__init__()
        self.mel = MelExtractor()
        self.spec_aug = SpecAugment()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=1,
            num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(feat.shape[1], n_classes)

    def forward(self, x):
        m = self.mel(x)
        m = self.spec_aug(m)
        m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        feat = feat.mean(dim=2) if feat.dim() == 4 else feat
        clip, fmax = self.head(feat)
        return clip, fmax


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = torch.maximum(lam * y, (1 - lam) * y[idx])
    return mixed_x, mixed_y


def smooth_targets(y, eps=LABEL_SMOOTH):
    # Multi-label smoothing: pull targets toward eps/2 (neg) and 1-eps/2 (pos)
    return y * (1.0 - eps) + 0.5 * eps


def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def build_val_truth():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary)}

    def parse_lbls(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc_clean = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
                .apply(lambda s: sorted({lbl for x in s for lbl in parse_lbls(x)}))
                .reset_index(name="label_list"))
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (sc_clean["filename"].str.replace(".ogg", "", regex=False)
                          + "_" + sc_clean["end_sec"].astype(str))
    meta_full = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    sc_idx = sc_clean.set_index("row_id")
    Y_SC = np.zeros((len(sc_clean), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_clean["label_list"]):
        for lbl in labs:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1
    Y_FULL = np.stack([Y_SC[sc_idx.index.get_loc(rid)] for rid in meta_full["row_id"]])
    return meta_full, Y_FULL


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
    total = 0.0; n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        if random.random() < MIXUP_P:
            x, y = mixup_data(x, y)
        y = smooth_targets(y)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            clip, fmax = model(x)
            loss = F.binary_cross_entropy_with_logits(clip, y) + \
                   F.binary_cross_entropy_with_logits(fmax, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0); n += x.size(0)
    return total / n


def train_config(cfg, train_df, label_to_idx, val_meta, Y_FULL):
    name = cfg["name"]; backbone = cfg["backbone"]
    cfg_out = OUT / name
    cfg_out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n Config {name}  backbone={backbone}\n{'='*60}")

    set_seed(SEED)
    train_ds = AudioDataset(train_df, label_to_idx, train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, drop_last=True)
    val_ds = ValSSDataset(val_meta)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = SEDModel(backbone, N_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    history = []
    best_auc = -1.0; best_ep = -1; best_preds = None
    no_improve = 0
    t0 = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        ep_t = time.time()
        loss = train_one_epoch(model, train_loader, opt, DEVICE)
        sched.step()
        preds = evaluate(model, val_loader, DEVICE, N_CLASSES)
        auc = macro_auc(Y_FULL, preds)
        dt = time.time() - ep_t
        print(f"  ep {ep:02d}  loss {loss:.4f}  val_auc {auc:.4f}  ({dt:.0f}s)")
        history.append({"epoch": ep, "loss": loss, "val_auc": auc, "time_s": dt})

        if auc > best_auc:
            best_auc = auc; best_ep = ep; best_preds = preds; no_improve = 0
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "backbone": backbone, "val_auc": auc}, cfg_out / "best_ckpt.pt")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  early stop (no improve {EARLY_STOP_PATIENCE} epochs)")
                break

    if best_preds is not None:
        np.savez_compressed(cfg_out / "val_scores.npz", preds=best_preds)
    elapsed = time.time() - t0
    print(f"  best {best_auc:.4f} at ep {best_ep}, {elapsed/60:.1f} min")
    return {"name": name, "backbone": backbone, "best_val_auc": best_auc,
            "best_epoch": best_ep, "history": history, "elapsed_min": elapsed/60}


# ─── Blend eval ─────────────────────────────────────────────────────────

def zscore(x):
    m = x.mean(0, keepdims=True); s = x.std(0, keepdims=True) + 1e-6
    return (x - m) / s


def eval_blends(Y_FULL, results):
    """For each successfully-trained config, measure blend Val-A against SED29 and Perch."""
    e28 = np.load(EXP28 / "best_oof.npz")
    perch_a = e28["val_a_smoothed"].astype(np.float32)
    sed29 = np.load(EXP29 / "val_scores.npz")["preds"].astype(np.float32)
    Pa = zscore(perch_a); S29 = zscore(sed29)

    summary = {"perch_alone": macro_auc(Y_FULL, perch_a),
               "sed29_alone": macro_auc(Y_FULL, sed29),
               "perch_sed29_a80": macro_auc(Y_FULL, 0.8*Pa + 0.2*S29)}
    print(f"\nReference:")
    for k, v in summary.items():
        print(f"  {k:30s}  Val-A {v:.4f}")

    for r in results:
        name = r["name"]
        pred_path = OUT / name / "val_scores.npz"
        if not pred_path.exists():
            continue
        new = np.load(pred_path)["preds"].astype(np.float32)
        SN = zscore(new)
        r["alone"] = macro_auc(Y_FULL, new)
        # 2-way: new + SED29
        best2 = -1.0; best2_w = None
        for w in np.arange(0.0, 1.01, 0.1):
            auc = macro_auc(Y_FULL, w*SN + (1-w)*S29)
            if auc > best2: best2 = auc; best2_w = w
        r["best_2way_new_sed29"] = best2
        r["best_2way_w_new"] = float(best2_w)
        # 2-way: Perch + new
        bestPN = -1.0; bestPN_w = None
        for w in np.arange(0.5, 1.01, 0.05):
            auc = macro_auc(Y_FULL, w*Pa + (1-w)*SN)
            if auc > bestPN: bestPN = auc; bestPN_w = w
        r["best_2way_perch_new"] = bestPN
        r["best_2way_w_perch"] = float(bestPN_w)
        # 3-way: Perch + SED29 + new simplex
        best3 = -1.0; best3_w = None
        for wp in np.arange(0.6, 0.91, 0.05):
            for w29 in np.arange(0.0, 1.0 - wp + 0.001, 0.05):
                wn = 1.0 - wp - w29
                if wn < -1e-6: continue
                auc = macro_auc(Y_FULL, wp*Pa + w29*S29 + wn*SN)
                if auc > best3:
                    best3 = auc; best3_w = (float(wp), float(w29), float(wn))
        r["best_3way"] = best3
        r["best_3way_w"] = best3_w

        print(f"\n{name}:")
        print(f"  alone               Val-A {r['alone']:.4f}")
        print(f"  2way (new+SED29)    Val-A {best2:.4f}  w_new={best2_w:.2f}")
        print(f"  2way (Perch+new)    Val-A {bestPN:.4f}  w_perch={bestPN_w:.2f}")
        print(f"  3way (P+SED29+new)  Val-A {best3:.4f}  w={best3_w}")
    return summary


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    t_all = time.time()
    print(f"Device: {DEVICE}  MAX_EPOCHS={MAX_EPOCHS}  WD={WD}  SpecAugment={SPEC_FREQ_MASK}/{SPEC_TIME_MASK}")

    primary, label_to_idx, train_df = load_labels()
    print(f"train_audio rows: {len(train_df)}, unique classes: {train_df['primary_label'].nunique()}")
    val_meta, Y_FULL = build_val_truth()
    print(f"val rows: {len(val_meta)}, active classes: {(Y_FULL.sum(0)>0).sum()}")

    results = []
    for cfg in CONFIGS:
        try:
            r = train_config(cfg, train_df, label_to_idx, val_meta, Y_FULL)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": cfg["name"], "error": str(e)})
        # Save partial results after each config
        (OUT / "results.json").write_text(json.dumps({
            "elapsed_min": (time.time() - t_all) / 60,
            "configs": CONFIGS,
            "results": results,
        }, indent=2))

    print("\n=== Blend evaluation ===")
    summary = eval_blends(Y_FULL, results)
    (OUT / "results.json").write_text(json.dumps({
        "elapsed_min": (time.time() - t_all) / 60,
        "configs": CONFIGS,
        "results": results,
        "reference": summary,
    }, indent=2))
    print(f"\nDone. Total {(time.time()-t_all)/60:.1f} min.")


if __name__ == "__main__":
    main()
