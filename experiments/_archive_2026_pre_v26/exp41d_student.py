#!/usr/bin/env python3
"""exp41d — Student SED with pseudo-labels + 2025 1st prize recipe.

Recipe:
  Teacher: SED29 (HGNet B0, train_audio only) — already done
  Student: HGNet B0 (same arch, train from scratch)
  Data: train_audio (35549) + labeled SS train (55 files) + pseudo (18,436 chunks)
  Mixup ratio 1.0: every sample mixed with random pseudo chunk
  drop_path: 0.15
  WeightedSampler: pseudo chunks weighted by (weight column)
  Eval: 11 held-out labeled SS files (same as exp38)

Output:
  experiments/exp41d_outputs/best_ckpt.pt
  experiments/exp41d_outputs/val_scores.npz (Val-A 59 files)
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
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"
EXP41 = ROOT / "experiments" / "exp41_outputs"
OUT = ROOT / "experiments" / "exp41d_outputs"
OUT.mkdir(exist_ok=True)

SR = 32000; CLIP_SEC = 20; CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5
N_WINDOWS = 12; N_CLASSES = 234
N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000

BATCH_SIZE = 32; EPOCHS = 20
LR = 1e-3; WD = 1e-2
NUM_WORKERS = 8
MIXUP_P = 1.0   # 2025 1st key finding: always mix with pseudo
MIXUP_ALPHA = 0.5
DROP_PATH = 0.15  # stochastic depth
SS_TRAIN_SHARE = 0.20   # labeled SS train portion of batch
PSEUDO_SHARE = 0.30     # pseudo portion of batch
EARLY_STOP = 5
EVAL_N_FILES = 11
SEED = 42

DEVICE = "cuda"
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ─── Data ─────────────────────────────────────────────────────────────
def build_ss():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)

    rng = np.random.RandomState(SEED)
    files = sorted(sc.filename.unique())
    rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    train_files = set(files[EVAL_N_FILES:])

    sc_train = sc[sc.filename.isin(train_files)].reset_index(drop=True)
    sc_eval = sc[sc.filename.isin(eval_files)].reset_index(drop=True)

    Y_eval = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["lbls"]):
        for l in labs:
            if l in l2i: Y_eval[i, l2i[l]] = 1
    return sc_train, sc_eval, Y_eval, l2i


def load_train_audio(l2i):
    t = pd.read_csv(DATA / "train.csv")
    t["primary_label"] = t["primary_label"].astype(str)
    def parse_sec(x):
        if pd.isna(x) or x == "[]": return []
        return [s.strip().strip("'\"") for s in str(x).strip("[]").split(",") if s.strip()]
    t["sec_labels"] = t["secondary_labels"].apply(parse_sec)
    return t


class AudioDS(Dataset):
    def __init__(self, df, l2i, root=DATA / "train_audio"):
        self.df = df.reset_index(drop=True); self.l2i = l2i; self.nc = len(l2i); self.root = root
    def __len__(self): return len(self.df)
    def _load(self, fn):
        try:
            y, sr = sf.read(self.root / fn, dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(1)
            if sr != SR:
                y = torchaudio.functional.resample(torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
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
        t = np.zeros(self.nc, dtype=np.float32)
        if row["primary_label"] in self.l2i: t[self.l2i[row["primary_label"]]] = 1.0
        for l in row["sec_labels"]:
            if l in self.l2i: t[self.l2i[l]] = 0.5
        return torch.from_numpy(y), torch.from_numpy(t), 1.0


class SSTrainDS(Dataset):
    def __init__(self, sc, l2i, root=DATA / "train_soundscapes"):
        self.df = sc.reset_index(drop=True); self.l2i = l2i; self.nc = len(l2i); self.root = root
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        end = int(row["end_sec"]); start = end - WINDOW_SEC
        try:
            y, sr = sf.read(self.root / row["filename"], dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(1)
            if sr != SR:
                y = torchaudio.functional.resample(torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
        except Exception:
            y = np.zeros(SR * 60, dtype=np.float32)
        c = ((start + end) // 2) * SR
        half = CLIP_SAMPLES // 2
        s = max(0, c - half); e = s + CLIP_SAMPLES
        if e > len(y): e = len(y); s = max(0, e - CLIP_SAMPLES)
        clip = y[s:e]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        t = np.zeros(self.nc, dtype=np.float32)
        for l in row["lbls"]:
            if l in self.l2i: t[self.l2i[l]] = 1.0
        return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(t), 1.0


class PseudoDS(Dataset):
    """Pseudo-labeled chunks from unlabeled SS."""
    def __init__(self, df, soft_labels, root=DATA / "train_soundscapes"):
        self.df = df.reset_index(drop=True)
        self.labels = soft_labels  # (N, 234)
        self.root = root
        assert len(df) == len(soft_labels)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        end = int(row["end_sec"]); start = end - WINDOW_SEC
        try:
            y, sr = sf.read(self.root / row["filename"], dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(1)
            if sr != SR:
                y = torchaudio.functional.resample(torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
        except Exception:
            y = np.zeros(SR * 60, dtype=np.float32)
        c = ((start + end) // 2) * SR
        half = CLIP_SAMPLES // 2
        s = max(0, c - half); e = s + CLIP_SAMPLES
        if e > len(y): e = len(y); s = max(0, e - CLIP_SAMPLES)
        clip = y[s:e]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(self.labels[idx].astype(np.float32)), float(row["weight"])


class ValDS(Dataset):
    def __init__(self, sc, root=DATA / "train_soundscapes"):
        self.df = sc.reset_index(drop=True); self.root = root
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        end = int(row["end_sec"]); start = end - WINDOW_SEC
        y, sr = sf.read(self.root / row["filename"], dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(1)
        c = ((start + end) // 2) * SR
        half = CLIP_SAMPLES // 2
        s = max(0, c - half); e = s + CLIP_SAMPLES
        if e > len(y): e = len(y); s = max(0, e - CLIP_SAMPLES)
        clip = y[s:e]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        return torch.from_numpy(clip.astype(np.float32)), idx


class MixDS(Dataset):
    """Concatenation with index-based source selection."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.sizes = [len(d) for d in datasets]
        self.offsets = np.cumsum([0] + self.sizes)
    def __len__(self): return int(self.offsets[-1])
    def __getitem__(self, idx):
        for i, (lo, hi) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            if lo <= idx < hi:
                return self.datasets[i][int(idx - lo)]
        raise IndexError


# ─── Model ────────────────────────────────────────────────────────────
class MelExt(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class SEDHead(nn.Module):
    def __init__(self, d, nc):
        super().__init__()
        self.att = nn.Conv1d(d, nc, 1); self.cla = nn.Conv1d(d, nc, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values


class SEDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = MelExt(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(
            BACKBONE, pretrained=True, in_chans=1, num_classes=0, global_pool="",
            drop_path_rate=DROP_PATH)
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(f.shape[1], N_CLASSES)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        if feat.dim() == 4: feat = feat.mean(2)
        return self.head(feat)


def mixup(x, y, w, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return (lam * x + (1 - lam) * x[idx],
            torch.maximum(lam * y, (1 - lam) * y[idx]),
            torch.maximum(lam * w, (1 - lam) * w[idx]))


def macro_auc(Y, S):
    k = Y.sum(0) > 0
    return float(roc_auc_score(Y[:, k], S[:, k], average="macro"))


def evaluate(model, dl, device):
    model.eval()
    preds = np.zeros((len(dl.dataset), N_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for x, idxs in dl:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip, _ = model(x)
            p = torch.sigmoid(clip).float().cpu().numpy()
            for i, j in zip(idxs.tolist(), range(len(p))):
                preds[i] = p[j]
    return preds


def main():
    t0 = time.time()
    set_seed(SEED)
    print(f"[exp41d] drop_path={DROP_PATH}  mixup_p={MIXUP_P}")

    # Load data
    sc_train, sc_eval, Y_eval, l2i = build_ss()
    print(f"SS split: train {sc_train.filename.nunique()} files  eval {sc_eval.filename.nunique()} files")

    tdf = load_train_audio(l2i)
    print(f"train_audio: {len(tdf)} rows")

    # Pseudo
    pseudo_df = pd.read_parquet(EXP41 / "pseudo_train_df.parquet")
    pseudo_labels = np.load(EXP41 / "pseudo_soft_labels.npz")["labels"]
    print(f"pseudo: {len(pseudo_df)} rows")

    a_ds = AudioDS(tdf, l2i)
    s_ds = SSTrainDS(sc_train, l2i)
    p_ds = PseudoDS(pseudo_df, pseudo_labels)
    mix = MixDS([a_ds, s_ds, p_ds])

    # Weighted sampler: composition a : s : p
    a_frac = 1.0 - SS_TRAIN_SHARE - PSEUDO_SHARE
    wa = a_frac / len(a_ds)
    ws = SS_TRAIN_SHARE / len(s_ds)
    # Pseudo weights: by row weight (confidence) within pseudo share
    pw = pseudo_df["weight"].values.astype(np.float64)
    pw = (pw / pw.sum()) * PSEUDO_SHARE
    weights = np.concatenate([
        np.full(len(a_ds), wa, dtype=np.float64),
        np.full(len(s_ds), ws, dtype=np.float64),
        pw,
    ])
    # Samples per epoch: match train_audio size for steady iteration count
    sampler = WeightedRandomSampler(weights, num_samples=len(a_ds), replacement=True)

    dl_train = DataLoader(mix, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, drop_last=True)
    dl_eval = DataLoader(ValDS(sc_eval), batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f"Per-epoch samples: {len(a_ds)}  (audio {a_frac:.0%} + SS {SS_TRAIN_SHARE:.0%} + pseudo {PSEUDO_SHARE:.0%})")

    model = SEDModel().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_auc = -1.0; best_ep = -1; best_preds = None; no_imp = 0
    history = []
    for ep in range(1, EPOCHS + 1):
        ep_t = time.time()
        model.train()
        total = 0.0; n = 0
        for x, y, w in dl_train:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            w = w.to(DEVICE, non_blocking=True).float()
            if random.random() < MIXUP_P:
                x, y, w = mixup(x, y, w)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip, fmax = model(x)
                # weighted BCE (per-sample weight on loss)
                loss_c = F.binary_cross_entropy_with_logits(clip, y, reduction="none").mean(1)
                loss_f = F.binary_cross_entropy_with_logits(fmax, y, reduction="none").mean(1)
                loss = ((loss_c + loss_f) * w).mean()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        preds = evaluate(model, dl_eval, DEVICE)
        auc = macro_auc(Y_eval, preds)
        dt = time.time() - ep_t
        avg = total / n
        print(f"Ep {ep:02d}/{EPOCHS}  loss {avg:.4f}  val_auc {auc:.4f}  ({dt:.0f}s)")
        history.append({"epoch": ep, "loss": avg, "val_auc": auc, "time_s": dt})
        if auc > best_auc:
            best_auc = auc; best_ep = ep; best_preds = preds; no_imp = 0
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "backbone": BACKBONE, "val_auc": auc}, OUT / "best_ckpt.pt")
        else:
            no_imp += 1
        (OUT / "results.json").write_text(json.dumps(
            {"history": history, "best_epoch": best_ep, "best_val_auc": best_auc,
             "elapsed_s": time.time() - t0}, indent=2))
        if no_imp >= EARLY_STOP:
            print(f"Early stop at ep {ep}")
            break

    if best_preds is not None:
        np.savez_compressed(OUT / "eval_scores.npz", preds=best_preds, Y_eval=Y_eval)
    print(f"\nDone. Best Val-A_v2 = {best_auc:.4f} at ep {best_ep}. {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
