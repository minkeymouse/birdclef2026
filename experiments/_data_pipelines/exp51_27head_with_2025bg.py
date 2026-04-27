#!/usr/bin/env python3
"""exp51 — 27-species dedicated head retrained with 2025 BG mixing.

Rebuilds exp44g's architecture (HGNet-B0 SED trained only on labeled SS 55
files for 27 double-blind species = 25 Insecta sonotypes + 2 Amphibia) but
uses 2025 soundscapes as the background-mix source instead of 2026 same-pool.
Goal: break the site-shortcut that caused v19 LB −0.022.

Recipe from exp44g:
  HGNet-B0, fp32, BN2d(N_MELS), LR 1e-4, pos oversample 6x
  synth_p=0.75, α~U(0.4, 0.8)
  PLUS: background sourced from 2025 quiet clips (exp49 BG pool)
"""
from __future__ import annotations
import json, random, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
BG_PATH = ROOT / "experiments" / "exp49_outputs" / "bg_quiet_2025.npz"
OUT = ROOT / "experiments" / "exp51_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000; CLIP_SEC = 20; CLIP_SAMPLES = SR * CLIP_SEC
FILE_SAMPLES = SR * 60; WIN_SEC = 5
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000
BATCH = 16; EPOCHS = 15
LR = 1e-4; WD = 1e-2
SYNTH_P = 0.75
BG_ALPHA_LO, BG_ALPHA_HI = 0.4, 0.8
POS_OVERSAMPLE = 6
WARMUP = 1; PATIENCE = 6
EVAL_N_FILES = 11; SEED = 42
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"; DEVICE = "cuda"

# 25 Insecta sonotypes + 2 Amphibia (exp44c target set)
TARGET_SPECIES = [f"47158son{i:02d}" for i in range(1, 26)] + ["47143", "47147"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_splits():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    # Filter TARGET to those present in primary
    target = [t for t in TARGET_SPECIES if t in primary]
    print(f"Target species: {len(target)}  (from TARGET_SPECIES: {len(TARGET_SPECIES)})")
    t2i = {t: i for i, t in enumerate(target)}

    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g.filename.unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES]); train_files = set(files[EVAL_N_FILES:])
    tr = sc_g[sc_g.filename.isin(train_files)].reset_index(drop=True)
    ev = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)

    def make_Y(df):
        Y = np.zeros((len(df), len(target)), dtype=np.float32)
        for i, labs in enumerate(df.lbls):
            for l in labs:
                if l in t2i: Y[i, t2i[l]] = 1
        return Y
    Y_tr = make_Y(tr); Y_ev = make_Y(ev)
    print(f"train: {len(tr)} rows  pos per class: {Y_tr.sum(axis=0).astype(int)}")
    print(f"eval : {len(ev)} rows  pos per class: {Y_ev.sum(axis=0).astype(int)}")
    return tr, Y_tr, ev, Y_ev, target, t2i


def load_clip(path, end_sec, train=True):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != SR:
            import torchaudio.functional as TF
            wav = TF.resample(torch.from_numpy(wav), sr, SR).numpy()
    except Exception:
        return np.zeros(CLIP_SAMPLES, dtype=np.float32)
    end_samp = end_sec * SR
    center = end_samp - (WIN_SEC / 2) * SR
    cs = int(max(0, center - CLIP_SAMPLES / 2))
    cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
    if train: cs = max(0, min(FILE_SAMPLES - CLIP_SAMPLES, cs + random.randint(-SR, SR)))
    clip = wav[cs:cs + CLIP_SAMPLES]
    if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
    return clip.astype(np.float32)


class Ds(Dataset):
    def __init__(self, df, Y, train=True):
        self.df = df.reset_index(drop=True); self.Y = Y; self.train = train
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip = load_clip(DATA / "train_soundscapes" / row.filename, int(row.end_sec), self.train)
        y = self.Y[idx]
        return torch.from_numpy(clip), torch.from_numpy(y)


class SEDModel(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=40)
        self.backbone = timm.create_model(BACKBONE, pretrained=True, in_chans=1,
                                          drop_rate=0.1, drop_path_rate=0.1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        C = feat.shape[1]
        self.att = nn.Conv1d(C, n_cls, 1)
        self.cla = nn.Conv1d(C, n_cls, 1)
    def forward(self, x, aug=False):
        m = self.adb(self.mel(x)).unsqueeze(1)
        m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        if aug and self.training: m = self.tm(self.fm(m))
        f = self.backbone(m)
        f = f.mean(dim=2) if f.dim() == 4 else f
        a = self.att(f); c = self.cla(f)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values


def main():
    set_seed(SEED)
    tr_df, Y_tr, ev_df, Y_ev, target, t2i = build_splits()
    n_cls = len(target)

    # Load 2025 BG pool
    bg_pool = None
    if BG_PATH.exists():
        bg_pool = np.load(BG_PATH)["windows"]
        print(f"2025 BG pool loaded: {bg_pool.shape}")
    else:
        print("WARNING: BG not found")

    # Per-sample weight: 6x for any positive row
    pos_mask = Y_tr.any(axis=1)
    weights = np.where(pos_mask, POS_OVERSAMPLE, 1).astype(np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(),
                                     num_samples=len(tr_df) * 2, replacement=True)
    ds = Ds(tr_df, Y_tr, train=True)
    loader = DataLoader(ds, batch_size=BATCH, sampler=sampler, num_workers=4,
                        pin_memory=True, drop_last=True, persistent_workers=True)

    model = SEDModel(n_cls=n_cls).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP)

    best = -1; pat = 0; hist = []
    for ep in range(1, EPOCHS + 1):
        if ep <= WARMUP:
            for pg in opt.param_groups: pg["lr"] = LR * ep / WARMUP
        t0 = time.time()
        model.train(); t_loss, t_n, nan_s = 0, 0, 0
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            if not torch.isfinite(x).all(): x = torch.nan_to_num(x, 0, 1, -1)
            # Synthetic BG mix (from 2025) when target positive
            if bg_pool is not None and random.random() < SYNTH_P:
                B = x.size(0)
                idx = np.random.randint(0, bg_pool.shape[0], size=B)
                bg = np.stack([bg_pool[i] for i in idx])
                # tile 5s × 4 → 20s
                bg_tiled = np.tile(bg, (1, 4))[:, :CLIP_SAMPLES]
                bg_t = torch.from_numpy(bg_tiled.astype(np.float32)).to(DEVICE)
                lam = np.random.uniform(BG_ALPHA_LO, BG_ALPHA_HI)
                # Mix: α * source + (1-α) * BG, labels unchanged
                x = lam * x + (1 - lam) * bg_t
            clip, fmax = model(x, aug=True)
            if not torch.isfinite(clip).all():
                nan_s += 1; opt.zero_grad(set_to_none=True); continue
            loss = F.binary_cross_entropy_with_logits(clip, y) + \
                   F.binary_cross_entropy_with_logits(fmax, y)
            if not torch.isfinite(loss):
                nan_s += 1; opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn): nan_s += 1; continue
            opt.step()
            t_loss += loss.item() * x.size(0); t_n += x.size(0)
        if ep > WARMUP: sched.step()

        # Eval
        model.eval()
        ev_preds = np.zeros((len(ev_df), n_cls), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(ev_df), BATCH):
                j = min(len(ev_df), i + BATCH)
                wavs = [load_clip(DATA / "train_soundscapes" / ev_df.iloc[k].filename,
                                   int(ev_df.iloc[k].end_sec), train=False) for k in range(i, j)]
                x = torch.from_numpy(np.stack(wavs)).to(DEVICE)
                clip, _ = model(x)
                ev_preds[i:j] = torch.sigmoid(clip).cpu().numpy()
        aucs = []
        for c in range(n_cls):
            if Y_ev[:, c].sum() == 0 or Y_ev[:, c].sum() == len(Y_ev): continue
            try: aucs.append(roc_auc_score(Y_ev[:, c], ev_preds[:, c]))
            except: pass
        ev_auc = float(np.mean(aucs)) if aucs else 0
        dt = time.time() - t0
        hist.append({"epoch": ep, "loss": t_loss/t_n if t_n else 0,
                     "val_auc": ev_auc, "n_cls": len(aucs), "nan_s": nan_s, "time_s": dt})
        print(f"  ep {ep:02d}  loss {t_loss/max(t_n,1):.4f}  val {ev_auc:.4f} ({len(aucs)})  nan={nan_s}  {dt:.0f}s", flush=True)
        if ev_auc > best:
            best = ev_auc; pat = 0
            torch.save({"state_dict": model.state_dict(),
                        "target_species": target, "epoch": ep, "val_auc": ev_auc,
                        "config": {"bg_source": "2025", "synth_p": SYNTH_P, "seed": SEED}},
                       OUT / "best_ckpt.pt")
        else:
            pat += 1
            if pat >= PATIENCE: print(f"early stop ep {ep}"); break
    print(f"\nBest val_auc: {best:.4f}")
    with open(OUT / "history.json", "w") as f:
        json.dump({"history": hist, "best_auc": best, "target_species": target}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
