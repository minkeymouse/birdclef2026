#!/usr/bin/env python3
"""
exp38 eval on full 59-file Val-A + substitute blend against SED29.

1. Inference exp38 best_ckpt on 708 Val-A rows (same as exp29 val_scores.npz).
2. Save exp38_outputs/val_scores_full.npz.
3. Run Perch + SED38 blend grid vs Perch + SED29 baseline.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP21 = ROOT / "experiments/exp21_outputs/perch_cache"
EXP28 = ROOT / "experiments/exp28_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP38 = ROOT / "experiments/exp38_outputs"

SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5
N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
N_CLASSES = 234
DEVICE = "cuda"


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True,
        )
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, x):
        m = self.mel(x); m = self.adb(m)
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
    def __init__(self):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(BACKBONE, pretrained=False, in_chans=1, num_classes=0, global_pool="")
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(f.shape[1], N_CLASSES)

    def forward(self, x):
        m = self.mel(x)
        m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        if feat.dim() == 4: feat = feat.mean(2)
        return self.head(feat)


class ValSSDataset(Dataset):
    def __init__(self, meta_full, ss_root=DATA / "train_soundscapes"):
        self.meta = meta_full.reset_index(drop=True)
        self.ss_root = ss_root

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        end_sec = int(row["row_id"].rsplit("_", 1)[1])
        start_sec = end_sec - WINDOW_SEC
        y, sr = sf.read(self.ss_root / row["filename"], dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(1)
        assert sr == SR
        c = ((start_sec + end_sec) // 2) * SR
        half = CLIP_SAMPLES // 2
        s = max(0, c - half); e = s + CLIP_SAMPLES
        if e > len(y): e = len(y); s = max(0, e - CLIP_SAMPLES)
        clip = y[s:e]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        return torch.from_numpy(clip.astype(np.float32)), idx


def macro_auc(Y, S):
    keep = Y.sum(0) > 0
    return float(roc_auc_score(Y[:, keep], S[:, keep], average="macro"))


def zscore(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd


def build_truth_and_meta():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)}))
          .reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    meta = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    Y_sc = np.zeros((len(sc), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc["lbls"]):
        for l in labs:
            if l in l2i: Y_sc[i, l2i[l]] = 1
    sc_idx = sc.set_index("row_id")
    Y = np.stack([Y_sc[sc_idx.index.get_loc(rid)] for rid in meta["row_id"]])
    return meta, Y


def run_inference(meta):
    ds = ValSSDataset(meta)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    model = SEDModel().to(DEVICE)
    ckpt = torch.load(EXP38 / "best_ckpt.pt", map_location=DEVICE)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()
    preds = np.zeros((len(meta), N_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for x, idxs in dl:
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip, _ = model(x)
            p = torch.sigmoid(clip).float().cpu().numpy()
            for i, j in zip(idxs.tolist(), range(len(p))):
                preds[i] = p[j]
    return preds


def main():
    meta, Y = build_truth_and_meta()
    print(f"Val-A meta: {len(meta)} rows, active classes: {int((Y.sum(0)>0).sum())}")

    print("=== exp38 inference on 59-file Val-A ===")
    sed38 = run_inference(meta)
    np.savez_compressed(EXP38 / "val_scores_full.npz", preds=sed38)
    print(f"SED38 alone Val-A: {macro_auc(Y, sed38):.4f}")

    sed29 = np.load(EXP29 / "val_scores.npz")["preds"]
    print(f"SED29 alone Val-A: {macro_auc(Y, sed29):.4f}")

    # Perch smoothed from exp28
    p = np.load(EXP28 / "best_oof.npz")
    perch = p["val_a_smoothed"]  # (708, 234) matching meta order
    print(f"Perch smoothed Val-A: {macro_auc(Y, perch):.4f}")

    zP = zscore(perch); z29 = zscore(sed29); z38 = zscore(sed38)

    results = {"perch": macro_auc(Y, perch),
               "sed29_alone": macro_auc(Y, sed29),
               "sed38_alone": macro_auc(Y, sed38)}

    # 2-way blends (substitution)
    print("\n=== Perch α·Z + (1-α)·SED29 ===")
    for a in np.arange(0.0, 1.01, 0.1):
        s = a * zP + (1 - a) * z29
        results[f"P+29_a{a:.1f}"] = macro_auc(Y, s)
        print(f"  α={a:.1f}: {results[f'P+29_a{a:.1f}']:.4f}")

    print("\n=== Perch α·Z + (1-α)·SED38 ===")
    for a in np.arange(0.0, 1.01, 0.1):
        s = a * zP + (1 - a) * z38
        results[f"P+38_a{a:.1f}"] = macro_auc(Y, s)
        print(f"  α={a:.1f}: {results[f'P+38_a{a:.1f}']:.4f}")

    # 3-way simplex
    print("\n=== 3-way: wP·Perch + w29·SED29 + w38·SED38 ===")
    best = (-1, None)
    grid = []
    for wP in np.arange(0.0, 1.01, 0.05):
        for w29 in np.arange(0.0, 1.01 - wP, 0.05):
            w38 = 1.0 - wP - w29
            if w38 < -1e-9: continue
            s = wP * zP + w29 * z29 + w38 * z38
            auc = macro_auc(Y, s)
            grid.append((wP, w29, w38, auc))
            if auc > best[0]:
                best = (auc, (wP, w29, w38))
    results["best_3way"] = {"val_a": best[0], "w": best[1]}
    print(f"Best 3-way: {best[0]:.4f}  (wP={best[1][0]:.2f}, w29={best[1][1]:.2f}, w38={best[1][2]:.2f})")

    # Gauss σ=0.5 post-smoothing on best 3-way
    from scipy.ndimage import gaussian_filter1d
    wP, w29, w38 = best[1]
    s_best = wP * zP + w29 * z29 + w38 * z38
    files = meta["filename"].values
    s_sm = np.zeros_like(s_best)
    for f in np.unique(files):
        m = files == f
        s_sm[m] = gaussian_filter1d(s_best[m], sigma=0.5, axis=0)
    results["best_3way_gauss0.5"] = macro_auc(Y, s_sm)
    print(f"Best 3-way + Gauss σ=0.5: {results['best_3way_gauss0.5']:.4f}")

    with open(EXP38 / "blend_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {EXP38}/blend_results.json")


if __name__ == "__main__":
    main()
