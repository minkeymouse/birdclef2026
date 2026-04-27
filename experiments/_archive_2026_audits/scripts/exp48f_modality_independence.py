#!/usr/bin/env python3
"""exp48f — Modality independence check using existing exp47 ckpt.

Use exp47 (train_audio + labeled SS, NO Perch distill) to produce predictions
on 11-file held-out. Compare per-class prediction CORRELATION with:
  - Perch alone (same mel backbone as exp29 in spirit)
  - SED29 (old teacher)
  - SED41f (Perch-distilled teacher)

Key question: is exp47's error INDEPENDENT from Perch's error?
If yes → adding exp47 to blend provides new information axis.
If no (errors correlated) → exp47 is redundant despite Perch-independent training.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch, torch.nn as nn
import timm, torchaudio
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP41F = ROOT / "experiments/exp41f_outputs"
EXP47 = ROOT / "experiments/exp47_outputs"
OUT = ROOT / "experiments/exp48_outputs"
SEED = 42; EVAL_N = 11; SR = 32000; CLIP_SEC = 20
N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000; DEVICE = "cuda"


def build_eval():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N])
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["lbls"]):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_eval, Y, primary, l2i


def align_43a(sc_eval):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(sc_eval), scs.shape[1]), np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]
    return S


def align_old(sc_eval, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(sc_eval), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x):
        return self.adb(self.mel(x)).unsqueeze(1)


class SEDHead(nn.Module):
    def __init__(self, feat_dim, n_cls):
        super().__init__()
        self.att = nn.Conv1d(feat_dim, n_cls, 1)
        self.cla = nn.Conv1d(feat_dim, n_cls, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values


class SEDModel(nn.Module):
    def __init__(self, backbone="hgnetv2_b0.ssld_stage2_ft_in1k", n_cls=234):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(backbone, pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(feat.shape[1], n_cls)
    def forward(self, x):
        m = self.mel(x)
        m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        f = feat.mean(dim=2) if feat.dim() == 4 else feat
        clip, _ = self.head(f)
        return clip


@torch.no_grad()
def predict_exp47(sc_eval, primary):
    ck = torch.load(EXP47 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    model = SEDModel(n_cls=len(primary)).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    n = len(sc_eval)
    out = np.zeros((n, len(primary)), dtype=np.float32)
    batch = 8; CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
    cache = {}
    for i in range(0, n, batch):
        j = min(n, i + batch)
        wavs = []
        for k in range(i, j):
            row = sc_eval.iloc[k]
            fn = row.filename
            if fn not in cache:
                w, sr = sf.read(DATA / "train_soundscapes" / fn, dtype="float32")
                if w.ndim > 1: w = w.mean(1)
                cache[fn] = w
            wav = cache[fn]
            end_sec = int(row.end_sec)
            target_c = (end_sec - 2.5) * SR
            cs = int(max(0, target_c - CLIP_SAMPLES/2))
            cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
            clip = wav[cs:cs + CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES:
                clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            wavs.append(clip.astype(np.float32))
        x = torch.from_numpy(np.stack(wavs)).to(DEVICE)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy()
        out[i:j] = p
    return out


def per_class_auc(Y, P):
    ev = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    return {c: float(roc_auc_score(Y[:, c], P[:, c])) for c in ev
            if np.isfinite(P[:, c]).all()}


def main():
    sc_eval, Y, primary, l2i = build_eval()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    # Get all 4 predictions on eval
    print("Loading predictions...")
    S_perch = align_43a(sc_eval); perch = sigmoid(S_perch)
    sed29 = align_old(sc_eval, EXP29 / "val_scores.npz")
    sed41f = align_old(sc_eval, EXP41F / "val_scores_full.npz")
    if sed29 is not None: sed29 = np.nan_to_num(sed29, nan=0)
    if sed41f is not None: sed41f = np.nan_to_num(sed41f, nan=0)
    print("Computing exp47 predictions...")
    exp47 = predict_exp47(sc_eval, primary)
    print(f"Shapes: perch {perch.shape}  sed29 {None if sed29 is None else sed29.shape}  "
          f"sed41f {None if sed41f is None else sed41f.shape}  exp47 {exp47.shape}")

    # Per-class error correlation between models
    # For each evaluable class, compute AUC, then Pearson correlation of
    # (prediction on positive rows - prediction on negative rows) across models.
    # Proxy for "error signature"
    print("\n=== Per-class prediction correlation (40 eval classes) ===")
    all_models = {"perch": perch, "sed29": sed29, "sed41f": sed41f, "exp47": exp47}

    evaluable = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    # Compute per-class AUC for each
    aucs = {name: per_class_auc(Y, arr) for name, arr in all_models.items() if arr is not None}
    for name in aucs:
        m = np.mean(list(aucs[name].values()))
        print(f"  {name:<8} macro on eval 40 cls: {m:.4f}")

    # Predictions pairwise correlation — global on full pred matrix
    from scipy.stats import pearsonr, spearmanr
    names = list(all_models.keys())
    print("\n[Global prediction Spearman correlation (flatten all 122×234)]")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            if all_models[n1] is None or all_models[n2] is None: continue
            a = all_models[n1].flatten(); b = all_models[n2].flatten()
            r, _ = spearmanr(a, b)
            print(f"  {n1:<8} vs {n2:<8}  Spearman ρ = {r:.3f}")

    # Error vector correlation: for each model, compute per-row (pred - y) residuals
    # Then compare residuals across models
    print("\n[Residual (pred - label) correlation on all 40 eval classes only]")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            if all_models[n1] is None or all_models[n2] is None: continue
            resid1 = (all_models[n1][:, evaluable] - Y[:, evaluable]).flatten()
            resid2 = (all_models[n2][:, evaluable] - Y[:, evaluable]).flatten()
            r, _ = pearsonr(resid1, resid2)
            print(f"  {n1:<8} vs {n2:<8}  residual-corr = {r:.3f}")

    # Blend test: v12 (P + SED29) vs v12' (P + exp47) vs P + 0.5*exp47 + 0.5*SED29
    def blend(P, name, w_p=0.8, w_s=0.2, second=None):
        b = w_p * zs(P) + w_s * zs(second)
        return sigmoid(b)

    print("\n=== Blend head-to-head on eval 40 cls ===")
    for name, S in [("sed29", sed29), ("sed41f", sed41f), ("exp47", exp47)]:
        if S is None: continue
        b = sigmoid(0.8 * zs(perch) + 0.2 * zs(S))
        aucs_b = per_class_auc(Y, b)
        m = np.mean([aucs_b[c] for c in evaluable if c in aucs_b])
        print(f"  Perch + 0.2*{name:<8}  macro={m:.4f}")

    # 3-way: Perch + exp47 + SED29
    if exp47 is not None and sed29 is not None:
        for w47 in [0.1, 0.15, 0.2, 0.3]:
            b3 = sigmoid(0.8 * zs(perch) + 0.2 * zs(sed29) + w47 * zs(exp47))
            aucs_b = per_class_auc(Y, b3)
            m = np.mean([aucs_b[c] for c in evaluable if c in aucs_b])
            print(f"  Perch + 0.2*sed29 + {w47}*exp47  macro={m:.4f}")

    # Bottom-8: does exp47 rescue any?
    print("\n[Bottom-8 per-model AUC]")
    bottom8 = ["516975", "67107", "326272", "bafcur1", "74113", "25073", "116570", "47158son11"]
    header = f"  {'label':<12}  {'taxon':<9}  " + "  ".join(f"{n:>8}" for n in names)
    print(header)
    for lbl in bottom8:
        c = primary.index(lbl) if lbl in primary else -1
        if c < 0: continue
        line = f"  {lbl:<12}  {species_taxon[c]:<9}  "
        for n in names:
            a = aucs.get(n, {}).get(c)
            line += f"{a:>8.3f}  " if a is not None else f"{'--':>8}  "
        print(line)


if __name__ == "__main__":
    main()
