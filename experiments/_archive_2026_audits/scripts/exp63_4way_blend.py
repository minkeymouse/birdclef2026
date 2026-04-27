#!/usr/bin/env python3
"""exp63 — 4-way blend with ConvNeXt-tiny SED added.

Tests global rule blend (no per-class routing — that's been proven LB-fragile).

Configs to test:
  v26 (ref): 0.7P + 0.3 exp50
  v30a: 0.7P + 0.2 exp50 + 0.1 ConvNeXt
  v30b: 0.7P + 0.15 exp50 + 0.15 ConvNeXt
  v30c: 0.7P + 0.1 exp50 + 0.2 ConvNeXt
  v30d: 0.7P + 0.3 ConvNeXt (replace exp50)
  v30e: 0.6P + 0.2 exp50 + 0.2 ConvNeXt
  v30f: 0.6P + 0.4 ConvNeXt (just ConvNeXt heavier)
  v30g: 0.65P + 0.175 exp50 + 0.175 ConvNeXt (more conservative)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch, torch.nn as nn
import timm, torchaudio
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, pearsonr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
EXP59 = ROOT / "experiments/exp59_outputs"
OUT = ROOT / "experiments/exp63_outputs"
OUT.mkdir(exist_ok=True)
SR = 32000; CLIP_SEC = 20; CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000; DEVICE = "cuda"
SEED = 42


def build_all():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    rng = np.random.RandomState(SEED); files = sorted(sc_g.filename.unique()); rng.shuffle(files)
    eval_files = set(files[:11])
    sc_g["split"] = ["eval" if f in eval_files else "train" for f in sc_g.filename]
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g.lbls):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_g, Y, primary, l2i


def align_43a(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(df), scs.shape[1]), np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]
    return S


class _Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class _Head(nn.Module):
    def __init__(self, f, n):
        super().__init__()
        self.att = nn.Conv1d(f, n, 1); self.cla = nn.Conv1d(f, n, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values


class _SED(nn.Module):
    def __init__(self, n_cls, backbone):
        super().__init__()
        self.mel = _Mel(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(backbone, pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = _Head(feat.shape[1], n_cls)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1,2); m = self.bn0(m); m = m.transpose(1,2)
        f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
        c, _ = self.head(f); return c


@torch.no_grad()
def predict_sed(model, df, n_cls):
    model.eval()
    out = np.zeros((len(df), n_cls), dtype=np.float32); cache = {}
    for i in range(0, len(df), 8):
        j = min(len(df), i + 8); wavs = []
        for k in range(i, j):
            row = df.iloc[k]
            if row.filename not in cache:
                w, sr = sf.read(DATA / "train_soundscapes" / row.filename, dtype="float32")
                if w.ndim > 1: w = w.mean(1)
                cache[row.filename] = w
            wav = cache[row.filename]
            cs = int(max(0, (int(row.end_sec) - 2.5) * SR - CLIP_SAMPLES/2))
            cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
            clip = wav[cs:cs + CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            wavs.append(clip.astype(np.float32))
        x = torch.from_numpy(np.stack(wavs)).to(DEVICE)
        out[i:j] = torch.sigmoid(model(x)).cpu().numpy()
    return out


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s
def gauss_pf(scores, df, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in df.filename.unique():
        m = (df.filename == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out


def per_class_auc(Y, P, min_pos=1):
    out = {}
    for c in range(Y.shape[1]):
        y = Y[:, c]
        if y.sum() < min_pos or y.sum() == len(y): continue
        if not np.isfinite(P[:, c]).all(): continue
        try: out[c] = float(roc_auc_score(y, P[:, c]))
        except: pass
    return out


def main():
    print("Loading...")
    sc_all, Y_all, primary, l2i = build_all()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    ck50 = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED(n_cls=234, backbone="hgnetv2_b0.ssld_stage2_ft_in1k").to(DEVICE)
    m50.load_state_dict(ck50["state_dict"])
    print("Computing exp50 preds...")
    P50 = predict_sed(m50, sc_all, 234); del m50
    torch.cuda.empty_cache()

    ck59 = torch.load(EXP59 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m59 = _SED(n_cls=234, backbone="convnext_tiny.fb_in22k_ft_in1k").to(DEVICE)
    m59.load_state_dict(ck59["state_dict"])
    print("Computing exp59 ConvNeXt preds...")
    P59 = predict_sed(m59, sc_all, 234); del m59
    torch.cuda.empty_cache()

    print(f"Perch {perch_prob.shape}, exp50 {P50.shape}, exp59 {P59.shape}")

    # Diversity check
    print("\n=== Diversity (Pearson on raw predictions) ===")
    flat_p = perch_prob.flatten(); flat_50 = P50.flatten(); flat_59 = P59.flatten()
    print(f"  Perch ↔ exp50: {pearsonr(flat_p, flat_50)[0]:.3f}")
    print(f"  Perch ↔ exp59: {pearsonr(flat_p, flat_59)[0]:.3f}")
    print(f"  exp50 ↔ exp59: {pearsonr(flat_50, flat_59)[0]:.3f}")

    zP = zs(perch_prob); z50 = zs(P50); z59 = zs(P59)

    def blend(wP, w50, w59):
        raw = wP * zP + w50 * z50 + w59 * z59
        return sigmoid(gauss_pf(raw, sc_all, 0.5))

    def eval_(P, P_ref, label):
        ev_mask = (sc_all.split == "eval").values
        Y_ev = Y_all[ev_mask]
        a = per_class_auc(Y_ev, P[ev_mask])
        a_ref = per_class_auc(Y_ev, P_ref[ev_mask])
        common = set(a) & set(a_ref)
        macro = np.mean([a[c] for c in common])
        macro_ref = np.mean([a_ref[c] for c in common])
        sp_row = []
        for i in range(len(sc_all)):
            r, _ = spearmanr(P_ref[i], P[i])
            if np.isfinite(r): sp_row.append(r)
        out = {"label": label, "macro_eval11": macro, "delta": macro - macro_ref,
                "sp_row": float(np.mean(sp_row))}
        for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
            cls = [c for c in common if species_taxon[c] == t]
            if cls:
                out[f"tx_{t}"] = float(np.mean([a[c] - a_ref[c] for c in cls]))
        return out

    v26 = blend(0.7, 0.3, 0.0)
    results = [eval_(v26, v26, "v26 ref")]

    print("\n=== 4-way blend candidates ===")
    candidates = [
        ("v30a P0.7 e50_0.2 e59_0.1", 0.7, 0.2, 0.1),
        ("v30b P0.7 e50_0.15 e59_0.15", 0.7, 0.15, 0.15),
        ("v30c P0.7 e50_0.1 e59_0.2", 0.7, 0.1, 0.2),
        ("v30d P0.7 e50_0.0 e59_0.3 (replace)", 0.7, 0.0, 0.3),
        ("v30e P0.6 e50_0.2 e59_0.2", 0.6, 0.2, 0.2),
        ("v30f P0.6 e50_0.0 e59_0.4", 0.6, 0.0, 0.4),
        ("v30g P0.65 e50_0.175 e59_0.175", 0.65, 0.175, 0.175),
        ("v30h P0.75 e50_0.125 e59_0.125", 0.75, 0.125, 0.125),
    ]
    for label, wP, w50, w59 in candidates:
        p = blend(wP, w50, w59)
        r = eval_(p, v26, label)
        results.append(r)
        print(f"  {label:<40}  macro {r['macro_eval11']:.4f}  Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  "
              f"Aves Δ {r.get('tx_Aves', 0):+.3f}")

    # Also test pure ConvNeXt vs Perch blend (mirror of v26 with exp50→exp59)
    print("\n=== ConvNeXt swap variants (replace exp50 with exp59) ===")
    for w59 in [0.20, 0.25, 0.30, 0.35, 0.40]:
        wP = 1 - w59
        p = blend(wP, 0.0, w59)
        r = eval_(p, v26, f"v30_swap P{wP:.2f} e59_{w59}")
        results.append(r)
        print(f"  w59={w59}  macro {r['macro_eval11']:.4f}  Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  "
              f"Aves Δ {r.get('tx_Aves', 0):+.3f}")

    df = pd.DataFrame(results).round(4)
    df.to_csv(OUT / "63_grid.csv", index=False)
    print("\n=== TOP 10 by held-out delta ===")
    print(df.sort_values("delta", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
