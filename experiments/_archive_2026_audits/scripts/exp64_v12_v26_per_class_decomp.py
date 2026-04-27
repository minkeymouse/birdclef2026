#!/usr/bin/env python3
"""exp64 — Class-by-class decomposition of LB +0.002 (v12 → v26).

Goal: identify WHICH classes carried the +0.002 LB improvement so we can
understand the mechanism and target it specifically.

Available data:
  - LB scores: v12=0.929, v24=0.930, v26=0.931
  - LB metric: macro AUC over evaluable classes (n_pos > 0 in test).
    Approx 150-200 classes evaluable on LB (we don't know exactly which).

Key insight: LB +0.002 averaged over ~200 classes = +0.4 cumulative AUC.
This could be:
  - 4 classes × +0.1 each
  - 40 classes × +0.01 each
  - 200 classes × +0.002 each (uniform improvement)

If we could see which classes moved, we could amplify the right ones.

Approach:
  1. Predictions on ALL 10,658 unlabeled SS for v12 vs v26 (proxy for LB distribution)
  2. Per-class measure: KS distance between v12 and v26 predictions
  3. On 66 labeled SS: per-class AUC change
  4. Cross-reference: classes with both KS shift AND AUC improvement (held-out)
  5. Hypothesis: those are the classes that drive LB gain
  6. Look at: are they specifically rare classes / non-Aves / Perch-weak?

Then: design lever that targets ONLY those classes, leaves others as v12.
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
from scipy.stats import ks_2samp, spearmanr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
OUT = ROOT / "experiments/exp64_outputs"
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


def align_old(df, p):
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(df), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


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


class _SED50(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.mel = _Mel(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model("hgnetv2_b0.ssld_stage2_ft_in1k",
                                          pretrained=False, in_chans=1,
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


def per_class_auc(Y, P):
    out = {}
    for c in range(Y.shape[1]):
        y = Y[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
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
    ta_cnt = pd.read_csv(DATA / "train.csv").groupby("primary_label").size().to_dict()

    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)
    ck50 = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED50(n_cls=234).to(DEVICE); m50.load_state_dict(ck50["state_dict"])
    P50 = predict_sed(m50, sc_all, 234); del m50
    torch.cuda.empty_cache()

    zP = zs(perch_prob); z29 = zs(S29); z50 = zs(P50)
    v12 = sigmoid(gauss_pf(0.8*zP + 0.2*z29, sc_all, 0.5))
    v24 = sigmoid(gauss_pf(0.8*zP + 0.2*z50, sc_all, 0.5))
    v26 = sigmoid(gauss_pf(0.7*zP + 0.3*z50, sc_all, 0.5))

    ev_mask = (sc_all.split == "eval").values

    # ─── Per-class movement decomposition v12 → v24 → v26 ───
    print("\n" + "="*70)
    print("Q1: Per-class KS distance between predictions (proxy for LB-distribution shift)")
    print("="*70)
    rows = []
    for c in range(234):
        try:
            ks_12_24, _ = ks_2samp(v12[:, c], v24[:, c])
            ks_12_26, _ = ks_2samp(v12[:, c], v26[:, c])
            ks_24_26, _ = ks_2samp(v24[:, c], v26[:, c])
        except Exception: continue
        rows.append({
            "class": primary[c], "taxon": species_taxon[c], "n_ta": ta_cnt.get(primary[c], 0),
            "n_pos_ss66": int(Y_all[:, c].sum()),
            "ks_v12_v24": ks_12_24, "ks_v12_v26": ks_12_26, "ks_v24_v26": ks_24_26,
            "v12_mean": float(v12[:, c].mean()), "v24_mean": float(v24[:, c].mean()),
            "v26_mean": float(v26[:, c].mean()),
        })
    df_ks = pd.DataFrame(rows)
    print(f"  {len(df_ks)} classes analyzed")
    print(f"  Mean KS v12↔v24: {df_ks.ks_v12_v24.mean():.3f}")
    print(f"  Mean KS v12↔v26: {df_ks.ks_v12_v26.mean():.3f}")
    print(f"  Mean KS v24↔v26: {df_ks.ks_v24_v26.mean():.3f}")
    print(f"\n  Classes with KS_v12_v26 > 0.3 (substantial shift): {(df_ks.ks_v12_v26 > 0.3).sum()}")
    print(f"  By taxon (KS > 0.3 v12→v26):")
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        sub = df_ks[(df_ks.taxon == t) & (df_ks.ks_v12_v26 > 0.3)]
        total_t = (df_ks.taxon == t).sum()
        if len(sub):
            print(f"    {t:<10}  {len(sub):3d} of {total_t} taxon classes shifted (KS > 0.3)")

    print(f"\n  Top 30 classes by KS_v12_v26 (most shifted):")
    for _, r in df_ks.nlargest(30, "ks_v12_v26").iterrows():
        print(f"    {r['class']:<14} ({r.taxon:<9}) n_pos={r.n_pos_ss66:3d} n_ta={r.n_ta:4d}  "
              f"ks={r.ks_v12_v26:.3f}  v12_mean={r.v12_mean:.4f}  v26_mean={r.v26_mean:.4f}")

    # ─── Q2: per-class AUC change v12 → v26 (held-out + train) ───
    print("\n" + "="*70)
    print("Q2: Per-class AUC change v12 → v26 (held-out 11 + train 55, on 66 SS)")
    print("="*70)
    a12 = per_class_auc(Y_all, v12)
    a26 = per_class_auc(Y_all, v26)
    common = set(a12) & set(a26)
    rows_a = []
    for c in common:
        rows_a.append({
            "class": primary[c], "taxon": species_taxon[c], "n_ta": ta_cnt.get(primary[c], 0),
            "n_pos": int(Y_all[:, c].sum()),
            "auc_v12": a12[c], "auc_v26": a26[c],
            "delta": a26[c] - a12[c],
            "ks_v12_v26": df_ks[df_ks["class"] == primary[c]].ks_v12_v26.iloc[0],
        })
    df_a = pd.DataFrame(rows_a)
    print(f"  {len(df_a)} classes evaluable on 66 SS")
    print(f"  v12 macro: {df_a.auc_v12.mean():.4f}")
    print(f"  v26 macro: {df_a.auc_v26.mean():.4f}  Δ {df_a.auc_v26.mean()-df_a.auc_v12.mean():+.4f}")
    print(f"\n  Distribution of v26 - v12 per class:")
    print(f"    >+0.05: {(df_a.delta > 0.05).sum():3d} classes")
    print(f"    >+0.01: {(df_a.delta > 0.01).sum():3d} classes")
    print(f"    [-0.01, +0.01]: {((df_a.delta > -0.01) & (df_a.delta < 0.01)).sum():3d} classes (no change)")
    print(f"    <-0.01: {(df_a.delta < -0.01).sum():3d} classes")
    print(f"    <-0.05: {(df_a.delta < -0.05).sum():3d} classes")

    # Top movers
    print(f"\n  Top 20 v26 GAINS:")
    for _, r in df_a.nlargest(20, "delta").iterrows():
        print(f"    {r['class']:<14} ({r.taxon:<9}) n_pos={r.n_pos:3d} n_ta={r.n_ta:4d}  "
              f"v12={r.auc_v12:.3f} → v26={r.auc_v26:.3f}  Δ{r.delta:+.3f}  ks={r.ks_v12_v26:.3f}")

    print(f"\n  Top 10 v26 LOSSES:")
    for _, r in df_a.nsmallest(10, "delta").iterrows():
        print(f"    {r['class']:<14} ({r.taxon:<9}) n_pos={r.n_pos:3d} n_ta={r.n_ta:4d}  "
              f"v12={r.auc_v12:.3f} → v26={r.auc_v26:.3f}  Δ{r.delta:+.3f}  ks={r.ks_v12_v26:.3f}")

    # ─── Q3: relationship — does KS shift correlate with auc gain? ───
    print("\n" + "="*70)
    print("Q3: KS shift vs AUC gain correlation")
    print("="*70)
    print(f"  Spearman(ks_v12_v26, |delta|): {spearmanr(df_a.ks_v12_v26, df_a.delta.abs())[0]:.3f}")
    print(f"  Spearman(ks_v12_v26, delta):   {spearmanr(df_a.ks_v12_v26, df_a.delta)[0]:.3f}")
    print(f"  → Positive direction: classes with bigger shift tend to GAIN more")

    # ─── Q4: Is there a class characteristic that predicts gain? ───
    print("\n" + "="*70)
    print("Q4: What class properties predict gain?")
    print("="*70)
    df_a["v12_low"] = df_a.auc_v12 < 0.6
    print(f"  v12_AUC < 0.6 classes ({df_a.v12_low.sum()}):")
    print(f"    Mean delta v26-v12: {df_a[df_a.v12_low].delta.mean():+.4f}")
    print(f"    n_ta = 0:  delta {df_a[df_a.v12_low & (df_a.n_ta == 0)].delta.mean():+.4f}")
    print(f"    n_ta > 0:  delta {df_a[df_a.v12_low & (df_a.n_ta > 0)].delta.mean():+.4f}")

    print(f"\n  v12_AUC ≥ 0.9 classes ({(df_a.auc_v12 >= 0.9).sum()}, mostly 'easy' for Perch):")
    print(f"    Mean delta v26-v12: {df_a[df_a.auc_v12 >= 0.9].delta.mean():+.4f}")

    print(f"\n  Per-taxon mean delta:")
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        sub = df_a[df_a.taxon == t]
        if len(sub):
            print(f"    {t:<10}  n={len(sub):2d}  mean Δ {sub.delta.mean():+.4f}  median {sub.delta.median():+.4f}")

    # ─── Q5: Consistency of Perch on LB — proxy via ALL 10K UNLABELED SS ───
    print("\n" + "="*70)
    print("Q5: Perch consistency proxy — ALL 10,658 unlabeled SS prediction stats")
    print("="*70)
    all_d = np.load(EXP43A / "perch_ss_all.npz")
    all_scs = all_d["scores"]
    all_perch = sigmoid(all_scs)
    # Per-class CV (coefficient of variation) across all unlabeled SS rows
    cv_per_class = []
    for c in range(234):
        m = all_perch[:, c].mean(); s = all_perch[:, c].std()
        cv_per_class.append({
            "class": primary[c], "taxon": species_taxon[c],
            "perch_mean": float(m), "perch_std": float(s),
            "perch_max": float(all_perch[:, c].max()),
            "perch_p99": float(np.quantile(all_perch[:, c], 0.99)),
            "perch_p1": float(np.quantile(all_perch[:, c], 0.01)),
        })
    df_cv = pd.DataFrame(cv_per_class)
    # Classes Perch confidently fires on (high p99): more "confident" = more LB-consistent
    print(f"  Classes with Perch p99 > 0.9 (Perch fires confidently somewhere):")
    print(f"    n={(df_cv.perch_p99 > 0.9).sum()}")
    print(f"    Per-taxon:")
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        sub = df_cv[(df_cv.taxon == t) & (df_cv.perch_p99 > 0.9)]
        total_t = (df_cv.taxon == t).sum()
        print(f"      {t:<10}  {len(sub):3d} of {total_t}  ({100*len(sub)/total_t:.0f}%)")

    print(f"\n  Classes with Perch p99 < 0.1 (Perch never fires — likely unmapped):")
    print(f"    n={(df_cv.perch_p99 < 0.1).sum()}")
    for _, r in df_cv[df_cv.perch_p99 < 0.1].head(15).iterrows():
        print(f"      {r['class']:<14} ({r.taxon:<9}) p99={r.perch_p99:.3f}")

    # Save
    df_a.to_csv(OUT / "64_v12_v26_per_class.csv", index=False)
    df_ks.to_csv(OUT / "64_ks_distance.csv", index=False)
    df_cv.to_csv(OUT / "64_perch_consistency.csv", index=False)


if __name__ == "__main__":
    main()
