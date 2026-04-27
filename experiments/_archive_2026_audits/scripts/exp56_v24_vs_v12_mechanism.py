#!/usr/bin/env python3
"""exp56 — WHY did v24 reach LB 0.930 while v12=0.929?

v24 and v12 share the ENTIRE pipeline except ONE thing: the 0.2-weight SED.
  v12: 0.8*Perch + 0.2*SED29
  v24: 0.8*Perch + 0.2*exp50

So +0.001 LB = (exp50 - SED29) contribution at 0.2 blend weight.

Analysis questions:
  (1) Where do SED29 and exp50 predictions DIFFER most — which classes?
  (2) Are the differences consistent with our oracle expectation? (exp50 helps
      classes Perch is weak on)
  (3) On held-out eval (11 files), which classes did v24 change vs v12?
  (4) Distribution shift test: for each class, KS-test between v12 and v24
      predictions on ALL 66 SS rows. Big shifts = candidate drivers.
  (5) Quantify the ensemble diversity metric that matters:
      resid-corr(Perch, SED) × w_SED × Δ_AUC_from_blending
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
from scipy.stats import spearmanr, ks_2samp, pearsonr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
OUT = ROOT / "experiments/exp56_outputs"
OUT.mkdir(exist_ok=True)
SR = 32000; CLIP_SEC = 20
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
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g.filename.unique()); rng.shuffle(files)
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


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)

class SEDHead(nn.Module):
    def __init__(self, f, n):
        super().__init__()
        self.att = nn.Conv1d(f, n, 1); self.cla = nn.Conv1d(f, n, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values

class SEDModel(nn.Module):
    def __init__(self, n_cls=234):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model("hgnetv2_b0.ssld_stage2_ft_in1k",
                                          pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(feat.shape[1], n_cls)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1,2); m = self.bn0(m); m = m.transpose(1,2)
        f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
        c, _ = self.head(f); return c


@torch.no_grad()
def predict_sed(df, ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = SEDModel().to(DEVICE); model.load_state_dict(ck["state_dict"]); model.eval()
    CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
    out = np.zeros((len(df), 234), dtype=np.float32); cache = {}
    for i in range(0, len(df), 8):
        j = min(len(df), i + 8); wavs = []
        for k in range(i, j):
            row = df.iloc[k]
            if row.filename not in cache:
                w, sr = sf.read(DATA / "train_soundscapes" / row.filename, dtype="float32")
                if w.ndim > 1: w = w.mean(1)
                cache[row.filename] = w
            wav = cache[row.filename]
            end_sec = int(row.end_sec)
            cs = int(max(0, (end_sec - 2.5) * SR - CLIP_SAMPLES/2))
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
    print("Loading preds...")
    sc_all, Y_all, primary, l2i = build_all()
    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)
    P50 = predict_sed(sc_all, EXP50 / "best_ckpt.pt")

    # Same pipeline as v12/v24: 0.8*zP + 0.2*zS + Gauss + sigmoid
    zP = zs(perch_prob); z29 = zs(S29); z50 = zs(P50)
    v12_prob = sigmoid(gauss_pf(0.8*zP + 0.2*z29, sc_all, 0.5))
    v24_prob = sigmoid(gauss_pf(0.8*zP + 0.2*z50, sc_all, 0.5))

    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])
    ta_cnt = pd.read_csv(DATA / "train.csv").groupby("primary_label").size().to_dict()
    n_pos = Y_all.sum(axis=0)

    # ── Q1: Per-class prediction shift v12 → v24 ──
    print("\n=== Q1: per-class prediction shift v12 → v24 ===")
    # For each class, compute mean and KS distance between v12 and v24 predictions
    shifts = []
    for c in range(234):
        d_mean = v24_prob[:, c].mean() - v12_prob[:, c].mean()
        ks_stat, ks_p = ks_2samp(v12_prob[:, c], v24_prob[:, c])
        spear_r, _ = spearmanr(v12_prob[:, c], v24_prob[:, c])
        shifts.append({
            "class": primary[c], "taxon": species_taxon[c],
            "n_pos": int(n_pos[c]), "n_ta": int(ta_cnt.get(primary[c], 0)),
            "v12_mean": float(v12_prob[:, c].mean()),
            "v24_mean": float(v24_prob[:, c].mean()),
            "mean_shift": float(d_mean),
            "ks_stat": float(ks_stat),
            "spearman_v12_v24": float(spear_r) if not np.isnan(spear_r) else None,
        })
    df_shift = pd.DataFrame(shifts)
    print(f"  Mean |ks_stat|: {df_shift.ks_stat.mean():.3f}")
    print(f"  Classes with ks_stat > 0.3 (large distribution shift): {(df_shift.ks_stat > 0.3).sum()}")
    print(f"  Classes with Spearman < 0.9 (significant rank change): {(df_shift.spearman_v12_v24 < 0.9).sum()}")

    # Top 20 by KS distance
    print(f"\n  Top-20 classes with biggest DISTRIBUTION SHIFT (v12→v24):")
    top_ks = df_shift.sort_values("ks_stat", ascending=False).head(20)
    print(top_ks[["class","taxon","n_pos","n_ta","v12_mean","v24_mean","mean_shift","ks_stat"]].to_string(index=False))

    # ── Q2: Join with eval AUC to see whether shift helped or hurt ──
    aucs_v12 = per_class_auc(Y_all, v12_prob)
    aucs_v24 = per_class_auc(Y_all, v24_prob)
    # Only for evaluable classes
    eval_common = set(aucs_v12) & set(aucs_v24)
    rows = []
    for c in eval_common:
        r = df_shift.iloc[c].to_dict()
        r["auc_v12"] = aucs_v12[c]
        r["auc_v24"] = aucs_v24[c]
        r["auc_gain"] = aucs_v24[c] - aucs_v12[c]
        rows.append(r)
    df_auc = pd.DataFrame(rows)

    print(f"\n=== Q2: classes where v24 auc improved by > 0.03 ===")
    gained = df_auc[df_auc.auc_gain > 0.03].sort_values("auc_gain", ascending=False)
    print(f"  {len(gained)} such classes:")
    print(gained[["class","taxon","n_pos","n_ta","auc_v12","auc_v24","auc_gain","ks_stat","mean_shift"]].head(20).to_string(index=False))

    print(f"\n=== Q2 cont.: classes where v24 auc REGRESSED by > 0.02 ===")
    lost = df_auc[df_auc.auc_gain < -0.02].sort_values("auc_gain")
    print(f"  {len(lost)} such classes:")
    if len(lost):
        print(lost[["class","taxon","n_pos","n_ta","auc_v12","auc_v24","auc_gain","ks_stat","mean_shift"]].head(20).to_string(index=False))
    else:
        print("  NONE — v24 monotonically improves or ties vs v12 on evaluable eval classes")

    # ── Q3: Held-out (11-eval) split ──
    ev_mask = (sc_all.split == "eval").values
    Y_ev = Y_all[ev_mask]; v12_ev = v12_prob[ev_mask]; v24_ev = v24_prob[ev_mask]
    aucs_v12_ev = per_class_auc(Y_ev, v12_ev)
    aucs_v24_ev = per_class_auc(Y_ev, v24_ev)
    common_ev = set(aucs_v12_ev) & set(aucs_v24_ev)
    ev_macro_v12 = np.mean([aucs_v12_ev[c] for c in common_ev])
    ev_macro_v24 = np.mean([aucs_v24_ev[c] for c in common_ev])
    print(f"\n=== Q3: Held-out 11-file eval macro ===")
    print(f"  v12: {ev_macro_v12:.4f}  v24: {ev_macro_v24:.4f}  Δ={ev_macro_v24-ev_macro_v12:+.4f}")
    # Per-taxon
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        cls = [c for c in common_ev if species_taxon[c] == t]
        if cls:
            d = np.mean([aucs_v24_ev[c] - aucs_v12_ev[c] for c in cls])
            print(f"    {t:<10} n={len(cls):2d}  v12={np.mean([aucs_v12_ev[c] for c in cls]):.3f}  v24={np.mean([aucs_v24_ev[c] for c in cls]):.3f}  Δ={d:+.3f}")

    # ── Q4: Diversity metric explanation ──
    print(f"\n=== Q4: diversity correlation metrics ===")
    # Per-class correlation of SED29 vs exp50 on predictions
    corrs = []
    for c in range(234):
        r, _ = pearsonr(S29[:, c], P50[:, c])
        if np.isfinite(r): corrs.append(r)
    print(f"  Per-class Pearson(SED29, exp50): mean={np.mean(corrs):.3f}")
    print(f"    distribution: 10%={np.percentile(corrs,10):.2f} 50%={np.percentile(corrs,50):.2f} 90%={np.percentile(corrs,90):.2f}")
    # Global:
    a = S29.flatten(); b = P50.flatten()
    g_r, _ = pearsonr(a, b)
    g_sp, _ = spearmanr(a, b)
    print(f"  Global flatten Pearson(SED29,exp50) = {g_r:.3f}, Spearman = {g_sp:.3f}")

    # ── Q5: correlate AUC gain with prediction-distribution shift ──
    print(f"\n=== Q5: relationship between AUC gain and mean_shift ===")
    r, p = spearmanr(df_auc.auc_gain, df_auc.mean_shift)
    print(f"  Spearman(auc_gain, mean_shift) = {r:.3f} p={p:.3g}")
    r2, p2 = spearmanr(df_auc.auc_gain, df_auc.ks_stat)
    print(f"  Spearman(auc_gain, ks_stat)    = {r2:.3f} p={p2:.3g}")

    # Save
    df_auc.to_csv(OUT / "56_per_class_v12_v24_shift.csv", index=False)
    df_shift.to_csv(OUT / "56_all_class_shift.csv", index=False)
    print(f"\nSaved → {OUT}/56_*.csv")

    # ── Q6: summary of WHY ──
    print(f"\n=== SUMMARY: what changed, v12 → v24 ===")
    total_classes_moved = (df_shift.ks_stat > 0.1).sum()
    print(f"  - {total_classes_moved}/234 classes had KS statistic > 0.1 (meaningful shift)")
    pos_classes = (df_auc.auc_gain > 0.01).sum()
    neg_classes = (df_auc.auc_gain < -0.01).sum()
    print(f"  - Local eval: {pos_classes} classes gained >0.01, {neg_classes} lost >0.01")
    if pos_classes > 0:
        top_gainers = df_auc.sort_values("auc_gain", ascending=False).head(5)
        print(f"\n  Top gainers (these likely explain most of LB +0.001):")
        for _, r in top_gainers.iterrows():
            print(f"    {r['class']:<12} ({r.taxon:<9}) v12={r.auc_v12:.3f} → v24={r.auc_v24:.3f}  (n_pos={int(r.n_pos)}, n_ta={int(r.n_ta)})")


if __name__ == "__main__":
    main()
