#!/usr/bin/env python3
"""exp68 — Multi-scale Gauss ensemble + Test-time rank-normalization.

Tests on v26 base:
  - Multi-σ Gauss: ensemble of σ ∈ {0.3, 0.5, 1.0}, average
  - Per-class adaptive σ (low-confidence classes → larger σ)
  - Test-time rank-normalization: per-class quantile transform
  - Per-class temperature calibration (z-norm on test predictions)
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
from scipy.stats import spearmanr, rankdata, norm

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
OUT = ROOT / "experiments/exp68_outputs"
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

    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    ck50 = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED(n_cls=234).to(DEVICE); m50.load_state_dict(ck50["state_dict"])
    P50 = predict_sed(m50, sc_all, 234); del m50
    torch.cuda.empty_cache()

    zP = zs(perch_prob); z50 = zs(P50)
    v26_raw = 0.7*zP + 0.3*z50
    v26 = sigmoid(gauss_pf(v26_raw, sc_all, 0.5))

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

    results = [eval_(v26, v26, "v26 ref")]

    # ─── Multi-σ Gauss ensemble ───
    print("\n=== Multi-σ Gauss ensemble ===")
    g_03 = gauss_pf(v26_raw, sc_all, 0.3)
    g_05 = gauss_pf(v26_raw, sc_all, 0.5)
    g_07 = gauss_pf(v26_raw, sc_all, 0.7)
    g_10 = gauss_pf(v26_raw, sc_all, 1.0)
    p_avg357 = sigmoid((g_03 + g_05 + g_07) / 3)
    p_avg35710 = sigmoid((g_03 + g_05 + g_07 + g_10) / 4)
    p_avg357_w = sigmoid(0.25*g_03 + 0.5*g_05 + 0.25*g_07)  # weighted toward σ=0.5
    p_avg510 = sigmoid((g_05 + g_10) / 2)

    for label, p in [
        ("avg σ={0.3, 0.5, 0.7}", p_avg357),
        ("avg σ={0.3, 0.5, 0.7, 1.0}", p_avg35710),
        ("w-avg 0.25/0.5/0.25 σ", p_avg357_w),
        ("avg σ={0.5, 1.0}", p_avg510),
    ]:
        r = eval_(p, v26, f"G1: {label}")
        results.append(r)
        print(f"  {r['label']:<55}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves {r.get('tx_Aves',0):+.3f}")

    # ─── Per-class adaptive σ ───
    # Compute Perch p99 from earlier exp65/66 cache
    all_d = np.load(EXP43A / "perch_ss_all.npz")
    all_perch = sigmoid(all_d["scores"])
    perch_p99 = np.array([np.quantile(all_perch[:, c], 0.99) for c in range(234)])
    print(f"\n=== Per-class adaptive σ (low-confidence → larger σ) ===")
    # Each class gets σ = 0.5 if p99 ≥ 0.3, else σ = 1.0
    p_adaptive = np.zeros_like(v26)
    for c in range(234):
        sig = 0.5 if perch_p99[c] >= 0.3 else 1.0
        out_c = np.zeros(len(sc_all))
        for fn in sc_all.filename.unique():
            m = (sc_all.filename == fn).values
            out_c[m] = gaussian_filter1d(v26_raw[m, c], sigma=sig, mode="nearest")
        p_adaptive[:, c] = out_c
    p_adaptive = sigmoid(p_adaptive)
    r = eval_(p_adaptive, v26, "G2: per-class adaptive σ (alive=0.5, dead=1.0)")
    results.append(r)
    print(f"  {r['label']:<55}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves {r.get('tx_Aves',0):+.3f}")

    # ─── Test-time rank normalization ───
    print(f"\n=== Test-time rank normalization ===")
    # For each class, replace prediction with its rank quantile mapped to standard normal
    def rank_normalize_per_class(P):
        N, C = P.shape
        out = np.zeros_like(P)
        for c in range(C):
            ranks = rankdata(P[:, c]) / (N + 1)
            out[:, c] = norm.ppf(ranks)
        # Map back to [0, 1] via sigmoid
        return sigmoid(out)

    p_rn = rank_normalize_per_class(v26)
    r = eval_(p_rn, v26, "RN1: per-class quantile → normal")
    results.append(r)
    print(f"  {r['label']:<55}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves {r.get('tx_Aves',0):+.3f}")

    # ─── Cross-class temperature: equalize prediction std ───
    target_std = 0.15
    p_tnorm = v26.copy()
    cur_std = p_tnorm.std(axis=0, keepdims=True) + 1e-8
    cur_mean = p_tnorm.mean(axis=0, keepdims=True)
    p_tnorm = cur_mean + (p_tnorm - cur_mean) * (target_std / cur_std)
    p_tnorm = np.clip(p_tnorm, 0, 1)
    r = eval_(p_tnorm, v26, "RN2: per-class std equalization to 0.15")
    results.append(r)
    print(f"  {r['label']:<55}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves {r.get('tx_Aves',0):+.3f}")

    # ─── Combined: best of each ───
    print(f"\n=== Combined: G1 + RN ===")
    p_combined = rank_normalize_per_class(p_avg357)
    r = eval_(p_combined, v26, "COMB: avg σ + rank normalization")
    results.append(r)
    print(f"  {r['label']:<55}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves {r.get('tx_Aves',0):+.3f}")

    df = pd.DataFrame(results).round(5)
    df.to_csv(OUT / "68_results.csv", index=False)
    print("\n=== ALL RANKED ===")
    print(df.sort_values("delta", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
