#!/usr/bin/env python3
"""exp60 — Extended candidate grid + LOFO detail.

Builds on exp57/exp58. Tests:
  - v28 (27-head additive) variants: w27 ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}
  - v28 + class-conditional combined
  - v28 with different base (v24 instead of v26)
  - v28 with different gauss sigma
  - Per-file LOFO breakdown
  - exp51 prediction site-conditional analysis
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch, torch.nn as nn
import timm, torchaudio
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
EXP51 = ROOT / "experiments/exp51_outputs"
OUT = ROOT / "experiments/exp60_outputs"
OUT.mkdir(exist_ok=True)
SR = 32000; CLIP_SEC = 20; CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000; DEVICE = "cuda"
SEED = 42
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_site(fn):
    m = FNAME_RE.match(fn); return m.group(2) if m else None


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
    sc_g["site"] = sc_g["filename"].apply(parse_site)
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


class _SEDFlat(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=40)
        self.backbone = timm.create_model("hgnetv2_b0.ssld_stage2_ft_in1k",
                                          pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.att = nn.Conv1d(feat.shape[1], n_cls, 1)
        self.cla = nn.Conv1d(feat.shape[1], n_cls, 1)
    def forward(self, x):
        m = self.adb(self.mel(x)).unsqueeze(1)
        m = m.transpose(1,2); m = self.bn0(m); m = m.transpose(1,2)
        f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
        a = self.att(f); c = self.cla(f)
        return (torch.softmax(a, dim=-1) * c).sum(-1)


@torch.no_grad()
def predict(model, df, n_cls):
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


def per_taxon_eval(P, P_ref, Y, sc_all, species_taxon, label):
    """Compute held-out 11-file metrics with per-taxon detail."""
    ev_mask = (sc_all.split == "eval").values
    Y_ev = Y[ev_mask]; P_ev = P[ev_mask]; P_ref_ev = P_ref[ev_mask]
    a = per_class_auc(Y_ev, P_ev); a_ref = per_class_auc(Y_ev, P_ref_ev)
    common = set(a) & set(a_ref)
    macro = np.mean([a[c] for c in common])
    macro_ref = np.mean([a_ref[c] for c in common])
    sp_row = np.mean([spearmanr(P_ref[i], P[i])[0] for i in range(len(P)) if np.isfinite(spearmanr(P_ref[i], P[i])[0])])
    out = {"label": label, "macro_eval11": macro, "delta": macro - macro_ref, "sp_row": sp_row}
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        cls = [c for c in common if species_taxon[c] == t]
        if cls:
            out[f"tx_{t}"] = float(np.mean([a[c] - a_ref[c] for c in cls]))
        else:
            out[f"tx_{t}"] = float("nan")
    return out


def main():
    print("Loading...")
    sc_all, Y_all, primary, l2i = build_all()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)
    ck50 = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED50(n_cls=234).to(DEVICE); m50.load_state_dict(ck50["state_dict"])
    P50 = predict(m50, sc_all, 234); del m50
    ck51 = torch.load(EXP51 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    target_species = ck51["target_species"]; n51 = len(target_species)
    m51 = _SEDFlat(n_cls=n51).to(DEVICE); m51.load_state_dict(ck51["state_dict"])
    P51_raw = predict(m51, sc_all, n51); del m51
    target_cols = [l2i[t] for t in target_species if t in l2i]
    P51 = np.zeros((len(sc_all), 234), dtype=np.float32)
    for i, t in enumerate(target_species):
        if t in l2i: P51[:, l2i[t]] = P51_raw[:, i]
    torch.cuda.empty_cache()

    zP = zs(perch_prob); z29 = zs(S29); z50 = zs(P50); z51 = zs(P51)

    # Bases
    v12_raw = 0.8*zP + 0.2*z29
    v12 = sigmoid(gauss_pf(v12_raw, sc_all, 0.5))
    v24_raw = 0.8*zP + 0.2*z50
    v24 = sigmoid(gauss_pf(v24_raw, sc_all, 0.5))
    v26_raw = 0.7*zP + 0.3*z50
    v26 = sigmoid(gauss_pf(v26_raw, sc_all, 0.5))

    print(f"Bases: v12 {v12.shape}  v24 {v24.shape}  v26 {v26.shape}")

    def add_27_head(base_raw, w27, sigma=0.5):
        raw = base_raw.copy()
        for c in target_cols:
            raw[:, c] = (1 - w27) * raw[:, c] + w27 * z51[:, c]
        return sigmoid(gauss_pf(raw, sc_all, sigma))

    results = []
    # Reference: v26
    results.append(per_taxon_eval(v26, v26, Y_all, sc_all, species_taxon, "v26 REF"))

    print("\n=== Variant grid ===")
    # 27-head additive on v26 base
    for w27 in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]:
        p = add_27_head(v26_raw, w27)
        r = per_taxon_eval(p, v26, Y_all, sc_all, species_taxon, f"v28 w27={w27} on v26")
        results.append(r)

    # 27-head additive on v24 base
    for w27 in [0.05, 0.10, 0.15, 0.20, 0.30]:
        p = add_27_head(v24_raw, w27)
        r = per_taxon_eval(p, v26, Y_all, sc_all, species_taxon, f"v28 w27={w27} on v24")
        results.append(r)

    # 27-head additive on v12 base (control: how good if base is v12, not v26?)
    for w27 in [0.10, 0.20, 0.30]:
        p = add_27_head(v12_raw, w27)
        r = per_taxon_eval(p, v26, Y_all, sc_all, species_taxon, f"v28 w27={w27} on v12")
        results.append(r)

    # 27-head + class-conditional combined
    for w27 in [0.10, 0.15]:
        for non_aves_w50 in [0.5, 0.7]:
            w_P = np.full(234, 0.7, dtype=np.float32)
            w_50 = np.full(234, 0.3, dtype=np.float32)
            non_mask = species_taxon != "Aves"
            w_P[non_mask] = 1 - non_aves_w50
            w_50[non_mask] = non_aves_w50
            base_raw = w_P[None,:] * zP + w_50[None,:] * z50
            p = add_27_head(base_raw, w27)
            r = per_taxon_eval(p, v26, Y_all, sc_all, species_taxon,
                               f"v28+CC w27={w27} non-Aves(P{1-non_aves_w50:.1f}/50:{non_aves_w50})")
            results.append(r)

    # Different sigma on v28 best
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        p = add_27_head(v26_raw, 0.10, sigma=sigma)
        r = per_taxon_eval(p, v26, Y_all, sc_all, species_taxon, f"v28 w27=0.10 σ={sigma}")
        results.append(r)

    df = pd.DataFrame(results).round(4)
    df.to_csv(OUT / "60_grid.csv", index=False)
    print("\n=== TOP 15 by held-out delta (with sp_row ≥ 0.99) ===")
    safe = df[(df.sp_row >= 0.99) & (df.label != "v26 REF")]
    print(safe.sort_values("delta", ascending=False).head(15).to_string(index=False))
    print("\n=== ALL config ranked by delta ===")
    print(df.sort_values("delta", ascending=False).to_string(index=False))

    # ─── LOFO per-file detailed analysis for top 3 candidates ───
    print("\n=== LOFO per-file detail for top candidates ===")
    eval_files = sorted(sc_all[sc_all.split == "eval"].filename.unique())
    print(f"Eval files: {len(eval_files)}")

    candidates_to_lofo = [
        ("v28 w27=0.10 on v26", lambda: add_27_head(v26_raw, 0.10)),
        ("v28 w27=0.20 on v26", lambda: add_27_head(v26_raw, 0.20)),
        ("v28 w27=0.10 on v24", lambda: add_27_head(v24_raw, 0.10)),
        ("v28+CC w27=0.10 non-Aves(0.3/0.7)", None),
    ]
    for label, fn in candidates_to_lofo[:3]:  # Skip last one as it's complex
        p = fn()
        print(f"\n  --- {label} ---")
        for f in eval_files:
            f_mask = (sc_all.filename == f).values
            Y_f = Y_all[f_mask]
            v26_f = v26[f_mask]
            p_f = p[f_mask]
            a26 = per_class_auc(Y_f, v26_f); ap = per_class_auc(Y_f, p_f)
            common_f = set(a26) & set(ap)
            if common_f:
                d = np.mean([ap[c] - a26[c] for c in common_f])
                site = parse_site(f)
                print(f"    {site} {f.split('_')[2]:<6}  Δ={d:+.4f}  ({len(common_f)} cls)")


if __name__ == "__main__":
    main()
