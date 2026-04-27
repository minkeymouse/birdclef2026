#!/usr/bin/env python3
"""exp48g — All-lever combined. Final best-of-breed config.

Levers tested:
  (1) site prior (leak-free, from train 55-file labeled SS)
  (2) confusion-cluster rewrite (leak-free, derived from train)
  (3) exp47 added as third teacher blend

Configs:
  Base: v12 (Perch + 0.2*SED29 + Gauss σ=0.5)
  Add exp47: Base + w47*exp47 in z-space before sigmoid
  Apply site prior: final *= (tau * P(sp|site) + (1-tau))
  Apply cluster rewrite: final[target] *= (1 + alpha * cluster_score)

Sweep:
  w47 ∈ {0.0, 0.15, 0.25}
  tau ∈ {0.0, 0.5, 0.75}
  alpha ∈ {0.0, 2.0, 4.0}

Report macro + per-taxon + bottom-8 for best few configs.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
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
EXP47 = ROOT / "experiments/exp47_outputs"
OUT = ROOT / "experiments/exp48_outputs"
SEED = 42; EVAL_N = 11; SR = 32000; CLIP_SEC = 20
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000
DEVICE = "cuda"

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_site(fn):
    m = FNAME_RE.match(fn); return m.group(2) if m else None


def build_splits():
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
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N])
    sc_train_df = sc_g[~sc_g.filename.isin(eval_files)].reset_index(drop=True)
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    l2i = {p: i for i, p in enumerate(primary)}
    def Ym(df):
        Y = np.zeros((len(df), len(primary)), dtype=np.uint8)
        for i, labs in enumerate(df["lbls"]):
            for l in labs:
                if l in l2i: Y[i, l2i[l]] = 1
        return Y
    return sc_train_df, Ym(sc_train_df), sc_eval, Ym(sc_eval), primary, l2i


def align_43a(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(df), scs.shape[1]), np.float32)
    for i, rid in enumerate(df["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]
    return S


def align_old(df, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(df), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(df["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s

def gauss_pf(scores, df, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in df["filename"].unique():
        m = (df["filename"] == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out

def per_class_auc(Y, P):
    ev = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    return {c: float(roc_auc_score(Y[:, c], P[:, c])) for c in ev
            if np.isfinite(P[:, c]).all()}


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)

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
def predict_exp47(df, primary):
    ck = torch.load(EXP47 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    model = SEDModel(n_cls=len(primary)).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
    n = len(df); out = np.zeros((n, len(primary)), dtype=np.float32)
    cache = {}; batch = 8
    for i in range(0, n, batch):
        j = min(n, i + batch)
        wavs = []
        for k in range(i, j):
            row = df.iloc[k]
            if row.filename not in cache:
                w, sr = sf.read(DATA / "train_soundscapes" / row.filename, dtype="float32")
                if w.ndim > 1: w = w.mean(1)
                cache[row.filename] = w
            wav = cache[row.filename]
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


def derive_clusters(Y_train, P_train, species_taxon, top_k=3, min_pos=3):
    aves_idx = np.array([c for c in range(Y_train.shape[1]) if species_taxon[c] == "Aves"])
    cm = {}
    for c in range(Y_train.shape[1]):
        if species_taxon[c] in ("Aves", "?"): continue
        if Y_train[:, c].sum() < min_pos: continue
        pos = np.where(Y_train[:, c] == 1)[0]
        mp = P_train[pos][:, aves_idx].mean(axis=0)
        cm[c] = aves_idx[np.argsort(mp)[-top_k:]].tolist()
    return cm


def build_site_prior(sc_train_df, l2i, sites_all):
    site_idx = {s: i for i, s in enumerate(sites_all)}
    sp = np.zeros((len(sites_all), len(l2i)), dtype=np.float32)
    for site, grp in sc_train_df.groupby("site"):
        si = site_idx.get(site)
        if si is None: continue
        cnt = np.zeros(len(l2i), dtype=np.float32)
        for _, r in grp.iterrows():
            for l in r.lbls:
                if l in l2i: cnt[l2i[l]] += 1
        sp[si] = cnt / (cnt.max() + 1e-8)
    return sp, site_idx


def main():
    sc_train_df, Y_train, sc_eval, Y_eval, primary, l2i = build_splits()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    # Base predictions on both
    def v12_preds(df):
        S_p = sigmoid(align_43a(df))
        S29 = np.nan_to_num(align_old(df, EXP29 / "val_scores.npz"), nan=0)
        return S_p, S29

    print("Loading base preds...")
    P_p_tr, P_29_tr = v12_preds(sc_train_df)
    P_p_ev, P_29_ev = v12_preds(sc_eval)
    print("Computing exp47 preds...")
    P_47_tr = predict_exp47(sc_train_df, primary)
    P_47_ev = predict_exp47(sc_eval, primary)

    # Site prior from train
    sites_all = sorted(set(sc_train_df.site.unique()) | set(sc_eval.site.unique()))
    sp_norm, site_idx = build_site_prior(sc_train_df, l2i, sites_all)
    eval_site_vec = np.ones((len(sc_eval), len(primary)), dtype=np.float32)
    for i, r in sc_eval.iterrows():
        si = site_idx.get(r.site)
        if si is not None: eval_site_vec[i] = sp_norm[si]

    # Cluster map from train (using v12 on train WITHOUT exp47)
    P_v12_train = sigmoid(gauss_pf(0.8 * zs(P_p_tr) + 0.2 * zs(P_29_tr), sc_train_df, 0.5))
    cm = derive_clusters(Y_train, P_v12_train, species_taxon, top_k=3)
    print(f"Derived {len(cm)} cluster mappings from train")

    def pipeline(P_p, P_29, P_47, df, Y, w47=0.0, tau=0.0, alpha=0.0, use_site_vec=None, cluster_map=None):
        """Build blend, apply gauss, site prior, cluster rewrite."""
        raw = 0.8 * zs(P_p) + 0.2 * zs(P_29)
        if w47 > 0:
            raw = raw + w47 * zs(P_47)
        smoothed = gauss_pf(raw, df, 0.5)
        prob = sigmoid(smoothed)
        if tau > 0 and use_site_vec is not None:
            prob = prob * (tau * use_site_vec + (1 - tau))
        if alpha > 0 and cluster_map is not None:
            for tc, trig in cluster_map.items():
                sc_arr = prob[:, trig].min(axis=1)
                prob[:, tc] = prob[:, tc] * (1 + alpha * sc_arr)
        return prob

    # Base
    base = pipeline(P_p_ev, P_29_ev, P_47_ev, sc_eval, Y_eval)
    base_aucs = per_class_auc(Y_eval, base)
    base_macro = np.mean(list(base_aucs.values()))
    print(f"v12 base macro: {base_macro:.4f}")

    # Grid
    results = []
    for w47 in [0.0, 0.15, 0.25]:
        for tau in [0.0, 0.5, 0.75]:
            for alpha in [0.0, 2.0, 4.0]:
                p = pipeline(P_p_ev, P_29_ev, P_47_ev, sc_eval, Y_eval, w47, tau, alpha,
                             eval_site_vec if tau > 0 else None,
                             cm if alpha > 0 else None)
                aucs = per_class_auc(Y_eval, p)
                m = np.mean([aucs[c] for c in base_aucs if c in aucs])
                per_tax = {}
                for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
                    cls = [c for c in base_aucs if species_taxon[c] == t and c in aucs]
                    if cls: per_tax[t] = np.mean([aucs[c] for c in cls])
                results.append({
                    "w47": w47, "tau": tau, "alpha": alpha,
                    "macro": m, "delta": m - base_macro,
                    **{f"tx_{t}": per_tax.get(t, float("nan")) for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]},
                })

    df_res = pd.DataFrame(results).sort_values("macro", ascending=False)
    print("\n=== Full grid (top-15 by macro) ===")
    print(df_res.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Full grid (bottom-5) ===")
    print(df_res.tail(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Per-class detail for the TOP config
    top = df_res.iloc[0]
    p_best = pipeline(P_p_ev, P_29_ev, P_47_ev, sc_eval, Y_eval,
                      top["w47"], top["tau"], top["alpha"],
                      eval_site_vec if top["tau"] > 0 else None,
                      cm if top["alpha"] > 0 else None)
    aucs_best = per_class_auc(Y_eval, p_best)
    deltas = [(c, aucs_best[c] - base_aucs[c]) for c in base_aucs if c in aucs_best]
    deltas.sort(key=lambda x: x[1])
    print(f"\n=== TOP CONFIG: w47={top['w47']:.2f} tau={top['tau']:.2f} alpha={top['alpha']:.2f}  macro={top['macro']:.4f} Δ{top['delta']:+.4f} ===")
    print("WORST 5 (watch for rare-taxa overfit):")
    for c, d in deltas[:5]:
        print(f"  {primary[c]:<12} ({species_taxon[c]:<8}) {base_aucs[c]:.3f} → {aucs_best[c]:.3f}  Δ{d:+.3f}")
    print("BEST 10:")
    for c, d in deltas[-10:]:
        print(f"  {primary[c]:<12} ({species_taxon[c]:<8}) {base_aucs[c]:.3f} → {aucs_best[c]:.3f}  Δ{d:+.3f}")

    # Anti-correlation proxy: Aves Δ vs overall Δ
    print("\n=== Anti-correlation safety check: Aves vs overall Δ per config ===")
    df_res["aves_delta"] = df_res["tx_Aves"] - df_res[df_res.w47.eq(0) & df_res.tau.eq(0) & df_res.alpha.eq(0)].iloc[0]["tx_Aves"]
    df_res = df_res.sort_values("aves_delta", ascending=False)
    print("Top-5 by Aves Δ (LOCAL-SAFE configs that increase Aves too):")
    print(df_res.head(5)[["w47", "tau", "alpha", "macro", "delta", "aves_delta"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nConfigs with near-zero Aves Δ BUT positive overall Δ (LB-safer candidates):")
    safe = df_res[(df_res.aves_delta.abs() < 0.01) & (df_res.delta > 0.03)].sort_values("delta", ascending=False)
    print(safe[["w47", "tau", "alpha", "macro", "delta", "aves_delta"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df_res.to_csv(OUT / "48g_grid.csv", index=False)
    print(f"\nSaved grid → {OUT}/48g_grid.csv")


if __name__ == "__main__":
    main()
