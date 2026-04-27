#!/usr/bin/env python3
"""exp48h — Final local validation of the full exp48g aggressive config
with the new models (exp50 instead of exp47, taxon_head_v49 instead of v45a).

Measures the same grid as exp48g (w50, tau, alpha) to find the best
submission candidate. Both defensive (w50=0) and aggressive configs
reported for user's choice.
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

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP47 = ROOT / "experiments/exp47_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
EXP49 = ROOT / "experiments/exp49_outputs"
EXP45A = ROOT / "experiments/exp45a_outputs"
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
    rng = np.random.RandomState(SEED); files = sorted(sc_g.filename.unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N])
    sc_train_df = sc_g[~sc_g.filename.isin(eval_files)].reset_index(drop=True)
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    l2i = {p: i for i, p in enumerate(primary)}
    def Y(df):
        y = np.zeros((len(df), len(primary)), dtype=np.uint8)
        for i, labs in enumerate(df.lbls):
            for l in labs:
                if l in l2i: y[i, l2i[l]] = 1
        return y
    return sc_train_df, Y(sc_train_df), sc_eval, Y(sc_eval), primary, l2i


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


def align_43a_emb(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    embs = d["emb"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    E = np.zeros((len(df), embs.shape[1]), np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: E[i] = embs[j]
    return E


def align_old(df, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(df), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(df.row_id.values):
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
        m = self.mel(x)
        m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        f = self.backbone(m)
        f = f.mean(dim=2) if f.dim() == 4 else f
        clip, _ = self.head(f)
        return clip


class TaxonHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=256, n_taxa=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, n_taxa))
    def forward(self, x): return self.net(x)


@torch.no_grad()
def predict_sed(df, ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = SEDModel().to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
    CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
    out = np.zeros((len(df), 234), dtype=np.float32); cache = {}; batch = 8
    for i in range(0, len(df), batch):
        j = min(len(df), i + batch); wavs = []
        for k in range(i, j):
            row = df.iloc[k]
            if row.filename not in cache:
                w, sr = sf.read(DATA / "train_soundscapes" / row.filename, dtype="float32")
                if w.ndim > 1: w = w.mean(1)
                cache[row.filename] = w
            wav = cache[row.filename]
            end_sec = int(row.end_sec)
            target_c = (end_sec - 2.5) * SR
            cs = int(max(0, target_c - CLIP_SAMPLES / 2))
            cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
            clip = wav[cs:cs + CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            wavs.append(clip.astype(np.float32))
        x = torch.from_numpy(np.stack(wavs)).to(DEVICE)
        logits = model(x)
        out[i:j] = torch.sigmoid(logits).cpu().numpy()
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


def apply_taxon_gate(probs, embs, taxon_ckpt_path):
    """V9 soft gate: multiply by taxon_prob+0.1."""
    ck = torch.load(taxon_ckpt_path, map_location=DEVICE, weights_only=False)
    m = TaxonHead().to(DEVICE); m.load_state_dict(ck["state_dict"]); m.eval()
    species_to_taxon = np.asarray(ck["species_to_taxon"], dtype=np.int64)
    with torch.no_grad():
        tp = torch.sigmoid(m(torch.from_numpy(embs).to(DEVICE))).cpu().numpy()
    gate = np.clip(tp[:, species_to_taxon] + 0.1, 0, 1)
    return probs * gate


def main():
    sc_train_df, Y_train, sc_eval, Y_eval, primary, l2i = build_splits()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    # Base preds
    print("Loading base preds...")
    P_p_tr = sigmoid(align_43a(sc_train_df)); P_p_ev = sigmoid(align_43a(sc_eval))
    E_p_ev = align_43a_emb(sc_eval)
    P_29_tr = np.nan_to_num(align_old(sc_train_df, EXP29 / "val_scores.npz"), nan=0)
    P_29_ev = np.nan_to_num(align_old(sc_eval, EXP29 / "val_scores.npz"), nan=0)

    # exp47 and exp50 preds on eval
    print("exp47 preds...")
    P_47_ev = predict_sed(sc_eval, EXP47 / "best_ckpt.pt")
    print("exp50 preds...")
    P_50_ev = predict_sed(sc_eval, EXP50 / "best_ckpt.pt")

    # v12 base + V9 gate (both v45a and v49)
    zP = zs(P_p_ev); z29 = zs(P_29_ev)
    v12_raw = 0.8 * zP + 0.2 * z29
    v12_smoothed = gauss_pf(v12_raw, sc_eval, 0.5)
    v12_prob = sigmoid(v12_smoothed)

    # Clusters from train (using v12 preds on train)
    zP_tr = zs(P_p_tr); z29_tr = zs(P_29_tr)
    v12_train = sigmoid(gauss_pf(0.8*zP_tr + 0.2*z29_tr, sc_train_df, 0.5))
    cm = derive_clusters(Y_train, v12_train, species_taxon, top_k=3)

    # Site prior from train
    sites = sorted(set(sc_train_df.site.unique()) | set(sc_eval.site.unique()))
    site_idx = {s: i for i, s in enumerate(sites)}
    sp = np.zeros((len(sites), 234), dtype=np.float32)
    for site, grp in sc_train_df.groupby("site"):
        si = site_idx[site]
        cnt = np.zeros(234, dtype=np.float32)
        for _, r in grp.iterrows():
            for l in r.lbls:
                if l in l2i: cnt[l2i[l]] += 1
        sp[si] = cnt / (cnt.max() + 1e-8)
    eval_site_vec = np.ones((len(sc_eval), 234), dtype=np.float32)
    for i, r in sc_eval.iterrows():
        si = site_idx.get(r.site)
        if si is not None: eval_site_vec[i] = sp[si]

    base_aucs = per_class_auc(Y_eval, v12_prob)
    base_macro = np.mean(list(base_aucs.values()))
    print(f"\nv12 base macro: {base_macro:.4f}")

    def pipeline(P_p, P_29, P_sed, df, Y, w_sed=0.0, use_v9_v45=False, use_v9_v49=False,
                  tau=0.0, alpha=0.0):
        raw = 0.8 * zs(P_p) + 0.2 * zs(P_29)
        if w_sed > 0 and P_sed is not None:
            raw = raw + w_sed * zs(P_sed)
        smoothed = gauss_pf(raw, df, 0.5)
        prob = sigmoid(smoothed)
        if use_v9_v45:
            prob = apply_taxon_gate(prob, E_p_ev, EXP45A / "taxon_head.pt")
        elif use_v9_v49:
            prob = apply_taxon_gate(prob, E_p_ev, EXP49 / "taxon_head_v49.pt")
        if tau > 0:
            prob = prob * (tau * eval_site_vec + (1 - tau))
        if alpha > 0:
            for tc, trig in cm.items():
                sc_arr = prob[:, trig].min(axis=1)
                prob[:, tc] = prob[:, tc] * (1 + alpha * sc_arr)
        return np.clip(prob, 0, 1)

    configs = [
        # (label, w_sed, which_sed, use_v9_v45, use_v9_v49, tau, alpha)
        ("v12 baseline",                                    0.0, None,     False, False, 0.0,  0.0),
        ("v12 + V9_v45a",                                   0.0, None,     True,  False, 0.0,  0.0),
        ("v12 + V9_v49 (2025+2026)",                        0.0, None,     False, True,  0.0,  0.0),
        ("v12 + site only",                                 0.0, None,     False, False, 0.5,  0.0),
        ("v12 + cluster only",                              0.0, None,     False, False, 0.0,  2.0),
        ("v22 DEF: v12 + V9_v45 + site + cluster",          0.0, None,     True,  False, 0.5,  2.0),
        ("v22' DEF: v12 + V9_v49 + site + cluster",         0.0, None,     False, True,  0.5,  2.0),
        ("v22'' DEF: v12 + V9_v49 + site + cluster (τ=.75)",0.0, None,     False, True,  0.75, 2.0),
        ("v23 AGG-exp47: + exp47 blend",                    0.25,P_47_ev,  False, True,  0.75, 2.0),
        ("v23' AGG-exp50: + exp50 blend",                   0.25,P_50_ev,  False, True,  0.75, 2.0),
        ("v23'' AGG-exp50 τ=.5",                            0.25,P_50_ev,  False, True,  0.5,  2.0),
        ("v23''' AGG-exp50 α=4",                            0.25,P_50_ev,  False, True,  0.75, 4.0),
    ]

    results = []
    for label, w_sed, P_sed, v45, v49, tau, alpha in configs:
        p = pipeline(P_p_ev, P_29_ev, P_sed, sc_eval, Y_eval,
                     w_sed=w_sed, use_v9_v45=v45, use_v9_v49=v49, tau=tau, alpha=alpha)
        aucs = per_class_auc(Y_eval, p)
        m = np.mean([aucs[c] for c in base_aucs if c in aucs])
        per_tax = {}
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
            cls = [c for c in base_aucs if species_taxon[c] == t and c in aucs]
            if cls: per_tax[t] = np.mean([aucs[c] for c in cls])
        results.append({"label": label, "macro": m, "delta": m - base_macro, **per_tax})
        print(f"  {label:<55}  macro {m:.4f} Δ{m-base_macro:+.4f}  "
              f"Aves Δ{per_tax.get('Aves',0)-0.822:+.3f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "48h_final_grid.csv", index=False)
    print(f"\nSaved → {OUT}/48h_final_grid.csv")

    # Pick top-3 by macro
    print("\nTop-3 configs:")
    for _, r in df.sort_values("macro", ascending=False).head(3).iterrows():
        print(f"  {r.label}  macro {r.macro:.4f} Δ{r.delta:+.4f}")


if __name__ == "__main__":
    main()
