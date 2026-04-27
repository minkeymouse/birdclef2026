#!/usr/bin/env python3
"""exp77 — v33 + mel-iVAE z-kNN as additional teacher for Perch-dead classes.

Combines v33 (file-max coherence) with mel-iVAE z-space cos-sim kNN scores
applied selectively to classes where Perch p99 < 0.1.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
import timm, torchaudio

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/_data_pipelines/exp43a_outputs"
EXP50 = ROOT / "experiments/_data_pipelines/exp50_outputs"
EXP76 = ROOT / "experiments/_audits_post_v26/exp76_outputs"
SR = 32000; CLIP_SEC = 20; CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000; DEVICE = "cuda"
T_POOL = 16; SEED = 42

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_meta(fn):
    m = FNAME_RE.match(fn); return (m.group(2), int(m.group(4)[:2])) if m else (None, -1)


def build_ss_data():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g[["site","hour"]] = sc_g.filename.apply(lambda f: pd.Series(parse_meta(f)))
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
    sc_all, Y_all, primary, l2i = build_ss_data()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    species_taxon = np.array([dict(zip(tax.primary_label.astype(str), tax.class_name)).get(p, "?") for p in primary])

    # Base preds
    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    ck50 = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED(n_cls=234).to(DEVICE); m50.load_state_dict(ck50["state_dict"])
    P50 = predict_sed(m50, sc_all, 234); del m50
    torch.cuda.empty_cache()

    zP = zs(perch_prob); z50 = zs(P50)
    v26_raw = 0.7*zP + 0.3*z50
    v26 = sigmoid(gauss_pf(v26_raw, sc_all, 0.5))

    # v33 = v26 + per-file max blend α=0.10 (group by filename, not assume 12/file)
    def file_max_blend(probs, df, alpha=0.10):
        out = probs.copy()
        for fn in df.filename.unique():
            m = (df.filename == fn).values
            block = probs[m]
            fmax = block.max(axis=0, keepdims=True)
            out[m] = (1 - alpha) * block + alpha * fmax
        return out.astype(np.float32)

    v33 = file_max_blend(v26, sc_all, alpha=0.10)

    # Load mel-iVAE z (from exp76)
    z_data = np.load(EXP76 / "z_raw_mel.npz")
    Z = z_data["Z"]
    print(f"mel-iVAE Z shape: {Z.shape}")

    # Compute per-class kNN scores using TRAINING positives only (no leak)
    tr_mask = (sc_all.split == "train").values
    Y_tr = Y_all[tr_mask]
    Z_tr = Z[tr_mask]

    # For each class, compute centroid in z-space + cos sim of all rows
    z_knn_scores = np.zeros_like(perch_prob)  # (N, 234)
    for c in range(234):
        if Y_tr[:, c].sum() < 3: continue
        centroid = Z_tr[Y_tr[:, c] == 1].mean(axis=0, keepdims=True)
        sims = cosine_similarity(Z, centroid).flatten()
        # rescale to [0, 1] approximately using sigmoid of cosine similarity
        z_knn_scores[:, c] = sigmoid(sims * 5)  # sharpen

    # Compute Perch p99 to identify dead classes
    all_d = np.load(EXP43A / "perch_ss_all.npz")
    all_perch = sigmoid(all_d["scores"])
    perch_p99 = np.array([np.quantile(all_perch[:, c], 0.99) for c in range(234)])

    # Build apply_mask: class c gets mel-iVAE boost if Perch p99 < some threshold
    print("\n=== Per-class strategy ===")
    for thresh in [0.1, 0.3, 0.5]:
        mask = perch_p99 < thresh
        print(f"  Perch p99 < {thresh}: {mask.sum()} classes")

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
        out = {"label": label, "macro": macro, "delta": macro - macro_ref,
                "sp_row": float(np.mean(sp_row))}
        for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
            cls = [c for c in common if species_taxon[c] == t]
            if cls:
                out[f"tx_{t}"] = float(np.mean([a[c] - a_ref[c] for c in cls]))
        return out

    results = [eval_(v33, v33, "v33 ref")]

    print("\n=== v33 + mel-iVAE z-kNN per-class additive ===")
    # Variant 1: Apply z-kNN ONLY on Perch-dead classes
    for thresh in [0.1, 0.3, 0.5]:
        dead = perch_p99 < thresh
        for w_z in [0.10, 0.20, 0.30, 0.40]:
            v34 = v33.copy()
            for c in np.where(dead)[0]:
                v34[:, c] = (1 - w_z) * v33[:, c] + w_z * z_knn_scores[:, c]
            r = eval_(v34, v33, f"v34 dead<{thresh} w_z={w_z}")
            results.append(r)

    # Variant 2: Apply z-kNN globally with low weight
    for w_z in [0.05, 0.10, 0.15]:
        v34 = (1 - w_z) * v33 + w_z * z_knn_scores
        r = eval_(v34, v33, f"v34 global w_z={w_z}")
        results.append(r)

    df = pd.DataFrame(results).round(5)
    print("\n=== TOP 15 by macro Δ ===")
    print(df.sort_values("delta", ascending=False).head(15).to_string(index=False))
    print("\n=== sp_row ≥ 0.99 candidates ===")
    safe = df[(df.sp_row >= 0.99) & (df.delta > 0.001)]
    print(safe.sort_values("delta", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
