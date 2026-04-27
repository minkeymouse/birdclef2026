#!/usr/bin/env python3
"""exp74 — Local validation of Tier 1 creative levers:
  L1 = Cross-model rank consensus (test-data-driven, universal)
  L2 = File-level temporal coherence (within-file SD signal)

Both are uncertainty-aware perturbations of v26 base, NOT lookup tables.
Goal: signal > 0.005 with sp_row > 0.95 and Aves Δ ≥ 0 → push to LB.
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
from scipy.stats import spearmanr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/_data_pipelines/exp43a_outputs"
EXP50 = ROOT / "experiments/_data_pipelines/exp50_outputs"
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
    species_taxon = np.array([dict(zip(tax.primary_label.astype(str), tax.class_name)).get(p, "?") for p in primary])

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
        out = {"label": label, "macro": macro, "delta": macro - macro_ref,
                "sp_row": float(np.mean(sp_row))}
        for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
            cls = [c for c in common if species_taxon[c] == t]
            if cls:
                out[f"tx_{t}"] = float(np.mean([a[c] - a_ref[c] for c in cls]))
        return out

    results = [eval_(v26, v26, "v26 ref")]

    # ─── Lever 1: Cross-model rank consensus ───
    print("\n=== L1: Cross-model rank consensus ===")
    # For each row, compute top-K from Perch and exp50, find consensus species
    def consensus_features(K=5):
        N, C = perch_prob.shape
        perch_topK = np.argpartition(-perch_prob, K, axis=1)[:, :K]   # (N, K)
        exp50_topK = np.argpartition(-P50, K, axis=1)[:, :K]
        # For each row, build set of consensus species
        consensus = np.zeros((N, C), dtype=np.float32)
        for r in range(N):
            p_set = set(perch_topK[r])
            e_set = set(exp50_topK[r])
            cons = p_set & e_set
            for c in cons:
                consensus[r, c] = 1.0
        return consensus, perch_topK, exp50_topK

    K_VALUES = [3, 5, 10]
    for K in K_VALUES:
        consensus, _, _ = consensus_features(K=K)
        # Variant L1a: For consensus species, BOOST prediction
        for boost in [0.05, 0.10, 0.15, 0.20]:
            v = v26 * (1 + boost * consensus)
            v = np.clip(v, 0, 1)
            r = eval_(v, v26, f"L1a boost K={K} α={boost}")
            results.append(r)
        # Variant L1b: For non-consensus species in top-K of either, SUPPRESS
        any_topK = np.zeros_like(consensus)
        # mark species in any model's top K
        ev_perch_topK = np.argpartition(-perch_prob, K, axis=1)[:, :K]
        ev_exp50_topK = np.argpartition(-P50, K, axis=1)[:, :K]
        for r in range(len(sc_all)):
            for c in ev_perch_topK[r]: any_topK[r, c] = 1.0
            for c in ev_exp50_topK[r]: any_topK[r, c] = 1.0
        non_consensus_topK = any_topK * (1 - consensus)
        for sup in [0.10, 0.20, 0.30]:
            v = v26 * (1 - sup * non_consensus_topK)
            v = np.clip(v, 0, 1)
            r = eval_(v, v26, f"L1b suppress non-consensus K={K} α={sup}")
            results.append(r)
        # Variant L1c: combined boost+suppress
        for boost, sup in [(0.10, 0.10), (0.15, 0.15)]:
            v = v26 * (1 + boost * consensus) * (1 - sup * non_consensus_topK)
            v = np.clip(v, 0, 1)
            r = eval_(v, v26, f"L1c K={K} +{boost}/-{sup}")
            results.append(r)

    # ─── Lever 2: File-level temporal coherence ───
    print("\n=== L2: File-level temporal coherence ===")
    # For each file, compute per-class SD across 12 windows
    file_sd = np.zeros_like(v26)
    file_max = np.zeros_like(v26)
    file_mean = np.zeros_like(v26)
    for fn in sc_all.filename.unique():
        m = (sc_all.filename == fn).values
        block = v26[m]
        sd = block.std(axis=0)
        mx = block.max(axis=0)
        mn = block.mean(axis=0)
        for i in np.where(m)[0]:
            file_sd[i] = sd
            file_max[i] = mx
            file_mean[i] = mn

    # L2a: boost classes with high within-file SD (signal present)
    for sd_thresh in [0.03, 0.05, 0.08, 0.10]:
        boost_mask = (file_sd > sd_thresh).astype(np.float32)
        for boost in [0.05, 0.10, 0.20]:
            v = v26 * (1 + boost * boost_mask)
            v = np.clip(v, 0, 1)
            r = eval_(v, v26, f"L2a SD>{sd_thresh} boost {boost}")
            results.append(r)

    # L2b: suppress classes with very low SD (flat = no info)
    for sd_thresh in [0.01, 0.015, 0.02]:
        flat_mask = (file_sd < sd_thresh).astype(np.float32)
        for sup in [0.05, 0.10, 0.20, 0.30]:
            v = v26 * (1 - sup * flat_mask)
            v = np.clip(v, 0, 1)
            r = eval_(v, v26, f"L2b SD<{sd_thresh} suppress {sup}")
            results.append(r)

    # L2c: replace per-row prediction with file-max (presence indicator)
    for alpha in [0.10, 0.20, 0.30]:
        v = (1 - alpha) * v26 + alpha * file_max
        r = eval_(v, v26, f"L2c blend with file_max α={alpha}")
        results.append(r)

    # L2d: combined L2a + L2b
    for sd_h in [0.05]:
        for sd_l in [0.015]:
            for boost in [0.10]:
                for sup in [0.15]:
                    boost_m = (file_sd > sd_h).astype(np.float32)
                    flat_m = (file_sd < sd_l).astype(np.float32)
                    v = v26 * (1 + boost * boost_m) * (1 - sup * flat_m)
                    v = np.clip(v, 0, 1)
                    r = eval_(v, v26, f"L2d sd>{sd_h}+{boost}/sd<{sd_l}-{sup}")
                    results.append(r)

    # ─── Combined L1 + L2 ───
    print("\n=== L1 + L2 combined ===")
    for K in [5]:
        consensus, _, _ = consensus_features(K=K)
        for sd_l in [0.015]:
            for boost in [0.10]:
                for sup in [0.15]:
                    flat_m = (file_sd < sd_l).astype(np.float32)
                    v = v26 * (1 + boost * consensus) * (1 - sup * flat_m)
                    v = np.clip(v, 0, 1)
                    r = eval_(v, v26, f"L1+L2 K={K} cons+{boost} flat-{sup}")
                    results.append(r)

    df = pd.DataFrame(results).round(5)
    print("\n=== TOP 15 by macro Δ ===")
    print(df.sort_values("delta", ascending=False).head(15).to_string(index=False))

    print("\n=== LB-safe candidates (sp_row ≥ 0.99 AND delta > 0.003) ===")
    safe = df[(df.sp_row >= 0.99) & (df.delta > 0.003)]
    if len(safe):
        print(safe.sort_values("delta", ascending=False).to_string(index=False))
    else:
        print("none")


if __name__ == "__main__":
    main()
