#!/usr/bin/env python3
"""exp65 — Class-conditional blend by Perch ALIVENESS (universal, not distribution-specific).

Insight from exp64: Perch's p99 across 10,658 unlabeled SS is a UNIVERSAL property
(reflects whether Perch's training data includes that class). Use this — not
OOF AUC — to determine per-class blend weights. Should transfer because Perch's
confidence on a species is determined by its training, not test distribution.

Rules:
  Bucket A (p99 < 0.1, "dead"): w_P = 0, w_50 = 1 (drop Perch noise entirely)
  Bucket B (0.1 ≤ p99 < 0.3, "semi-dead"): w_P = 0.3, w_50 = 0.7
  Bucket C (p99 ≥ 0.3, "alive"): w_P = 0.7, w_50 = 0.3 (v26 default)

Compare: v26 (global), v30_aliveness (proposed), variants.
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
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
OUT = ROOT / "experiments/exp65_outputs"
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

    # Compute Perch p99 on ALL 10,658 unlabeled SS — universal property
    print("Computing Perch p99 on all 10K unlabeled SS...")
    all_d = np.load(EXP43A / "perch_ss_all.npz")
    all_perch = sigmoid(all_d["scores"])
    perch_p99 = np.array([np.quantile(all_perch[:, c], 0.99) for c in range(234)])

    # Bucket classes by p99
    bucket_a = (perch_p99 < 0.1)  # dead
    bucket_b = (perch_p99 >= 0.1) & (perch_p99 < 0.3)  # semi-dead
    bucket_c = (perch_p99 >= 0.3)  # alive
    print(f"\nBuckets: A_dead={bucket_a.sum()}  B_semi={bucket_b.sum()}  C_alive={bucket_c.sum()}")
    print(f"  A breakdown by taxon:")
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        print(f"    {t:<10}: A_dead={(bucket_a & (species_taxon == t)).sum()}, "
              f"B_semi={(bucket_b & (species_taxon == t)).sum()}, "
              f"C_alive={(bucket_c & (species_taxon == t)).sum()}")

    zP = zs(perch_prob); z50 = zs(P50)

    def blend(wP, w50):
        """Apply per-class weight arrays."""
        if np.isscalar(wP):
            wP = np.full(234, wP, dtype=np.float32)
        if np.isscalar(w50):
            w50 = np.full(234, w50, dtype=np.float32)
        raw = wP[None, :] * zP + w50[None, :] * z50
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

    # Baseline: v26 global
    v26 = blend(0.7, 0.3)
    results = [eval_(v26, v26, "v26 global ref")]

    print("\n=== Aliveness-routed configurations ===")

    # Config 1: bucket-based, conservative
    wP = np.full(234, 0.7, dtype=np.float32)
    w50 = np.full(234, 0.3, dtype=np.float32)
    wP[bucket_a] = 0.0; w50[bucket_a] = 1.0      # dead → all exp50
    wP[bucket_b] = 0.3; w50[bucket_b] = 0.7      # semi → exp50-heavy
    p_v30 = blend(wP, w50)
    r = eval_(p_v30, v26, "v30 alive-route (dead:0/1, semi:0.3/0.7, alive:0.7/0.3)")
    results.append(r)
    print(f"  {r['label']:<60}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves Δ {r.get('tx_Aves', 0):+.3f}")

    # Config 2: only dead bucket (most conservative)
    wP = np.full(234, 0.7, dtype=np.float32)
    w50 = np.full(234, 0.3, dtype=np.float32)
    wP[bucket_a] = 0.0; w50[bucket_a] = 1.0
    p_v31 = blend(wP, w50)
    r = eval_(p_v31, v26, "v31 dead-only (dead:0/1, rest:v26)")
    results.append(r)
    print(f"  {r['label']:<60}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves Δ {r.get('tx_Aves', 0):+.3f}")

    # Config 3: more aggressive on dead
    wP = np.full(234, 0.7, dtype=np.float32)
    w50 = np.full(234, 0.3, dtype=np.float32)
    wP[bucket_a] = 0.0; w50[bucket_a] = 1.0
    wP[bucket_b] = 0.0; w50[bucket_b] = 1.0     # semi also full exp50
    p_v32 = blend(wP, w50)
    r = eval_(p_v32, v26, "v32 dead+semi=1 exp50, alive: v26")
    results.append(r)
    print(f"  {r['label']:<60}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves Δ {r.get('tx_Aves', 0):+.3f}")

    # Config 4: continuous (smoothed) — w_50 = 1 - p99
    wP_cont = perch_p99.astype(np.float32) * 0.7
    w50_cont = 1.0 - wP_cont
    # Clip to reasonable range
    wP_cont = np.clip(wP_cont, 0.1, 0.7)
    w50_cont = np.clip(w50_cont, 0.3, 0.9)
    p_v33 = blend(wP_cont, w50_cont)
    r = eval_(p_v33, v26, "v33 continuous (wP = 0.7 * p99, w50 = 1-wP)")
    results.append(r)
    print(f"  {r['label']:<60}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves Δ {r.get('tx_Aves', 0):+.3f}")

    # Config 5: extreme — 1.0 weight on exp50 for dead
    wP = np.full(234, 0.7, dtype=np.float32)
    w50 = np.full(234, 0.3, dtype=np.float32)
    wP[bucket_a] = 0.0; w50[bucket_a] = 1.5   # boost beyond 1
    p_v34 = blend(wP, w50)
    r = eval_(p_v34, v26, "v34 dead-boost (wP=0, w50=1.5 on dead)")
    results.append(r)
    print(f"  {r['label']:<60}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves Δ {r.get('tx_Aves', 0):+.3f}")

    # Config 6: drop Perch only on Insecta (taxon-based but motivated by Perch deadness)
    wP = np.full(234, 0.7, dtype=np.float32)
    w50 = np.full(234, 0.3, dtype=np.float32)
    insecta_mask = species_taxon == "Insecta"
    wP[insecta_mask] = 0.0; w50[insecta_mask] = 1.0
    p_v35 = blend(wP, w50)
    r = eval_(p_v35, v26, "v35 Insecta-only drop Perch")
    results.append(r)
    print(f"  {r['label']:<60}  m {r['macro_eval11']:.4f} Δ{r['delta']:+.4f}  sp {r['sp_row']:.4f}  Aves Δ {r.get('tx_Aves', 0):+.3f}")

    df_r = pd.DataFrame(results).round(5)
    df_r.to_csv(OUT / "65_results.csv", index=False)
    print("\n=== RANKED ===")
    print(df_r.sort_values("delta", ascending=False).to_string(index=False))

    # Show which classes get bucket A treatment
    print("\n=== Bucket A (dead) classes ===")
    for c in np.where(bucket_a)[0]:
        print(f"  {primary[c]:<14} ({species_taxon[c]:<9}) p99={perch_p99[c]:.3f}")

    print("\n=== Bucket B (semi-dead) classes — SAMPLE 20 ===")
    for c in np.where(bucket_b)[0][:20]:
        print(f"  {primary[c]:<14} ({species_taxon[c]:<9}) p99={perch_p99[c]:.3f}")


if __name__ == "__main__":
    main()
