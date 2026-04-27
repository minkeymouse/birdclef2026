#!/usr/bin/env python3
"""exp62 — OOF per-class adaptive blend (lightweight MoE).

Question: can we use different blend weights for different classes,
estimated WITHOUT leak?

Method:
  1. Split 66 labeled SS files into 5 folds (random by file)
  2. For each fold:
       - Compute per-class AUC for Perch alone and exp50 alone on held-out fold
       - Aggregate across all 5 folds → OOF per-class AUC for each teacher
  3. Per-class blend weight rule (no looking at held-out 11):
     - if exp50_OOF >> Perch_OOF (gap > 0.15) AND exp50_OOF > 0.7: w_50 = 0.7
     - if Perch_OOF > exp50_OOF: w_50 = 0.2 (default-ish)
     - default for sparse/unsure: w_50 = 0.3 (matches v26 global)
  4. Apply on the 11 held-out file eval to validate generalization.

Compares:
  v26 (global w_50=0.3) vs v29-routing (per-class)
  Reports: per-class outcomes, macro improvement, sp_row vs v26.
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
OUT = ROOT / "experiments/exp62_outputs"
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
def zs(X):
    m = X.mean(0,keepdims=True); s = X.std(0,keepdims=True) + 1e-8
    return (X - m) / s
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

    # Load preds
    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)
    ck50 = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED50(n_cls=234).to(DEVICE); m50.load_state_dict(ck50["state_dict"])
    P50 = predict_sed(m50, sc_all, 234); del m50
    torch.cuda.empty_cache()

    # Apply Gauss smoothing to ALL preds (consistent with v26 pipeline)
    perch_smooth = sigmoid(gauss_pf(zs(perch_prob), sc_all, 0.5))
    sed29_smooth = sigmoid(gauss_pf(zs(S29), sc_all, 0.5))
    p50_smooth = sigmoid(gauss_pf(zs(P50), sc_all, 0.5))

    # ─── 5-fold OOF per-class AUC for each teacher (on TRAIN 55 files) ───
    print("\n=== 5-fold OOF per-class AUC on train 55 files ===")
    train_files = sorted(sc_all[sc_all.split == "train"].filename.unique())
    print(f"  Train files: {len(train_files)}")
    rng = np.random.RandomState(SEED)
    fold_assign = rng.choice(5, size=len(train_files))
    file_to_fold = dict(zip(train_files, fold_assign))

    # Per-class OOF AUC: collect predictions on each held-out fold, concat
    oof_perch = np.zeros((sc_all.split == "train").sum(), dtype=np.float32)
    oof_p50 = np.zeros((sc_all.split == "train").sum(), dtype=np.float32)
    # Actually we just need per-class AUC on the train portion using OOF AUC.
    # Since predictions are produced by the SAME model regardless of which fold,
    # OOF here just means evaluate on each fold separately and concat.
    # The model is NOT retrained per fold (we don't have that infrastructure).
    # So this is essentially per-fold AUC averaged — not strict OOF, but a
    # generalization-stability indicator since each fold uses different files.

    n_classes = 234
    perch_oof_auc = {c: [] for c in range(n_classes)}
    p50_oof_auc = {c: [] for c in range(n_classes)}
    for fold in range(5):
        fold_mask = sc_all.filename.isin([f for f in train_files if file_to_fold[f] == fold]).values
        fold_mask = fold_mask & (sc_all.split == "train").values
        if fold_mask.sum() == 0: continue
        Y_f = Y_all[fold_mask]
        P_f = perch_smooth[fold_mask]; P50_f = p50_smooth[fold_mask]
        for c in range(n_classes):
            y = Y_f[:, c]
            if y.sum() == 0 or y.sum() == len(y): continue
            try:
                perch_oof_auc[c].append(roc_auc_score(y, P_f[:, c]))
                p50_oof_auc[c].append(roc_auc_score(y, P50_f[:, c]))
            except: pass

    perch_avg = {c: float(np.mean(v)) for c, v in perch_oof_auc.items() if len(v) >= 2}
    p50_avg = {c: float(np.mean(v)) for c, v in p50_oof_auc.items() if len(v) >= 2}
    common = sorted(set(perch_avg) & set(p50_avg))
    print(f"  {len(common)} classes evaluated in OOF (≥2 folds)")

    # ─── Routing rule ───
    print("\n=== Per-class routing rule ===")
    print(f"  Default: w_P = 0.7, w_50 = 0.3 (v26 global)")
    print(f"  Rule R1: if Perch_OOF < 0.5 AND p50_OOF > 0.7: w_P = 0.2, w_50 = 0.8")
    print(f"  Rule R2: if p50_OOF < 0.5 AND Perch_OOF > 0.7: w_P = 1.0, w_50 = 0.0 (skip exp50)")
    print(f"  Rule R3: if both > 0.7 AND p50 > Perch by 0.15: w_P = 0.4, w_50 = 0.6")

    weights_R1 = {}
    weights_R2 = {}
    weights_R3 = {}
    for c in common:
        p_auc = perch_avg[c]; e_auc = p50_avg[c]
        # Default
        wP, w50 = 0.7, 0.3
        if p_auc < 0.5 and e_auc > 0.7:
            wP, w50 = 0.2, 0.8
            weights_R1[c] = (wP, w50, p_auc, e_auc)
        elif e_auc < 0.5 and p_auc > 0.7:
            wP, w50 = 1.0, 0.0
            weights_R2[c] = (wP, w50, p_auc, e_auc)
        elif e_auc > 0.7 and p_auc > 0.7 and (e_auc - p_auc) > 0.15:
            wP, w50 = 0.4, 0.6
            weights_R3[c] = (wP, w50, p_auc, e_auc)

    print(f"\n  R1 candidates ({len(weights_R1)}): Perch reversed, exp50 strong → exp50-dominant")
    for c, (wP, w50, p, e) in list(weights_R1.items())[:15]:
        print(f"    {primary[c]:<14} ({species_taxon[c]:<9}) Perch_OOF={p:.3f}  exp50_OOF={e:.3f}  → w50={w50}")
    print(f"\n  R2 candidates ({len(weights_R2)}): exp50 reversed, Perch strong → drop exp50")
    for c, (wP, w50, p, e) in list(weights_R2.items())[:15]:
        print(f"    {primary[c]:<14} ({species_taxon[c]:<9}) Perch_OOF={p:.3f}  exp50_OOF={e:.3f}  → w50={w50}")
    print(f"\n  R3 candidates ({len(weights_R3)}): both strong, exp50 better → tilt toward exp50")
    for c, (wP, w50, p, e) in list(weights_R3.items())[:15]:
        print(f"    {primary[c]:<14} ({species_taxon[c]:<9}) Perch_OOF={p:.3f}  exp50_OOF={e:.3f}  → w50={w50}")

    # Build per-class weight arrays
    wP_arr = np.full(234, 0.7, dtype=np.float32)
    w50_arr = np.full(234, 0.3, dtype=np.float32)
    for c, (wP, w50, _, _) in {**weights_R1, **weights_R2, **weights_R3}.items():
        wP_arr[c] = wP; w50_arr[c] = w50

    # ─── Apply on held-out 11 files ───
    print("\n=== Apply per-class blend on held-out 11 files ===")
    zP_full = zs(perch_prob); z50_full = zs(P50)
    v26_raw = 0.7 * zP_full + 0.3 * z50_full
    v26 = sigmoid(gauss_pf(v26_raw, sc_all, 0.5))
    v29_raw = wP_arr[None, :] * zP_full + w50_arr[None, :] * z50_full
    v29 = sigmoid(gauss_pf(v29_raw, sc_all, 0.5))

    ev_mask = (sc_all.split == "eval").values
    Y_ev = Y_all[ev_mask]
    aucs_v26 = per_class_auc(Y_ev, v26[ev_mask])
    aucs_v29 = per_class_auc(Y_ev, v29[ev_mask])
    common_ev = sorted(set(aucs_v26) & set(aucs_v29))
    macro_v26 = np.mean([aucs_v26[c] for c in common_ev])
    macro_v29 = np.mean([aucs_v29[c] for c in common_ev])
    print(f"\n  v26 macro held-out: {macro_v26:.4f}")
    print(f"  v29 (per-class) macro held-out: {macro_v29:.4f}  Δ={macro_v29-macro_v26:+.4f}")

    # Per-taxon
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        cls = [c for c in common_ev if species_taxon[c] == t]
        if cls:
            d26 = np.mean([aucs_v26[c] for c in cls])
            d29 = np.mean([aucs_v29[c] for c in cls])
            print(f"    {t:<10}  n={len(cls):2d}  v26={d26:.3f}  v29={d29:.3f}  Δ={d29-d26:+.4f}")

    # Spearman per row
    sp_row = []
    for i in range(len(sc_all)):
        r, _ = spearmanr(v26[i], v29[i])
        if np.isfinite(r): sp_row.append(r)
    print(f"\n  Spearman row mean: {np.mean(sp_row):.4f}  min: {np.min(sp_row):.4f}")

    # Report classes that changed most
    print("\n  Top class movements v26 → v29:")
    deltas = sorted([(c, aucs_v29[c] - aucs_v26[c]) for c in common_ev], key=lambda x: x[1])
    print("    Worst drops:")
    for c, d in deltas[:5]:
        print(f"      {primary[c]:<14} ({species_taxon[c]:<9}) v26={aucs_v26[c]:.3f}  v29={aucs_v29[c]:.3f}  Δ{d:+.3f}  "
              f"P_OOF={perch_avg.get(c, '?'):.3f}  e_OOF={p50_avg.get(c, '?'):.3f}")
    print("    Best gains:")
    for c, d in deltas[-10:]:
        print(f"      {primary[c]:<14} ({species_taxon[c]:<9}) v26={aucs_v26[c]:.3f}  v29={aucs_v29[c]:.3f}  Δ{d:+.3f}  "
              f"P_OOF={perch_avg.get(c, '?'):.3f}  e_OOF={p50_avg.get(c, '?'):.3f}")

    # ─── Try SOFT blending: w_50[c] = sigmoid((p50 - perch) * scale) ───
    print("\n=== Soft-routed per-class blend (sigmoid(diff)) ===")
    for scale in [3.0, 5.0, 10.0]:
        wP_soft = np.full(234, 0.7, dtype=np.float32)
        w50_soft = np.full(234, 0.3, dtype=np.float32)
        for c in common:
            diff = p50_avg[c] - perch_avg[c]
            t = float(1.0 / (1.0 + np.exp(-scale * diff)))  # 0..1
            # Map t in [0,1] to w50 in [0.1, 0.7], wP = 1 - w50
            w50_c = 0.1 + 0.6 * t
            w50_soft[c] = w50_c; wP_soft[c] = 1 - w50_c
        v_soft_raw = wP_soft[None, :] * zP_full + w50_soft[None, :] * z50_full
        v_soft = sigmoid(gauss_pf(v_soft_raw, sc_all, 0.5))
        a_soft = per_class_auc(Y_ev, v_soft[ev_mask])
        c_s = set(a_soft) & set(aucs_v26)
        m_s = np.mean([a_soft[c] for c in c_s])
        sp_s = np.mean([spearmanr(v26[i], v_soft[i])[0] for i in range(len(sc_all))
                         if np.isfinite(spearmanr(v26[i], v_soft[i])[0])])
        # Per-taxon
        print(f"  scale={scale}  macro_v_soft={m_s:.4f}  Δ={m_s-macro_v26:+.4f}  sp_row={sp_s:.4f}")


if __name__ == "__main__":
    main()
