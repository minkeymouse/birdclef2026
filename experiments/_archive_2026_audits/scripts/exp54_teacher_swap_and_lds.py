#!/usr/bin/env python3
"""exp54 — Round 2 adaptive/SSM exploration.

Focus:
  54a: exp50 teacher swap (replace SED29 with exp50 at same 0.8/0.2 weight)
  54b: exp47+exp50 3-way additive blend into v12
  54c: Kalman/LDS on logit sequence within file (not predictions)
  54d: Per-class adaptive gain based on test prediction spread
  54e: Teacher agreement filtering — only keep predictions where ≥2 teachers agree
  54f: MC dropout-like averaging via TTA-lite (mask random 5% of Perch score classes, re-evaluate)
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
EXP29 = ROOT / "experiments/exp29_outputs"
EXP41F = ROOT / "experiments/exp41f_outputs"
EXP47 = ROOT / "experiments/exp47_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
OUT = ROOT / "experiments/exp53_outputs"
SR = 32000; CLIP_SEC = 20
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000; DEVICE = "cuda"


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
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g.lbls):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_g, Y, primary, l2i


def align_43a(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]; embs = d["emb"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(df), scs.shape[1]), np.float32)
    E = np.zeros((len(df), embs.shape[1]), np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]; E[i] = embs[j]
    return S, E


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


@torch.no_grad()
def predict_sed(df, ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = SEDModel().to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
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
            target_c = (end_sec - 2.5) * SR
            cs = int(max(0, target_c - CLIP_SAMPLES/2))
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
    ev = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    return {c: float(roc_auc_score(Y[:, c], P[:, c])) for c in ev
            if np.isfinite(P[:, c]).all()}


def spearman_summary(A, B):
    r_row = []
    for i in range(A.shape[0]):
        r, _ = spearmanr(A[i], B[i])
        if np.isfinite(r): r_row.append(r)
    r_col = []
    for c in range(A.shape[1]):
        r, _ = spearmanr(A[:, c], B[:, c])
        if np.isfinite(r): r_col.append(r)
    return float(np.mean(r_row)), float(np.mean(r_col))


def eval_r(Y, P, P_ref, species_taxon, label):
    aucs = per_class_auc(Y, P)
    aucs_ref = per_class_auc(Y, P_ref)
    common = set(aucs) & set(aucs_ref)
    macro = np.mean([aucs[c] for c in common])
    macro_ref = np.mean([aucs_ref[c] for c in common])
    per_tax = {}
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        cls = [c for c in common if species_taxon[c] == t]
        if cls:
            per_tax[t] = np.mean([aucs[c] - aucs_ref[c] for c in cls])
    sr_row, sr_col = spearman_summary(P_ref, P)
    return {"label": label, "macro": macro, "delta": macro - macro_ref,
            "sp_row": sr_row, "sp_col": sr_col,
            **{f"tx_{t}_delta": per_tax.get(t, np.nan) for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}}


# ── Scheme 54c: LDS on logit sequence ──
def scheme_54c(logit_blend, df, q_proc=0.1, r_obs=0.3):
    """1D Kalman smoother on per-class logit time series within file.
    state = class logit, proc noise q_proc, obs noise r_obs."""
    out = logit_blend.copy()
    for fn in df.filename.unique():
        m = (df.filename == fn).values
        idx = np.where(m)[0]
        if len(idx) < 2: continue
        seq = logit_blend[idx]  # (T, C)
        T, C = seq.shape
        smoothed = np.zeros_like(seq)
        x_prev = seq[0]; P_prev = np.ones(C) * r_obs
        for t in range(T):
            x_pred = x_prev; P_pred = P_prev + q_proc
            K = P_pred / (P_pred + r_obs)
            x_new = x_pred + K * (seq[t] - x_pred)
            P_new = (1 - K) * P_pred
            smoothed[t] = x_new
            x_prev, P_prev = x_new, P_new
        out[idx] = smoothed
    return out


# ── Scheme 54d: per-class adaptive gain ──
def scheme_54d(probs, target_std=0.2):
    """Rescale each class so std across rows matches target_std."""
    out = probs.copy()
    cur_std = probs.std(axis=0, keepdims=True) + 1e-8
    gain = target_std / cur_std
    out = probs.mean(axis=0, keepdims=True) + gain * (probs - probs.mean(axis=0, keepdims=True))
    return np.clip(out, 0, 1)


# ── Scheme 54e: teacher agreement filter ──
def scheme_54e(probs_list, threshold=0.3):
    """Pull prediction toward low confidence when teachers disagree beyond threshold."""
    all_p = np.stack(probs_list, axis=0)  # (T, N, C)
    mean_p = all_p.mean(axis=0)
    std_p = all_p.std(axis=0)
    # Where disagreement large, pull toward mean
    pull = np.clip(std_p / threshold, 0, 1)
    return (1 - pull) * mean_p + pull * 0.5


# ── Scheme 54f: MC-dropout-like: random mask perturbation on Perch score ──
def scheme_54f(perch_score, n_samples=5, mask_frac=0.05):
    """Not really MC dropout, but: perturb Perch scores by randomly zeroing
    5% of class columns per pass, collecting outputs. Mean of perturbations.
    Tests robustness of Perch output to missing columns."""
    N, C = perch_score.shape
    rng = np.random.RandomState(42)
    acc = np.zeros_like(perch_score)
    for _ in range(n_samples):
        mask_cols = rng.choice(C, size=int(C * mask_frac), replace=False)
        p = perch_score.copy()
        p[:, mask_cols] = 0  # zero out
        acc += p
    return acc / n_samples


def main():
    print("Loading base preds...")
    sc_all, Y_all, primary, l2i = build_all()
    S_p, E_p = align_43a(sc_all)
    perch_prob = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)
    S41f = np.nan_to_num(align_old(sc_all, EXP41F / "val_scores_full.npz"), nan=0)

    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    zP = zs(perch_prob); z29 = zs(S29); z41 = zs(S41f)
    v12_raw = 0.8 * zP + 0.2 * z29
    v12_prob = sigmoid(gauss_pf(v12_raw, sc_all, 0.5))
    print(f"rows: {len(sc_all)}  v12 eval classes: {len(per_class_auc(Y_all, v12_prob))}")

    print("Loading exp47/exp50 preds...")
    P47 = predict_sed(sc_all, EXP47 / "best_ckpt.pt")
    P50 = predict_sed(sc_all, EXP50 / "best_ckpt.pt")
    z47 = zs(P47); z50 = zs(P50)
    print("  done")

    results = [eval_r(Y_all, v12_prob, v12_prob, species_taxon, "v12 (ref)")]

    print("\n--- 54a: teacher swap (SED29 → exp50 at same 0.2 weight) ---")
    for w in [0.1, 0.2, 0.3]:
        raw = 0.8 * zP + w * z50
        p = sigmoid(gauss_pf(raw, sc_all, 0.5))
        r = eval_r(Y_all, p, v12_prob, species_taxon, f"54a_exp50_w={w}")
        results.append(r)
        print(f"  w={w}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  sp_row {r['sp_row']:.3f}  Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 54a': teacher swap (SED29 → exp47) ---")
    for w in [0.1, 0.2]:
        raw = 0.8 * zP + w * z47
        p = sigmoid(gauss_pf(raw, sc_all, 0.5))
        r = eval_r(Y_all, p, v12_prob, species_taxon, f"54a_exp47_w={w}")
        results.append(r)
        print(f"  w={w}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  sp_row {r['sp_row']:.3f}  Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 54b: 3-way additive (P + SED29 + exp50) ---")
    for w29, w50 in [(0.15, 0.05), (0.1, 0.1), (0.05, 0.15), (0.1, 0.2), (0.15, 0.15)]:
        w_p = 1 - w29 - w50
        raw = w_p * zP + w29 * z29 + w50 * z50
        p = sigmoid(gauss_pf(raw, sc_all, 0.5))
        r = eval_r(Y_all, p, v12_prob, species_taxon, f"54b_3way_p={w_p:.2f}_s29={w29}_s50={w50}")
        results.append(r)
        print(f"  P={w_p:.2f} s29={w29} s50={w50}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  sp_row {r['sp_row']:.3f}  Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 54c: LDS/Kalman on logits ---")
    for q in [0.05, 0.1, 0.3]:
        smoothed_logit = scheme_54c(v12_raw, sc_all, q_proc=q, r_obs=0.3)
        # apply gauss too for fairness
        smoothed = gauss_pf(smoothed_logit, sc_all, 0.5)
        p = sigmoid(smoothed)
        r = eval_r(Y_all, p, v12_prob, species_taxon, f"54c_kalman_q={q}")
        results.append(r)
        print(f"  q={q}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  sp_row {r['sp_row']:.3f}  Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 54c-noG: LDS only, no gauss ---")
    for q in [0.1, 0.3, 1.0]:
        smoothed_logit = scheme_54c(v12_raw, sc_all, q_proc=q, r_obs=0.3)
        p = sigmoid(smoothed_logit)
        r = eval_r(Y_all, p, v12_prob, species_taxon, f"54c-noG_kalman_q={q}")
        results.append(r)
        print(f"  q={q}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  sp_row {r['sp_row']:.3f}  Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 54d: per-class adaptive gain ---")
    for target_std in [0.15, 0.25, 0.35]:
        p = scheme_54d(v12_prob, target_std=target_std)
        r = eval_r(Y_all, p, v12_prob, species_taxon, f"54d_gain_std={target_std}")
        results.append(r)
        print(f"  σ={target_std}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  sp_row {r['sp_row']:.3f}  Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 54e: teacher agreement filter (P+SED29+exp50) ---")
    for th in [0.3, 0.5, 1.0]:
        out = scheme_54e([perch_prob, sigmoid(S29), P50], threshold=th)
        # smooth
        raw = zs(out)
        p = sigmoid(gauss_pf(raw, sc_all, 0.5))
        r = eval_r(Y_all, p, v12_prob, species_taxon, f"54e_agree_th={th}")
        results.append(r)
        print(f"  th={th}  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  sp_row {r['sp_row']:.3f}  Aves_Δ {r['tx_Aves_delta']:+.4f}")

    print("\n--- 54f: Perch MC mask perturbation ---")
    perch_perturbed = scheme_54f(S_p, n_samples=5, mask_frac=0.05)
    raw = 0.8 * zs(sigmoid(perch_perturbed)) + 0.2 * z29
    p = sigmoid(gauss_pf(raw, sc_all, 0.5))
    r = eval_r(Y_all, p, v12_prob, species_taxon, "54f_mc_mask")
    results.append(r)
    print(f"  macro {r['macro']:.4f}  Δ{r['delta']:+.4f}  sp_row {r['sp_row']:.3f}  Aves_Δ {r['tx_Aves_delta']:+.4f}")

    # Save & summarize
    df = pd.DataFrame(results).sort_values("delta", ascending=False)
    df.to_csv(OUT / "54_results.csv", index=False)
    print("\n=== SUMMARY (top-10 by macro Δ) ===")
    print(df.head(10)[["label","macro","delta","sp_row","sp_col","tx_Aves_delta"]].to_string(index=False))

    safe = df[(df.sp_row >= 0.95) & (df.delta > 0.001) & (df.label != "v12 (ref)")]
    print(f"\n=== LB-safe (Spearman≥0.95, Δ>0.001) ===")
    print(safe[["label","macro","delta","sp_row","tx_Aves_delta"]].to_string(index=False))

    print(f"\nSaved → {OUT}/54_results.csv")


if __name__ == "__main__":
    main()
