#!/usr/bin/env python3
"""exp48c — DSP-based Insecta presence detector (no training, physics rule).

Target: 25 single-site Insect sonotypes (47158sonXX). Features:
  - Spectral centroid > 4kHz (shift mostly in HF)
  - Spectral bandwidth narrow (standard deviation of spectrum low relative to centroid)
  - Temporal envelope auto-correlation high at ~10-50Hz (regular stridulation pulse)
  - Spectral flatness low (tonal not noisy)

Rule: insect score = mean(normalized centroid, inverse bandwidth, pulse_AC, 1-flatness)

Apply on 11-file eval:
  - boost insect columns in rows with high insect score
  - compare macro AUC before/after
  - also compare per-insect-class AUC
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
OUT = ROOT / "experiments/exp48_outputs"
OUT.mkdir(exist_ok=True)
SEED = 42; EVAL_N = 11
SR = 32000; WIN = 5 * SR  # 5-sec window


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
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N])
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    l2i = {p: i for i, p in enumerate(primary)}
    Y_eval = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["lbls"]):
        for l in labs:
            if l in l2i: Y_eval[i, l2i[l]] = 1
    return sc_eval, Y_eval, primary, l2i


def align_43a(sc_eval):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(sc_eval), scs.shape[1]), np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]
    return S


def align_old(sc_eval, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(sc_eval), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s

def gauss_pf(scores, sc_eval, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in sc_eval["filename"].unique():
        m = (sc_eval["filename"] == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out


def per_class_auc(Y, P):
    ev = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    return {c: float(roc_auc_score(Y[:, c], P[:, c])) for c in ev
            if np.isfinite(P[:, c]).all()}


def compute_insect_dsp(wav, sr=SR):
    """Return scalar 'insect score' in [0,1] for a 5-sec clip.
    Higher = more likely to contain insect stridulation.

    Features:
      - spec_centroid relative to Nyquist (push to HF → 1)
      - inverse bandwidth (narrow → 1)
      - 1 - spectral_flatness (tonal → 1)
      - amplitude envelope AC at ~10-50ms lag (regular pulse → 1)
    """
    if len(wav) == 0: return 0.0
    if len(wav) < sr: wav = np.pad(wav, (0, sr - len(wav)))
    # STFT
    S = np.abs(librosa.stft(wav, n_fft=2048, hop_length=512))  # (1025, T)
    S = np.maximum(S, 1e-10)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)  # (1025,)
    # Spectral centroid per frame
    power = S ** 2
    centroid = (freqs[:, None] * power).sum(axis=0) / (power.sum(axis=0) + 1e-10)
    # Spectral bandwidth (std of frequency around centroid)
    bandwidth = np.sqrt(((freqs[:, None] - centroid[None, :]) ** 2 * power).sum(axis=0)
                        / (power.sum(axis=0) + 1e-10))
    # Spectral flatness per frame (geometric / arithmetic mean)
    log_S = np.log(S)
    geomean = np.exp(log_S.mean(axis=0))
    arithmean = S.mean(axis=0)
    flatness = geomean / (arithmean + 1e-10)
    # Amplitude envelope
    hop = 512
    envelope = librosa.feature.rms(y=wav, frame_length=2048, hop_length=hop)[0]
    env_norm = (envelope - envelope.mean()) / (envelope.std() + 1e-8)
    # AC at lags 10-50ms (320-1600 samples at 32k; in frame units with hop=512 that's 0.6-3 frames)
    # Better: AC on envelope at frame-lag 1-10 frames (≈16-160 ms per lag)
    ac = correlate(env_norm, env_norm, mode="full")
    ac = ac[len(env_norm) - 1:]
    ac = ac / (ac[0] + 1e-10)  # normalize peak to 1
    # Take max AC in lag 1..10
    pulse_ac = ac[1:11].max() if len(ac) > 10 else 0.0

    # Normalize features to [0,1]-ish
    f_centroid = float(np.clip(centroid.mean() / (sr / 2), 0, 1))  # 0-1 of Nyquist
    # bandwidth — narrow is <500Hz, wide is >4000Hz
    f_narrow = float(np.clip(1.0 - bandwidth.mean() / 4000.0, 0, 1))
    f_tonal = float(np.clip(1.0 - flatness.mean(), 0, 1))
    f_pulse = float(np.clip(pulse_ac, 0, 1))

    score = 0.25 * f_centroid + 0.25 * f_narrow + 0.25 * f_tonal + 0.25 * f_pulse
    return score, {"centroid_hz": float(centroid.mean()), "bandwidth_hz": float(bandwidth.mean()),
                   "flatness": float(flatness.mean()), "pulse_ac": float(pulse_ac),
                   "f_centroid": f_centroid, "f_narrow": f_narrow,
                   "f_tonal": f_tonal, "f_pulse": f_pulse, "score": score}


def main():
    sc_eval, Y, primary, l2i = build_splits()
    print(f"Eval: {len(sc_eval)} rows from {len(sc_eval.filename.unique())} files")

    insect_cols = [c for c in range(len(primary)) if primary[c].startswith("47158son")]
    print(f"Insect sonotype cols: {len(insect_cols)}")

    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    # Compute insect score per row
    print("\n[Extract insect DSP score per 5-sec window]")
    insect_scores = np.zeros(len(sc_eval), dtype=np.float32)
    insect_features = []
    per_file_cache = {}
    for i, row in sc_eval.iterrows():
        fn = row.filename
        if fn not in per_file_cache:
            try:
                wav, sr = sf.read(DATA / "train_soundscapes" / fn, dtype="float32")
                if wav.ndim > 1: wav = wav.mean(1)
                if sr != SR:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
                per_file_cache[fn] = wav
            except Exception as e:
                per_file_cache[fn] = np.zeros(SR * 60, dtype=np.float32)
        wav = per_file_cache[fn]
        end_sec = int(row.end_sec)
        start = max(0, (end_sec - 5) * SR)
        end = min(len(wav), end_sec * SR)
        clip = wav[start:end]
        if len(clip) < WIN: clip = np.pad(clip, (0, WIN - len(clip)))
        try:
            score, feats = compute_insect_dsp(clip)
        except Exception:
            score, feats = 0.0, {}
        insect_scores[i] = score
        insect_features.append(feats)

    print(f"Insect scores: mean={insect_scores.mean():.3f}  std={insect_scores.std():.3f}  "
          f"min={insect_scores.min():.3f}  max={insect_scores.max():.3f}")

    # Validate: do rows with actual insect positives have higher insect_score?
    any_insect = Y[:, insect_cols].sum(axis=1) > 0
    print(f"Rows with ANY insect positive: {any_insect.sum()}/{len(sc_eval)}")
    if any_insect.any():
        pos_scores = insect_scores[any_insect]
        neg_scores = insect_scores[~any_insect]
        print(f"  mean insect_score on positive rows: {pos_scores.mean():.3f}")
        print(f"  mean insect_score on negative rows: {neg_scores.mean():.3f}")
        # AUC of insect_score as binary detector
        try:
            det_auc = roc_auc_score(any_insect.astype(int), insect_scores)
            print(f"  AUC of insect_score for any-insect-present binary: {det_auc:.3f}")
        except Exception: pass

    # Per-class detector check
    print("\n[Per-insect-class detector AUC]")
    for c in insect_cols:
        y = Y[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
        try: auc = roc_auc_score(y, insect_scores)
        except Exception: auc = float("nan")
        print(f"  {primary[c]:<12}  n_pos={y.sum():2d}  dsp_auc={auc:.3f}")

    # Build v12 base
    S_perch = align_43a(sc_eval)
    perch_prob = sigmoid(S_perch)
    S_sed29 = align_old(sc_eval, EXP29 / "val_scores.npz")
    zP = zs(perch_prob); z29 = zs(np.nan_to_num(S_sed29, nan=0))
    v12_raw = 0.8*zP + 0.2*z29
    v12_prob = sigmoid(gauss_pf(v12_raw, sc_eval, 0.5))

    base_aucs = per_class_auc(Y, v12_prob)
    base_macro = np.mean(list(base_aucs.values()))
    print(f"\n[Overlay test]")
    print(f"v12 base macro: {base_macro:.4f} ({len(base_aucs)} cls)")

    # Overlay: p_new[:, insect_cols] *= insect_scores  (scaled)
    # Variant A: multiplicative boost  final[:,insect] = base[:,insect] * (1 + insect_score * w)
    # Variant B: additive  final[:,insect] = base[:,insect] + insect_score * w
    # Variant C: rank blend  final[:,insect] = (1-w) * base[:,insect] + w * insect_score_broadcast
    print("\n=== Variant A (multiplicative boost) ===")
    for w in [0.5, 1.0, 2.0, 4.0]:
        p_new = v12_prob.copy()
        p_new[:, insect_cols] = p_new[:, insect_cols] * (1 + insect_scores[:, None] * w)
        aucs = per_class_auc(Y, p_new)
        insect_sub = [c for c in insect_cols if c in aucs]
        m_ins = np.mean([aucs[c] for c in insect_sub]) if insect_sub else 0
        m_ins_base = np.mean([base_aucs[c] for c in insect_sub])
        m_all = np.mean([aucs[c] for c in base_aucs if c in aucs])
        print(f"  w={w}  insect macro {m_ins_base:.3f} → {m_ins:.3f}  Δ={m_ins-m_ins_base:+.3f}  overall {m_all:.4f} Δ{m_all-base_macro:+.4f}")

    print("\n=== Variant C (rank blend) ===")
    for w in [0.25, 0.5, 0.75, 1.0]:
        p_new = v12_prob.copy()
        p_new[:, insect_cols] = (1 - w) * p_new[:, insect_cols] + w * insect_scores[:, None]
        aucs = per_class_auc(Y, p_new)
        insect_sub = [c for c in insect_cols if c in aucs]
        m_ins = np.mean([aucs[c] for c in insect_sub]) if insect_sub else 0
        m_ins_base = np.mean([base_aucs[c] for c in insect_sub])
        m_all = np.mean([aucs[c] for c in base_aucs if c in aucs])
        print(f"  w={w}  insect macro {m_ins_base:.3f} → {m_ins:.3f}  Δ={m_ins-m_ins_base:+.3f}  overall {m_all:.4f} Δ{m_all-base_macro:+.4f}")

    # Diagnostic: what do features look like per row
    # Sample rows where Y has insect positive to check feature signature
    print("\n[Feature signature on positive vs negative insect rows]")
    if any_insect.any():
        pos_f = [insect_features[i] for i in range(len(insect_features)) if any_insect[i]]
        neg_f = [insect_features[i] for i in range(len(insect_features)) if not any_insect[i]]
        for key in ["centroid_hz", "bandwidth_hz", "flatness", "pulse_ac"]:
            if all(key in f for f in pos_f + neg_f):
                pm = np.mean([f[key] for f in pos_f])
                nm = np.mean([f[key] for f in neg_f])
                print(f"  {key:<14}  pos={pm:.4f}  neg={nm:.4f}")

    with open(OUT / "48c_insect_dsp.json", "w") as f:
        json.dump({
            "insect_score_mean": float(insect_scores.mean()),
            "any_insect_detector_works": bool(any_insect.any()),
            "base_macro": base_macro,
        }, f, indent=2, default=float)


if __name__ == "__main__":
    main()
