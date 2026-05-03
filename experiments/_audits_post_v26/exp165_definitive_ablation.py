#!/usr/bin/env python3
"""exp165 — Definitive component ablation: what exactly is the +0.009?

Run all factor combinations on labeled SS to isolate contributions:

  Factor 1 (SED choice): exp50 vs Tucker_5fold vs Tucker_fold0_only
  Factor 2 (fusion):     linear vs rank-pct
  Factor 3 (rescues):    none vs all_3 vs each_individual

Build a controlled grid; identify which factor(s) drive the gain.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS, DATA, SR)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate

EPS = 1e-6
TUCKER_DIR = ROOT / "model-weights" / "tucker_sed"

# Tucker mel
N_MELS_T = 256; N_FFT_T = 2048; HOP_T = 512
FMIN_T = 20; FMAX_T = 16000; TOP_DB_T = 80
WIN_SAMPLES = SR * 5
N_WINDOWS = 12
SS_DIR = DATA / "train_soundscapes"

# Mattia rescue config
FAKE_ONLY_THR = 0.50
SED_LOW_THR = 0.05
FAKE_ONLY_BLEND = 0.12
PROTO_CONT_RADIUS = 3
PROTO_CONT_DF = 2.0
PROTO_CONT_SCALE = 1.20
PROTO_CONT_RANK_THR = 0.88
PROTO_LOCAL_RANK_THR = 0.75
SED_CONT_LOW_THR = 0.12
PROTO_CONT_BLEND = 0.15
SED_ONLY_RANK_THR = 0.95
FAKE_RANK_LOW_THR = 0.80
SED_ONLY_BLEND = 0.12


def rank_pct(arr):
    return pd.DataFrame(arr).rank(axis=0, pct=True).to_numpy(dtype=np.float32)


def t_dist_kernel(radius, df, scale):
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = (1.0 + (offs / scale) ** 2 / df) ** (-(df + 1.0) / 2.0)
    return (k / k.sum()).astype(np.float32)


def proto_context_rank(pa, file_ids, radius, df, scale):
    kernel = t_dist_kernel(radius, df, scale)
    pa_ctx = pa.copy()
    R = radius
    for fid in pd.unique(file_ids):
        m = file_ids == fid
        x = pa[m]
        if len(x) > 1:
            xp = np.pad(x, ((R, R), (0, 0)), mode="edge")
            pa_ctx[m] = sum(kernel[i] * xp[i:i + len(x)] for i in range(2 * R + 1))
    return pa_ctx


def fuse(streamA, streamB, file_ids, mode="linear", sed_w=0.30,
          rescues="none"):
    """Apply fusion to (streamA, streamB).

    mode: 'linear' | 'rank'
    rescues: 'none' | 'fake' | 'cont' | 'spike' | 'all'
    Returns fused predictions in [0,1].
    """
    pa = np.clip(streamA, EPS, 1.0 - EPS)
    pb = np.clip(streamB, EPS, 1.0 - EPS)

    if mode == "linear":
        pred = pa * (1.0 - sed_w) + pb * sed_w
        if rescues == "none":
            return pred
        # Rescues require rank-pct context, even in linear mode we compute them.
        xa = rank_pct(pa); xb = rank_pct(pb)
    else:  # rank
        xa = rank_pct(pa); xb = rank_pct(pb)
        pred = xa * (1.0 - sed_w) + xb * sed_w
        if rescues == "none":
            return pred

    fake_only = (pa > FAKE_ONLY_THR) & (pb < SED_LOW_THR)
    pa_ctx = proto_context_rank(pa, file_ids, PROTO_CONT_RADIUS, PROTO_CONT_DF, PROTO_CONT_SCALE)
    xctx = rank_pct(pa_ctx)
    proto_cont = ((xctx > PROTO_CONT_RANK_THR) & (xa > PROTO_LOCAL_RANK_THR)
                  & (pb < SED_CONT_LOW_THR) & (~fake_only))
    sed_only = ((xb > SED_ONLY_RANK_THR) & (xa < FAKE_RANK_LOW_THR)
                & (~fake_only) & (~proto_cont))

    if rescues in ("fake", "all"):
        pred = np.where(fake_only, (1.0 - FAKE_ONLY_BLEND) * pred + FAKE_ONLY_BLEND * xa, pred)
    if rescues in ("cont", "all"):
        pred = np.where(proto_cont,
                         (1.0 - PROTO_CONT_BLEND) * pred + PROTO_CONT_BLEND * np.maximum(xa, xctx),
                         pred)
    if rescues in ("spike", "all"):
        pred = np.where(sed_only, (1.0 - SED_ONLY_BLEND) * pred + SED_ONLY_BLEND * xb, pred)
    return pred


def get_tucker_fold0_scores(sc_g):
    """Score Tucker fold 0 only (NOT ensemble) for ablation."""
    cache = EXP80 / "tucker_fold0_labeled.npz"
    if cache.exists():
        return np.load(cache)["scores"]

    print("Computing Tucker fold0 only on labeled SS ...", flush=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(TUCKER_DIR / "sed_fold0.onnx"), sess_options=so,
                                 providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    files = sorted(sc_g.filename.unique())
    fname_idx = {f: [] for f in files}
    for i, r in sc_g.iterrows():
        fname_idx[r.filename].append((i, int(r.end_sec)))

    out = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    t0 = time.time()
    for fi, fn in enumerate(files):
        path = SS_DIR / fn
        try:
            y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            continue
        if y.ndim == 2: y = y.mean(axis=1)
        if sr0 != SR:
            y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
        n = 60 * SR
        if len(y) < n: y = np.pad(y, (0, n - len(y)))
        else: y = y[:n]
        chunks = [y[i*WIN_SAMPLES:(i+1)*WIN_SAMPLES] for i in range(N_WINDOWS)]
        mels = []
        for x in chunks:
            s = librosa.feature.melspectrogram(
                y=x, sr=SR, n_fft=N_FFT_T, hop_length=HOP_T,
                n_mels=N_MELS_T, fmin=FMIN_T, fmax=FMAX_T, power=2.0)
            s = librosa.power_to_db(s, top_db=TOP_DB_T)
            s = (s - s.mean()) / (s.std() + 1e-6)
            mels.append(s)
        mel_b = np.stack(mels)[:, None].astype(np.float32)
        logits = sess.run(None, {in_name: mel_b})[0]  # (12, 234)
        if logits.min() < 0 or logits.max() > 1:
            probs = 1.0 / (1.0 + np.exp(-logits))
        else:
            probs = logits
        for row_idx, end_sec in fname_idx[fn]:
            window_idx = max(0, min(N_WINDOWS - 1, end_sec // 5 - 1))
            out[row_idx] = probs[window_idx]
        if (fi + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{fi+1}/{len(files)}] {el:.0f}s", flush=True)
    np.savez_compressed(cache, scores=out)
    print(f"  saved {cache.name}")
    return out


def main():
    print("=== exp165: definitive component ablation ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    file_ids = sc_g["filename"].astype(str).values

    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]
    tucker_5fold = np.load(EXP80 / "tucker_sed_5fold_labeled.npz")["scores"]
    tucker_fold0 = get_tucker_fold0_scores(sc_g)

    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)
    ev_mask = sc_g.split.values == "eval"

    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 LINEAR ref")]

    # === Factor 1: SED choice (linear, no rescues, W=0.30) ===
    for sed_name, sed_arr in [("exp50", exp50),
                                ("Tucker_fold0_only", tucker_fold0),
                                ("Tucker_5fold", tucker_5fold)]:
        p = fuse(v33, sed_arr, file_ids, mode="linear", sed_w=0.30, rescues="none")
        rows.append(evaluate(p, v33, ev_mask, Y, sp_taxon,
                              f"linear SED={sed_name} W=0.30"))

    # === Factor 2: linear vs rank, no rescues, fixed Tucker_5fold ===
    for mode in ["linear", "rank"]:
        p = fuse(v33, tucker_5fold, file_ids, mode=mode, sed_w=0.30, rescues="none")
        rows.append(evaluate(p, v33, ev_mask, Y, sp_taxon,
                              f"FUSION={mode} SED=Tucker_5fold W=0.30 rescues=none"))

    # === Factor 3: rescue contribution (rank, Tucker_5fold) ===
    for r in ["none", "fake", "cont", "spike", "all"]:
        p = fuse(v33, tucker_5fold, file_ids, mode="rank", sed_w=0.30, rescues=r)
        rows.append(evaluate(p, v33, ev_mask, Y, sp_taxon,
                              f"rank Tucker_5fold rescues={r}"))

    # === Cross factors (best SED in best fusion with all rescues) ===
    for sed_name, sed_arr in [("exp50", exp50),
                                ("Tucker_fold0_only", tucker_fold0),
                                ("Tucker_5fold", tucker_5fold)]:
        p = fuse(v33, sed_arr, file_ids, mode="rank", sed_w=0.30, rescues="all")
        rows.append(evaluate(p, v33, ev_mask, Y, sp_taxon,
                              f"FULL: rank SED={sed_name} rescues=all W=0.30"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal",
            "Amphib", "Reptil", "predicted"]
    print("\n=== Audit (in order, factor by factor) ===")
    print(res[cols].to_string(index=False))
    print()
    print("=== Audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
