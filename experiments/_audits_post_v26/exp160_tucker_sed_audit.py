#!/usr/bin/env python3
"""exp160 — Tucker bc2026-distilled-sed-public 5-fold ensemble audit.

Run Tucker's 5-fold ONNX distilled SED on the 122 labeled SS eval rows,
ensemble-average the per-fold scores, and compare blend variants:

  - v33 baseline = 0.7 P + 0.3 exp50 (linear)
  - v33-style swap: 0.7 P + 0.3 tucker_5fold (replace exp50)
  - 4-way: 0.7 P + 0.15 exp50 + 0.15 tucker_5fold

Mel preprocessing (from public 0.943 notebook):
  SR=32000, n_mels=256, n_fft=2048, hop=512, fmin=20, fmax=16000
  power=2.0, top_db=80, per-spec z-score: (s - s.mean()) / (s.std() + 1e-6)

Caches scores at exp80_outputs/tucker_sed_5fold_labeled.npz.

Predicted: macro Δ +0.005 to +0.007 vs v33 ref. Profile guard: Aves Δ ≥ 0,
sp_row > 0.998, no single-class catastrophe.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import onnxruntime as ort
import torch  # noqa - just for compat with _lib

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS, DATA, SR)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate

TUCKER_DIR = ROOT / "model-weights" / "tucker_sed"
N_FOLDS = 5

# Mel config (Tucker)
N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80

WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
SS_DIR = DATA / "train_soundscapes"


def make_sed_session(onnx_path: Path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                 providers=["CPUExecutionProvider"])


def audio_to_mel(chunks):
    """Return (B, 1, n_mels, T) float32."""
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def file_to_chunks(path: Path):
    y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    n = 60 * SR
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    else:
        y = y[:n]
    chunks = [y[i*WINDOW_SAMPLES:(i+1)*WINDOW_SAMPLES] for i in range(N_WINDOWS)]
    return chunks


def get_tucker_scores(sc_g):
    """Run 5-fold ensemble on labeled SS files, return (n_rows, 234) ensemble probs."""
    cache = EXP80 / "tucker_sed_5fold_labeled.npz"
    if cache.exists():
        print(f"  Loading cached Tucker scores from {cache.name}")
        return np.load(cache)["scores"]

    files = sorted(sc_g.filename.unique())
    print(f"  Running Tucker 5-fold on {len(files)} labeled SS files")

    # Load 5 sessions
    sessions = []
    for i in range(N_FOLDS):
        p = TUCKER_DIR / f"sed_fold{i}.onnx"
        if not p.exists():
            raise FileNotFoundError(p)
        sessions.append(make_sed_session(p))
    in_name = sessions[0].get_inputs()[0].name
    print(f"  Sessions loaded; input: {in_name}")

    # row_id mapping
    fname_idx = {f: [] for f in files}
    for i, r in sc_g.iterrows():
        fname_idx[r.filename].append((i, int(r.end_sec)))

    scores_out = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    t0 = time.time()
    for fi, fn in enumerate(files):
        path = SS_DIR / fn
        chunks = file_to_chunks(path)              # 12 × 5-sec
        mel_batch = audio_to_mel(chunks)           # (12, 1, 256, T)

        # Average across 5 folds
        fold_outs = []
        for sess in sessions:
            out = sess.run(None, {in_name: mel_batch})[0]  # (12, 234)
            fold_outs.append(out)
        ensemble = np.mean(np.stack(fold_outs, axis=0), axis=0)  # (12, 234)
        # If logits, sigmoid; if already probs, leave
        if ensemble.min() < 0 or ensemble.max() > 1:
            ensemble = 1.0 / (1.0 + np.exp(-ensemble))

        # Map back to row indices via end_sec
        for row_idx, end_sec in fname_idx[fn]:
            window_idx = (end_sec // WINDOW_SEC) - 1  # end_sec=5 → window 0
            window_idx = max(0, min(N_WINDOWS - 1, window_idx))
            scores_out[row_idx] = ensemble[window_idx]

        if (fi + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (fi + 1) * (len(files) - fi - 1)
            print(f"  [{fi+1}/{len(files)}] {elapsed:.0f}s ETA {eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  Tucker inference done: {elapsed:.0f}s")
    np.savez_compressed(cache, scores=scores_out)
    print(f"  Saved {cache.name}")
    return scores_out


def main():
    print("=== exp160: Tucker 5-fold SED audit ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]

    tucker = get_tucker_scores(sc_g)
    print(f"\n  Tucker scores: {tucker.shape}, range "
          f"[{tucker.min():.5f}, {tucker.max():.5f}]")

    from scipy.stats import pearsonr
    print(f"\n  Pearson:")
    print(f"    Perch  ↔ Tucker: {pearsonr(perch_prob.flatten(), tucker.flatten())[0]:.3f}")
    print(f"    exp50  ↔ Tucker: {pearsonr(exp50.flatten(), tucker.flatten())[0]:.3f}")

    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref (0.7P+0.3 exp50)")]

    # 1) Full SED swap: replace exp50 with tucker_5fold
    base_swap = 0.7 * perch_prob + 0.3 * tucker
    g = apply_v9_gate(base_swap, perch_emb, sp_taxon, offset=0.1)
    rows.append(evaluate(file_max_blend(g, sc_g, alpha=0.10), v33, ev_mask, Y, sp_taxon,
                          "v55: 0.7P + 0.3 tucker (swap exp50)"))

    # 2) 4-way: split SED weight between exp50 and tucker
    for w_t in [0.10, 0.15, 0.20]:
        w_50 = 0.30 - w_t
        b = 0.7 * perch_prob + w_50 * exp50 + w_t * tucker
        g = apply_v9_gate(b, perch_emb, sp_taxon, offset=0.1)
        rows.append(evaluate(file_max_blend(g, sc_g, alpha=0.10), v33, ev_mask, Y, sp_taxon,
                              f"4-way: 0.7P + {w_50:.2f}exp50 + {w_t:.2f}tucker"))

    # 3) Additive on top of v33
    for w_t in [0.05, 0.10, 0.15]:
        P = (1.0 - w_t) * v33 + w_t * tucker
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon,
                              f"v33 + {w_t}*tucker additive"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal",
            "Amphib", "Reptil", "predicted"]
    print("\n=== Audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
