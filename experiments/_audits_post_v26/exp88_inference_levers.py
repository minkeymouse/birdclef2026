#!/usr/bin/env python3
"""exp88 — Local audit of inference-time levers we never properly tested.

Tests on 122 held-out eval rows:
  T1. SED50 chunk-shift TTA (run SED50 with chunk_start_offset=10s, average)
  T2. Logit-space Gaussian smoothing (vs current prob-space σ=0.5)
  T3. Per-class temperature scaling (OOF-fit on train, apply on eval)
  T4. File-level cross-class z-score normalization
  T5. Combined: T1 + T2

Reports (macro_d, sp_row, per-taxon Δ, predicted_LB_class) for each.

Goal: identify any lever in framework Category 1 (universal physics) or
Category 2 (training-time invariance) that adds a NEW source independent
of {Perch xeno-canto, exp50 2025-BG, file-max coherence}.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, MW, DATA, ROOT, N_CLS, SR)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def get_cached(name):
    return np.load(EXP80 / name)["scores"]


def build_v33_base(perch_prob, exp50, perch_emb, sc_g, sp_taxon):
    """v33 = (0.7P + 0.3 exp50) → V9 gate → file-max α=0.10. Returns post-blend probs."""
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    return file_max_blend(gated, sc_g, alpha=0.10)


def get_exp50_shifted_scores(sc_g, shift_sec: int = 10):
    """Build exp50 inference cache with chunk start shifted by shift_sec.
    Original: chunks at 0, 20, 40 sec. Shifted: chunks at shift_sec, shift_sec+20, shift_sec+40.
    Tail wrap to keep 3 chunks of 20s within 60-sec audio.
    """
    cache = EXP80 / f"exp50_scores_shift{shift_sec}.npz"
    if cache.exists():
        return np.load(cache)["scores"]

    print(f"  building exp50 shift={shift_sec}s scores cache...", flush=True)
    import torchaudio, timm
    SED_N_MELS, SED_N_FFT, SED_HOP = 128, 2048, 512
    SED_FMIN, SED_FMAX = 50, 14000
    SED_CHUNK_SEC = 20
    SED_CHUNK_SAMPLES = SR * SED_CHUNK_SEC

    class _MelExt(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=SED_N_FFT, hop_length=SED_HOP, n_mels=SED_N_MELS,
                f_min=SED_FMIN, f_max=SED_FMAX, power=2.0, center=True)
            self.adb = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)

    class _Head(nn.Module):
        def __init__(self, fd, nc):
            super().__init__()
            self.att = nn.Conv1d(fd, nc, 1); self.cla = nn.Conv1d(fd, nc, 1)
        def forward(self, x):
            a, c = self.att(x), self.cla(x)
            return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values

    class _SED(nn.Module):
        def __init__(self, bb='hgnetv2_b0.ssld_stage2_ft_in1k'):
            super().__init__()
            self.mel = _MelExt(); self.bn0 = nn.BatchNorm2d(SED_N_MELS)
            self.backbone = timm.create_model(bb, pretrained=False, in_chans=1, num_classes=0, global_pool='')
            with torch.no_grad():
                f = self.backbone(torch.zeros(1, 1, SED_N_MELS, 100))
            self.head = _Head(f.shape[1], N_CLS)
        def forward(self, x):
            m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
            f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
            clip, _ = self.head(f); return clip

    st = torch.load(MW / "exp50_hgnet_sed.pt", map_location="cuda", weights_only=False)
    bb = st.get('backbone', 'hgnetv2_b0.ssld_stage2_ft_in1k')
    model = _SED(bb).to("cuda").eval()
    model.load_state_dict(st['state_dict'])

    SS_DIR = DATA / "train_soundscapes"
    files = sorted(sc_g.filename.unique())
    fname_idx = {f: [] for f in files}
    for i, r in sc_g.iterrows():
        fname_idx[r.filename].append((i, int(r.end_sec)))

    out = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    t0 = time.time()
    BATCH_F = 8
    shift_samples = shift_sec * SR
    audio_len = 60 * SR

    with torch.inference_mode():
        for s in range(0, len(files), BATCH_F):
            batch = files[s:s+BATCH_F]
            chunks = []; meta = []
            for bi, fn in enumerate(batch):
                y, _ = sf.read(SS_DIR / fn, dtype="float32", always_2d=False)
                if y.ndim == 2: y = y.mean(axis=1)
                if len(y) < audio_len: y = np.pad(y, (0, audio_len - len(y)))
                y = y[:audio_len]
                # Shifted chunks: start at shift, shift+20s, shift+40s. Wrap if past end.
                for ci in range(3):
                    st_idx = (shift_samples + ci * SED_CHUNK_SAMPLES) % audio_len
                    end_idx = st_idx + SED_CHUNK_SAMPLES
                    if end_idx <= audio_len:
                        chunks.append(y[st_idx:end_idx])
                    else:
                        # wrap around
                        chunks.append(np.concatenate([y[st_idx:], y[:end_idx - audio_len]]))
                    meta.append((bi, ci))
            x = torch.from_numpy(np.stack(chunks)).to("cuda")
            clip = model(x)
            p = torch.sigmoid(clip).cpu().numpy()
            for k, (bi, ci) in enumerate(meta):
                fn = batch[bi]
                # Map shifted chunk back to which 5-sec windows it covers
                shift_chunk_start = (shift_samples + ci * SED_CHUNK_SAMPLES) % audio_len
                shift_chunk_end = shift_chunk_start + SED_CHUNK_SAMPLES
                for row_idx, end_sec in fname_idx[fn]:
                    win_start_sample = (end_sec - 5) * SR  # window starts at this sample
                    win_end_sample = end_sec * SR
                    # If window center is in this chunk's range, assign
                    win_center = (win_start_sample + win_end_sample) // 2
                    if shift_chunk_end <= audio_len:
                        in_chunk = shift_chunk_start <= win_center < shift_chunk_end
                    else:
                        # wrapped chunk
                        in_chunk = (win_center >= shift_chunk_start) or (win_center < shift_chunk_end - audio_len)
                    if in_chunk:
                        out[row_idx] = p[k]
    cache.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache, scores=out)
    print(f"  cached → {cache} ({time.time()-t0:.1f}s)", flush=True)
    return out


# ---------------------------------------------------------------- Lever implementations
def T1_sed50_tta(perch_prob, exp50_orig, exp50_shifted, perch_emb, sc_g, sp_taxon, w_tta=0.5):
    """SED50 prediction = (1-w_tta)*orig + w_tta*shifted, then plug into v33 base."""
    sed50_avg = (1 - w_tta) * exp50_orig + w_tta * exp50_shifted
    return build_v33_base(perch_prob, sed50_avg, perch_emb, sc_g, sp_taxon)


def T2_logit_gauss(probs, sc_g, sigma=0.5):
    """Apply Gaussian temporal smoothing in LOGIT space across 12 windows/file."""
    from scipy.ndimage import gaussian_filter1d
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    out = logit.copy()
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = logit[idx]  # (T, C)
        smoothed = gaussian_filter1d(sub, sigma=sigma, axis=0, mode='nearest')
        out[idx] = smoothed
    return (1.0 / (1.0 + np.exp(-out))).astype(np.float32)


def T3_per_class_temperature(v33, Y, train_mask, eval_mask):
    """Fit per-class temperature on train, apply on eval. Universal calibration."""
    out = v33.copy()
    eps = 1e-6
    p = np.clip(v33, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    for c in range(N_CLS):
        if Y[train_mask, c].sum() < 5: continue
        # Find temp T that minimizes BCE on train
        best_T, best_bce = 1.0, 1e9
        y_tr = Y[train_mask, c].astype(np.float32)
        l_tr = logit[train_mask, c]
        for T in [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0]:
            pp = 1.0 / (1.0 + np.exp(-l_tr / T))
            pp = np.clip(pp, eps, 1 - eps)
            bce = -(y_tr * np.log(pp) + (1 - y_tr) * np.log(1 - pp)).mean()
            if bce < best_bce:
                best_bce, best_T = bce, T
        # Apply best_T to eval
        l_ev = logit[eval_mask, c]
        out[eval_mask, c] = 1.0 / (1.0 + np.exp(-l_ev / best_T))
    return out.astype(np.float32)


def T4_file_zscore(probs, sc_g):
    """Per file, z-score per class across 12 windows, then sigmoid back."""
    out = probs.copy()
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = logit[idx]  # (T, C)
        m = sub.mean(axis=0, keepdims=True)
        s = sub.std(axis=0, keepdims=True) + 1e-6
        z = (sub - m) / s
        out[idx] = 1.0 / (1.0 + np.exp(-z))
    return out.astype(np.float32)


def main():
    print("=== exp88: inference-time levers local audit ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    print(f"loaded perch + exp50", flush=True)

    # T1: build shifted exp50 cache (chunk start at 10s instead of 0s)
    print("Building exp50 shift=10s cache (one-time, ~1 min)...", flush=True)
    exp50_shift10 = get_exp50_shifted_scores(sc_g, shift_sec=10)
    print(f"exp50 shift10: {exp50_shift10.shape}, range [{exp50_shift10.min():.3f}, {exp50_shift10.max():.3f}]", flush=True)

    # Diagnostic: how different is shifted from original?
    from scipy.stats import pearsonr
    print(f"\nPearson(exp50_orig, exp50_shift10) = {pearsonr(exp50.flatten(), exp50_shift10.flatten())[0]:.4f}")
    print(f"Mean |orig - shifted| = {np.abs(exp50 - exp50_shift10).mean():.4f}")

    # Build v33 reference
    v33_ref = build_v33_base(perch_prob, exp50, perch_emb, sc_g, sp_taxon)
    ev_mask = sc_g.split.values == "eval"
    tr_mask = sc_g.split.values == "train"

    rows = [evaluate(v33_ref, v33_ref, ev_mask, Y, sp_taxon, "v33 ref")]

    # ===== T1: SED50 TTA =====
    print("\n=== T1: SED50 chunk-shift TTA (avg orig + shift10) ===", flush=True)
    for w in [0.3, 0.5, 0.7]:
        P = T1_sed50_tta(perch_prob, exp50, exp50_shift10, perch_emb, sc_g, sp_taxon, w_tta=w)
        rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, f"T1 SED50_TTA w={w}"))

    # ===== T2: Logit-space Gauss =====
    print("\n=== T2: Logit-space Gaussian smoothing ===", flush=True)
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        P = T2_logit_gauss(v33_ref, sc_g, sigma=sigma)
        rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, f"T2 logit-Gauss σ={sigma}"))

    # ===== T3: Per-class temperature =====
    print("\n=== T3: Per-class temperature (OOF fit on train) ===", flush=True)
    P = T3_per_class_temperature(v33_ref, Y, tr_mask, ev_mask)
    rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, "T3 Per-class temp"))

    # ===== T4: File-level cross-class z-score =====
    print("\n=== T4: File-level z-score normalization ===", flush=True)
    P = T4_file_zscore(v33_ref, sc_g)
    rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, "T4 File z-score"))

    # ===== T5: Combined T1 + T2 =====
    print("\n=== T5: SED50_TTA + logit-Gauss (combined) ===", flush=True)
    for w_tta in [0.5]:
        for sigma in [0.3, 0.5]:
            P_tta = T1_sed50_tta(perch_prob, exp50, exp50_shift10, perch_emb, sc_g, sp_taxon, w_tta=w_tta)
            P_combo = T2_logit_gauss(P_tta, sc_g, sigma=sigma)
            rows.append(evaluate(P_combo, v33_ref, ev_mask, Y, sp_taxon, f"T5 TTA(w={w_tta}) + logit-Gauss(σ={sigma})"))

    df = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== ALL RESULTS sorted by macro_d desc ===")
    print(df[cols].sort_values("macro_d", ascending=False).to_string(index=False))
    df.to_csv(EXP80 / "exp88_inference_levers.csv", index=False)
    print(f"\nSaved → {EXP80}/exp88_inference_levers.csv")

    print("\n=== Top class-A candidates (sp_row ≥ 0.99 AND Aves Δ ≥ 0) ===")
    safe = df[df.predicted.str.startswith("A") & (df.label != "v33 ref")]
    if len(safe) > 0:
        top = safe.sort_values("Aves", ascending=False).head(8)
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
