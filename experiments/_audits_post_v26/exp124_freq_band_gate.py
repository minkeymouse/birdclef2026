#!/usr/bin/env python3
"""exp124 — Frequency-band inference gate (universal physics, no learning).

bird-bias fix that is site-invariant by construction:
  - Mammals/large frogs: dominant <1 kHz
  - Birds: dominant 1-8 kHz
  - Insects (sonotypes): often >5 kHz, narrow band
  - These are physical sound facts, NOT site-conditional

Per 5-sec window:
  E_low  = mel-spec energy in 200-1000 Hz
  E_mid  = mel-spec energy in 1-5 kHz
  E_high = mel-spec energy in 5-15 kHz
  E_total = sum

Gate rules (multiplicative on logits):
  if E_low_ratio > τ_low  AND E_mid_ratio < τ_mid_low:
    → low-freq dominant: suppress all Aves logits by factor α
  Insecta gate: if E_high_ratio > τ_high:
    → high-freq dominant: keep Insecta + Aves slight suppress

NO labeled data used. Pure physics.
Audit on 122 eval w/ proper sp_row + per-taxon.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torchaudio
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS, DATA, SR, TAXA)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SS_DIR = DATA / "train_soundscapes"

WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
N_FFT = 2048; HOP = 512; N_MELS = 128
F_MIN, F_MAX = 50, 14000

# Frequency band boundaries (in Hz)
LOW_BAND = (200, 1000)      # mammals, large frogs
MID_BAND = (1000, 5000)     # birds primary
HIGH_BAND = (5000, 14000)   # high-pitch insects, some birds


def mel_freq_centers(n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX):
    """Get center frequency of each mel bin (Hz)."""
    # torchaudio mel filter centers
    mel_min = 2595 * np.log10(1 + f_min/700)
    mel_max = 2595 * np.log10(1 + f_max/700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts/2595) - 1)
    centers = hz_pts[1:-1]  # n_mels centers
    return centers


def compute_band_energies(sc_g):
    """For each 5-sec window in labeled SS, compute E_low, E_mid, E_high.

    Returns: (n_rows, 3) array.
    """
    cache = EXP80 / "exp124_band_energies.npz"
    if cache.exists():
        return np.load(cache)["bands"]

    print("Computing band energies on labeled SS...")
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        f_min=F_MIN, f_max=F_MAX, power=2.0, center=True
    ).to(DEVICE)

    # Mel bin → which band
    centers = mel_freq_centers(N_MELS)
    low_mask = (centers >= LOW_BAND[0]) & (centers < LOW_BAND[1])
    mid_mask = (centers >= MID_BAND[0]) & (centers < MID_BAND[1])
    high_mask = (centers >= HIGH_BAND[0]) & (centers <= HIGH_BAND[1])
    print(f"  bin counts: low {low_mask.sum()}, mid {mid_mask.sum()}, high {high_mask.sum()}")

    # row_id mapping
    files = sorted(sc_g.filename.unique())
    fname_idx = {f: [] for f in files}
    for i, r in sc_g.iterrows():
        fname_idx[r.filename].append((i, int(r.end_sec)))

    bands = np.zeros((len(sc_g), 3), dtype=np.float32)
    t0 = time.time()
    for fi, fn in enumerate(files):
        wav, _ = sf.read(SS_DIR / fn, dtype="float32", always_2d=False)
        if wav.ndim == 2: wav = wav.mean(axis=1)
        if len(wav) < 60*SR: wav = np.pad(wav, (0, 60*SR - len(wav)))
        wav = wav[:60*SR]
        # Process all 12 windows of this file
        for row_idx, end_sec in fname_idx[fn]:
            # 5-sec window ending at end_sec
            start = (end_sec - WINDOW_SEC) * SR
            chunk = wav[max(0, start):start + WINDOW_SAMPLES]
            if len(chunk) < WINDOW_SAMPLES: chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
            x = torch.from_numpy(chunk.astype(np.float32))[None].to(DEVICE)
            with torch.no_grad():
                spec = mel(x).squeeze(0)  # (n_mels, T)
                # mean energy per mel bin across time
                bin_energy = spec.mean(dim=1).cpu().numpy()
            E_low = bin_energy[low_mask].sum()
            E_mid = bin_energy[mid_mask].sum()
            E_high = bin_energy[high_mask].sum()
            bands[row_idx] = [E_low, E_mid, E_high]
        if fi % 10 == 0:
            print(f"  {fi}/{len(files)} files done, {(time.time()-t0):.1f}s elapsed", flush=True)

    np.savez_compressed(cache, bands=bands)
    print(f"  Saved {cache} in {(time.time()-t0)/60:.1f} min")
    return bands


def apply_freq_gate(probs, bands, sp_taxon, alpha_aves=0.5, tau_low=0.4):
    """Multiplicatively suppress Aves logits when low-band dominant.

    probs: (n_rows, 234) sigmoid scores
    bands: (n_rows, 3) [E_low, E_mid, E_high]
    """
    aves_mask = sp_taxon == "Aves"
    out = probs.copy()

    band_total = bands.sum(axis=1, keepdims=True) + 1e-9
    band_ratio = bands / band_total  # (n_rows, 3): low, mid, high ratios
    low_dom = band_ratio[:, 0] > tau_low  # (n_rows,)

    # For low-dominant rows, multiply Aves logits by (1 - alpha)
    # Equivalent to suppressing Aves for these rows
    n_modified = low_dom.sum()
    if n_modified > 0:
        out[low_dom, :][:, aves_mask] = probs[low_dom, :][:, aves_mask] * (1 - alpha_aves)
        # Direct assignment in mixed-bool mask
        rows = np.where(low_dom)[0]
        for r in rows:
            out[r, aves_mask] = probs[r, aves_mask] * (1 - alpha_aves)
    return out, n_modified


def main():
    print("=== exp124: Frequency-band inference gate (no learning) ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    # Compute band energies (cached)
    bands = compute_band_energies(sc_g)
    print(f"\n  bands shape {bands.shape}, mean total {bands.sum(axis=1).mean():.2f}")

    # Diagnostic: which rows have low-band dominant?
    band_total = bands.sum(axis=1) + 1e-9
    low_ratio = bands[:, 0] / band_total
    mid_ratio = bands[:, 1] / band_total
    high_ratio = bands[:, 2] / band_total
    print(f"\n  low_ratio distribution: mean {low_ratio.mean():.3f}, "
          f"median {np.median(low_ratio):.3f}, p90 {np.percentile(low_ratio, 90):.3f}")
    print(f"  mid_ratio: mean {mid_ratio.mean():.3f}")
    print(f"  high_ratio: mean {high_ratio.mean():.3f}")

    # Cross-tab: for rows with TRUE non-Aves positive, what's the low_ratio distribution?
    aves_mask = sp_taxon == "Aves"
    has_aves_pos = (Y[:, aves_mask] > 0).any(axis=1)
    has_non_aves_pos = (Y[:, ~aves_mask] > 0).any(axis=1)
    pure_non_aves = has_non_aves_pos & ~has_aves_pos
    pure_aves = has_aves_pos & ~has_non_aves_pos

    print(f"\n  Pure-Aves rows: {int(pure_aves.sum())}, mean low_ratio {low_ratio[pure_aves].mean():.3f}")
    print(f"  Pure-non-Aves rows: {int(pure_non_aves.sum())}, mean low_ratio {low_ratio[pure_non_aves].mean():.3f}")
    print(f"  Mixed rows: {int((has_aves_pos & has_non_aves_pos).sum())}, mean low_ratio {low_ratio[has_aves_pos & has_non_aves_pos].mean():.3f}")

    # Build v33
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # Apply frequency gate
    ev_mask = sc_g.split.values == "eval"
    rows_audit = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    # Sweep parameters
    for tau in [0.3, 0.4, 0.5]:
        for alpha in [0.2, 0.3, 0.5]:
            v33_gated, n_mod = apply_freq_gate(v33, bands, sp_taxon, alpha_aves=alpha, tau_low=tau)
            rows_audit.append(evaluate(v33_gated, v33, ev_mask, Y, sp_taxon,
                                          f"freq-gate τ={tau} α={alpha} (n_mod={n_mod})"))

    res = pd.DataFrame(rows_audit)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== Audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
