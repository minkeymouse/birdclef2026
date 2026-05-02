#!/usr/bin/env python3
"""exp158 — Source separation POC for synthetic-recombination training.

Question: does Demucs (music-pretrained) usefully separate bird/insect
calls from forest BG when applied to BirdCLEF audio? If yes, recombine
foreground from one site with BG from another → site-decoupled training.

Test: take 3 labeled SS files (one per site if possible), separate into
4 stems (drums/bass/other/vocals), inspect spectral content of each stem
to see which stem captures species calls vs which captures BG.

Output: exp158_outputs/{file_id}_stems.npz with all 4 stems + STFT mean
per band, plus a verdict whether Demucs is suitable.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
SS_DIR = DATA / "train_soundscapes"
OUT = ROOT / "experiments/_audits_post_v26/exp158_outputs"
OUT.mkdir(exist_ok=True)

SR_TARGET = 44100  # demucs native
SR_BIRD = 32000


def main():
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    print("Loading Demucs htdemucs...")
    t0 = time.time()
    model = get_model("htdemucs")
    model.to("cuda").eval()
    print(f"Loaded in {time.time()-t0:.1f}s. Stems: {model.sources}")
    print(f"Model SR: {model.samplerate}, channels: {model.audio_channels}")

    # Pick 3 labeled SS files (assume different sites)
    test_files = sorted(SS_DIR.glob("*.ogg"))[:3]
    if not test_files:
        # fall back to test_soundscapes if labeled SS not directly accessible
        test_files = sorted((DATA / "train_soundscapes").glob("*.ogg"))[:3]
    if not test_files:
        print("No SS files found")
        sys.exit(1)
    print(f"\nTesting on {len(test_files)} files")

    resampler_up = torchaudio.transforms.Resample(SR_BIRD, SR_TARGET).cuda()
    resampler_down = torchaudio.transforms.Resample(SR_TARGET, SR_BIRD).cuda()

    for fp in test_files:
        print(f"\n=== {fp.name} ===")
        wav, sr = sf.read(fp, dtype="float32", always_2d=False)
        print(f"  raw: shape={wav.shape}, sr={sr}, duration={len(wav)/sr:.1f}s")
        if sr != SR_BIRD:
            print(f"  WARN: expected {SR_BIRD}, got {sr}")
        # Take first 10 sec
        wav = wav[:10 * SR_BIRD]
        if wav.ndim == 1:
            wav = wav[None]  # (1, T)

        # Upsample to demucs SR (44100), make stereo
        x = torch.from_numpy(wav).float().cuda()  # (1, T)
        x = resampler_up(x)  # (1, T*44100/32000)
        x = x.repeat(2, 1)[None]  # (1, 2, T) batch dim + stereo

        with torch.no_grad():
            stems = apply_model(model, x, device="cuda", progress=False)
        # stems: (B=1, S=4, C=2, T) — stem dimension: drums, bass, other, vocals
        stems_np = stems.squeeze(0).mean(dim=1).cpu().numpy()  # (S, T) mono mean

        # Spectral content per stem
        band_energies = {}
        for si, name in enumerate(model.sources):
            s = stems_np[si]
            # FFT energy in bird-relevant bands (1-8 kHz)
            fft = np.abs(np.fft.rfft(s)) ** 2
            freqs = np.fft.rfftfreq(len(s), 1 / SR_TARGET)
            E_low = fft[(freqs >= 200) & (freqs < 1000)].sum()
            E_mid = fft[(freqs >= 1000) & (freqs < 5000)].sum()
            E_high = fft[(freqs >= 5000) & (freqs < 14000)].sum()
            E_total = E_low + E_mid + E_high + 1e-9
            band_energies[name] = (E_low/E_total, E_mid/E_total, E_high/E_total, E_total)
            print(f"  stem '{name:>8s}': low={E_low/E_total:.3f} mid={E_mid/E_total:.3f} high={E_high/E_total:.3f} total={E_total:.0f}")

        # Save stems for audit
        np.savez_compressed(OUT / f"{fp.stem}_stems.npz",
                             stems=stems_np.astype(np.float32),
                             sources=np.array(model.sources))
    print(f"\nSaved → {OUT}")
    print("\nVerdict guidance: if 'vocals' or 'other' stem has dominant mid-band (1-5 kHz)")
    print("energy that matches species calls, separation is usable. If energies are spread")
    print("evenly across stems, demucs music-pretrained is a poor fit for bioacoustics.")


if __name__ == "__main__":
    main()
