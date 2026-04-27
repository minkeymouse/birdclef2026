"""GPU mel extraction for SS files (60s → 12 × 5-sec → pooled (T_POOL, N_MELS))."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

SR = 32000
N_WIN = 12
T_POOL = 16
N_MELS = 128
N_FFT = 2048
HOP = 512
FMIN = 50
FMAX = 14000


def make_mel_pool_gpu(device: str = "cuda"):
    """Return (mel_t, adb_t, sr_5s) GPU transforms matching exp78/exp80 settings."""
    import torchaudio
    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        f_min=FMIN, f_max=FMAX, power=2.0, center=True).to(device)
    adb_t = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)
    return mel_t, adb_t, SR * 5


def extract_pool_one_file(path: Path, mel_t, adb_t, sr_5s, device: str = "cuda") -> np.ndarray:
    """Returns (12, T_POOL, N_MELS) for one 60-sec file. Pads/truncates to 60s."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if sr != SR: raise ValueError(f"unexpected sr {sr}")
    if len(y) < SR * 60: y = np.pad(y, (0, SR * 60 - len(y)))
    y = y[:SR * 60]
    wins = torch.from_numpy(np.stack([y[i*sr_5s:(i+1)*sr_5s] for i in range(N_WIN)])).to(device)
    with torch.no_grad():
        m = adb_t(mel_t(wins))                # (12, n_mels, T_frames)
        T = m.shape[-1]
        bins = torch.linspace(0, T, T_POOL + 1, device=device).long()
        pooled = torch.stack([m[:, :, bins[k]:bins[k+1]].mean(dim=-1) for k in range(T_POOL)], dim=1)
        return pooled.cpu().numpy()


def extract_pool_many(file_paths: list[Path], device: str = "cuda",
                      log_every: int = 200) -> tuple[np.ndarray, list[str]]:
    """Returns (n_files * 12, T_POOL, N_MELS) and the per-row filename list."""
    mel_t, adb_t, sr_5s = make_mel_pool_gpu(device)
    pools = []
    fnames_per_row = []
    t0 = time.time()
    for fi, p in enumerate(file_paths):
        try:
            pooled = extract_pool_one_file(Path(p), mel_t, adb_t, sr_5s, device)
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        pools.append(pooled)
        fnames_per_row.extend([Path(p).name] * N_WIN)
        if log_every and (fi + 1) % log_every == 0:
            print(f"  {fi+1}/{len(file_paths)} files  elapsed {time.time()-t0:.1f}s", flush=True)
    return np.concatenate(pools), fnames_per_row
