"""Tucker bc2026-distilled-sed-public 5-fold ONNX SED inference.

Self-contained, locally runnable. Replaces the cell-11 source dump.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

from .paths import TUCKER_DIR

# Tucker mel preprocessing config
SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
N_CLASSES = 234

N_MELS = 256
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 16000
TOP_DB = 80
N_FOLDS = 5


def make_session(onnx_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=so,
                                 providers=["CPUExecutionProvider"])


def load_5fold(tucker_dir=None):
    """Load 5 ONNX folds. Returns list of sessions."""
    d = tucker_dir or TUCKER_DIR
    if d is None:
        raise FileNotFoundError("Tucker SED dir not found; set TUCKER_DIR")
    sessions = []
    for i in range(N_FOLDS):
        p = Path(d) / f"sed_fold{i}.onnx"
        if not p.exists():
            raise FileNotFoundError(p)
        sessions.append(make_session(p))
    return sessions


def audio_to_mel(chunks):
    """List of 5-sec audio chunks -> (B, 1, n_mels, T) mel-spec batch."""
    mels = []
    for x in chunks:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def file_to_chunks(path):
    """Read 60-sec audio file and split into 12 5-sec chunks."""
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
    return [y[i*WINDOW_SAMPLES:(i+1)*WINDOW_SAMPLES] for i in range(N_WINDOWS)]


def predict_file(sessions, path, ensemble=True):
    """Run 5-fold ensemble on one file. Returns (12, 234) probs."""
    chunks = file_to_chunks(path)
    mel_b = audio_to_mel(chunks)
    in_name = sessions[0].get_inputs()[0].name
    fold_outs = [s.run(None, {in_name: mel_b})[0] for s in sessions]
    out = np.mean(np.stack(fold_outs, axis=0), axis=0) if ensemble else fold_outs[0]
    if out.min() < 0 or out.max() > 1:
        out = 1.0 / (1.0 + np.exp(-out))
    return out.astype(np.float32)


def predict_files(sessions, paths, verbose=False):
    """Run on multiple files. Returns (n_files * N_WINDOWS, N_CLASSES)."""
    out = np.zeros((len(paths) * N_WINDOWS, N_CLASSES), dtype=np.float32)
    t0 = time.time()
    for fi, p in enumerate(paths):
        out[fi*N_WINDOWS:(fi+1)*N_WINDOWS] = predict_file(sessions, p)
        if verbose and (fi + 1) % 20 == 0:
            el = time.time() - t0
            eta = el / (fi + 1) * (len(paths) - fi - 1)
            print(f"  [{fi+1}/{len(paths)}] {el:.0f}s ETA {eta:.0f}s", flush=True)
    return out
