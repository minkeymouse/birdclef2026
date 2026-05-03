# ─────────────────────────────────────────────────────────────────────────
# v58 Tucker 5-fold LINEAR W=0.30 (was W=0.10 in v55, rank in v56)
#
# exp165 definitive ablation showed:
#   - linear v33 + Tucker W=0.30 rescues=none  → macro_d +0.117 sp_row 0.991
#   - rank v33 + Tucker rescues=all            → macro_d +0.087 sp_row 0.409
#   - Linear BEATS rank in our pipeline by +0.039 macro_d AND keeps sp_row.
#
# Why our v55 (W=0.10) under-delivered: dose was too conservative, not the
# wrong architecture. With W=0.30 the local macro_d quadruples (+0.077 → +0.117)
# while sp_row stays > 0.99 — the cleanest A-profile in our entire experiment
# history.
#
# Why we don't need rank+rescues: our v33 streamA is already calibrated through
# V9 gate + Gauss + file-max α=0.10. Linear blend on calibrated probabilities
# preserves the absolute confidence ordering that rank-pct would erase.
# Mattia's rank+rescues fix HIS uncalibrated streamA (raw Perch+ProtoSSM); our
# pipeline doesn't need that fix.
#
# Local audit (exp165, 122 eval rows, v33 baseline = 0):
#   macro_d +0.117  sp_row 0.991  Aves +0.041  Insecta +0.168
#   Mam +0.182  Amphib +0.108  Reptil +0.081  ALL TAXA POSITIVE
# ─────────────────────────────────────────────────────────────────────────
TUCKER_DIRS = [
    Path('/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public'),
    Path('/kaggle/input/bc2026-distilled-sed-public'),
]
_tucker_dir = next((p for p in TUCKER_DIRS if p.exists()), None)
W_TUCKER = 0.30  # raised from v55's 0.10 per exp165 dose-response

if _tucker_dir is None:
    print("v58 Tucker linear W=0.30: dir missing, skipping")
else:
    import librosa as _librosa
    import onnxruntime as _ort

    _N_MELS_T = 256
    _N_FFT_T = 2048
    _HOP_T = 512
    _FMIN_T = 20
    _FMAX_T = 16000
    _TOP_DB_T = 80
    _N_FOLDS_T = 5

    def _audio_to_mel_tucker(chunks):
        mels = []
        for x in chunks:
            s = _librosa.feature.melspectrogram(
                y=x, sr=SR, n_fft=_N_FFT_T, hop_length=_HOP_T,
                n_mels=_N_MELS_T, fmin=_FMIN_T, fmax=_FMAX_T, power=2.0,
            )
            s = _librosa.power_to_db(s, top_db=_TOP_DB_T)
            s = (s - s.mean()) / (s.std() + 1e-6)
            mels.append(s)
        return np.stack(mels)[:, None].astype(np.float32)

    def _file_to_chunks_tucker(path):
        y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr0 != SR:
            y = _librosa.resample(y, orig_sr=sr0, target_sr=SR)
        n = 60 * SR
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        else:
            y = y[:n]
        return [y[i*WINDOW_SAMPLES:(i+1)*WINDOW_SAMPLES] for i in range(N_WINDOWS)]

    so = _ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _tucker_sessions = []
    for i in range(_N_FOLDS_T):
        p = _tucker_dir / f"sed_fold{i}.onnx"
        if not p.exists():
            print(f"v58 Tucker: missing {p.name}, skipping")
            _tucker_sessions = None
            break
        _tucker_sessions.append(
            _ort.InferenceSession(str(p), sess_options=so,
                                   providers=["CPUExecutionProvider"])
        )

    if _tucker_sessions:
        print(f"\nv58: Tucker 5-fold loaded; running on {len(test_paths)} test files...",
              flush=True)
        _in_name = _tucker_sessions[0].get_inputs()[0].name
        tucker_scores = np.zeros((len(test_paths) * N_WINDOWS, N_CLASSES),
                                  dtype=np.float32)
        _t0 = time.time()
        for fi, path in enumerate(test_paths):
            chunks = _file_to_chunks_tucker(path)
            mel_b = _audio_to_mel_tucker(chunks)
            fold_outs = [s.run(None, {_in_name: mel_b})[0] for s in _tucker_sessions]
            ens = np.mean(np.stack(fold_outs, axis=0), axis=0)
            if ens.min() < 0 or ens.max() > 1:
                ens = 1.0 / (1.0 + np.exp(-ens))
            tucker_scores[fi*N_WINDOWS:(fi+1)*N_WINDOWS] = ens.astype(np.float32)
            if (fi + 1) % 50 == 0:
                el = time.time() - _t0
                eta = el / (fi + 1) * (len(test_paths) - fi - 1)
                print(f"  v58 Tucker [{fi+1}/{len(test_paths)}] {el:.0f}s ETA {eta:.0f}s",
                      flush=True)
        print(f"  v58 Tucker inference: {time.time()-_t0:.0f}s")

        # === LINEAR blend (NOT rank-pct) at W=0.30 ===
        # probs is current v33 output (calibrated through V9 + Gauss + file-max)
        probs = ((1.0 - W_TUCKER) * probs + W_TUCKER * tucker_scores).astype(np.float32)
        probs = np.clip(probs, 0.0, 1.0)
        print(f"  v58 linear blend applied W_TUCKER={W_TUCKER}; probs range "
              f"[{probs.min():.5f}, {probs.max():.5f}]")

        del _tucker_sessions, tucker_scores
