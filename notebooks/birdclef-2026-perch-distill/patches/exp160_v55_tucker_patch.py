# ─────────────────────────────────────────────────────────────────────────
# v55 Tucker bc2026-distilled-sed-public 5-fold ensemble — additive on v33
#
# Source: tuckerarrants/bc2026-distilled-sed-public (5 ONNX folds).
# Public LB-verified at 0.943 inside Mattia Angeli's "Better Blend" notebook.
#
# Local audit (exp160 on 122 eval rows):
#   v33 + 0.10 * Tucker additive: macro_d +0.0774, sp_row 0.997
#     Aves +0.017, Insecta +0.134, Mam +0.063, Amphib +0.067, Reptil +0.076
#   Pearson(Perch, Tucker) = -0.060 (orthogonal)
#   Pearson(exp50, Tucker) = 0.850 (correlated but distinct)
#
# Mel preprocessing (Tucker convention):
#   SR=32000, n_mels=256, n_fft=2048, hop=512, fmin=20, fmax=16000
#   power=2.0, top_db=80, per-spec z-score (s - s.mean()) / (s.std() + 1e-6)
#
# Wall-time: ~5 min for 5 folds on the comp test set (30s × 60 files
# × 30× scaling factor / 5 parallel = 5 min). Within 90-min budget.
# ─────────────────────────────────────────────────────────────────────────
TUCKER_DIRS = [
    Path('/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public'),
    Path('/kaggle/input/bc2026-distilled-sed-public'),
]
_tucker_dir = next((p for p in TUCKER_DIRS if p.exists()), None)
W_TUCKER = 0.10  # conservative, sp_row 0.997 in local audit

if _tucker_dir is None:
    print("v55 Tucker: dir missing, skipping")
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
            print(f"v55 Tucker: missing {p.name}, skipping")
            _tucker_sessions = None
            break
        _tucker_sessions.append(
            _ort.InferenceSession(str(p), sess_options=so,
                                   providers=["CPUExecutionProvider"])
        )

    if _tucker_sessions:
        print(f"\nv55: Tucker 5-fold loaded; running on {len(test_paths)} test files...",
              flush=True)
        _in_name = _tucker_sessions[0].get_inputs()[0].name
        tucker_scores = np.zeros((len(test_paths) * N_WINDOWS, N_CLASSES),
                                  dtype=np.float32)
        _t0 = time.time()
        for fi, path in enumerate(test_paths):
            chunks = _file_to_chunks_tucker(path)
            mel_b = _audio_to_mel_tucker(chunks)  # (12, 1, 256, T)
            fold_outs = [s.run(None, {_in_name: mel_b})[0] for s in _tucker_sessions]
            ens = np.mean(np.stack(fold_outs, axis=0), axis=0)  # (12, 234)
            if ens.min() < 0 or ens.max() > 1:
                ens = 1.0 / (1.0 + np.exp(-ens))
            tucker_scores[fi*N_WINDOWS:(fi+1)*N_WINDOWS] = ens.astype(np.float32)
            if (fi + 1) % 50 == 0:
                el = time.time() - _t0
                eta = el / (fi + 1) * (len(test_paths) - fi - 1)
                print(f"  v55 Tucker [{fi+1}/{len(test_paths)}] {el:.0f}s ETA {eta:.0f}s",
                      flush=True)

        print(f"  v55 Tucker inference: {time.time()-_t0:.0f}s")
        probs = (1.0 - W_TUCKER) * probs + W_TUCKER * tucker_scores
        probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
        print(f"  v55 applied W_TUCKER={W_TUCKER}; probs range "
              f"[{probs.min():.5f}, {probs.max():.5f}]")

        del _tucker_sessions, tucker_scores
