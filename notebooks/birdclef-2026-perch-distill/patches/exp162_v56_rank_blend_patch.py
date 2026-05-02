# ─────────────────────────────────────────────────────────────────────────
# v56 Tucker 5-fold + RANK-PERCENTILE blend (vs v55 linear blend)
#
# v55 (linear additive W=0.10) returned LB ≤ 0.932 (no improvement vs v33
# team best). Local audit had macro_d +0.0774. Anti-correlation likely
# due to: scale mismatch between Tucker SED probs and v33 output. Linear
# blend mixes raw probabilities of different scales → distorts ranking.
#
# v56 fix: per-class RANK-PERCENTILE transform on both streams, then blend
# in rank space. This is the public 0.943 recipe (Mattia "Better Blend").
#
# Local audit (exp162 Mvar: v33-as-streamA + rank-blend Tucker SED_W=0.30):
#   macro_d +0.0784, sp_row 0.404
#   Aves +0.043 (POSITIVE! linear had Aves only +0.017)
#   Insecta +0.107, Mam +0.128, Amphib +0.065, Reptil +0.055
#   sp_row 0.404 reflects rank space change; macro AUC (per-class
#   monotonic) is the LB metric, so sp_row collapse is benign here.
#
# Implementation: same Tucker inference as v55, but the final blend
# operation is `xa * (1 - W) + xb * W` where xa, xb are per-class
# rank-pct of (probs, tucker_scores).
# ─────────────────────────────────────────────────────────────────────────
TUCKER_DIRS = [
    Path('/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public'),
    Path('/kaggle/input/bc2026-distilled-sed-public'),
]
_tucker_dir = next((p for p in TUCKER_DIRS if p.exists()), None)
W_TUCKER = 0.30  # rank space; bigger than v55's 0.10 (per local sweep)

if _tucker_dir is None:
    print("v56 Tucker rank-blend: dir missing, skipping")
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
            print(f"v56 Tucker: missing {p.name}, skipping")
            _tucker_sessions = None
            break
        _tucker_sessions.append(
            _ort.InferenceSession(str(p), sess_options=so,
                                   providers=["CPUExecutionProvider"])
        )

    if _tucker_sessions:
        print(f"\nv56: Tucker 5-fold loaded; running on {len(test_paths)} test files...",
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
                print(f"  v56 Tucker [{fi+1}/{len(test_paths)}] {el:.0f}s ETA {eta:.0f}s",
                      flush=True)
        print(f"  v56 Tucker inference: {time.time()-_t0:.0f}s")

        # === Rank-percentile blend (vs v55's linear blend) ===
        # Per-class rank-pct on each stream, then convex combination
        _streamA = probs.copy()
        _streamB = tucker_scores
        # Use pandas for fast vectorized rank
        _xa = pd.DataFrame(_streamA).rank(axis=0, pct=True).to_numpy(dtype=np.float32)
        _xb = pd.DataFrame(_streamB).rank(axis=0, pct=True).to_numpy(dtype=np.float32)
        probs = (_xa * (1.0 - W_TUCKER) + _xb * W_TUCKER).astype(np.float32)
        probs = np.clip(probs, 0.0, 1.0)
        print(f"  v56 rank-blend applied (W_TUCKER={W_TUCKER}); probs range "
              f"[{probs.min():.5f}, {probs.max():.5f}] (rank-pct space)")

        del _tucker_sessions, tucker_scores, _xa, _xb, _streamA
