"""Tucker bc2026-distilled-sed-public 5-fold ONNX SED inference."""
# ── Cell 11: Tucker Arrants distilled SED ONNX inference ──────────────

import librosa
from scipy.ndimage import gaussian_filter1d

N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80


def find_sed_dir():
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.onnx"))
    if not hits:
        raise FileNotFoundError(
            "sed_fold0.onnx not found. "
            "Attach tuckerarrants/bc2026-distilled-sed-public to this notebook."
        )
    return hits[0].parent


def make_sed_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(
        str(path),
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )


def audio_to_mel(chunks):
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


def file_to_sed_chunks(path):
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

    chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
    ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC

    return chunks, ends


def sigmoid_sed(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


# Load the 5 SED fold models
sed_dir = find_sed_dir()

sed_fold_paths = sorted(
    sed_dir.glob("sed_fold*.onnx"),
    key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1))
)

sed_sessions = [make_sed_session(p) for p in sed_fold_paths]

print(f"SED dir: {sed_dir}")
print(f"SED folds loaded: {[p.name for p in sed_fold_paths]}")


# Run on the exact same test files used by Cell 9/10
sed_rows, sed_preds = [], []
_t0_sed = time.time()


for i, path in enumerate(test_paths, 1):
    chunks, ends = file_to_sed_chunks(path)
    mel = audio_to_mel(chunks)

    p_sum = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)

    for sess in sed_sessions:
        outs = sess.run(None, {sess.get_inputs()[0].name: mel})

        clip_logits = outs[0]             # (12, 234)
        frame_max   = outs[1].max(axis=1) # (12, 234)

        p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)

    p_mean = p_sum / len(sed_sessions)

    if len(p_mean) > 1:
        p_mean = gaussian_filter1d(
            p_mean,
            sigma=0.65,
            axis=0,
            mode="nearest"
        ).astype(np.float32)

    stem = path.stem

    sed_rows.extend([f"{stem}_{int(t)}" for t in ends])
    sed_preds.append(p_mean)

    if i == 1 or i % 50 == 0 or i == len(test_paths):
        print(f"SED: {i}/{len(test_paths)} | {time.time()-_t0_sed:.1f}s")


sed_preds_arr = np.concatenate(sed_preds, axis=0)

sed_sub = pd.DataFrame(
    np.clip(sed_preds_arr, 0.0, 1.0),
    columns=PRIMARY_LABELS
)

sed_sub.insert(0, "row_id", sed_rows)

sed_sub.to_csv("submission_sed.csv", index=False)

print(f"Saved submission_sed.csv: {sed_sub.shape}")