# ─────────────────────────────────────────────────────────────────────────
# v34 mel-iVAE z-kNN blend (exp78 artifacts)
#
# Mel-iVAE was trained on raw mel-spectrograms of 739 labeled SS rows
# (T_POOL=16 × N_MELS=128, train-split mean/std standardized).
# For each test row: extract 5-sec mel → pool to (16, 128) → standardize
# → encoder forward → mu_q (32-d). Cosine sim to per-class centroids
# (computed from train positives ≥3) → sigmoid(sim * 5) → z_knn_scores.
# Blend: probs = (1 - w_z) * probs + w_z * z_knn_scores.
#
# Local exp77 (122 held-out eval rows, on top of v33):
#   global w_z=0.05  →  macro Δ +0.026, sp_row 0.990 (very safe rank-pres)
#                    →  Aves +0.008, Insecta +0.068, Reptilia +0.104
# Aves Δ slightly positive → anti-correlation rule says LB neutral/positive.
# Mechanism: orthogonal raw-mel signal on Perch-dead classes (Insecta sonotypes
# + Caiman + non-Aves where Perch never trained).
# ─────────────────────────────────────────────────────────────────────────
import torch as _t34
import torchaudio as _ta34
import time as _time34

_IVAE_ENC_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/ivae_encoder.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/ivae_encoder.pt'),
]
_IVAE_STATS_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/ivae_mel_stats.npz'),
    Path('/kaggle/input/birdclef2026-model-weights/ivae_mel_stats.npz'),
]
_IVAE_CENT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/ivae_z_centroids.npz'),
    Path('/kaggle/input/birdclef2026-model-weights/ivae_z_centroids.npz'),
]
_ivae_enc_path = next((p for p in _IVAE_ENC_PATHS if p.exists()), None)
_ivae_stats_path = next((p for p in _IVAE_STATS_PATHS if p.exists()), None)
_ivae_cent_path = next((p for p in _IVAE_CENT_PATHS if p.exists()), None)

W_IVAE_Z = 0.05  # local: global w_z=0.05 → macro Δ +0.026, sp_row 0.990

if _ivae_enc_path is None or _ivae_stats_path is None or _ivae_cent_path is None or W_IVAE_Z <= 0:
    print(f"v34 mel-iVAE: SKIPPED (enc={_ivae_enc_path}, stats={_ivae_stats_path}, cent={_ivae_cent_path}, w={W_IVAE_Z})")
else:
    print(f"\nv34 mel-iVAE z-kNN blend (w_z={W_IVAE_Z})")
    _ck34 = _t34.load(str(_ivae_enc_path), map_location='cpu', weights_only=False)
    _stats34 = np.load(_ivae_stats_path)
    _cent34 = np.load(_ivae_cent_path)
    _train_mean = _stats34['mean'].astype(np.float32)
    _train_std = _stats34['std'].astype(np.float32)
    _T_POOL_IV = int(_stats34['T_POOL'])
    _N_MELS_IV = int(_stats34['N_MELS'])
    _z_centroids = _cent34['centroids'].astype(np.float32)
    _cent_valid = _cent34['valid'].astype(bool)
    _in_dim = int(_ck34['in_dim'])
    _z_dim = int(_ck34['z_dim'])
    _n_aux = int(_ck34['n_aux'])
    print(f"  artifacts: in_dim={_in_dim} z_dim={_z_dim} n_aux={_n_aux} valid_centroids={_cent_valid.sum()}/234")

    class _IVAEEnc(nn.Module):
        def __init__(self, in_dim, z_dim, n_aux, hidden=512):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, 256), nn.GELU(),
                nn.Linear(256, 2 * z_dim))
            self.aux_mlp = nn.Sequential(
                nn.Linear(n_aux, 64), nn.GELU(),
                nn.Linear(64, 2 * z_dim))
            self.z_dim = z_dim
        def encode(self, x):
            h = self.enc(x)
            mu_q, _ = h.chunk(2, dim=-1)
            return mu_q

    _ivae = _IVAEEnc(_in_dim, _z_dim, _n_aux).eval()
    _ivae.load_state_dict(_ck34['encoder_state_dict'], strict=False)

    # Mel extractor matches exp78: librosa(n_fft=2048, hop=512, n_mels=128, fmin=50, fmax=14000)
    _mel_iv = _ta34.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=2048, hop_length=512, n_mels=_N_MELS_IV,
        f_min=50.0, f_max=14000.0, power=2.0, center=True
    )
    _adb_iv = _ta34.transforms.AmplitudeToDB(stype='power', top_db=80)

    _SR_5S = SR * 5
    _Z_TEST = np.zeros((probs.shape[0], _z_dim), dtype=np.float32)

    _t0_iv = _time34.time()
    with _t34.inference_mode():
        for _fi, _path in enumerate(test_paths):
            _audio = read_soundscape_60s(_path)
            for _wi in range(N_WINDOWS):
                _st = _wi * _SR_5S
                _w = _audio[_st:_st + _SR_5S]
                if len(_w) < _SR_5S:
                    _w = np.pad(_w, (0, _SR_5S - len(_w)))
                _wt = _t34.from_numpy(_w).float().unsqueeze(0)
                _m = _mel_iv(_wt)
                _m = _adb_iv(_m).squeeze(0).numpy()  # (n_mels, T)
                _T = _m.shape[1]
                _bins = np.linspace(0, _T, _T_POOL_IV + 1).astype(int)
                _pooled = np.zeros((_T_POOL_IV, _N_MELS_IV), dtype=np.float32)
                for _k in range(_T_POOL_IV):
                    _ch = _m[:, _bins[_k]:_bins[_k+1]]
                    if _ch.size > 0:
                        _pooled[_k] = _ch.mean(axis=1)
                _x = _pooled.flatten()
                _x = (_x - _train_mean) / _train_std
                _xt = _t34.from_numpy(_x).float().unsqueeze(0)
                _mu = _ivae.encode(_xt).numpy()[0]
                _Z_TEST[_fi * N_WINDOWS + _wi] = _mu
    print(f"  iVAE encoded {probs.shape[0]} rows in {_time34.time() - _t0_iv:.1f}s")

    # Per-class z_knn_scores: sigmoid(cos_sim * 5) for valid classes; 0 for invalid
    _z_norm = _Z_TEST / (np.linalg.norm(_Z_TEST, axis=1, keepdims=True) + 1e-8)
    _c_norm = _z_centroids / (np.linalg.norm(_z_centroids, axis=1, keepdims=True) + 1e-8)
    _cos = _z_norm @ _c_norm.T  # (n_rows, 234)
    _z_knn = np.zeros_like(probs)
    _valid_idx = np.where(_cent_valid)[0]
    _z_knn[:, _valid_idx] = 1.0 / (1.0 + np.exp(-5.0 * _cos[:, _valid_idx]))

    probs = ((1.0 - W_IVAE_Z) * probs + W_IVAE_Z * _z_knn).astype(np.float32)
    probs = np.clip(probs, 0.0, 1.0)
    print(f"  blend applied. probs range: [{probs.min():.5f}, {probs.max():.5f}]")
    del _ivae, _Z_TEST, _z_knn, _cos, _z_norm, _c_norm
    gc.collect()
