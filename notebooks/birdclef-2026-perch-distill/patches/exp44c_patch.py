# ─────────────────────────────────────────────────────────────────────────
# exp44f — exp44c 27-species head overlay on 27 double-blind species columns.
# Insert as new cell BEFORE `submission = pd.DataFrame(probs, columns=...)`
# in cell 50.
#
# Prerequisite: exp44c_27species_head.pt uploaded to
#   ultimatumgame/birdclef2026-model-weights
#
# What this does:
#   1. Identifies 27 double-blind species columns (Insecta sonXX + 2 Amphibia).
#   2. Loads HGNet B0 SED27 checkpoint.
#   3. For each test window, extracts 20s clip centered around the 5s target.
#   4. Runs exp44c → 27 species probabilities.
#   5. Overwrites probs[:, unmapped_cols] = exp44c_probs (replace, not blend).
#   6. Runtime cost: ~30-60s on Kaggle CPU (4M params × 7200 clips).
# ─────────────────────────────────────────────────────────────────────────

# --- Identify 27 double-blind column indices ---
_tax = pd.read_csv(BASE / "taxonomy.csv")
_perch_labels = set(open('/kaggle/input/google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1/assets/labels.csv').read().strip().split('\n')) \
    if Path('/kaggle/input/google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1/assets/labels.csv').exists() \
    else set(open('/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2_labels.csv').read().strip().split('\n')) \
    if Path('/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2_labels.csv').exists() else set()

# Fall back: hardcoded list from exp43r audit
_DOUBLE_BLIND_LABELS = [
    '1491113', '25073',  # Amphibia: Adenomera guarani, Chiasmocleis mehelyi
    '47158son01', '47158son02', '47158son03', '47158son04', '47158son05',
    '47158son06', '47158son07', '47158son08', '47158son09', '47158son10',
    '47158son11', '47158son12', '47158son13', '47158son14', '47158son15',
    '47158son16', '47158son17', '47158son18', '47158son19', '47158son20',
    '47158son21', '47158son22', '47158son23', '47158son24', '47158son25',
]
_unmapped_cols = [i for i, p in enumerate(PRIMARY_LABELS) if str(p) in _DOUBLE_BLIND_LABELS]
assert len(_unmapped_cols) == 27, f"Expected 27 unmapped cols, got {len(_unmapped_cols)}"
print(f"exp44c overlay target: {len(_unmapped_cols)} double-blind columns")

# --- Load exp44c ckpt ---
_EXP44C_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp44c_27species_head.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/exp44c_27species_head.pt'),
]
_exp44c_ckpt = next((p for p in _EXP44C_CKPT_PATHS if p.exists()), None)

if _exp44c_ckpt is None:
    print("exp44c ckpt NOT found — skipping overlay. probs unchanged.")
else:
    import torch, torch.nn as nn, torch.nn.functional as F
    import torchaudio, timm

    _SED27_CHUNK_SEC = 20
    _SED27_CHUNK_SAMPLES = SR * _SED27_CHUNK_SEC
    _SED27_N_MELS, _SED27_N_FFT, _SED27_HOP = 128, 2048, 512
    _SED27_FMIN, _SED27_FMAX = 50, 14000

    class _Mel44c(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=_SED27_N_FFT, hop_length=_SED27_HOP,
                win_length=_SED27_N_FFT, n_mels=_SED27_N_MELS,
                f_min=_SED27_FMIN, f_max=_SED27_FMAX, power=2.0, center=True)
            self.adb = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        def forward(self, x): return self.adb(self.m(x)).unsqueeze(1)

    class _SED27(nn.Module):
        def __init__(self, n_classes):
            super().__init__()
            self.mel = _Mel44c()
            self.bn0 = nn.BatchNorm2d(_SED27_N_MELS)
            self.backbone = timm.create_model(
                'hgnetv2_b0.ssld_stage2_ft_in1k', pretrained=False, in_chans=1,
                num_classes=0, global_pool='')
            with torch.no_grad():
                feat = self.backbone(torch.zeros(1, 1, _SED27_N_MELS, 200))
            C = feat.shape[1]
            self.att = nn.Conv1d(C, n_classes, 1)
            self.cls = nn.Conv1d(C, n_classes, 1)
        def forward(self, x):
            m = self.mel(x)
            m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
            feat = self.backbone(m)
            f = feat.mean(dim=2) if feat.dim() == 4 else feat
            att = torch.softmax(self.att(f), dim=-1)
            cls = self.cls(f)
            return (att * cls).sum(-1), cls.max(-1).values

    _ckpt = torch.load(_exp44c_ckpt, map_location='cpu', weights_only=False)
    _labels_27 = _ckpt['labels_27']
    assert len(_labels_27) == 27

    # Map each exp44c class i → PRIMARY_LABELS index
    _ckpt_to_primary = []
    for ci, lbl in enumerate(_labels_27):
        if str(lbl) in PRIMARY_LABELS:
            _ckpt_to_primary.append((ci, PRIMARY_LABELS.index(str(lbl))))
    assert len(_ckpt_to_primary) == 27

    _sed27 = _SED27(n_classes=27)
    _sed27.load_state_dict(_ckpt['state_dict'])
    _sed27.eval()

    print(f"exp44c loaded (epoch={_ckpt.get('epoch')}, val_auc={_ckpt.get('val_auc'):.4f})")

    # --- Run exp44c inference on each test window (20s clip) ---
    _t0 = time.time()
    _exp44c_probs_27 = np.zeros((probs.shape[0], 27), dtype=np.float32)

    # meta_test has row_id, filename, end_sec etc aligned with probs rows
    _WINDOW_SAMPLES = int(SR * 5)
    _N_WINDOWS = N_WINDOWS  # 12 per file in perch-distill

    # Cache audio per file to avoid re-reading
    _audio_cache = {}
    _test_root = BASE / "test_soundscapes"
    if not _test_root.exists() or not any(_test_root.glob("*.ogg")):
        # Fallback for dev run with small sample
        _test_root = Path("/kaggle/input/birdclef-2026/test_soundscapes")

    for i in range(len(meta_test)):
        fname = meta_test.iloc[i]["filename"]
        # Extract end_sec from row_id like BC2026_..._5, _10, ..., _60
        rid = str(meta_test.iloc[i]["row_id"])
        try:
            end_sec = int(rid.rsplit("_", 1)[-1])
        except ValueError:
            end_sec = 5 + (i % _N_WINDOWS) * 5

        if fname not in _audio_cache:
            try:
                fp = _test_root / fname
                if not fp.exists():
                    # Fallback to alternative test paths
                    for cand_root in [BASE / "test_soundscapes", Path("/kaggle/input/birdclef-2026/test_soundscapes")]:
                        if (cand_root / fname).exists():
                            fp = cand_root / fname; break
                wav, _ = sf.read(str(fp), dtype="float32", always_2d=False)
                if wav.ndim > 1: wav = wav.mean(axis=1)
                if len(wav) < SR * 60:
                    wav = np.pad(wav, (0, SR * 60 - len(wav)))
                _audio_cache[fname] = wav[: SR * 60]
            except Exception as e:
                _audio_cache[fname] = np.zeros(SR * 60, dtype=np.float32)

        wav = _audio_cache[fname]
        target_center = (end_sec - 2.5) * SR
        start = int(max(0, target_center - _SED27_CHUNK_SAMPLES / 2))
        start = min(start, len(wav) - _SED27_CHUNK_SAMPLES)
        clip = wav[start : start + _SED27_CHUNK_SAMPLES]
        if len(clip) < _SED27_CHUNK_SAMPLES:
            clip = np.pad(clip, (0, _SED27_CHUNK_SAMPLES - len(clip)))
        x = torch.from_numpy(clip).float().unsqueeze(0)
        with torch.no_grad():
            clip_logit, _ = _sed27(x)
        p = torch.sigmoid(clip_logit)[0].cpu().numpy().astype(np.float32)
        _exp44c_probs_27[i] = p

    print(f"exp44c inference done in {time.time() - _t0:.1f}s")
    _audio_cache.clear(); gc.collect()

    # --- Overwrite 27 unmapped columns ---
    for ckpt_i, primary_i in _ckpt_to_primary:
        probs[:, primary_i] = _exp44c_probs_27[:, ckpt_i]

    print(f"Overwrote {len(_unmapped_cols)} double-blind species columns with exp44c preds")
    print(f"  exp44c probs range: [{_exp44c_probs_27.min():.4f}, {_exp44c_probs_27.max():.4f}]")
