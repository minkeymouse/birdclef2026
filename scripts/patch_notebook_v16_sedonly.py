#!/usr/bin/env python3
"""Patch notebook v16: SED38b single-model submission (Boredom-style test).

- Adds CFG['sed_single_mode'] = True to config cell
- Replaces SED block in Cell 50: load SED38b (train_audio + 66 labeled SS),
  override final_test_scores = logit(sed38b_probs). Subsequent post-proc runs
  on that. Perch/ProtoSSM still run (wasted compute ~60 min) but result discarded.
"""
import json
import sys

NB = 'notebooks/birdclef-2026-perch-distill/notebook.ipynb'

NEW_SED_BLOCK = """# --- v16 SED-SINGLE MODE (Boredom-style test): SED38b (train_audio + 66 labeled SS) ---
SED_SINGLE_MODE = True
SED38B_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp38b_hgnet_sed.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/exp38b_hgnet_sed.pt'),
]
SED38B_CKPT = next((p for p in SED38B_CKPT_PATHS if p.exists()), None)
POST_BLEND_GAUSS_SIGMA = 0.5

if SED38B_CKPT is not None:
    import torchaudio
    import timm

    SED_N_MELS, SED_N_FFT, SED_HOP = 128, 2048, 512
    SED_FMIN, SED_FMAX = 50, 14000
    SED_CHUNK_SEC = 20
    SED_CHUNK_SAMPLES = SR * SED_CHUNK_SEC

    class _MelExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=SED_N_FFT, hop_length=SED_HOP, n_mels=SED_N_MELS,
                f_min=SED_FMIN, f_max=SED_FMAX, power=2.0, center=True)
            self.adb = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        def forward(self, x):
            m = self.mel(x); m = self.adb(m); return m.unsqueeze(1)

    class _SEDHead(nn.Module):
        def __init__(self, feat_dim, n_classes):
            super().__init__()
            self.att = nn.Conv1d(feat_dim, n_classes, 1)
            self.cla = nn.Conv1d(feat_dim, n_classes, 1)
        def forward(self, x):
            a = self.att(x); c = self.cla(x)
            w = torch.softmax(a, dim=-1)
            return (w * c).sum(-1), c.max(-1).values

    class _SEDModel(nn.Module):
        def __init__(self, backbone_name='hgnetv2_b0.ssld_stage2_ft_in1k', n_classes=N_CLASSES):
            super().__init__()
            self.mel = _MelExtractor()
            self.bn0 = nn.BatchNorm2d(SED_N_MELS)
            self.backbone = timm.create_model(
                backbone_name, pretrained=False, in_chans=1, num_classes=0, global_pool='')
            with torch.no_grad():
                feat = self.backbone(torch.zeros(1, 1, SED_N_MELS, 100))
            self.head = _SEDHead(feat.shape[1], n_classes)
        def forward(self, x):
            m = self.mel(x)
            m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
            feat = self.backbone(m)
            feat = feat.mean(dim=2) if feat.dim() == 4 else feat
            clip, fmax = self.head(feat)
            return clip, fmax

    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print(f"\\n[SED-SINGLE] Loading SED38b ckpt: {SED38B_CKPT}")
    _st = torch.load(str(SED38B_CKPT), map_location='cpu', weights_only=False)
    _bb = _st.get('backbone', 'hgnetv2_b0.ssld_stage2_ft_in1k')
    _sed = _SEDModel(backbone_name=_bb).eval()
    _sed.load_state_dict(_st['state_dict'])
    print(f"  backbone={_bb}  epoch={_st.get('epoch', '?')}")

    SED_BATCH_FILES = 8
    n_test_f = len(test_paths)
    sed_scores = np.zeros((n_test_f * N_WINDOWS, N_CLASSES), dtype=np.float32)
    _t0 = time.time()
    with torch.inference_mode():
        for _s in tqdm(range(0, n_test_f, SED_BATCH_FILES), desc='SED38b'):
            _batch = test_paths[_s:_s + SED_BATCH_FILES]
            _bn = len(_batch)
            _chunks = np.empty((_bn * 3, SED_CHUNK_SAMPLES), dtype=np.float32)
            for _bi, _path in enumerate(_batch):
                _audio = read_soundscape_60s(_path)
                for _ci in range(3):
                    _st_ = _ci * SED_CHUNK_SAMPLES
                    _chunks[_bi * 3 + _ci] = _audio[_st_:_st_ + SED_CHUNK_SAMPLES]
            _x = torch.from_numpy(_chunks)
            _clip, _ = _sed(_x)
            _p = torch.sigmoid(_clip).cpu().numpy().astype(np.float32)
            for _bi in range(_bn):
                for _ci in range(3):
                    _row = (_s + _bi) * N_WINDOWS + _ci * 4
                    sed_scores[_row:_row + 4] = _p[_bi * 3 + _ci]
            del _x, _clip, _p
    print(f"SED38b inference: {time.time()-_t0:.0f}s")

    if SED_SINGLE_MODE:
        # Override: use pure SED38b in logit space; downstream post-proc (temp,
        # delta smooth, etc.) runs on this. No Perch/ProtoSSM contribution.
        _eps = 1e-6
        _p = np.clip(sed_scores, _eps, 1 - _eps)
        final_test_scores = np.log(_p / (1 - _p)).astype(np.float32)
        print(f"[SED-SINGLE] final_test_scores overridden with SED38b logits; Perch/ProtoSSM discarded")
        # Apply Gauss smoothing in logit space (same as v12)
        from scipy.ndimage import gaussian_filter1d as _gf
        _view = final_test_scores.reshape(-1, N_WINDOWS, N_CLASSES)
        final_test_scores = _gf(_view, sigma=POST_BLEND_GAUSS_SIGMA, axis=1, mode='nearest').reshape(final_test_scores.shape)
        print(f"Post Gauss smoothing sigma={POST_BLEND_GAUSS_SIGMA}")
    else:
        # (v12-style blend path — not reached in sed_single_mode)
        pass

    del _sed, sed_scores
    gc.collect()
else:
    print("\\nSED38b ckpt not found; aborting SED-single mode")
"""


def main():
    nb = json.load(open(NB))
    cell = nb['cells'][50]
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    start = src.find("# --- 3-way z-score blend")
    end = src.find("scaled_scores = final_test_scores / class_temperatures")
    if start < 0 or end < 0:
        print(f"ERROR markers: start={start} end={end}")
        sys.exit(1)
    new = src[:start] + NEW_SED_BLOCK + "\n" + src[end:]
    cell['source'] = new
    json.dump(nb, open(NB, 'w'), indent=1)
    print(f"Patched Cell 50. {len(src)} -> {len(new)}")


if __name__ == "__main__":
    main()
