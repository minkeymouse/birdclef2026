#!/usr/bin/env python3
"""Patch perch-distill notebook to add 3-way blend: Perch + SED29 + SED38.

Reads notebook.ipynb, locates cell 50 (Cell 18 post-processing), replaces the
SED29 2-way block with a 3-way blend that loads both SED29 and SED38 and
applies weights (wP=0.70, w29=0.15, w38=0.15).
"""
import json
import sys
from pathlib import Path

NB_PATH = Path("notebooks/birdclef-2026-perch-distill/notebook.ipynb")

NEW_SED_BLOCK = """# --- 3-way z-score blend: Perch + SED29 + SED38 (exp38b full-retrain) ---
SED29_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp29_hgnet_sed.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/exp29_hgnet_sed.pt'),
]
SED38_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp38b_hgnet_sed.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/exp38b_hgnet_sed.pt'),
]
SED29_CKPT = next((p for p in SED29_CKPT_PATHS if p.exists()), None)
SED38_CKPT = next((p for p in SED38_CKPT_PATHS if p.exists()), None)

# v14 ablation: SED29 → SED38 clean swap, keep Perch weight at v12's 0.80
W_PERCH = 0.80
W_SED29 = 0.0
W_SED38 = 0.20
POST_BLEND_GAUSS_SIGMA = 0.5

if SED29_CKPT is not None or SED38_CKPT is not None:
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

    def _load_sed(ckpt_path):
        m = _SEDModel().eval()
        st = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        m.load_state_dict(st['state_dict'])
        return m, st.get('val_auc', '?')

    def _run_sed(model, paths, desc='SED'):
        SED_BATCH_FILES = 8
        n_f = len(paths)
        out = np.zeros((n_f * N_WINDOWS, N_CLASSES), dtype=np.float32)
        t0 = time.time()
        with torch.inference_mode():
            for _s in tqdm(range(0, n_f, SED_BATCH_FILES), desc=desc):
                _batch = paths[_s:_s + SED_BATCH_FILES]
                _bn = len(_batch)
                _chunks = np.empty((_bn * 3, SED_CHUNK_SAMPLES), dtype=np.float32)
                for _bi, _path in enumerate(_batch):
                    _audio = read_soundscape_60s(_path)
                    for _ci in range(3):
                        _st = _ci * SED_CHUNK_SAMPLES
                        _chunks[_bi * 3 + _ci] = _audio[_st:_st + SED_CHUNK_SAMPLES]
                _x = torch.from_numpy(_chunks)
                _clip, _ = model(_x)
                _p = torch.sigmoid(_clip).cpu().numpy().astype(np.float32)
                for _bi in range(_bn):
                    for _ci in range(3):
                        _row = (_s + _bi) * N_WINDOWS + _ci * 4
                        out[_row:_row + 4] = _p[_bi * 3 + _ci]
                del _x, _clip, _p
        print(f"{desc} inference: {time.time()-t0:.0f}s")
        return out

    torch.set_num_threads(max(1, os.cpu_count() or 4))

    sed29_scores = None
    sed38_scores = None
    if SED29_CKPT is not None:
        print(f"\\nSED29 ckpt: {SED29_CKPT}")
        _m, _v = _load_sed(SED29_CKPT)
        print(f"  val_auc={_v}")
        sed29_scores = _run_sed(_m, test_paths, desc='SED29')
        del _m; gc.collect()
    if SED38_CKPT is not None:
        print(f"\\nSED38 ckpt: {SED38_CKPT}")
        _m, _v = _load_sed(SED38_CKPT)
        print(f"  val_auc={_v}")
        sed38_scores = _run_sed(_m, test_paths, desc='SED38')
        del _m; gc.collect()

    _blend_classes = np.concatenate([idx_mapped_active_event, idx_mapped_active_texture])
    _blend_classes = np.unique(_blend_classes).astype(np.int32)

    def _zscore(x):
        m = x.mean(0, keepdims=True)
        s = x.std(0, keepdims=True) + 1e-6
        return (x - m) / s, m, s

    _fs_sub = final_test_scores[:, _blend_classes]
    _fs_z, _fs_m, _fs_s = _zscore(_fs_sub)

    # Renormalize weights over available models
    avail = {'perch': True,
             'sed29': sed29_scores is not None,
             'sed38': sed38_scores is not None}
    raw_w = {'perch': W_PERCH, 'sed29': W_SED29, 'sed38': W_SED38}
    tot = sum(w for k, w in raw_w.items() if avail[k])
    w = {k: (raw_w[k] / tot if avail[k] else 0.0) for k in raw_w}
    print(f"\\nBlend weights (renormalized): perch={w['perch']:.3f} "
          f"sed29={w['sed29']:.3f} sed38={w['sed38']:.3f}")

    _blend_z = w['perch'] * _fs_z
    if avail['sed29']:
        _sd29_z, _, _ = _zscore(sed29_scores[:, _blend_classes])
        _blend_z = _blend_z + w['sed29'] * _sd29_z
    if avail['sed38']:
        _sd38_z, _, _ = _zscore(sed38_scores[:, _blend_classes])
        _blend_z = _blend_z + w['sed38'] * _sd38_z

    final_test_scores[:, _blend_classes] = (_blend_z * _fs_s + _fs_m).astype(np.float32)
    print(f"3-way blend applied on {len(_blend_classes)} mapped active classes")

    from scipy.ndimage import gaussian_filter1d as _gf
    _view = final_test_scores.reshape(-1, N_WINDOWS, N_CLASSES)
    final_test_scores = _gf(_view, sigma=POST_BLEND_GAUSS_SIGMA, axis=1, mode='nearest').reshape(final_test_scores.shape)
    print(f"Post-blend Gauss smoothing sigma={POST_BLEND_GAUSS_SIGMA}")

    del sed29_scores, sed38_scores
    gc.collect()
else:
    print("\\nNo SED ckpt found; skipping SED blend")
"""


def main():
    nb = json.load(open(NB_PATH))
    cell = nb['cells'][50]
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

    # Find boundaries of the old SED block
    start_marker = "# --- SED29 z-score blend"
    end_marker = "scaled_scores = final_test_scores / class_temperatures"
    si = src.find(start_marker)
    ei = src.find(end_marker)
    if si < 0 or ei < 0:
        print(f"ERROR: markers not found. start={si} end={ei}")
        sys.exit(1)

    new_src = src[:si] + NEW_SED_BLOCK + "\n" + src[ei:]
    cell['source'] = new_src
    json.dump(nb, open(NB_PATH, 'w'), indent=1)
    print(f"Patched cell 50. Old size {len(src)} → new size {len(new_src)}")


if __name__ == "__main__":
    main()
