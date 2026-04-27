#!/usr/bin/env python3
"""exp72 — Diagnostic: do exp50/Perch already predict the right species
on our newly downloaded iNaturalist clips? If YES, retraining redundant.
If NO, external clips contain signal we don't have."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch, torch.nn as nn
import timm, torchaudio
import onnxruntime as ort

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXT = ROOT / "data/external"
EXP50 = ROOT / "experiments/_data_pipelines/exp50_outputs"
ONNX_PATH = Path("/tmp/perch_v2.onnx")
SR = 32000
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000
DEVICE = "cuda"


class _Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class _Head(nn.Module):
    def __init__(self, f, n):
        super().__init__()
        self.att = nn.Conv1d(f, n, 1); self.cla = nn.Conv1d(f, n, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values


class _SED(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.mel = _Mel(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model("hgnetv2_b0.ssld_stage2_ft_in1k",
                                          pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = _Head(feat.shape[1], n_cls)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1,2); m = self.bn0(m); m = m.transpose(1,2)
        f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
        c, _ = self.head(f); return c


def load_clip(path, target_sec=20):
    """Load + resample + center-crop / loop-pad to target_sec."""
    target = target_sec * SR
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != SR:
            import torchaudio.functional as TF
            wav = TF.resample(torch.from_numpy(wav), sr, SR).numpy()
        if len(wav) == 0: return np.zeros(target, dtype=np.float32)
        if len(wav) < target:
            wav = np.tile(wav, target // len(wav) + 1)[:target]
        else:
            s = (len(wav) - target) // 2
            wav = wav[s:s + target]
        return wav.astype(np.float32)
    except Exception as e:
        print(f"  load fail {path}: {e}")
        return np.zeros(target, dtype=np.float32)


def main():
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    l2i = {p: i for i, p in enumerate(primary)}

    # Collect external clips
    clips = []  # list of (species, path)
    for sp_dir in EXT.iterdir():
        if not sp_dir.is_dir() or sp_dir.name.startswith("_"): continue
        sp = sp_dir.name
        if sp not in l2i: continue
        for f in sp_dir.iterdir():
            if f.suffix.lower() in {".ogg", ".mp3", ".wav"} and f.is_file():
                clips.append((sp, f))
    print(f"Total external clips: {len(clips)}")

    # Group by species
    by_sp = {}
    for sp, p in clips:
        by_sp.setdefault(sp, []).append(p)
    for sp, ps in by_sp.items():
        print(f"  {sp}: {len(ps)} clips")

    # Load Perch ONNX
    if not ONNX_PATH.exists():
        print(f"Perch ONNX not at {ONNX_PATH}, skip Perch")
        sess = None
    else:
        sess = ort.InferenceSession(str(ONNX_PATH),
                                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name

    # Load exp50
    ck = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED(n_cls=234).to(DEVICE); m50.load_state_dict(ck["state_dict"]); m50.eval()

    # For each species, run prediction on its clips, check rank of correct class
    print(f"\n{'sp':<14} {'n':>3} {'Perch self':>11} {'P_rank':>7} {'exp50 self':>11} {'e50_rank':>9}")
    for sp, paths in by_sp.items():
        sp_idx = l2i[sp]
        perch_self_vals = []; perch_ranks = []
        e50_self_vals = []; e50_ranks = []
        for p in paths:
            wav20 = load_clip(p, 20)  # for exp50
            wav5 = load_clip(p, 5)    # for Perch (input is 5-sec)
            with torch.no_grad():
                # exp50
                x20 = torch.from_numpy(wav20[None]).to(DEVICE)
                e50_logits = m50(x20)
                e50_probs = torch.sigmoid(e50_logits).cpu().numpy()[0]
                e50_self_vals.append(e50_probs[sp_idx])
                rank_e50 = (e50_probs > e50_probs[sp_idx]).sum()  # higher predictions
                e50_ranks.append(rank_e50)
                # Perch
                if sess is not None:
                    perch_out = sess.run(["embedding", "label"], {iname: wav5[None]})
                    perch_logits = perch_out[1][0]
                    # Map perch's 14k labels to our 234 — load mapping if exists
                    # For diagnostic, just use exp50 mapping (we want raw signal not blend)
                    # Skip perch for now since we'd need full mapping
        if e50_self_vals:
            print(f"  {sp:<14} {len(paths):>3} "
                  f"{'-':>11} {'-':>7} "
                  f"{np.mean(e50_self_vals):>11.3f} {np.mean(e50_ranks):>9.1f}")

    # Top-class prediction analysis: for each clip, what does exp50 predict as top?
    print("\n=== Top-3 exp50 predictions per clip ===")
    for sp, paths in list(by_sp.items())[:5]:
        sp_idx = l2i[sp]
        print(f"\n  --- {sp} (idx={sp_idx}) ---")
        for p in paths[:5]:  # first 5 per species
            wav20 = load_clip(p, 20)
            with torch.no_grad():
                x20 = torch.from_numpy(wav20[None]).to(DEVICE)
                e50_probs = torch.sigmoid(m50(x20)).cpu().numpy()[0]
            top3 = np.argsort(-e50_probs)[:3]
            top3_str = ", ".join(f"{primary[i]}:{e50_probs[i]:.2f}" for i in top3)
            self_p = e50_probs[sp_idx]
            print(f"    {p.name:<30}  self={self_p:.3f}  top3=[{top3_str}]")


if __name__ == "__main__":
    main()
