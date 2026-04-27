#!/usr/bin/env python3
"""exp94 — Same TP/FP/TN/FN signal hunt for exp50 SED's pre-head features.

exp50 SED architecture:
  audio (20s) → mel (128, T) → bn0 → backbone HGNet B0
  → feat (B, C, F, T') → mean over F → (B, C, T')
  → head: att(C → 234), cla(C → 234) → softmax(att) ⊙ cla → clip, fmax

The pre-head feature (B, C, T') has time axis T'. Hook to capture, then FFT
along time. Test whether SED shows similar TN-vs-FN separability that
Perch's spatial_embedding showed (AUC 0.71 on low_high_ratio, 0.69 on
ac_dc_ratio, 0.74 on top1).

If yes: both backbones agree on FN signal — strong universal lever.
If no: signal is Perch-specific.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torchaudio, soundfile as sf, timm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, MW, DATA, ROOT, N_CLS, SR)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend


def get_cached(name): return np.load(EXP80 / name)["scores"]


def extract_sed_pre_head(sc_g, ckpt_path):
    """Run exp50, hook the pre-head features. Returns (n_rows, channels, time)."""
    cache = EXP80 / "sed50_prehead_labeled.npz"
    if cache.exists():
        d = np.load(cache)
        return d["feat"], d["clip"]

    print(f"  building SED pre-head feature cache (~3 min)...", flush=True)

    SED_N_MELS, SED_N_FFT, SED_HOP = 128, 2048, 512
    SED_FMIN, SED_FMAX = 50, 14000
    SED_CHUNK_SEC = 20
    SED_CHUNK_SAMPLES = SR * SED_CHUNK_SEC

    class _MelExt(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=SED_N_FFT, hop_length=SED_HOP, n_mels=SED_N_MELS,
                f_min=SED_FMIN, f_max=SED_FMAX, power=2.0, center=True)
            self.adb = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)

    class _Head(nn.Module):
        def __init__(self, fd, nc):
            super().__init__()
            self.att = nn.Conv1d(fd, nc, 1); self.cla = nn.Conv1d(fd, nc, 1)
        def forward(self, x):
            a, c = self.att(x), self.cla(x)
            return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values

    class _SED(nn.Module):
        def __init__(self, bb='hgnetv2_b0.ssld_stage2_ft_in1k'):
            super().__init__()
            self.mel = _MelExt(); self.bn0 = nn.BatchNorm2d(SED_N_MELS)
            self.backbone = timm.create_model(bb, pretrained=False, in_chans=1, num_classes=0, global_pool='')
            with torch.no_grad():
                f = self.backbone(torch.zeros(1, 1, SED_N_MELS, 100))
            self.head = _Head(f.shape[1], N_CLS)
            self._captured_feat = None
        def forward(self, x):
            m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
            feat = self.backbone(m)
            feat = feat.mean(dim=2) if feat.dim() == 4 else feat   # (B, C, T')
            self._captured_feat = feat.detach().cpu()
            clip, _ = self.head(feat)
            return clip, feat

    st = torch.load(MW / "exp50_hgnet_sed.pt", map_location="cuda", weights_only=False)
    bb = st.get('backbone', 'hgnetv2_b0.ssld_stage2_ft_in1k')
    model = _SED(bb).to("cuda").eval()
    model.load_state_dict(st['state_dict'])

    SS_DIR = DATA / "train_soundscapes"
    files = sorted(sc_g.filename.unique())
    fname_idx = {f: [] for f in files}
    for i, r in sc_g.iterrows():
        fname_idx[r.filename].append((i, int(r.end_sec)))

    # Determine feature shape from one forward pass
    n_rows = len(sc_g)
    BATCH_F = 8
    audio_len = 60 * SR

    feat_tensor = None
    clip_arr = np.zeros((n_rows, N_CLS), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(files), BATCH_F):
            batch = files[s:s+BATCH_F]
            chunks = []; meta = []
            for bi, fn in enumerate(batch):
                y, _ = sf.read(SS_DIR / fn, dtype="float32", always_2d=False)
                if y.ndim == 2: y = y.mean(axis=1)
                if len(y) < audio_len: y = np.pad(y, (0, audio_len - len(y)))
                y = y[:audio_len]
                for ci in range(3):
                    st_idx = ci * SED_CHUNK_SAMPLES
                    chunks.append(y[st_idx:st_idx + SED_CHUNK_SAMPLES])
                    meta.append((bi, ci))
            x = torch.from_numpy(np.stack(chunks)).to("cuda")
            clip, feat = model(x)
            p = torch.sigmoid(clip).cpu().numpy()
            f_cpu = feat.cpu().numpy()   # (B, C, T')
            if feat_tensor is None:
                C, T_prime = f_cpu.shape[1], f_cpu.shape[2]
                print(f"  feature shape per chunk: C={C}, T'={T_prime}", flush=True)
                # We'll average across the 3 chunks per file. So per-row feat is (C, T').
                feat_tensor = np.zeros((n_rows, C, T_prime), dtype=np.float32)
            for k, (bi, ci) in enumerate(meta):
                fn = batch[bi]
                for row_idx, end_sec in fname_idx[fn]:
                    win_idx = (end_sec - 5) // 5
                    if win_idx // 4 == ci:
                        clip_arr[row_idx] = p[k]
                        feat_tensor[row_idx] = f_cpu[k]
            if (s // BATCH_F) % 10 == 0:
                print(f"    {s}/{len(files)} files, {time.time()-t0:.1f}s", flush=True)
    cache.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache, feat=feat_tensor, clip=clip_arr)
    print(f"  cached → {cache}", flush=True)
    return feat_tensor, clip_arr


def compute_spectral_features(feat):
    """feat: (n, C, T'). FFT along time. Returns dict."""
    fft = np.fft.rfft(feat, axis=2)
    mag = np.abs(fft)
    n_bins = mag.shape[2]
    total = mag.sum(axis=(1, 2))
    dc = mag[:, :, 0].sum(axis=-1)
    ac = mag[:, :, 1:].sum(axis=(1, 2))
    # low/high split based on n_bins
    low_end = max(2, n_bins // 4)
    high_start = max(low_end + 1, 3 * n_bins // 4)
    low_band = mag[:, :, 1:low_end].sum(axis=(1, 2))
    high_band = mag[:, :, high_start:].sum(axis=(1, 2))
    low_high_ratio = low_band / (high_band + 1e-6)
    eps = 1e-12
    pp = mag[:, :, 1:] / (mag[:, :, 1:].sum(axis=2, keepdims=True) + eps)
    spec_ent = -(pp * np.log(pp + eps)).sum(axis=2) / np.log(n_bins - 1)
    spec_ent_mean = spec_ent.mean(axis=-1)
    peak_freq = mag[:, :, 1:].argmax(axis=2).mean(axis=-1)
    time_var = feat.var(axis=2).mean(axis=-1)
    return {
        "total_energy": total,
        "dc_energy": dc,
        "ac_energy": ac,
        "ac_dc_ratio": ac / (dc + 1e-6),
        "low_high_ratio": low_high_ratio,
        "spec_ent_mean": spec_ent_mean,
        "peak_freq_mean": peak_freq,
        "time_var": time_var,
    }


def main():
    print("=== exp94: SED pre-head FFT + TP/FP/TN/FN comparison with Perch ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")

    print("Extracting SED50 pre-head features (one-time, cache after)...", flush=True)
    sed_feat, sed_clip = extract_sed_pre_head(sc_g, MW / "exp50_hgnet_sed.pt")
    print(f"sed_feat: {sed_feat.shape}, sed_clip: {sed_clip.shape}", flush=True)

    print("Computing SED FFT spectral features...", flush=True)
    feats_sed = compute_spectral_features(sed_feat)
    print("  features:", list(feats_sed.keys()))

    # v33 baseline for TP/FP labels
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    col_var = perch_prob.var(axis=0)
    unmapped_idx = np.where(col_var < 1e-6)[0]
    aves_mask = sp_taxon == "Aves"
    candidate_classes = []
    for c in range(N_CLS):
        if not aves_mask[c]: continue
        if c in unmapped_idx: continue
        n_pos = Y[:, c].sum()
        n_neg = (Y[:, c] == 0).sum()
        if n_pos >= 5 and n_neg >= 50:
            candidate_classes.append(c)
    print(f"\nMapped Aves classes (≥5 pos, ≥50 neg): {len(candidate_classes)}")

    # SED-specific output features
    sed_top1 = sed_clip.max(axis=-1)
    sed_top5_mass = np.sort(sed_clip, axis=-1)[:, -5:].sum(axis=-1) / (sed_clip.sum(axis=-1) + 1e-12)
    eps = 1e-12
    pp_sed = sed_clip / (sed_clip.sum(axis=-1, keepdims=True) + eps)
    sed_ent = -(pp_sed * np.log(pp_sed + eps)).sum(axis=-1) / np.log(N_CLS)

    # Build quadrants
    THRESH = 0.5
    quadrant_perch_top1 = {q: [] for q in ("TP", "FP", "TN", "FN")}
    quadrant_sed_top1 = {q: [] for q in ("TP", "FP", "TN", "FN")}
    quadrant_sed_ent = {q: [] for q in ("TP", "FP", "TN", "FN")}
    quadrant_sed_top5 = {q: [] for q in ("TP", "FP", "TN", "FN")}
    quadrant_emb_L2 = {q: [] for q in ("TP", "FP", "TN", "FN")}
    quadrant_sed_feats = {q: {f: [] for f in feats_sed} for q in ("TP", "FP", "TN", "FN")}

    for c in candidate_classes:
        pred = v33[:, c] > THRESH
        for i in range(len(sc_g)):
            y = Y[i, c]; p = pred[i]
            if y == 1 and p:    q = "TP"
            elif y == 1 and not p: q = "FN"
            elif y == 0 and p:  q = "FP"
            else:                q = "TN"
            quadrant_perch_top1[q].append(perch_prob[i].max())
            quadrant_sed_top1[q].append(sed_top1[i])
            quadrant_sed_ent[q].append(sed_ent[i])
            quadrant_sed_top5[q].append(sed_top5_mass[i])
            quadrant_emb_L2[q].append(np.linalg.norm(perch_emb[i]))
            for f in feats_sed:
                quadrant_sed_feats[q][f].append(feats_sed[f][i])

    print(f"\nQuadrant counts:")
    for q in ("TP", "FN", "FP", "TN"):
        print(f"  {q}: {len(quadrant_perch_top1[q])}")

    # Pairwise AUCs: TP_vs_FP, TN_vs_FN, CORR_vs_WRONG
    print(f"\n=== TP vs FP / TN vs FN / CORRECT vs WRONG (SED + comparison features) ===")
    print(f"  {'feat':<22} {'TP_vs_FP':>10} {'TN_vs_FN':>10} {'CORR_vs_WRONG':>14}")

    def pair_auc(tp_a, fp_a, tn_a, fn_a):
        rows = {}
        try:
            rows["TP_vs_FP"] = roc_auc_score(np.concatenate([np.zeros(len(tp_a)), np.ones(len(fp_a))]),
                                              np.concatenate([tp_a, fp_a]))
        except: rows["TP_vs_FP"] = np.nan
        try:
            rows["TN_vs_FN"] = roc_auc_score(np.concatenate([np.zeros(len(tn_a)), np.ones(len(fn_a))]),
                                              np.concatenate([tn_a, fn_a]))
        except: rows["TN_vs_FN"] = np.nan
        try:
            corr = np.concatenate([tp_a, tn_a]); wrong = np.concatenate([fp_a, fn_a])
            rows["CORR_vs_WRONG"] = roc_auc_score(np.concatenate([np.zeros(len(corr)), np.ones(len(wrong))]),
                                                    np.concatenate([corr, wrong]))
        except: rows["CORR_vs_WRONG"] = np.nan
        return rows

    # SED output features
    perch_top1_arr = lambda q: np.array(quadrant_perch_top1[q])
    sed_top1_arr = lambda q: np.array(quadrant_sed_top1[q])
    sed_ent_arr = lambda q: np.array(quadrant_sed_ent[q])
    sed_top5_arr = lambda q: np.array(quadrant_sed_top5[q])

    for nm, getter in [("perch_top1", perch_top1_arr),
                        ("sed_top1", sed_top1_arr),
                        ("sed_entropy_norm", sed_ent_arr),
                        ("sed_top5_mass", sed_top5_arr)]:
        r = pair_auc(getter("TP"), getter("FP"), getter("TN"), getter("FN"))
        print(f"  {nm:<22} {r['TP_vs_FP']:>10.3f} {r['TN_vs_FN']:>10.3f} {r['CORR_vs_WRONG']:>14.3f}")

    # SED FFT features
    print(f"\n  --- SED pre-head FFT features ---")
    for f in feats_sed:
        getter_f = lambda q, ff=f: np.array(quadrant_sed_feats[q][ff])
        r = pair_auc(getter_f("TP"), getter_f("FP"), getter_f("TN"), getter_f("FN"))
        print(f"  sed_{f:<18} {r['TP_vs_FP']:>10.3f} {r['TN_vs_FN']:>10.3f} {r['CORR_vs_WRONG']:>14.3f}")

    print(f"\n=== Summary comparison Perch (exp93) vs SED (exp94) ===")
    print(f"  Perch top1 TN_vs_FN: 0.740 — strongest signal in Perch")
    print(f"  Perch low_high_ratio: 0.713")
    print(f"  Perch ac_dc_ratio: 0.685")
    print(f"  SED features above — does same direction emerge?")


if __name__ == "__main__":
    main()
