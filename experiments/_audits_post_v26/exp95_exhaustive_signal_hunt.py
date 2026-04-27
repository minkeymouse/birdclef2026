#!/usr/bin/env python3
"""exp95 — Exhaustive TP/FP/TN/FN signal hunt.

Three concurrent experiments:
  A. ConvNeXt-tiny (exp59) pre-head features — pure CNN test of
     transformer-vs-CNN hypothesis. If exp59 also has muted FN
     signature like SED50, the hypothesis "Perch's transformer-augmented
     architecture is what enables FN signature" is supported.

  B. SED50 attention LAYER OUTPUT — extract `att(feat)` after softmax
     and the `cla(feat) * softmax(att(feat))` combined output. Tests
     whether SED's attention (1D conv) encodes FN signal that pre-head
     features didn't show.

  C. Additional features for TP/FP/TN/FN separation:
     - Logit margins (top1-top2, top1-mean) at Perch and SED outputs
     - Embedding sparsity (gini coefficient, kurtosis)
     - Cross-class prediction concentration (effective # predictions)
     - Cross-model agreement: |Perch[i,c] - exp50[i,c]| per row+class
     - exp50 raw score on class c (does SED itself flag FN class?)
     - File-level mean/max for class c (within-file signal)
     - Hour of day, site indicator
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torchaudio, soundfile as sf, timm
from sklearn.metrics import roc_auc_score
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, MW, DATA, ROOT, N_CLS, SR)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend


def get_cached(name): return np.load(EXP80 / name)["scores"]


def extract_sed_features(sc_g, ckpt_path, backbone, cache_basename):
    """Run SED, return (pre_head_feat, att_output, clip)."""
    cache = EXP80 / cache_basename
    if cache.exists():
        d = np.load(cache)
        return d["feat"], d["att_out"], d["clip"]
    print(f"  building {cache_basename} (~3 min)...", flush=True)

    SED_N_MELS, SED_N_FFT, SED_HOP = 128, 2048, 512
    SED_FMIN, SED_FMAX = 50, 14000
    SED_CHUNK_SAMPLES = SR * 20

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
            self.att = nn.Conv1d(fd, nc, 1)
            self.cla = nn.Conv1d(fd, nc, 1)
        def forward(self, x):
            a, c = self.att(x), self.cla(x)
            w = torch.softmax(a, dim=-1)   # (B, n_cls, T') — attention weights over time
            clip = (w * c).sum(-1)
            # att_out = (w * c) before sum: (B, n_cls, T') — class-wise attended scores per timestep
            att_out_max = (w * c).max(-1).values   # (B, n_cls) — max over time
            return clip, att_out_max

    class _SED(nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.mel = _MelExt(); self.bn0 = nn.BatchNorm2d(SED_N_MELS)
            self.backbone = timm.create_model(bb, pretrained=False, in_chans=1, num_classes=0, global_pool='')
            with torch.no_grad():
                f = self.backbone(torch.zeros(1, 1, SED_N_MELS, 100))
            self.head = _Head(f.shape[1], N_CLS)
        def forward(self, x):
            m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
            feat = self.backbone(m)
            feat = feat.mean(dim=2) if feat.dim() == 4 else feat   # (B, C, T')
            clip, att_max = self.head(feat)
            return clip, att_max, feat

    st = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    bb = st.get('backbone') or st.get('config', {}).get('backbone') or backbone
    model = _SED(bb).to("cuda").eval()
    model.load_state_dict(st['state_dict'])

    SS_DIR = DATA / "train_soundscapes"
    files = sorted(sc_g.filename.unique())
    fname_idx = {f: [] for f in files}
    for i, r in sc_g.iterrows():
        fname_idx[r.filename].append((i, int(r.end_sec)))

    n_rows = len(sc_g)
    BATCH_F = 8
    feat_t = None; att_arr = np.zeros((n_rows, N_CLS), dtype=np.float32); clip_arr = np.zeros((n_rows, N_CLS), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(files), BATCH_F):
            batch = files[s:s+BATCH_F]
            chunks = []; meta = []
            for bi, fn in enumerate(batch):
                y, _ = sf.read(SS_DIR / fn, dtype="float32", always_2d=False)
                if y.ndim == 2: y = y.mean(axis=1)
                if len(y) < 60*SR: y = np.pad(y, (0, 60*SR - len(y)))
                y = y[:60*SR]
                for ci in range(3):
                    chunks.append(y[ci*SED_CHUNK_SAMPLES:(ci+1)*SED_CHUNK_SAMPLES])
                    meta.append((bi, ci))
            x = torch.from_numpy(np.stack(chunks)).to("cuda")
            clip, att_max, feat = model(x)
            p = torch.sigmoid(clip).cpu().numpy()
            f_cpu = feat.cpu().numpy()
            a_cpu = att_max.cpu().numpy()
            if feat_t is None:
                C, T_p = f_cpu.shape[1], f_cpu.shape[2]
                feat_t = np.zeros((n_rows, C, T_p), dtype=np.float32)
            for k, (bi, ci) in enumerate(meta):
                fn = batch[bi]
                for row_idx, end_sec in fname_idx[fn]:
                    win_idx = (end_sec - 5) // 5
                    if win_idx // 4 == ci:
                        clip_arr[row_idx] = p[k]
                        feat_t[row_idx] = f_cpu[k]
                        att_arr[row_idx] = a_cpu[k]
            if (s // BATCH_F) % 10 == 0:
                print(f"    {s}/{len(files)} files, {time.time()-t0:.1f}s", flush=True)
    np.savez_compressed(cache, feat=feat_t, att_out=att_arr, clip=clip_arr)
    print(f"  cached → {cache}", flush=True)
    return feat_t, att_arr, clip_arr


def fft_feats(feat):
    """feat: (n, C, T'). FFT along time axis. Returns dict of per-row features."""
    fft = np.fft.rfft(feat, axis=2)
    mag = np.abs(fft)
    n_bins = mag.shape[2]
    eps = 1e-12
    total = mag.sum(axis=(1, 2))
    dc = mag[:, :, 0].sum(axis=-1)
    ac = mag[:, :, 1:].sum(axis=(1, 2))
    low_end = max(2, n_bins // 4)
    high_start = max(low_end + 1, 3 * n_bins // 4)
    low_band = mag[:, :, 1:low_end].sum(axis=(1, 2))
    high_band = mag[:, :, high_start:].sum(axis=(1, 2))
    pp = mag[:, :, 1:] / (mag[:, :, 1:].sum(axis=2, keepdims=True) + eps)
    spec_ent = -(pp * np.log(pp + eps)).sum(axis=2) / np.log(n_bins - 1)
    return {
        "total": total, "dc": dc, "ac": ac,
        "ac_dc_ratio": ac / (dc + 1e-6),
        "low_high_ratio": low_band / (high_band + 1e-6),
        "spec_ent_mean": spec_ent.mean(axis=-1),
        "peak_freq_mean": mag[:, :, 1:].argmax(axis=2).mean(axis=-1),
        "time_var": feat.var(axis=2).mean(axis=-1),
    }


def gini_coef(x):
    """Gini per-row of |x|. x: (n, d)."""
    a = np.abs(x).copy()
    a.sort(axis=-1)
    n = a.shape[-1]
    idx = np.arange(1, n + 1)
    num = (2 * idx - n - 1) * a
    return num.sum(axis=-1) / (n * a.sum(axis=-1) + 1e-12)


def pair_auc(tp_a, fp_a, tn_a, fn_a):
    out = {}
    try:
        out["TP_vs_FP"] = roc_auc_score(np.concatenate([np.zeros(len(tp_a)), np.ones(len(fp_a))]),
                                         np.concatenate([tp_a, fp_a]))
    except: out["TP_vs_FP"] = np.nan
    try:
        out["TN_vs_FN"] = roc_auc_score(np.concatenate([np.zeros(len(tn_a)), np.ones(len(fn_a))]),
                                         np.concatenate([tn_a, fn_a]))
    except: out["TN_vs_FN"] = np.nan
    try:
        corr = np.concatenate([tp_a, tn_a]); wrong = np.concatenate([fp_a, fn_a])
        out["CORR_vs_WRONG"] = roc_auc_score(np.concatenate([np.zeros(len(corr)), np.ones(len(wrong))]),
                                              np.concatenate([corr, wrong]))
    except: out["CORR_vs_WRONG"] = np.nan
    return out


def main():
    print("=== exp95: exhaustive TP/FP/TN/FN signal hunt ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")
    exp59 = get_cached("exp59_scores_labeled.npz")

    # === A. Extract exp59 ConvNeXt features ===
    print("Extracting exp59 ConvNeXt-tiny pre-head + att-output features...", flush=True)
    cnxt_feat, cnxt_att, cnxt_clip = extract_sed_features(
        sc_g, ROOT / "experiments/_archive_2026_audits/outputs/exp59_outputs/best_ckpt.pt",
        "convnext_tiny.fb_in22k_ft_in1k", "exp59_features_labeled.npz")
    print(f"exp59 feat: {cnxt_feat.shape}", flush=True)

    # === B. Re-load SED50 features (already cached from exp94) + att output ===
    print("Loading/extracting SED50 features + att output...", flush=True)
    sed_feat, sed_att, sed_clip = extract_sed_features(
        sc_g, MW / "exp50_hgnet_sed.pt",
        "hgnetv2_b0.ssld_stage2_ft_in1k", "exp50_features_full_labeled.npz")
    print(f"sed50 feat: {sed_feat.shape}", flush=True)

    # Compute FFT features for both
    print("\nComputing FFT features for SED50, exp59 ConvNeXt...", flush=True)
    f_sed = fft_feats(sed_feat)
    f_cnxt = fft_feats(cnxt_feat)

    # Load Perch spatial cache + features
    print("Loading Perch spatial features (from exp93 cache)...", flush=True)
    p_spatial = np.load(EXP80 / "spatial_emb_labeled.npz")["spatial"]
    f_perch = fft_feats(p_spatial.transpose(0, 2, 1))   # transpose to (n, C, T)
    print(f"Perch spatial: {p_spatial.shape}", flush=True)

    # Build v33 to get TP/FP/TN/FN labels
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    col_var = perch_prob.var(axis=0)
    unmapped_idx = np.where(col_var < 1e-6)[0]
    aves_mask = sp_taxon == "Aves"
    candidate_classes = [c for c in range(N_CLS)
                          if aves_mask[c] and c not in unmapped_idx
                          and Y[:, c].sum() >= 5 and (Y[:, c] == 0).sum() >= 50]
    print(f"\nCandidate Aves classes: {len(candidate_classes)}")

    # Per-row scalar features (don't depend on class)
    n_rows = len(sc_g)
    perch_top1 = perch_prob.max(axis=-1)
    perch_top1_top2 = np.sort(perch_prob, axis=-1)[:, -1] - np.sort(perch_prob, axis=-1)[:, -2]
    perch_eff_dim = (perch_prob.sum(-1) ** 2) / ((perch_prob ** 2).sum(-1) + 1e-12)   # ~ # active classes
    sed_top1 = sed_clip.max(-1)
    sed_top1_top2 = np.sort(sed_clip, axis=-1)[:, -1] - np.sort(sed_clip, axis=-1)[:, -2]
    sed_eff_dim = (sed_clip.sum(-1) ** 2) / ((sed_clip ** 2).sum(-1) + 1e-12)
    cnxt_top1 = cnxt_clip.max(-1)
    cnxt_eff_dim = (cnxt_clip.sum(-1) ** 2) / ((cnxt_clip ** 2).sum(-1) + 1e-12)
    emb_L2 = np.linalg.norm(perch_emb, axis=-1)
    emb_max = np.abs(perch_emb).max(-1)
    emb_gini = gini_coef(perch_emb)
    emb_kurt = kurtosis(perch_emb, axis=-1, fisher=True)
    perch_sed_corr_per_row = np.array([np.corrcoef(perch_prob[i], exp50[i])[0, 1] for i in range(n_rows)])

    # Quadrant gather
    THRESH = 0.5
    Q_keys = ("TP", "FP", "TN", "FN")
    qset = {q: {"row_idx": [], "class_idx": []} for q in Q_keys}
    for c in candidate_classes:
        pred = v33[:, c] > THRESH
        for i in range(n_rows):
            y = Y[i, c]; p = pred[i]
            q = "TP" if (y and p) else ("FN" if (y and not p) else ("FP" if (not y and p) else "TN"))
            qset[q]["row_idx"].append(i)
            qset[q]["class_idx"].append(c)

    print(f"\nQuadrant counts:")
    for q in ("TP", "FN", "FP", "TN"):
        print(f"  {q}: {len(qset[q]['row_idx'])}")

    # Define helper for per-row feature
    def gather_row(values, q):
        return np.array([values[i] for i in qset[q]["row_idx"]])

    # Class-dependent features
    def gather_perch_on_c(q):
        return np.array([perch_prob[i, c] for i, c in zip(qset[q]["row_idx"], qset[q]["class_idx"])])
    def gather_sed_on_c(q):
        return np.array([exp50[i, c] for i, c in zip(qset[q]["row_idx"], qset[q]["class_idx"])])
    def gather_cnxt_on_c(q):
        return np.array([exp59[i, c] for i, c in zip(qset[q]["row_idx"], qset[q]["class_idx"])])
    def gather_v33_on_c(q):
        return np.array([v33[i, c] for i, c in zip(qset[q]["row_idx"], qset[q]["class_idx"])])
    def gather_disagreement(q):
        return np.array([abs(perch_prob[i, c] - exp50[i, c]) for i, c in zip(qset[q]["row_idx"], qset[q]["class_idx"])])
    def gather_file_max_on_c(q):
        # File-level max for class c per row
        out = []
        for i, c in zip(qset[q]["row_idx"], qset[q]["class_idx"]):
            fn = sc_g.iloc[i].filename
            file_idx = sc_g[sc_g.filename == fn].index
            fm = perch_prob[file_idx, c].max()
            out.append(fm)
        return np.array(out)

    # === Run all features through TP/FP/TN/FN AUC ===
    print("\n=== AUC table for all features ===")
    print(f"  {'feature':<30} {'TP_vs_FP':>10} {'TN_vs_FN':>10} {'CORR_vs_WRONG':>14}")

    feature_groups = {
        "OUTPUT (per-row)": {
            "perch_top1": perch_top1,
            "perch_top1_top2_margin": perch_top1_top2,
            "perch_eff_dim": perch_eff_dim,
            "sed_top1": sed_top1,
            "sed_top1_top2_margin": sed_top1_top2,
            "sed_eff_dim": sed_eff_dim,
            "cnxt_top1": cnxt_top1,
            "cnxt_eff_dim": cnxt_eff_dim,
            "perch_sed_corr_per_row": perch_sed_corr_per_row,
        },
        "EMBEDDING (per-row)": {
            "emb_L2": emb_L2,
            "emb_max": emb_max,
            "emb_gini": emb_gini,
            "emb_kurtosis": emb_kurt,
        },
        "PERCH spatial-FFT": f_perch,
        "SED50 pre-head FFT": f_sed,
        "ConvNeXt pre-head FFT": f_cnxt,
        "PER-CLASS (row, c)": {  # gather differently
            "perch_on_c": "per_c_perch",
            "sed_on_c": "per_c_sed",
            "cnxt_on_c": "per_c_cnxt",
            "v33_on_c": "per_c_v33",
            "perch_sed_disagreement": "per_c_disagree",
            "file_max_perch_c": "per_c_filemax",
        },
    }

    summary_rows = []
    for group_name, group_feats in feature_groups.items():
        print(f"\n  --- {group_name} ---")
        for fname, fvalues in group_feats.items():
            try:
                if isinstance(fvalues, str):
                    if fvalues == "per_c_perch":
                        tp, fp, tn, fn = gather_perch_on_c("TP"), gather_perch_on_c("FP"), gather_perch_on_c("TN"), gather_perch_on_c("FN")
                    elif fvalues == "per_c_sed":
                        tp, fp, tn, fn = gather_sed_on_c("TP"), gather_sed_on_c("FP"), gather_sed_on_c("TN"), gather_sed_on_c("FN")
                    elif fvalues == "per_c_cnxt":
                        tp, fp, tn, fn = gather_cnxt_on_c("TP"), gather_cnxt_on_c("FP"), gather_cnxt_on_c("TN"), gather_cnxt_on_c("FN")
                    elif fvalues == "per_c_v33":
                        tp, fp, tn, fn = gather_v33_on_c("TP"), gather_v33_on_c("FP"), gather_v33_on_c("TN"), gather_v33_on_c("FN")
                    elif fvalues == "per_c_disagree":
                        tp, fp, tn, fn = gather_disagreement("TP"), gather_disagreement("FP"), gather_disagreement("TN"), gather_disagreement("FN")
                    elif fvalues == "per_c_filemax":
                        tp, fp, tn, fn = gather_file_max_on_c("TP"), gather_file_max_on_c("FP"), gather_file_max_on_c("TN"), gather_file_max_on_c("FN")
                else:
                    tp = gather_row(fvalues, "TP"); fp = gather_row(fvalues, "FP")
                    tn = gather_row(fvalues, "TN"); fn = gather_row(fvalues, "FN")
                r = pair_auc(tp, fp, tn, fn)
                print(f"  {fname:<30} {r['TP_vs_FP']:>10.3f} {r['TN_vs_FN']:>10.3f} {r['CORR_vs_WRONG']:>14.3f}")
                summary_rows.append({"group": group_name, "feature": fname,
                                       **r,
                                       "tp_mean": tp.mean(), "fn_mean": fn.mean(),
                                       "fp_mean": fp.mean(), "tn_mean": tn.mean()})
            except Exception as e:
                print(f"  {fname:<30}  ERROR: {e}")

    df = pd.DataFrame(summary_rows)
    df.to_csv(EXP80 / "exp95_signal_hunt.csv", index=False)
    print(f"\nSaved → {EXP80}/exp95_signal_hunt.csv")

    # === Top features by max separation strength ===
    print("\n=== Top 15 features by max(|TN_vs_FN-0.5|, |TP_vs_FP-0.5|, |CORR_vs_WRONG-0.5|) ===")
    df["best_strength"] = df[["TP_vs_FP", "TN_vs_FN", "CORR_vs_WRONG"]].sub(0.5).abs().max(axis=1)
    print(df.sort_values("best_strength", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
