#!/usr/bin/env python3
"""exp58 — Thorough analysis of v28 (v26 + exp51 27-head additive) candidate.

NOT a grid sweep — deep-dive per-class/per-row/per-site analysis to validate
the candidate before spending a Kaggle submission slot.

Analyses:
  Q1. Per-class AUC on held-out 11-file: which classes does v28 specifically
      improve, regress, or unchanged? Compare to v26.
  Q2. exp51 prediction quality on the 27 target classes themselves:
      - On positive rows: prediction magnitude
      - On negative rows: prediction magnitude
      - AUC for each of 27 classes
  Q3. Site-shortcut residual test: per-site mean prediction of exp51 on
      held-out + unlabeled files. If exp51 strongly site-correlated, it has
      not actually learned acoustic features.
  Q4. Leave-one-file-out (LOFO) on the 11 eval files: each file's macro
      improvement under v28. Variance estimate.
  Q5. exp51 vs Perch correlation (on the 27 columns): are they truly
      providing independent info?
  Q6. v22 failure mode comparison: KS distribution shift on all 234 cols
      between v26 → v28 vs v26 → v22 (we already have v22 data).
  Q7. False-positive analysis: rows where 27 species are NOT positive,
      what does exp51 output there?
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch, torch.nn as nn
import timm, torchaudio
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, ks_2samp, pearsonr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
EXP51 = ROOT / "experiments/exp51_outputs"
OUT = ROOT / "experiments/exp58_outputs"
OUT.mkdir(exist_ok=True)
SR = 32000; CLIP_SEC = 20; CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000; DEVICE = "cuda"
SEED = 42
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_site(fn):
    m = FNAME_RE.match(fn); return m.group(2) if m else None


def build_all():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g["site"] = sc_g["filename"].apply(parse_site)
    rng = np.random.RandomState(SEED); files = sorted(sc_g.filename.unique()); rng.shuffle(files)
    eval_files = set(files[:11])
    sc_g["split"] = ["eval" if f in eval_files else "train" for f in sc_g.filename]
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g.lbls):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_g, Y, primary, l2i


def align_43a(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(df), scs.shape[1]), np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]
    return S


def align_old(df, p):
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(df), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


# Models
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


class _SED50(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.mel = _Mel()
        self.bn0 = nn.BatchNorm2d(N_MELS)
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


class _SEDFlat(nn.Module):
    """exp51 architecture (flat heads, no .head wrapper)."""
    def __init__(self, n_cls):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=40)
        self.backbone = timm.create_model("hgnetv2_b0.ssld_stage2_ft_in1k",
                                          pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.att = nn.Conv1d(feat.shape[1], n_cls, 1)
        self.cla = nn.Conv1d(feat.shape[1], n_cls, 1)
    def forward(self, x):
        m = self.adb(self.mel(x)).unsqueeze(1)
        m = m.transpose(1,2); m = self.bn0(m); m = m.transpose(1,2)
        f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
        a = self.att(f); c = self.cla(f)
        return (torch.softmax(a, dim=-1) * c).sum(-1)


@torch.no_grad()
def predict(model, df, n_cls):
    model.eval()
    out = np.zeros((len(df), n_cls), dtype=np.float32); cache = {}
    for i in range(0, len(df), 8):
        j = min(len(df), i + 8); wavs = []
        for k in range(i, j):
            row = df.iloc[k]
            if row.filename not in cache:
                w, sr = sf.read(DATA / "train_soundscapes" / row.filename, dtype="float32")
                if w.ndim > 1: w = w.mean(1)
                cache[row.filename] = w
            wav = cache[row.filename]
            cs = int(max(0, (int(row.end_sec) - 2.5) * SR - CLIP_SAMPLES/2))
            cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
            clip = wav[cs:cs + CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            wavs.append(clip.astype(np.float32))
        x = torch.from_numpy(np.stack(wavs)).to(DEVICE)
        out[i:j] = torch.sigmoid(model(x)).cpu().numpy()
    return out


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s
def gauss_pf(scores, df, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in df.filename.unique():
        m = (df.filename == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out
def per_class_auc(Y, P):
    out = {}
    for c in range(Y.shape[1]):
        y = Y[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
        if not np.isfinite(P[:, c]).all(): continue
        try: out[c] = float(roc_auc_score(y, P[:, c]))
        except: pass
    return out


def main():
    print("Loading...")
    sc_all, Y_all, primary, l2i = build_all()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)

    # Load exp50
    ck50 = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED50(n_cls=234).to(DEVICE); m50.load_state_dict(ck50["state_dict"])
    print("Computing exp50...")
    P50 = predict(m50, sc_all, 234)
    del m50; torch.cuda.empty_cache()

    # Load exp51
    ck51 = torch.load(EXP51 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    target_species = ck51["target_species"]
    n51 = len(target_species)
    print(f"exp51 target_species: {n51} ({target_species[:5]}...)")
    m51 = _SEDFlat(n_cls=n51).to(DEVICE); m51.load_state_dict(ck51["state_dict"])
    print("Computing exp51...")
    P51_raw = predict(m51, sc_all, n51)  # shape (N, 27)
    del m51; torch.cuda.empty_cache()

    # Map exp51 to 234-col format
    target_cols = [l2i[t] for t in target_species if t in l2i]
    target_cols_idx = np.array(target_cols, dtype=np.int64)
    target_cols_in_target = np.array([target_species.index(primary[c]) for c in target_cols])
    P51_full = np.zeros((len(sc_all), 234), dtype=np.float32)
    for i, t in enumerate(target_species):
        if t in l2i: P51_full[:, l2i[t]] = P51_raw[:, i]

    zP = zs(perch_prob); z29 = zs(S29); z50 = zs(P50); z51 = zs(P51_full)
    v26_raw = 0.7 * zP + 0.3 * z50
    v26_prob = sigmoid(gauss_pf(v26_raw, sc_all, 0.5))

    # v28 candidate: w27 = 0.10 added to 27 columns only
    def v28_blend(w27):
        raw = v26_raw.copy()
        for c in target_cols:
            raw[:, c] = (1 - w27) * raw[:, c] + w27 * z51[:, c]
        return sigmoid(gauss_pf(raw, sc_all, 0.5))

    print("\n" + "="*70)
    print("Q1: Per-class AUC v26 vs v28 (held-out 11 files only)")
    print("="*70)
    ev_mask = (sc_all.split == "eval").values
    Y_ev = Y_all[ev_mask]
    v26_ev = v26_prob[ev_mask]
    v28_ev_010 = v28_blend(0.10)[ev_mask]
    v28_ev_015 = v28_blend(0.15)[ev_mask]
    v28_ev_020 = v28_blend(0.20)[ev_mask]
    aucs_v26_ev = per_class_auc(Y_ev, v26_ev)
    aucs_v28_ev = per_class_auc(Y_ev, v28_ev_010)
    common = sorted(set(aucs_v26_ev) & set(aucs_v28_ev))
    print(f"\nFor each evaluable class on held-out 11 files (sorted by Δ):")
    rows = []
    for c in common:
        d = aucs_v28_ev[c] - aucs_v26_ev[c]
        rows.append({
            "class": primary[c], "taxon": species_taxon[c], "n_pos": int(Y_ev[:, c].sum()),
            "in_27target": c in target_cols, "v26_auc": aucs_v26_ev[c], "v28_auc": aucs_v28_ev[c],
            "delta": d
        })
    df_q1 = pd.DataFrame(rows).sort_values("delta")
    print(f"  {len(df_q1)} evaluable classes")
    print(f"\n  Worst class drops (v28 < v26):")
    for _, r in df_q1.head(10).iterrows():
        print(f"    {r['class']:<14} ({r.taxon:<9}) n_pos={r.n_pos:3d} 27t={'Y' if r.in_27target else ' '}  "
              f"v26={r.v26_auc:.3f}  v28={r.v28_auc:.3f}  Δ{r.delta:+.4f}")
    print(f"\n  Best class gains (v28 > v26):")
    for _, r in df_q1.tail(10).iterrows():
        print(f"    {r['class']:<14} ({r.taxon:<9}) n_pos={r.n_pos:3d} 27t={'Y' if r.in_27target else ' '}  "
              f"v26={r.v26_auc:.3f}  v28={r.v28_auc:.3f}  Δ{r.delta:+.4f}")
    print(f"\n  Per-taxon mean Δ on held-out:")
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        sub = df_q1[df_q1.taxon == t]
        if len(sub):
            print(f"    {t:<10}  n={len(sub):2d}  mean Δ {sub.delta.mean():+.4f}  median {sub.delta.median():+.4f}")
    print(f"\n  In 27-target columns: n={df_q1.in_27target.sum()}  mean Δ {df_q1[df_q1.in_27target].delta.mean():+.4f}")
    print(f"  Outside 27 (rest):    n={(~df_q1.in_27target).sum()}  mean Δ {df_q1[~df_q1.in_27target].delta.mean():+.4f}")

    print("\n" + "="*70)
    print("Q2: exp51 prediction quality on 27 target classes (held-out)")
    print("="*70)
    Y_27_ev = Y_ev[:, target_cols_idx]  # (N_ev, 27)
    P51_27_ev = P51_full[ev_mask][:, target_cols_idx]
    aucs_p51 = []
    for i in range(n51):
        y = Y_27_ev[:, i]
        if y.sum() == 0 or y.sum() == len(y): continue
        try:
            auc = roc_auc_score(y, P51_27_ev[:, i])
            pos_mean = P51_27_ev[y == 1, i].mean()
            neg_mean = P51_27_ev[y == 0, i].mean()
            aucs_p51.append({
                "class": target_species[i], "n_pos": int(y.sum()),
                "exp51_auc": auc, "pos_mean": pos_mean, "neg_mean": neg_mean,
                "separation": pos_mean - neg_mean,
            })
        except Exception: pass
    df_p51 = pd.DataFrame(aucs_p51).sort_values("exp51_auc", ascending=False)
    print(f"  {len(df_p51)} evaluable 27-target classes:")
    print(df_p51[["class","n_pos","exp51_auc","pos_mean","neg_mean","separation"]].to_string(index=False))
    print(f"\n  Mean exp51 AUC on these (27-target eval-only): {df_p51.exp51_auc.mean():.3f}")
    print(f"  Mean pos prediction: {df_p51.pos_mean.mean():.3f}, mean neg: {df_p51.neg_mean.mean():.3f}")

    print("\n" + "="*70)
    print("Q3: Site-shortcut residual test on 27-class predictions")
    print("="*70)
    # Per-site: mean exp51 prediction across all rows (no labels needed) on 11 eval files
    sites = sorted(sc_all[ev_mask].site.unique())
    print(f"  Held-out sites: {sites}")
    print(f"  Per-site mean exp51 prediction (per 27-target class):")
    print(f"  {'site':<6}", *[f"{t[-3:]:<8}" for t in target_species[:8]])
    for s in sites:
        s_mask = (sc_all.site == s) & ev_mask
        if s_mask.sum() == 0: continue
        means = P51_full[s_mask][:, target_cols_idx].mean(axis=0)
        print(f"  {s:<6}", *[f"{means[i]:.3f}".ljust(8) for i in range(min(8, n51))])
    # Variance across sites = high site-dependence (BAD)
    print(f"\n  Across-site variance per class (mean prediction stdev across 6 sites):")
    perclass_per_site = np.zeros((len(sites), n51))
    for si, s in enumerate(sites):
        s_mask = (sc_all.site == s) & ev_mask
        if s_mask.sum() > 0:
            perclass_per_site[si] = P51_full[s_mask][:, target_cols_idx].mean(axis=0)
    perclass_std = perclass_per_site.std(axis=0)
    perclass_max = perclass_per_site.max(axis=0)
    perclass_min = perclass_per_site.min(axis=0)
    for i in np.argsort(-perclass_std)[:10]:
        print(f"    {target_species[i]:<14} stdev_across_sites={perclass_std[i]:.3f}  "
              f"min={perclass_min[i]:.3f}  max={perclass_max[i]:.3f}")

    print("\n" + "="*70)
    print("Q4: Leave-one-file-out (LOFO) variance estimate of v28 vs v26")
    print("="*70)
    eval_files = sorted(sc_all[ev_mask].filename.unique())
    print(f"  11 eval files: per-file macro Δ (v28 @ w=0.10 vs v26):")
    file_deltas = []
    for fn in eval_files:
        f_mask = (sc_all.filename == fn).values
        Y_f = Y_all[f_mask]
        v26_f = v26_prob[f_mask]
        v28_f = v28_blend(0.10)[f_mask]
        a26 = per_class_auc(Y_f, v26_f); a28 = per_class_auc(Y_f, v28_f)
        c = sorted(set(a26) & set(a28))
        if c:
            d = np.mean([a28[k] - a26[k] for k in c])
            file_deltas.append(d)
            site = parse_site(fn)
            print(f"    {fn:<45} site={site} n_cls={len(c):2d}  Δ {d:+.4f}")
    print(f"\n  LOFO mean Δ = {np.mean(file_deltas):+.4f}  std = {np.std(file_deltas):.4f}  n={len(file_deltas)}")

    print("\n" + "="*70)
    print("Q5: exp51 vs Perch correlation (on 27 target columns)")
    print("="*70)
    # Pearson per row, per col
    perch_27 = perch_prob[ev_mask][:, target_cols_idx]
    p51_27 = P51_full[ev_mask][:, target_cols_idx]
    corrs = []
    for i in range(n51):
        try:
            r, _ = pearsonr(perch_27[:, i], p51_27[:, i])
            if np.isfinite(r): corrs.append((target_species[i], r))
        except Exception: pass
    df_corr = pd.DataFrame(corrs, columns=["class", "pearson"])
    print(f"  Per-class Pearson(Perch, exp51) on 27 cols (held-out):")
    print(df_corr.sort_values("pearson", ascending=False).to_string(index=False))
    flat_perch = perch_27.flatten(); flat_p51 = p51_27.flatten()
    g_p, _ = pearsonr(flat_perch, flat_p51)
    g_s, _ = spearmanr(flat_perch, flat_p51)
    print(f"\n  Global flatten Pearson = {g_p:.3f}, Spearman = {g_s:.3f}")
    print(f"  → Lower correlation = more independent = better blend candidate")

    print("\n" + "="*70)
    print("Q6: KS distribution shift v26 → v28 (on ALL 234 columns)")
    print("="*70)
    # Compare scale of perturbation. v22 had massive shift; v28 should be small/local.
    v28_full = v28_blend(0.10)
    ks_per_col = []
    for c in range(234):
        try:
            ks_stat, _ = ks_2samp(v26_prob[:, c], v28_full[:, c])
            ks_per_col.append((c, ks_stat))
        except Exception: pass
    df_ks = pd.DataFrame(ks_per_col, columns=["col", "ks"])
    df_ks["class"] = [primary[c] for c in df_ks.col]
    df_ks["taxon"] = [species_taxon[c] for c in df_ks.col]
    df_ks["in_27"] = df_ks.col.isin(target_cols)
    print(f"  Cols with KS > 0.1: {(df_ks.ks > 0.1).sum()} / 234")
    print(f"  Cols with KS > 0.3: {(df_ks.ks > 0.3).sum()}")
    print(f"  Mean KS on 27-target cols: {df_ks[df_ks.in_27].ks.mean():.3f}")
    print(f"  Mean KS on non-27 cols:    {df_ks[~df_ks.in_27].ks.mean():.6f}")
    print(f"  Compare: v22 had 30 cols with KS > 0.3 (broad disturbance).")
    print(f"  v28 expected: ~25 cols (only 27-target) with KS shift, rest near zero.")

    print("\n" + "="*70)
    print("Q7: False-positive analysis: rows where 27 species ABSENT")
    print("="*70)
    # Rows where NO 27-target species is positive
    no_27_mask = (Y_all[:, target_cols_idx].sum(axis=1) == 0)
    print(f"  Rows with NO 27-target positive: {no_27_mask.sum()} / {len(sc_all)}")
    # On these rows, what's the mean prediction across 27 cols?
    p51_no27 = P51_full[no_27_mask][:, target_cols_idx]
    p51_with27 = P51_full[~no_27_mask][:, target_cols_idx]
    print(f"  Mean exp51 prediction on those rows (27 cols): {p51_no27.mean():.4f}")
    print(f"  Mean exp51 prediction on rows WITH ≥1 target: {p51_with27.mean():.4f}")
    print(f"  Ratio: with/without = {p51_with27.mean()/max(p51_no27.mean(),1e-6):.1f}x")
    # Per-class FP analysis
    print(f"\n  Per-class FP rate (exp51 > 0.5 on negative rows, in 27-target):")
    for i, t in enumerate(target_species):
        c = l2i.get(t)
        if c is None: continue
        neg_mask = Y_all[:, c] == 0
        if neg_mask.sum() == 0: continue
        fp_rate = (P51_full[neg_mask, c] > 0.5).mean()
        print(f"    {t:<14}  FP_rate@0.5 = {fp_rate:.3f}  ({neg_mask.sum()} neg rows)")

    # Save aggregates
    out_data = {
        "lofo_mean": float(np.mean(file_deltas)),
        "lofo_std": float(np.std(file_deltas)),
        "lofo_files": list(eval_files),
        "lofo_per_file": list(zip(eval_files, file_deltas)),
        "n_classes_with_v28_drop": int((df_q1.delta < -0.005).sum()),
        "n_classes_with_v28_gain": int((df_q1.delta > 0.005).sum()),
        "p51_mean_auc_on_27_eval": float(df_p51.exp51_auc.mean()),
        "perch_p51_correlation_global_pearson": float(g_p),
        "ks_mean_on_27_target": float(df_ks[df_ks.in_27].ks.mean()),
        "ks_mean_outside": float(df_ks[~df_ks.in_27].ks.mean()),
    }
    with open(OUT / "58_summary.json", "w") as f:
        json.dump(out_data, f, indent=2, default=float)
    print(f"\nSaved → {OUT}/58_summary.json")
    df_q1.to_csv(OUT / "58_q1_per_class.csv", index=False)
    df_p51.to_csv(OUT / "58_q2_p51_eval.csv", index=False)


if __name__ == "__main__":
    main()
