#!/usr/bin/env python3
"""exp61 — Concrete failure-mode audit of v26 (LB 0.931) and v28 (LB 0.929).

NOT a guess at "ceiling". Specifically identify:
  Q1. v26's WORST 30 classes by per-class AUC on held-out 11 files
  Q2. v26's per-site weakness (which site has worst v26 macro?)
  Q3. v28 vs v26 row-level differences: where does adding exp51 27-head help/hurt?
  Q4. exp51 false-positive analysis on UNLABELED 10k+ SS files: does
      exp51 fire spuriously on certain sites/times?
  Q5. Per-class exp50 vs Perch: which classes is exp50 strong/weak on relative
      to Perch (informs class-conditional)
  Q6. "Invisible" classes (n_pos < 3 in 66 SS): what does v26 predict for
      them? Are predictions stable across sites?
  Q7. What classes does Perch get wrong but exp50 also gets wrong? (consensus
      failures - these are TRULY hard)
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

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
EXP51 = ROOT / "experiments/exp51_outputs"
OUT = ROOT / "experiments/exp61_outputs"
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
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g.filename.unique()); rng.shuffle(files)
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


class _SEDFlat(nn.Module):
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
def predict_sed(model, df, n_cls):
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


def per_class_auc(Y, P, min_pos=1):
    out = {}
    for c in range(Y.shape[1]):
        y = Y[:, c]
        if y.sum() < min_pos or y.sum() == len(y): continue
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
    ta_cnt = pd.read_csv(DATA / "train.csv").groupby("primary_label").size().to_dict()

    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)

    ck50 = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED50(n_cls=234).to(DEVICE); m50.load_state_dict(ck50["state_dict"])
    P50 = predict_sed(m50, sc_all, 234); del m50

    ck51 = torch.load(EXP51 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    target_species = ck51["target_species"]; n51 = len(target_species)
    m51 = _SEDFlat(n_cls=n51).to(DEVICE); m51.load_state_dict(ck51["state_dict"])
    P51_raw = predict_sed(m51, sc_all, n51); del m51
    target_cols = [l2i[t] for t in target_species if t in l2i]
    P51 = np.zeros((len(sc_all), 234), dtype=np.float32)
    for i, t in enumerate(target_species):
        if t in l2i: P51[:, l2i[t]] = P51_raw[:, i]
    torch.cuda.empty_cache()

    zP = zs(perch_prob); z29 = zs(S29); z50 = zs(P50)
    v26_raw = 0.7*zP + 0.3*z50
    v26 = sigmoid(gauss_pf(v26_raw, sc_all, 0.5))
    v28_raw = v26_raw.copy()
    z51 = zs(P51)
    for c in target_cols:
        v28_raw[:, c] = 0.9 * v28_raw[:, c] + 0.1 * z51[:, c]
    v28 = sigmoid(gauss_pf(v28_raw, sc_all, 0.5))

    ev_mask = (sc_all.split == "eval").values
    Y_ev = Y_all[ev_mask]
    v26_ev = v26[ev_mask]; v28_ev = v28[ev_mask]
    perch_ev = perch_prob[ev_mask]; sed29_ev = S29[ev_mask]; p50_ev = P50[ev_mask]
    sc_ev = sc_all[ev_mask]
    aucs_v26 = per_class_auc(Y_ev, v26_ev)
    aucs_v28 = per_class_auc(Y_ev, v28_ev)

    # ─── Q1: v26 worst classes on held-out ───
    print("\n" + "="*70)
    print("Q1: v26 weakest classes on held-out 11 files")
    print("="*70)
    worst = sorted(aucs_v26.items(), key=lambda x: x[1])[:30]
    print(f"  {'class':<14} {'taxon':<10} {'n_pos':>5} {'n_ta':>5} {'v26':>6} {'v28':>6} {'Δ':>7}")
    for c, auc in worst:
        v28_a = aucs_v28.get(c, float("nan"))
        d = v28_a - auc if np.isfinite(v28_a) else float("nan")
        n_pos = int(Y_ev[:, c].sum())
        print(f"  {primary[c]:<14} {species_taxon[c]:<10} {n_pos:>5} "
              f"{ta_cnt.get(primary[c], 0):>5}  {auc:.3f}  {v28_a:.3f}  {d:+.3f}")

    # ─── Q2: per-site v26 macro ───
    print("\n" + "="*70)
    print("Q2: v26 per-site held-out macro")
    print("="*70)
    sites_ev = sorted(sc_ev.site.unique())
    for s in sites_ev:
        s_mask = (sc_ev.site == s).values
        if s_mask.sum() == 0: continue
        a = per_class_auc(Y_ev[s_mask], v26_ev[s_mask])
        if a:
            print(f"  {s}  n_rows={s_mask.sum():3d}  n_cls={len(a):2d}  v26 macro={np.mean(list(a.values())):.4f}")

    # ─── Q3: row-level v26 vs v28 ───
    print("\n" + "="*70)
    print("Q3: v28 vs v26 per-class held-out diff (sorted by gain/loss)")
    print("="*70)
    common = sorted(set(aucs_v26) & set(aucs_v28))
    rows = []
    for c in common:
        d = aucs_v28[c] - aucs_v26[c]
        rows.append((primary[c], species_taxon[c], int(Y_ev[:, c].sum()), c in target_cols,
                     aucs_v26[c], aucs_v28[c], d))
    rows.sort(key=lambda x: x[6])
    print(f"\n  Worst 8 v28 drops:")
    for r in rows[:8]:
        print(f"    {r[0]:<14} ({r[1]:<9}) n_pos={r[2]:3d} 27t={'Y' if r[3] else ' '}  "
              f"v26={r[4]:.3f}  v28={r[5]:.3f}  Δ{r[6]:+.4f}")
    print(f"\n  Best 8 v28 gains:")
    for r in rows[-8:]:
        print(f"    {r[0]:<14} ({r[1]:<9}) n_pos={r[2]:3d} 27t={'Y' if r[3] else ' '}  "
              f"v26={r[4]:.3f}  v28={r[5]:.3f}  Δ{r[6]:+.4f}")

    # ─── Q4: exp51 site-conditional behavior on FULL Perch SS data ───
    print("\n" + "="*70)
    print("Q4: exp51 false-positive site analysis (10K+ unlabeled SS not loaded — using 11 eval)")
    print("="*70)
    # On rows where target species ABSENT, what does exp51 say per site?
    print(f"  Held-out rows where 27-target species absent: per-site exp51 prediction")
    P51_ev = P51[ev_mask]
    no_target_mask = (Y_ev[:, target_cols].sum(axis=1) == 0)
    for s in sites_ev:
        s_mask = (sc_ev.site == s).values & no_target_mask
        if s_mask.sum() == 0: continue
        # Mean exp51 prediction across 27 cols
        m = P51_ev[s_mask][:, target_cols].mean()
        std = P51_ev[s_mask][:, target_cols].std()
        max_pred = P51_ev[s_mask][:, target_cols].max()
        print(f"  {s}  n={s_mask.sum():3d}  exp51 mean={m:.4f}  std={std:.3f}  max={max_pred:.3f}")

    # ─── Q5: exp50 vs Perch per class ───
    print("\n" + "="*70)
    print("Q5: per-class exp50 (alone) vs Perch (alone) — orthogonality")
    print("="*70)
    p50_alone = sigmoid(gauss_pf(zs(P50), sc_all, 0.5))[ev_mask]
    perch_alone = sigmoid(gauss_pf(zs(perch_prob), sc_all, 0.5))[ev_mask]
    aucs_p50 = per_class_auc(Y_ev, p50_alone)
    aucs_perch = per_class_auc(Y_ev, perch_alone)
    rows = []
    for c in set(aucs_p50) & set(aucs_perch) & set(aucs_v26):
        rows.append({
            "class": primary[c], "taxon": species_taxon[c], "n_pos": int(Y_ev[:, c].sum()),
            "perch_alone": aucs_perch[c], "p50_alone": aucs_p50[c],
            "v26_blend": aucs_v26[c], "p50_better": aucs_p50[c] > aucs_perch[c],
        })
    df = pd.DataFrame(rows)
    print(f"\n  Classes where exp50 alone > Perch alone (oracle blend candidates):")
    df_p50 = df[df.p50_better].sort_values("p50_alone", ascending=False)
    print(f"  {len(df_p50)} of {len(df)} classes")
    for _, r in df_p50.head(15).iterrows():
        print(f"    {r['class']:<14} ({r.taxon:<9}) Perch={r.perch_alone:.3f}  exp50={r.p50_alone:.3f}  "
              f"v26={r.v26_blend:.3f}")

    # ─── Q6: invisible classes ───
    print("\n" + "="*70)
    print("Q6: 'invisible' classes (n_pos < 3 in 66 SS) — what does v26 predict?")
    print("="*70)
    n_pos_full = Y_all.sum(axis=0)
    invisible = [c for c in range(234) if 0 < n_pos_full[c] < 3]
    print(f"  {len(invisible)} classes with 1-2 positives in 66 SS:")
    for c in invisible[:15]:
        v26_pred_mean = v26[:, c].mean()
        v26_pred_std = v26[:, c].std()
        per_site = []
        for s in sorted(sc_all.site.unique()):
            s_mask = (sc_all.site == s).values
            if s_mask.sum() > 0:
                per_site.append(v26[s_mask, c].mean())
        print(f"  {primary[c]:<14} ({species_taxon[c]:<9}) n_pos={int(n_pos_full[c])} n_ta={ta_cnt.get(primary[c], 0):4d}  "
              f"v26 mean={v26_pred_mean:.4f}  cross-site σ={np.std(per_site):.3f}")

    # ─── Q7: consensus failures (Perch + exp50 both fail) ───
    print("\n" + "="*70)
    print("Q7: Consensus failures — both Perch and exp50 ALONE < 0.6 AUC")
    print("="*70)
    consensus_fail = []
    for c in set(aucs_p50) & set(aucs_perch):
        if aucs_p50[c] < 0.6 and aucs_perch[c] < 0.6:
            consensus_fail.append((primary[c], species_taxon[c], int(Y_ev[:, c].sum()),
                                    aucs_perch[c], aucs_p50[c], aucs_v26.get(c, float("nan"))))
    print(f"  {len(consensus_fail)} classes where both Perch and exp50 below 0.6:")
    for r in sorted(consensus_fail, key=lambda x: -x[2]):
        print(f"    {r[0]:<14} ({r[1]:<9}) n_pos={r[2]:3d}  Perch={r[3]:.3f}  exp50={r[4]:.3f}  v26={r[5]:.3f}")

    # ─── Q8: which species is exp50 CONFIDENTLY WRONG on ───
    print("\n" + "="*70)
    print("Q8: Species where exp50 is CONFIDENTLY WRONG (FP mean > 0.3 on negatives, AUC < 0.6)")
    print("="*70)
    for c in set(aucs_p50):
        y = Y_ev[:, c]
        neg = (y == 0)
        pos = (y == 1)
        if neg.sum() == 0 or pos.sum() == 0: continue
        if aucs_p50[c] >= 0.6: continue
        neg_mean = p50_ev[neg, c].mean()
        pos_mean = p50_ev[pos, c].mean()
        if neg_mean > pos_mean:
            print(f"    {primary[c]:<14} ({species_taxon[c]:<9}) n_pos={int(pos.sum()):2d}  "
                  f"exp50_AUC={aucs_p50[c]:.3f}  pos_mean={pos_mean:.3f}  neg_mean={neg_mean:.3f} (REVERSED!)")


if __name__ == "__main__":
    main()
