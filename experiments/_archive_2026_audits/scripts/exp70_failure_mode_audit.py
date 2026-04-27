#!/usr/bin/env python3
"""exp70 — Concrete row-level failure mode audit on v26.

For each underperforming class on the 11-file held-out, examine:
  - WHICH rows are positive (n_pos, which files, which sites)
  - On those positive rows, Perch's top-K predictions (what is Perch firing on?)
  - On those positive rows, exp50's top-K predictions
  - On NEGATIVE rows where v26 falsely fires high (FP), the same
  - Specific masking rule: where would mask help?

Failure types we're targeting:
  Type 1: Perch overconfidence on Aves species when non-Aves is present
  Type 2: Perch reversed/missed on classes where exp50 succeeds (litnig1 etc)
  Type 3: 47158son cluster confusion
  Type 4: Mammalia/Reptilia → Aves confusion
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch, torch.nn as nn
import timm, torchaudio
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import re

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
OUT = ROOT / "experiments/exp70_outputs"
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
    ck50 = torch.load(EXP50 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    m50 = _SED(n_cls=234).to(DEVICE); m50.load_state_dict(ck50["state_dict"])
    P50 = predict_sed(m50, sc_all, 234); del m50
    torch.cuda.empty_cache()

    zP = zs(perch_prob); z50 = zs(P50)
    v26_raw = 0.7*zP + 0.3*z50
    v26 = sigmoid(gauss_pf(v26_raw, sc_all, 0.5))

    ev_mask = (sc_all.split == "eval").values
    Y_ev = Y_all[ev_mask]; sc_ev = sc_all[ev_mask].reset_index(drop=True)
    perch_ev = perch_prob[ev_mask]; p50_ev = P50[ev_mask]; v26_ev = v26[ev_mask]
    aucs_v26_ev = per_class_auc(Y_ev, v26_ev)

    # Identify worst-performing classes on held-out
    worst_classes = sorted([(c, a) for c, a in aucs_v26_ev.items() if a < 0.7])
    print(f"\n=== {len(worst_classes)} classes with v26 AUC < 0.7 on held-out ===")

    # ─── For each, deep dive ───
    aves_idx = np.array([c for c in range(234) if species_taxon[c] == "Aves"])

    findings = []
    for c, auc in worst_classes:
        cls_name = primary[c]
        cls_taxon = species_taxon[c]
        pos_rows = np.where(Y_ev[:, c] == 1)[0]
        neg_rows = np.where(Y_ev[:, c] == 0)[0]
        n_pos = len(pos_rows)
        if n_pos == 0: continue

        # Perch on positive rows: what's it firing on?
        perch_on_pos = perch_ev[pos_rows]
        p50_on_pos = p50_ev[pos_rows]
        # Mean Perch / exp50 prediction on this class
        perch_self = perch_on_pos[:, c]
        p50_self = p50_on_pos[:, c]
        # Top Perch predictions on positive rows (all classes except c)
        perch_top_idx = np.argsort(-perch_on_pos.mean(axis=0))[:5]
        p50_top_idx = np.argsort(-p50_on_pos.mean(axis=0))[:5]
        perch_top_aves = [i for i in np.argsort(-perch_on_pos.mean(axis=0)) if species_taxon[i] == "Aves"][:5]

        # FP analysis: rows where v26 strongly fires (top 10%) but no positive
        v26_self_neg = v26_ev[neg_rows, c]
        if len(v26_self_neg) > 0:
            top10pct = np.argsort(-v26_self_neg)[:max(1, len(v26_self_neg) // 10)]
            top_fp_rows = neg_rows[top10pct]
            v26_fp_mean = v26_self_neg[top10pct].mean()
            # On FP rows what was actually positive?
            actual_pos_on_fp = []
            for r in top_fp_rows[:5]:
                actual = [primary[i] for i in np.where(Y_ev[r] == 1)[0]]
                actual_pos_on_fp.append((r, actual))
        else:
            v26_fp_mean = 0
            actual_pos_on_fp = []

        # Failure type classification
        ft = "unknown"
        if perch_self.mean() < 0.05 and p50_self.mean() > 0.3:
            ft = "Type1: Perch dead, exp50 alive"
        elif perch_self.mean() < 0.5 and p50_self.mean() < 0.3:
            ft = "Type2: Both weak"
        elif p50_self.mean() > 0.5 and perch_self.mean() < p50_self.mean() / 2:
            ft = "Type3: Perch under-confident vs exp50"

        findings.append({
            "class": cls_name, "taxon": cls_taxon, "n_pos": n_pos, "v26_auc": auc,
            "perch_self_mean": float(perch_self.mean()), "perch_self_max": float(perch_self.max()),
            "p50_self_mean": float(p50_self.mean()), "p50_self_max": float(p50_self.max()),
            "perch_top1_other": primary[perch_top_idx[0]] if perch_top_idx[0] != c else primary[perch_top_idx[1]],
            "p50_top1": primary[p50_top_idx[0]],
            "v26_fp_topmean": float(v26_fp_mean),
            "ft": ft,
        })

    print(f"\n  {'class':<14} {'taxon':<10} {'n':>3} {'auc':>5} {'P_self':>7} {'P50_self':>9} {'P_top1':<10} {'P50_top1':<12} {'fp_top':>7} {'type':<35}")
    for f in findings:
        print(f"  {f['class']:<14} {f['taxon']:<10} {f['n_pos']:>3} {f['v26_auc']:>5.2f} "
              f"{f['perch_self_mean']:>7.3f} {f['p50_self_mean']:>9.3f}  "
              f"{f['perch_top1_other']:<10} {f['p50_top1']:<12} {f['v26_fp_topmean']:>7.3f}  {f['ft']:<35}")

    # ─── Specific masking experiment ───
    print("\n=== Masking experiment: suppress Perch where exp50 disagrees ===")

    # Rule MASK1: For each class C and row r, if Perch[r,C] > 0.5 BUT exp50[r,C] < 0.1
    #            → suppress Perch on this (r, C). Use exp50 for those cells.
    perch_high_exp50_low = (perch_prob > 0.5) & (P50 < 0.1)
    print(f"  Perch > 0.5 AND exp50 < 0.1 cells: {perch_high_exp50_low.sum()} of {234*len(sc_all):,}")

    perch_masked = perch_prob.copy()
    perch_masked[perch_high_exp50_low] = P50[perch_high_exp50_low]  # replace with exp50
    zP_m = zs(perch_masked)
    v26_mask1 = sigmoid(gauss_pf(0.7*zP_m + 0.3*z50, sc_all, 0.5))
    a_mask1 = per_class_auc(Y_ev, v26_mask1[ev_mask])
    common = set(aucs_v26_ev) & set(a_mask1)
    macro_v26 = np.mean([aucs_v26_ev[c] for c in common])
    macro_mask1 = np.mean([a_mask1[c] for c in common])
    print(f"  MASK1 (Perch_high AND exp50_low → use exp50): macro {macro_mask1:.4f}  Δ {macro_mask1-macro_v26:+.4f}")

    # Rule MASK2: Mask Aves only — Perch top-1 Aves on row, but exp50 top-1 is non-Aves
    print("\n  MASK2: row-level suppress Perch's top Aves where exp50's top is non-Aves")
    perch_masked2 = perch_prob.copy()
    suppress_count = 0
    for r in range(len(sc_all)):
        # Perch's top-1 Aves
        p_top_aves = -1
        for ci in np.argsort(-perch_prob[r]):
            if species_taxon[ci] == "Aves":
                p_top_aves = ci; break
        # exp50's top-1
        p50_top = np.argmax(P50[r])
        # If exp50 top-1 is non-Aves AND high (>0.5) AND Perch top-Aves is high (>0.4)
        if (species_taxon[p50_top] != "Aves" and P50[r, p50_top] > 0.5
                and p_top_aves >= 0 and perch_prob[r, p_top_aves] > 0.4):
            # Suppress that Aves prediction
            perch_masked2[r, p_top_aves] *= 0.3  # suppress to 30%
            suppress_count += 1
    print(f"    Suppressed {suppress_count} (row, Aves) pairs")
    zP_m2 = zs(perch_masked2)
    v26_mask2 = sigmoid(gauss_pf(0.7*zP_m2 + 0.3*z50, sc_all, 0.5))
    a_mask2 = per_class_auc(Y_ev, v26_mask2[ev_mask])
    macro_mask2 = np.mean([a_mask2[c] for c in common if c in a_mask2])
    print(f"  MASK2 macro: {macro_mask2:.4f}  Δ {macro_mask2-macro_v26:+.4f}")

    # Rule MASK3: zero Perch entirely on Perch-dead classes (rather than just route)
    print("\n  MASK3: zero Perch on classes with p99<0.1 (just compute exp50 dominant)")
    all_d = np.load(EXP43A / "perch_ss_all.npz")
    all_perch_full = sigmoid(all_d["scores"])
    perch_p99 = np.array([np.quantile(all_perch_full[:, c], 0.99) for c in range(234)])
    dead_mask = perch_p99 < 0.1  # 19 classes
    perch_masked3 = perch_prob.copy()
    # Zero out Perch on dead classes (so blend uses exp50 fully on them)
    perch_masked3[:, dead_mask] = 0.0
    zP_m3 = zs(perch_masked3 + 1e-6)  # avoid div by zero in z-score
    v26_mask3 = sigmoid(gauss_pf(0.7*zP_m3 + 0.3*z50, sc_all, 0.5))
    a_mask3 = per_class_auc(Y_ev, v26_mask3[ev_mask])
    macro_mask3 = np.mean([a_mask3[c] for c in common if c in a_mask3])
    print(f"    {dead_mask.sum()} dead classes Perch zeroed")
    print(f"  MASK3 macro: {macro_mask3:.4f}  Δ {macro_mask3-macro_v26:+.4f}")

    # Rule MASK4: percentile-based suppression — for each Perch row, if a class's prediction
    #             is in the top-3 Aves but exp50 strongly disagrees (says <0.1), suppress
    print("\n  MASK4: targeted top-3 Aves suppression where exp50 strongly disagrees (<0.1)")
    perch_masked4 = perch_prob.copy()
    suppress_count4 = 0
    for r in range(len(sc_all)):
        # Top-3 Aves
        aves_preds = [(ci, perch_prob[r, ci]) for ci in aves_idx]
        aves_preds.sort(key=lambda x: -x[1])
        for ci, p in aves_preds[:3]:
            if p > 0.4 and P50[r, ci] < 0.1:
                perch_masked4[r, ci] *= 0.3
                suppress_count4 += 1
    print(f"    Suppressed {suppress_count4} cells")
    zP_m4 = zs(perch_masked4)
    v26_mask4 = sigmoid(gauss_pf(0.7*zP_m4 + 0.3*z50, sc_all, 0.5))
    a_mask4 = per_class_auc(Y_ev, v26_mask4[ev_mask])
    macro_mask4 = np.mean([a_mask4[c] for c in common if c in a_mask4])
    print(f"  MASK4 macro: {macro_mask4:.4f}  Δ {macro_mask4-macro_v26:+.4f}")

    print(f"\n=== Summary ===")
    print(f"  v26 baseline: {macro_v26:.4f}")
    print(f"  MASK1 (Perch>0.5 & exp50<0.1): {macro_mask1:.4f}  Δ{macro_mask1-macro_v26:+.4f}")
    print(f"  MASK2 (top-Aves suppress when exp50 non-Aves): {macro_mask2:.4f}  Δ{macro_mask2-macro_v26:+.4f}")
    print(f"  MASK3 (zero Perch on dead cls): {macro_mask3:.4f}  Δ{macro_mask3-macro_v26:+.4f}")
    print(f"  MASK4 (top-3 Aves where exp50 disagrees): {macro_mask4:.4f}  Δ{macro_mask4-macro_v26:+.4f}")

    pd.DataFrame(findings).to_csv(OUT / "70_failure_modes.csv", index=False)
    print(f"\nSaved → {OUT}/70_failure_modes.csv")


if __name__ == "__main__":
    main()
