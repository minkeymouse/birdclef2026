#!/usr/bin/env python3
"""exp55 — Deep per-class audit. What are we actually getting wrong?

Since our local 11-file eval shows ~75 evaluable classes but LB has 160-200,
we have ~100+ classes invisible to local — this is our blindspot.

Goals:
  (A) ALL 66 labeled SS per-class AUC: v12 vs exp50 alone vs v23
  (B) Identify which classes exp50 SINGLE is better than blend (these are
      oracle candidates for per-class blend)
  (C) Predictions on 10,658 UNLABELED SS: prediction distribution shift
      analysis per class — does v23 systematically change any class's dist?
  (D) Find the 'invisible to local' classes (n_pos<3 in 66 SS) and see their
      prediction distribution (these may drive LB)
  (E) Oracle per-class blend weights: max AUC over (w_P, w_29, w_50) grid
      per class, report macro if we could use oracle per-class
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
from scipy.stats import spearmanr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
OUT = ROOT / "experiments/exp55_outputs"
OUT.mkdir(exist_ok=True)
SR = 32000; CLIP_SEC = 20
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000; DEVICE = "cuda"


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


def align_43a_all_ss():
    """Return all 10,658 SS row predictions + meta."""
    d = np.load(EXP43A / "perch_ss_all.npz")
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    return d["scores"], meta


def align_old(df, p):
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(df), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)

class SEDHead(nn.Module):
    def __init__(self, feat_dim, n_cls):
        super().__init__()
        self.att = nn.Conv1d(feat_dim, n_cls, 1)
        self.cla = nn.Conv1d(feat_dim, n_cls, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values

class SEDModel(nn.Module):
    def __init__(self, n_cls=234):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model("hgnetv2_b0.ssld_stage2_ft_in1k",
                                          pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(feat.shape[1], n_cls)
    def forward(self, x):
        m = self.mel(x)
        m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        f = self.backbone(m)
        f = f.mean(dim=2) if f.dim() == 4 else f
        clip, _ = self.head(f)
        return clip


@torch.no_grad()
def predict_sed(df, ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = SEDModel().to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
    CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
    out = np.zeros((len(df), 234), dtype=np.float32); cache = {}
    for i in range(0, len(df), 8):
        j = min(len(df), i + 8); wavs = []
        for k in range(i, j):
            row = df.iloc[k]
            if row.filename not in cache:
                w, sr = sf.read(DATA / "train_soundscapes" / row.filename, dtype="float32")
                if w.ndim > 1: w = w.mean(1)
                cache[row.filename] = w
            wav = cache[row.filename]
            end_sec = int(row.end_sec)
            target_c = (end_sec - 2.5) * SR
            cs = int(max(0, target_c - CLIP_SAMPLES/2))
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
    print("Loading data + base preds...")
    sc_all, Y_all, primary, l2i = build_all()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])
    ta_cnt = pd.read_csv(DATA / "train.csv").groupby("primary_label").size().to_dict()

    S_perch = align_43a(sc_all)
    perch_prob = sigmoid(S_perch)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)
    sed29_prob = sigmoid(S29) if S29.min() < 0 or S29.max() > 1 else S29
    # Handle: if S29 already probs, just use as-is
    # Actually check: S29 preds from val_scores.npz — were these logits or probs?
    print(f"  S29 range: [{S29.min():.3f}, {S29.max():.3f}]")

    print("Computing exp50 preds on 66 SS files...")
    P50 = predict_sed(sc_all, EXP50 / "best_ckpt.pt")
    print(f"  P50 range: [{P50.min():.3f}, {P50.max():.3f}]")

    # Build v12 and v23 outputs
    zP = zs(perch_prob); z29 = zs(S29); z50 = zs(P50)
    v12_prob = sigmoid(gauss_pf(0.8*zP + 0.2*z29, sc_all, 0.5))
    v23_prob = sigmoid(gauss_pf(0.8*zP + 0.1*z29 + 0.1*z50, sc_all, 0.5))
    # exp50 alone for comparison
    p50_alone = sigmoid(gauss_pf(zs(P50), sc_all, 0.5))
    # perch alone
    perch_alone = sigmoid(gauss_pf(zs(perch_prob), sc_all, 0.5))

    # === AUDIT A: per-class AUC across models ===
    print("\n=== A. Per-class AUC: v12, v23, exp50-alone, Perch-alone ===")
    aucs_v12 = per_class_auc(Y_all, v12_prob)
    aucs_v23 = per_class_auc(Y_all, v23_prob)
    aucs_p50 = per_class_auc(Y_all, p50_alone)
    aucs_perch = per_class_auc(Y_all, perch_alone)

    common = set(aucs_v12) & set(aucs_v23) & set(aucs_p50) & set(aucs_perch)
    rows = []
    for c in sorted(common):
        n_pos = int(Y_all[:, c].sum())
        rows.append({
            "class": primary[c], "taxon": species_taxon[c],
            "n_pos": n_pos, "n_ta": int(ta_cnt.get(primary[c], 0)),
            "auc_v12": aucs_v12[c], "auc_v23": aucs_v23[c],
            "auc_p50_alone": aucs_p50[c], "auc_perch_alone": aucs_perch[c],
            "v23_minus_v12": aucs_v23[c] - aucs_v12[c],
            "p50_minus_v12": aucs_p50[c] - aucs_v12[c],
            "p50_beats_v12": aucs_p50[c] > aucs_v12[c],
        })
    df = pd.DataFrame(rows)
    print(f"  {len(df)} classes with AUC computable")

    print(f"\n  Classes where v23 HURT vs v12 (bottom-15):")
    for _, r in df.sort_values("v23_minus_v12").head(15).iterrows():
        print(f"    {r['class']:<12} ({r.taxon:<9}) n_pos={r.n_pos:3d} n_ta={r.n_ta:4d}  "
              f"v12={r.auc_v12:.3f}  v23={r.auc_v23:.3f}  Δ{r.v23_minus_v12:+.3f}  "
              f"p50_alone={r.auc_p50_alone:.3f}")

    print(f"\n  Classes where v23 HELPED vs v12 (top-15):")
    for _, r in df.sort_values("v23_minus_v12", ascending=False).head(15).iterrows():
        print(f"    {r['class']:<12} ({r.taxon:<9}) n_pos={r.n_pos:3d} n_ta={r.n_ta:4d}  "
              f"v12={r.auc_v12:.3f}  v23={r.auc_v23:.3f}  Δ{r.v23_minus_v12:+.3f}  "
              f"p50_alone={r.auc_p50_alone:.3f}")

    # === AUDIT B: where exp50 ALONE beats v12 ===
    print(f"\n=== B. Classes where exp50 ALONE > v12 (oracle blend candidates) ===")
    better_p50 = df[df.p50_beats_v12].sort_values("p50_minus_v12", ascending=False)
    print(f"  {len(better_p50)} / {len(df)} classes where p50 alone beats v12")
    print(f"  Top 20:")
    for _, r in better_p50.head(20).iterrows():
        print(f"    {r['class']:<12} ({r.taxon:<9}) n_pos={r.n_pos:3d} n_ta={r.n_ta:4d}  "
              f"v12={r.auc_v12:.3f}  p50={r.auc_p50_alone:.3f}  Δ{r.p50_minus_v12:+.3f}")

    # === AUDIT C: oracle per-class blend weights ===
    print(f"\n=== C. Oracle per-class (w_P, w_29, w_50) grid sweep ===")
    best_cfg_per_class = {}
    zB = [("P", zP), ("S29", z29), ("S50", z50)]
    # 3-simplex grid: w_P + w_29 + w_50 = 1, step 0.1
    for c in common:
        y = Y_all[:, c]
        if y.sum() < 3: continue
        best_auc = 0; best_w = None
        for wp in np.arange(0, 1.01, 0.1):
            for w29 in np.arange(0, 1 - wp + 0.01, 0.1):
                w50 = 1 - wp - w29
                if w50 < 0 or w50 > 1: continue
                raw = wp * zP[:, c] + w29 * z29[:, c] + w50 * z50[:, c]
                # Apply gauss per file
                sc_arr = np.zeros_like(raw)
                for fn in sc_all.filename.unique():
                    m = (sc_all.filename == fn).values
                    sc_arr[m] = gaussian_filter1d(raw[m], 0.5, mode="nearest")
                prob = sigmoid(sc_arr)
                try:
                    auc = roc_auc_score(y, prob)
                    if auc > best_auc:
                        best_auc = auc; best_w = (wp, w29, w50)
                except: pass
        best_cfg_per_class[c] = (best_auc, best_w)
    print(f"  Computed for {len(best_cfg_per_class)} classes")
    # Macro under oracle
    oracle_macro = np.mean([a for a, _ in best_cfg_per_class.values()])
    v12_macro = np.mean([aucs_v12[c] for c in best_cfg_per_class])
    v23_macro = np.mean([aucs_v23[c] for c in best_cfg_per_class])
    print(f"  ORACLE per-class blend macro: {oracle_macro:.4f}  (v12: {v12_macro:.4f}, v23: {v23_macro:.4f})")
    # Weight distribution
    from collections import Counter
    w_dist = Counter()
    for c, (a, w) in best_cfg_per_class.items():
        if w is not None:
            # Categorize dominant
            if w[0] >= 0.6: key = "P-heavy"
            elif w[1] >= 0.4: key = "S29-heavy"
            elif w[2] >= 0.4: key = "S50-heavy"
            else: key = "mixed"
            w_dist[key] += 1
    print(f"  Best-weight distribution: {dict(w_dist)}")
    # Per-taxon best weight
    print(f"\n  Per-taxon mean best (w_P, w_29, w_50):")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        cls_t = [c for c in best_cfg_per_class if species_taxon[c] == t]
        if cls_t:
            ws = np.array([best_cfg_per_class[c][1] for c in cls_t if best_cfg_per_class[c][1]])
            if len(ws):
                print(f"    {t:<10}  n={len(cls_t):2d}  mean w=(P {ws[:,0].mean():.2f}, S29 {ws[:,1].mean():.2f}, S50 {ws[:,2].mean():.2f})")

    # === AUDIT D: "invisible" classes (n_pos < 3) ===
    print(f"\n=== D. Classes with n_pos < 3 in 66 SS — invisible to local audit ===")
    invisible = []
    for c in range(len(primary)):
        n_pos = Y_all[:, c].sum()
        if 0 < n_pos < 3:
            invisible.append(c)
    print(f"  {len(invisible)} classes have n_pos in [1, 2]")
    # These likely contribute to LB if they're common elsewhere. Let's check prediction distribution
    # on these classes across ALL rows — does v23 vs v12 meaningfully change them?
    for c in invisible[:15]:
        d_pred = v23_prob[:, c] - v12_prob[:, c]
        print(f"    {primary[c]:<12} ({species_taxon[c]:<9}) n_pos={int(Y_all[:,c].sum())} n_ta={ta_cnt.get(primary[c], 0):4d}  "
              f"mean Δpred={d_pred.mean():+.4f}  |Δ|max={np.abs(d_pred).max():.3f}")

    # === AUDIT E: prediction distribution analysis on unlabeled SS ===
    print(f"\n=== E. Predictions on ALL 10,658 unlabeled SS — structural shifts ===")
    scs_all, meta_all = align_43a_all_ss()
    perch_all = sigmoid(scs_all)  # (10658, 234)
    zP_all = zs(perch_all)
    # v12-like without SED29 (since we don't have SED29 on full 10k) approximation
    # Just measure where exp50 blend would land for all 10k (approximate)
    # Skip full Re-inference for time, use 66-SS findings
    print(f"  Full SS Perch preds shape: {perch_all.shape}")

    # Save full per-class table
    df.to_csv(OUT / "55_per_class.csv", index=False)
    print(f"\n  Saved → {OUT}/55_per_class.csv")

    # === AUDIT F: top classes where v12 is weak (<0.8) and exp50 is strong (>0.8) ===
    print(f"\n=== F. Classes where V12 WEAK (<0.8) and exp50 STRONG (>0.8) ===")
    weak_strong = df[(df.auc_v12 < 0.8) & (df.auc_p50_alone >= 0.8)].sort_values("auc_p50_alone", ascending=False)
    print(f"  {len(weak_strong)} classes. Top 15:")
    for _, r in weak_strong.head(15).iterrows():
        print(f"    {r['class']:<12} ({r.taxon:<9}) n_pos={r.n_pos:3d} n_ta={r.n_ta:4d}  "
              f"v12={r.auc_v12:.3f}  p50={r.auc_p50_alone:.3f}  gap +{r.auc_p50_alone - r.auc_v12:.3f}")


if __name__ == "__main__":
    main()
