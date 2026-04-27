#!/usr/bin/env python3
"""exp57 — Local grid evaluation of next-submission candidates.

After v26 (0.7P + 0.3 exp50) hit LB 0.931, evaluate further lever variants
locally on the 66 labeled SS files, prioritizing:
  - High Spearman vs v26 (low risk of LB regression)
  - High macro on held-out 11-file (rare-class signal preserved)
  - Per-taxon Aves not negative

Candidates:
  C1: v26 baseline (reference)
  C2: v27 class-conditional Aves(0.7/0.3) non-Aves(0.3/0.7)
  C3: v27' class-conditional Aves(0.8/0.2) non-Aves(0.5/0.5)
  C4: v27'' class-conditional Aves(0.7/0.3) non-Aves(0.5/0.5)
  C5: peak triangulation w_50 = 0.25
  C6: peak triangulation w_50 = 0.35
  C7: peak triangulation w_50 = 0.40
  C8: 27-head additive: v26 + 0.15*exp51 on 27 sonotype/Amphibia columns
  C9: 27-head additive heavier: v26 + 0.25*exp51 on 27 columns
  C10: v26 + Gauss σ=0.4 (less smoothing)
  C11: v26 + Gauss σ=0.7 (more smoothing)
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
from scipy.stats import spearmanr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP50 = ROOT / "experiments/exp50_outputs"
EXP51 = ROOT / "experiments/exp51_outputs"
OUT = ROOT / "experiments/exp57_outputs"
OUT.mkdir(exist_ok=True)
SR = 32000; CLIP_SEC = 20; CLIP_SAMPLES = SR * CLIP_SEC; FILE_SAMPLES = SR * 60
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000; DEVICE = "cuda"
SEED = 42


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
def predict_sed(df, ckpt_path, n_cls=234):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if "state_dict" in ck:
        # Determine n_cls from ckpt
        sd = ck["state_dict"]
        for k, v in sd.items():
            if "head.cla.weight" in k or "cla.weight" in k:
                n_cls_actual = v.shape[0]
                break
        else:
            n_cls_actual = n_cls
    else:
        n_cls_actual = n_cls
    model = _SED(n_cls=n_cls_actual).to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
    out = np.zeros((len(df), n_cls_actual), dtype=np.float32); cache = {}
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
    return out, ck.get("target_species", None)


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


def evaluate(P, P_ref, Y, sc_all, species_taxon, label):
    aucs = per_class_auc(Y, P)
    aucs_ref = per_class_auc(Y, P_ref)
    common = set(aucs) & set(aucs_ref)
    macro = np.mean([aucs[c] for c in common])
    macro_ref = np.mean([aucs_ref[c] for c in common])
    # Held-out only (split=eval)
    ev_mask = (sc_all.split == "eval").values
    Y_ev = Y[ev_mask]; P_ev = P[ev_mask]; P_ref_ev = P_ref[ev_mask]
    aucs_ev = per_class_auc(Y_ev, P_ev)
    aucs_ref_ev = per_class_auc(Y_ev, P_ref_ev)
    common_ev = set(aucs_ev) & set(aucs_ref_ev)
    macro_ev = np.mean([aucs_ev[c] for c in common_ev])
    macro_ref_ev = np.mean([aucs_ref_ev[c] for c in common_ev])
    # Spearman per row
    sp_row = []
    for i in range(P.shape[0]):
        r, _ = spearmanr(P_ref[i], P[i])
        if np.isfinite(r): sp_row.append(r)
    # Per-taxon held-out delta
    tax_d = {}
    for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        cls = [c for c in common_ev if species_taxon[c] == t]
        if cls:
            tax_d[t] = np.mean([aucs_ev[c] - aucs_ref_ev[c] for c in cls])
    return {
        "label": label,
        "macro_66": macro, "delta_66": macro - macro_ref,
        "macro_eval11": macro_ev, "delta_eval11": macro_ev - macro_ref_ev,
        "spearman_row_mean": float(np.mean(sp_row)),
        **{f"tx_{t}": tax_d.get(t, np.nan) for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]},
    }


def main():
    print("Loading...")
    sc_all, Y_all, primary, l2i = build_all()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])
    n_aves = (species_taxon == "Aves").sum()
    n_non = (species_taxon != "Aves").sum()
    print(f"Aves={n_aves}  non-Aves={n_non}")

    S_p = align_43a(sc_all); perch_prob = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)
    P50, _ = predict_sed(sc_all, EXP50 / "best_ckpt.pt")
    print(f"Perch {perch_prob.shape}, S29 {S29.shape}, P50 {P50.shape}")

    # Try exp51 — different ckpt structure (no .head wrapper)
    P51 = None; target_species = None
    try:
        ck51 = torch.load(EXP51 / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
        target_species = ck51.get("target_species", None)
        n51 = ck51["state_dict"]["cla.weight"].shape[0]
        # Build flat-headed model
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
                C = feat.shape[1]
                self.att = nn.Conv1d(C, n_cls, 1)
                self.cla = nn.Conv1d(C, n_cls, 1)
            def forward(self, x):
                m = self.adb(self.mel(x)).unsqueeze(1)
                m = m.transpose(1,2); m = self.bn0(m); m = m.transpose(1,2)
                f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
                a = self.att(f); c = self.cla(f)
                w = torch.softmax(a, dim=-1)
                return (w * c).sum(-1)
        model51 = _SEDFlat(n_cls=n51).to(DEVICE)
        model51.load_state_dict(ck51["state_dict"]); model51.eval()
        P51 = np.zeros((len(sc_all), n51), dtype=np.float32); cache = {}
        with torch.no_grad():
            for i in range(0, len(sc_all), 8):
                j = min(len(sc_all), i + 8); wavs = []
                for k in range(i, j):
                    row = sc_all.iloc[k]
                    if row.filename not in cache:
                        w_, sr = sf.read(DATA / "train_soundscapes" / row.filename, dtype="float32")
                        if w_.ndim > 1: w_ = w_.mean(1)
                        cache[row.filename] = w_
                    wav = cache[row.filename]
                    cs = int(max(0, (int(row.end_sec) - 2.5) * SR - CLIP_SAMPLES/2))
                    cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
                    clip = wav[cs:cs + CLIP_SAMPLES]
                    if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
                    wavs.append(clip.astype(np.float32))
                x = torch.from_numpy(np.stack(wavs)).to(DEVICE)
                P51[i:j] = torch.sigmoid(model51(x)).cpu().numpy()
        print(f"P51 {P51.shape}, targets={target_species[:3] if target_species else 'unknown'}")
        # Remap to 234 columns
        if target_species:
            P51_full = np.zeros((len(sc_all), 234), dtype=np.float32)
            for i, t in enumerate(target_species):
                if t in l2i: P51_full[:, l2i[t]] = P51[:, i]
            P51 = P51_full
    except Exception as e:
        print(f"exp51 load failed: {e}")
        P51 = None

    zP = zs(perch_prob); z29 = zs(S29); z50 = zs(P50)

    # Reference v26
    v26_raw = 0.7 * zP + 0.3 * z50
    v26_prob = sigmoid(gauss_pf(v26_raw, sc_all, 0.5))
    print(f"v26 ref macro_66 (computed inline):")
    aucs_v26 = per_class_auc(Y_all, v26_prob)
    print(f"  {len(aucs_v26)} classes, macro {np.mean(list(aucs_v26.values())):.4f}")

    results = []
    # --- Reference ---
    results.append(evaluate(v26_prob, v26_prob, Y_all, sc_all, species_taxon, "C1: v26 (REF)"))

    # --- C2: v27 class-conditional 0.7/0.3 vs 0.3/0.7 ---
    def class_cond_blend(w_aves_P, w_aves_50, w_non_P, w_non_50):
        w_P = np.full(234, w_aves_P, dtype=np.float32)
        w_50 = np.full(234, w_aves_50, dtype=np.float32)
        non_mask = species_taxon != "Aves"
        w_P[non_mask] = w_non_P
        w_50[non_mask] = w_non_50
        raw = w_P[None,:] * zP + w_50[None,:] * z50
        return sigmoid(gauss_pf(raw, sc_all, 0.5))

    print("\n--- Class-conditional ---")
    for label, (waP, wa5, wnP, wn5) in [
        ("C2: v27 cc Aves(0.7/0.3) non(0.3/0.7)", (0.7, 0.3, 0.3, 0.7)),
        ("C3: v27' cc Aves(0.8/0.2) non(0.5/0.5)", (0.8, 0.2, 0.5, 0.5)),
        ("C4: v27'' cc Aves(0.7/0.3) non(0.5/0.5)", (0.7, 0.3, 0.5, 0.5)),
        ("C4b: cc Aves(0.7/0.3) non(0.4/0.6)", (0.7, 0.3, 0.4, 0.6)),
        ("C4c: cc Aves(0.75/0.25) non(0.4/0.6)", (0.75, 0.25, 0.4, 0.6)),
    ]:
        p = class_cond_blend(waP, wa5, wnP, wn5)
        r = evaluate(p, v26_prob, Y_all, sc_all, species_taxon, label)
        results.append(r)
        print(f"  {label:<45}  m66 {r['macro_66']:.4f} Δ{r['delta_66']:+.4f}  "
              f"m11 {r['macro_eval11']:.4f} Δ{r['delta_eval11']:+.4f}  sp {r['spearman_row_mean']:.3f}  "
              f"Aves {r['tx_Aves']:+.3f}")

    # --- Peak triangulation ---
    print("\n--- Peak triangulation w_50 ---")
    for w50 in [0.22, 0.25, 0.28, 0.32, 0.35, 0.40]:
        wp = 1 - w50
        raw = wp * zP + w50 * z50
        p = sigmoid(gauss_pf(raw, sc_all, 0.5))
        r = evaluate(p, v26_prob, Y_all, sc_all, species_taxon, f"C5_{w50}: w_P={wp} w_50={w50}")
        results.append(r)
        print(f"  w50={w50}  m66 {r['macro_66']:.4f} Δ{r['delta_66']:+.4f}  "
              f"m11 {r['macro_eval11']:.4f} Δ{r['delta_eval11']:+.4f}  sp {r['spearman_row_mean']:.3f}  "
              f"Aves {r['tx_Aves']:+.3f}")

    # --- 27-head additive on top of v26 ---
    if P51 is not None:
        print("\n--- v26 + exp51 27-head additive (only on 27 columns) ---")
        # Find which columns are the 27 targets
        target_cols = []
        if target_species is not None:
            target_cols = [l2i[t] for t in target_species if t in l2i]
        else:
            # Fallback: all 47158son* + 47143/47147
            target_cols = [l2i[p] for p in primary if p.startswith("47158son")] + \
                          [l2i.get("47143", -1), l2i.get("47147", -1)]
            target_cols = [c for c in target_cols if c >= 0]
        print(f"  27-head target columns: {len(target_cols)}")
        if len(target_cols) > 0 and P51.sum() > 0:
            z51 = zs(P51)  # already 234-col with zeros outside targets
            for w27 in [0.10, 0.15, 0.20, 0.25, 0.30]:
                # Apply additive ONLY to target columns
                raw = 0.7 * zP + 0.3 * z50  # v26 base
                for c in target_cols:
                    raw[:, c] = (1 - w27) * raw[:, c] + w27 * z51[:, c]
                p = sigmoid(gauss_pf(raw, sc_all, 0.5))
                r = evaluate(p, v26_prob, Y_all, sc_all, species_taxon, f"C8_{w27}: v26 + exp51@{w27}")
                results.append(r)
                print(f"  w27={w27}  m66 {r['macro_66']:.4f} Δ{r['delta_66']:+.4f}  "
                      f"m11 {r['macro_eval11']:.4f} Δ{r['delta_eval11']:+.4f}  sp {r['spearman_row_mean']:.3f}  "
                      f"Aves {r['tx_Aves']:+.3f}")

    # --- Gauss sigma variations ---
    print("\n--- Gauss sigma sweep ---")
    for sigma in [0.3, 0.4, 0.5, 0.7, 1.0]:
        p = sigmoid(gauss_pf(0.7 * zP + 0.3 * z50, sc_all, sigma))
        r = evaluate(p, v26_prob, Y_all, sc_all, species_taxon, f"C10_{sigma}: v26 σ={sigma}")
        results.append(r)
        print(f"  σ={sigma}  m66 {r['macro_66']:.4f} Δ{r['delta_66']:+.4f}  "
              f"m11 {r['macro_eval11']:.4f} Δ{r['delta_eval11']:+.4f}  sp {r['spearman_row_mean']:.3f}  "
              f"Aves {r['tx_Aves']:+.3f}")

    # --- Save and rank ---
    df = pd.DataFrame(results)
    df.to_csv(OUT / "57_grid.csv", index=False)
    print("\n=== RANKED by held-out (m11) Δ ===")
    print(df.sort_values("delta_eval11", ascending=False).head(15)[
        ["label", "macro_66", "macro_eval11", "delta_eval11",
         "spearman_row_mean", "tx_Aves"]
    ].to_string(index=False))

    # LB-safe candidates (Spearman ≥ 0.99 + Δ_eval11 ≥ 0)
    print("\n=== LB-SAFE candidates (Spearman ≥ 0.99 AND held-out macro Δ ≥ 0) ===")
    safe = df[(df.spearman_row_mean >= 0.99) & (df.delta_eval11 >= 0)]
    safe = safe[safe.label != "C1: v26 (REF)"]
    print(safe.sort_values("delta_eval11", ascending=False)[
        ["label", "macro_66", "macro_eval11", "delta_eval11",
         "spearman_row_mean", "tx_Aves"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
