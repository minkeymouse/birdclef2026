#!/usr/bin/env python3
"""exp137 — Pseudo refinement via FP/FN detectors (exp99 era).

Pipeline:
  1. Run exp59 (ConvNeXt-tiny SED) on unlabeled SS to get scores
  2. For each pseudo-positive (row, class) in v3 where class is in 15-candidate
     Aves list: build feature vector, apply FP detector → drop if P(FP) > τ_fp
  3. For each (row, class) where v33 < 0.3 BUT class is in candidate list:
     apply FN detector → ADD as pseudo-positive if P(FN) > τ_fn

Detector candidate classes (15 Aves):
  bafcur1, bufpar, chacha1, chvcon1, compau, hyamac1, litnig1, magant1,
  nacnig1, orwpar, purjay1, redjun, trsowl, undtin1, whtdov

Features (9-dim):
  perch_on_c, exp50_on_c, exp59_on_c, perch_sed_disagree, perch_low_sed_high,
  file_mean_e50_c, file_std_e50_c, file_uniform_e50_c, v33_on_c

Output: pseudo_soundscapes_labels_v5.csv = v3 - high_FP + high_FN_recovered
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torchaudio
import timm
import soundfile as sf

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
EXP59_CKPT = ROOT / "model-weights/exp59_convnext_sed.pt"
LR_FP = ROOT / "model-weights/lr_fp_detector.npz"
LR_FN = ROOT / "model-weights/lr_fn_detector.npz"
LR_META = ROOT / "model-weights/lr_correction_meta.npz"
PERCH_UNLAB = ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz"
EXP50_UNLAB = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
V126_SCORES = ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz"
V3_CSV = DATA / "pseudo_soundscapes_labels_v3.csv"
OUT_CSV = DATA / "pseudo_soundscapes_labels_v5.csv"
OUT_DIAG = ROOT / "experiments/_data_pipelines/exp137_outputs"
OUT_DIAG.mkdir(parents=True, exist_ok=True)
EXP59_SCORES_NPZ = OUT_DIAG / "exp59_unlabeled_scores.npz"

DEVICE = "cuda"
SR = 32000
N_WINDOWS = 12
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = SR * 60
N_FFT = 2048; HOP = 512; N_MELS = 128; F_MIN, F_MAX = 50, 14000
SED_CHUNK_SEC = 20
SED_CHUNK_SAMPLES = SR * SED_CHUNK_SEC
N_CLS = 234

TAU_FP = 0.7    # drop pseudo-positive if P(FP) > 0.7
TAU_FN = 0.7    # add pseudo-positive if P(FN) > 0.7 (and v33 < 0.3)


class _MelExt(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=F_MIN, f_max=F_MAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class _SEDHead(nn.Module):
    def __init__(self, fd, nc):
        super().__init__()
        self.att = nn.Conv1d(fd, nc, 1); self.cla = nn.Conv1d(fd, nc, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values


class _SED(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.mel = _MelExt(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(bb, pretrained=False, in_chans=1, num_classes=0, global_pool='')
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = _SEDHead(f.shape[1], N_CLS)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
        clip, _ = self.head(f); return clip


def run_exp59_on_unlabeled():
    if EXP59_SCORES_NPZ.exists():
        print(f"  Loading cached exp59 scores from {EXP59_SCORES_NPZ}")
        return np.load(EXP59_SCORES_NPZ)["scores"]

    print("Running exp59 (ConvNeXt-tiny) on unlabeled SS...")
    files = sorted((DATA / "train_soundscapes").glob("*.ogg"))
    print(f"  {len(files)} files")

    st = torch.load(str(EXP59_CKPT), map_location=DEVICE, weights_only=False)
    bb = st.get("config", {}).get("backbone", "convnext_tiny.fb_in22k_ft_in1k")
    m = _SED(bb).to(DEVICE).eval()
    m.load_state_dict(st["state_dict"])

    n_files = len(files)
    out = np.zeros((n_files * N_WINDOWS, N_CLS), dtype=np.float32)
    BATCH_F = 8
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, n_files, BATCH_F):
            batch = files[s:s+BATCH_F]
            chunks = []
            for fn in batch:
                wav, _ = sf.read(str(fn), dtype="float32", always_2d=False)
                if wav.ndim == 2: wav = wav.mean(axis=1)
                if len(wav) < FILE_SAMPLES: wav = np.pad(wav, (0, FILE_SAMPLES - len(wav)))
                wav = wav[:FILE_SAMPLES]
                for ci in range(3):
                    st_idx = ci * SED_CHUNK_SAMPLES
                    chunks.append(wav[st_idx:st_idx + SED_CHUNK_SAMPLES])

            x = torch.from_numpy(np.stack(chunks).astype(np.float32)).to(DEVICE)
            logits = m(x)
            probs = torch.sigmoid(logits).cpu().numpy()

            for bi, fn in enumerate(batch):
                for ci in range(3):
                    chunk_prob = probs[bi*3 + ci]
                    for win_in_chunk in range(4):
                        end_sec = ci * 20 + (win_in_chunk + 1) * 5
                        file_idx = s + bi
                        win_idx = (end_sec // 5) - 1
                        global_idx = file_idx * N_WINDOWS + win_idx
                        out[global_idx] = chunk_prob

            if s % 80 == 0:
                elapsed = time.time() - t0
                print(f"  {s + BATCH_F}/{n_files} files, {elapsed:.0f}s elapsed", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")
    np.savez_compressed(EXP59_SCORES_NPZ, scores=out)
    print(f"  Saved → {EXP59_SCORES_NPZ}")
    return out


def apply_lr_detector(detector, X):
    """Apply LR detector with stored scaling. Returns sigmoid(z) probabilities."""
    mu = detector["scaler_mean"]
    sd = detector["scaler_scale"]
    X_s = (X - mu) / sd
    z = X_s @ detector["coef"] + detector["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def main():
    print("=== exp137 — FP/FN detector pseudo refinement ===\n", flush=True)

    # Load detectors
    lr_fp = np.load(LR_FP)
    lr_fn = np.load(LR_FN)
    lr_meta = np.load(LR_META, allow_pickle=True)
    candidate_classes = lr_meta["candidate_classes"]
    feature_names = lr_meta["feature_names"]
    print(f"  Detectors: FP coef {lr_fp['coef'].shape}, FN coef {lr_fn['coef'].shape}")
    print(f"  Candidate classes: {len(candidate_classes)}")
    print(f"  Features: {feature_names}")

    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    candidate_labels = [primary[c] for c in candidate_classes]
    print(f"  Candidate labels: {candidate_labels}\n")

    # Step 1: Run exp59 inference on unlabeled
    exp59_scores = run_exp59_on_unlabeled()
    print(f"  exp59 scores: {exp59_scores.shape}\n")

    # Load other signals
    print("Loading other signals...")
    perch_unlab = np.load(PERCH_UNLAB, mmap_mode="r")
    perch_logits = np.array(perch_unlab["scores"])
    perch_probs = (1.0 / (1.0 + np.exp(-np.clip(perch_logits, -30, 30)))).astype(np.float32)

    exp50_unlab = np.load(EXP50_UNLAB, allow_pickle=True)
    exp50 = exp50_unlab["scores"]
    filenames_unlab = exp50_unlab["filenames"].astype(str)
    row_ids_unlab = exp50_unlab["row_ids"].astype(str)

    v33_data = np.load(V126_SCORES, allow_pickle=True)
    v33 = v33_data["v33"].astype(np.float32)

    n_rows = len(perch_probs)
    n_files = n_rows // N_WINDOWS

    # File-level stats of exp50
    print("Computing file-level exp50 stats...")
    e50_view = exp50.reshape(n_files, N_WINDOWS, N_CLS)
    file_mean = np.broadcast_to(e50_view.mean(axis=1, keepdims=True), e50_view.shape).reshape(-1, N_CLS)
    file_std = np.broadcast_to(e50_view.std(axis=1, keepdims=True), e50_view.shape).reshape(-1, N_CLS)
    file_uniform = 1.0 - (file_std / (file_mean + 1e-6))

    # Build feature matrix for ALL (row, candidate_class) pairs
    # Feature order: perch_on_c, exp50_on_c, exp59_on_c, perch_sed_disagree,
    #                perch_low_sed_high, file_mean_e50_c, file_std_e50_c,
    #                file_uniform_e50_c, v33_on_c
    print("Computing FP/FN probabilities for candidate classes...")
    fp_probs = np.zeros((n_rows, len(candidate_classes)), dtype=np.float32)
    fn_probs = np.zeros((n_rows, len(candidate_classes)), dtype=np.float32)

    for i, c in enumerate(candidate_classes):
        cidx = int(c)
        X = np.stack([
            perch_probs[:, cidx],
            exp50[:, cidx],
            exp59_scores[:, cidx],
            np.abs(perch_probs[:, cidx] - exp50[:, cidx]),
            np.maximum(0, exp50[:, cidx] - perch_probs[:, cidx]),
            file_mean[:, cidx],
            file_std[:, cidx],
            file_uniform[:, cidx],
            v33[:, cidx],
        ], axis=1)
        fp_probs[:, i] = apply_lr_detector(lr_fp, X)
        fn_probs[:, i] = apply_lr_detector(lr_fn, X)

    print(f"  fp_probs distribution: mean {fp_probs.mean():.3f}, p90 {np.percentile(fp_probs, 90):.3f}, p99 {np.percentile(fp_probs, 99):.3f}")
    print(f"  fn_probs distribution: mean {fn_probs.mean():.3f}, p90 {np.percentile(fn_probs, 90):.3f}, p99 {np.percentile(fn_probs, 99):.3f}")

    # Apply filter to v3 pseudo
    print(f"\n=== Applying FP filter (drop pseudo-positive if P(FP) > {TAU_FP}) ===")
    df_v3 = pd.read_csv(V3_CSV)
    print(f"  v3 entries: {len(df_v3)}")

    # Build (filename, end_sec) → row_idx lookup
    import re
    rid_to_idx = {}
    for i, rid in enumerate(row_ids_unlab):
        m = re.search(r"_(\d+)$", str(rid))
        if m:
            rid_to_idx[(filenames_unlab[i], int(m.group(1)))] = i

    candidate_label_to_i = {primary[c]: i for i, c in enumerate(candidate_classes)}
    df_v3["end"] = df_v3["end"].astype(int)

    keep_mask = np.ones(len(df_v3), dtype=bool)
    n_dropped = 0
    n_in_candidate = 0
    for i, row in df_v3.iterrows():
        if row.primary_label not in candidate_label_to_i:
            continue
        n_in_candidate += 1
        c_i = candidate_label_to_i[row.primary_label]
        key = (row.filename, int(row.end))
        if key not in rid_to_idx: continue
        r_idx = rid_to_idx[key]
        if fp_probs[r_idx, c_i] > TAU_FP:
            keep_mask[i] = False
            n_dropped += 1

    print(f"  Pseudo-positives in candidate list: {n_in_candidate}")
    print(f"  Dropped (high P(FP)): {n_dropped} ({100*n_dropped/max(n_in_candidate,1):.1f}% of candidates)")
    df_filtered = df_v3[keep_mask].copy()
    print(f"  After FP filter: {len(df_filtered)} (was {len(df_v3)})")

    # FN recovery: for (row, candidate_class) where v33 < 0.3 BUT P(FN) > τ → ADD
    print(f"\n=== Applying FN recovery (add if v33<0.3 AND P(FN) > {TAU_FN}) ===")
    new_pos = []
    for r_idx in range(n_rows):
        for c_i, c_global in enumerate(candidate_classes):
            cidx = int(c_global)
            if v33[r_idx, cidx] >= 0.3: continue  # already considered or pseudo positive
            if fn_probs[r_idx, c_i] > TAU_FN:
                # Look up filename + end_sec
                m = re.search(r"_(\d+)$", row_ids_unlab[r_idx])
                if not m: continue
                end_sec = int(m.group(1))
                new_pos.append({
                    "filename": filenames_unlab[r_idx],
                    "start": str(end_sec - 5),
                    "end": str(end_sec),
                    "primary_label": primary[cidx],
                    "v33_score": float(v33[r_idx, cidx]),
                    "perch_score": float(perch_probs[r_idx, cidx]),
                    "exp50_score": float(exp50[r_idx, cidx]),
                    "source": "fn_recovery",
                })

    print(f"  Recovered FN positives: {len(new_pos)}")
    if new_pos:
        df_new = pd.DataFrame(new_pos)
        df_v5 = pd.concat([df_filtered, df_new], ignore_index=True)
        df_v5 = df_v5.drop_duplicates(subset=["filename", "start", "end", "primary_label"], keep="first")
    else:
        df_v5 = df_filtered

    print(f"\n=== Final v5 ===")
    print(f"  Total: {len(df_v5)} (v3 was {len(df_v3)})")
    print(f"  Diff: {len(df_v5) - len(df_v3):+d}")

    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    df_v5["taxon"] = df_v5.primary_label.map(sp2tax).fillna("?")
    print(f"\n  Per-taxon:")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        n_t = (df_v5.taxon == t).sum()
        n_classes = df_v5[df_v5.taxon == t].primary_label.nunique()
        print(f"    {t}: {n_t} entries, {n_classes} classes")

    df_v5.to_csv(OUT_CSV, index=False)
    print(f"\n  Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
