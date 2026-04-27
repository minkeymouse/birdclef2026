#!/usr/bin/env python3
"""exp79c — Probe Insecta-detection on UNLABELED pool.

Sample N=300 unlabeled SS files (~3,600 rows), extract mel via torchaudio
on GPU, encode through exp78 iVAE, and ask: do the top Insecta cos-sim
candidates come from (a) the 4 train-Insecta sites (S08/S15/S19/S23) — site
shortcut, or (b) other sites — real species signal?

If (a): iVAE Insecta-detection is a site-fingerprint detector → can't expand
training data via this path.
If (b): real acoustic signal → user's pseudo-label idea is viable.
"""
from __future__ import annotations
import re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/_data_pipelines/exp43a_outputs"
MW = ROOT / "model-weights"
OUT = ROOT / "experiments/_audits_post_v26/exp79_outputs"
OUT.mkdir(exist_ok=True, parents=True)
DEVICE = "cuda"
SR = 32000; N_WIN = 12
T_POOL = 16; N_MELS = 128
N_FFT = 2048; HOP = 512; FMIN = 50; FMAX = 14000
SEED = 42

FN_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")


class IVAEEnc(nn.Module):
    def __init__(self, in_dim, z_dim, n_aux, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256, 2 * z_dim))
        self.aux_mlp = nn.Sequential(
            nn.Linear(n_aux, 64), nn.GELU(),
            nn.Linear(64, 2 * z_dim))
    def encode(self, x):
        h = self.enc(x); mu, _ = h.chunk(2, dim=-1); return mu


def main():
    print("=== exp79c: Unlabeled Insecta probe ===\n")

    # Load Perch meta to know which files are unlabeled
    perch_meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    labeled_files = set(pd.read_csv(DATA / "train_soundscapes_labels.csv").filename.unique())
    unlab_files = [f for f in perch_meta.filename.unique() if f not in labeled_files]
    print(f"unlabeled files: {len(unlab_files)}")
    rng = np.random.RandomState(SEED)
    sample_files = rng.choice(unlab_files, size=300, replace=False).tolist()

    # Add 100 each from S22 (dominant) + sample from rare sites for diversity
    by_site = {}
    for f in unlab_files:
        m = FN_RE.match(f); s = m.group(2) if m else "?"
        by_site.setdefault(s, []).append(f)
    print("site coverage:", {k: len(v) for k, v in sorted(by_site.items(), key=lambda x: -len(x[1]))})

    # Force balance: ≥30 from each site that has data
    balanced = []
    for s, fs in by_site.items():
        n_pick = min(30, len(fs))
        balanced.extend(rng.choice(fs, size=n_pick, replace=False).tolist())
    sample_files = list(set(balanced))[:300]
    print(f"sampled {len(sample_files)} files for probe")

    # Load iVAE artifacts
    ck = torch.load(MW / "ivae_encoder.pt", map_location=DEVICE, weights_only=False)
    stats = np.load(MW / "ivae_mel_stats.npz")
    cent = np.load(MW / "ivae_z_centroids.npz")
    train_mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(DEVICE)
    train_std = torch.from_numpy(stats["std"].astype(np.float32)).to(DEVICE)
    z_centroids = cent["centroids"].astype(np.float32)
    cent_valid = cent["valid"].astype(bool)

    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    l2t = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([l2t.get(p, "?") for p in primary])

    enc = IVAEEnc(int(ck["in_dim"]), int(ck["z_dim"]), int(ck["n_aux"])).to(DEVICE).eval()
    enc.load_state_dict(ck["encoder_state_dict"], strict=False)

    # Mel extractor on GPU
    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        f_min=FMIN, f_max=FMAX, power=2.0, center=True
    ).to(DEVICE)
    adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(DEVICE)

    # Process files
    SR_5S = SR * 5
    rows = []
    t0 = time.time()
    soundscape_dir = DATA / "train_soundscapes"

    print("Processing files...")
    for fi, fname in enumerate(sample_files):
        try:
            y, sr = sf.read(soundscape_dir / fname, dtype="float32", always_2d=False)
        except Exception as e:
            continue
        if sr != SR: continue
        if y.ndim == 2: y = y.mean(axis=1)
        if len(y) < SR * 60:
            y = np.pad(y, (0, SR * 60 - len(y)))
        y = y[:SR * 60]
        y_t = torch.from_numpy(y).to(DEVICE)

        # 12 windows of 5s
        wins = torch.stack([y_t[i*SR_5S:(i+1)*SR_5S] for i in range(N_WIN)])  # (12, 160000)
        with torch.no_grad():
            m = mel_t(wins)        # (12, n_mels, T)
            m = adb(m)             # dB
            # Pool to T_POOL
            T = m.shape[-1]
            bins = torch.linspace(0, T, T_POOL + 1, device=DEVICE).long()
            pooled = torch.stack([m[:, :, bins[k]:bins[k+1]].mean(dim=-1) for k in range(T_POOL)], dim=1)
            # pooled: (12, T_POOL, n_mels)
            x = pooled.reshape(N_WIN, -1)
            x = (x - train_mean) / train_std
            z = enc.encode(x).cpu().numpy()  # (12, 32)

        m_re = FN_RE.match(fname); site = m_re.group(2); hour = int(m_re.group(4)[:2])
        for wi in range(N_WIN):
            rows.append({
                "filename": fname,
                "site": site, "hour": hour,
                "start": wi * 5, "end": (wi + 1) * 5,
                "z": z[wi],
            })
        if (fi + 1) % 50 == 0:
            print(f"  {fi+1}/{len(sample_files)} files  elapsed {time.time()-t0:.1f}s")

    print(f"Done extracting. {len(rows)} windows in {time.time()-t0:.1f}s")

    # Build z matrix
    Z = np.stack([r["z"] for r in rows])
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "z"} for r in rows])
    print(f"Z shape: {Z.shape}")

    # cos sim to all 234 centroids
    z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    c_norm = z_centroids / (np.linalg.norm(z_centroids, axis=1, keepdims=True) + 1e-8)
    cos = z_norm @ c_norm.T   # (n_rows, 234)
    cos[:, ~cent_valid] = -np.inf

    # Top-1 species per row
    df["ivae_top1"] = [primary[i] for i in cos.argmax(axis=1)]
    df["ivae_top1_taxon"] = [species_taxon[i] for i in cos.argmax(axis=1)]
    df["ivae_top1_cos"] = cos.max(axis=1)

    # Per-row max Insecta-centroid cos sim
    insecta_idx = np.where(cent_valid & (species_taxon == "Insecta"))[0]
    print(f"Valid Insecta centroids: {len(insecta_idx)} ({[primary[i] for i in insecta_idx]})")
    df["insecta_max_cos"] = cos[:, insecta_idx].max(axis=1) if len(insecta_idx) else -np.inf
    df["insecta_max_lbl"] = [primary[insecta_idx[j]] for j in cos[:, insecta_idx].argmax(axis=1)]

    # === Probe Q1: top-100 Insecta-cos candidates — what site? ===
    print("\n=== Q1: Top-100 Insecta-cos candidates by site ===")
    top100 = df.nlargest(100, "insecta_max_cos")
    site_dist = top100.site.value_counts().to_dict()
    print(f"  site distribution: {site_dist}")
    print(f"  predicted Insecta species:")
    print(top100.insecta_max_lbl.value_counts().head(10).to_dict())

    print("\n=== Q1b: Top-100 Insecta-cos candidates — which sites are 'Insecta sites'? ===")
    INS_TRAIN_SITES = {"S08", "S15", "S19", "S23"}
    in_ins = top100.site.isin(INS_TRAIN_SITES).sum()
    print(f"  {in_ins}/100 from train-Insecta sites (S08/S15/S19/S23)")
    print(f"  {100-in_ins}/100 from other sites")

    # === Probe Q2: per-site mean Insecta-cos ===
    print("\n=== Q2: per-site mean Insecta-cos (high = strong site shortcut) ===")
    by_site = df.groupby("site")["insecta_max_cos"].agg(["mean", "max", "count"]).round(3)
    by_site = by_site.sort_values("mean", ascending=False)
    print(by_site.to_string())

    # === Probe Q3: per-site COUNT of top-1=Insecta rows ===
    print("\n=== Q3: per-site count of rows where iVAE top1 IS an Insecta sonotype ===")
    df_ins_top1 = df[df["ivae_top1_taxon"] == "Insecta"]
    print(f"  total: {len(df_ins_top1)}/{len(df)} rows ({100*len(df_ins_top1)/len(df):.1f}%)")
    by_site2 = df_ins_top1.site.value_counts()
    print(by_site2.to_string())

    df.to_parquet(OUT / "unlabeled_probe.parquet")
    top100.to_csv(OUT / "top100_insecta_candidates.csv", index=False)
    print(f"\nSaved → {OUT}/unlabeled_probe.parquet")
    print(f"Saved → {OUT}/top100_insecta_candidates.csv  (audit these)")


if __name__ == "__main__":
    main()
