#!/usr/bin/env python3
"""exp78 — Save iVAE artifacts for Kaggle notebook integration.

Outputs:
  - ivae_encoder.pt: iVAE encoder state_dict (small MLP)
  - mel_stats.npz: mean/std for mel input standardization (train-only)
  - z_centroids.npz: per-class z-centroids derived from labeled SS train positives
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP76 = ROOT / "experiments/_audits_post_v26/exp76_outputs"
OUT = ROOT / "model-weights"  # save to Kaggle model-weights dataset
OUT.mkdir(exist_ok=True, parents=True)
DEVICE = "cuda"; SEED = 42
SR = 32000
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000
T_POOL = 16

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_meta(fn):
    m = FNAME_RE.match(fn); return (m.group(2), int(m.group(4)[:2])) if m else (None, -1)


def build_ss_data():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g[["site","hour"]] = sc_g.filename.apply(lambda f: pd.Series(parse_meta(f)))
    rng = np.random.RandomState(SEED); files = sorted(sc_g.filename.unique()); rng.shuffle(files)
    eval_files = set(files[:11])
    sc_g["split"] = ["eval" if f in eval_files else "train" for f in sc_g.filename]
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g.lbls):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_g, Y, primary, l2i


class IVAE(nn.Module):
    def __init__(self, in_dim, z_dim=32, n_aux=10, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256, 2 * z_dim))
        self.aux_mlp = nn.Sequential(
            nn.Linear(n_aux, 64), nn.GELU(),
            nn.Linear(64, 2 * z_dim))
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 256), nn.GELU(),
            nn.Linear(256, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, in_dim))
        self.z_dim = z_dim
    def forward(self, x, aux):
        h = self.enc(x)
        mu_q, lv_q = h.chunk(2, dim=-1)
        h_a = self.aux_mlp(aux)
        mu_p, lv_p = h_a.chunk(2, dim=-1)
        std_q = (0.5 * lv_q).exp()
        z = mu_q + std_q * torch.randn_like(mu_q)
        return self.dec(z), mu_q, lv_q, mu_p, lv_p, z


def kl(mu_q, lv_q, mu_p, lv_p):
    return 0.5 * (lv_p - lv_q - 1 + (lv_q - lv_p).exp() + (mu_q - mu_p).pow(2) * (-lv_p).exp()).sum(-1).mean()


def extract_mel_pooled(wav_5sec):
    """Same mel extraction as exp76. Returns (T_POOL, N_MELS) float32."""
    if len(wav_5sec) < SR * 5:
        wav_5sec = np.pad(wav_5sec, (0, SR * 5 - len(wav_5sec)))
    mel = librosa.feature.melspectrogram(y=wav_5sec[:SR*5], sr=SR, n_fft=N_FFT, hop_length=HOP,
                                          n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    T = mel_db.shape[1]
    bins = np.linspace(0, T, T_POOL + 1).astype(int)
    pooled = np.zeros((T_POOL, N_MELS), dtype=np.float32)
    for k in range(T_POOL):
        chunk = mel_db[:, bins[k]:bins[k+1]]
        if chunk.size > 0:
            pooled[k] = chunk.mean(axis=1)
    return pooled


def main():
    print("Loading data...")
    sc_all, Y_all, primary, l2i = build_ss_data()
    sites = sorted(sc_all.site.unique())
    site_idx = {s: i for i, s in enumerate(sites)}
    n_sites = len(sites)

    # Use cached mel from exp76
    mel_cache = EXP76 / "mel_cache.npz"
    if not mel_cache.exists():
        print(f"Mel cache missing at {mel_cache}, run exp76 first")
        return
    mels = np.load(mel_cache)["mel"]
    print(f"mel shape: {mels.shape}")

    # Build flat input
    X_flat = mels.reshape(len(sc_all), -1).astype(np.float32)

    # CRITICAL: standardize using TRAIN ONLY stats
    tr_mask = (sc_all.split == "train").values
    train_mean = X_flat[tr_mask].mean(0)
    train_std = X_flat[tr_mask].std(0) + 1e-6
    X_flat = (X_flat - train_mean) / train_std
    print(f"flat dim: {X_flat.shape[1]}")

    # Auxiliary
    aux = np.zeros((len(sc_all), n_sites + 1), dtype=np.float32)
    for i, r in sc_all.iterrows():
        aux[i, site_idx[r.site]] = 1.0
        aux[i, -1] = r.hour / 24.0

    # Train iVAE on train split only
    model = IVAE(in_dim=X_flat.shape[1], z_dim=32, n_aux=n_sites + 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X_tr = torch.from_numpy(X_flat[tr_mask]).to(DEVICE)
    A_tr = torch.from_numpy(aux[tr_mask]).to(DEVICE)
    BETA = 0.05
    print("Training iVAE 200 ep (train-only stats)...")
    for ep in range(200):
        model.train(); opt.zero_grad()
        x_recon, mu_q, lv_q, mu_p, lv_p, z = model(X_tr, A_tr)
        recon = F.mse_loss(x_recon, X_tr) * X_tr.shape[1]
        kl_loss = kl(mu_q, lv_q, mu_p, lv_p)
        loss = recon + BETA * kl_loss
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == 199:
            print(f"  ep {ep:03d}  recon {recon.item():.3f}  kl {kl_loss.item():.3f}")

    # Get z for all
    model.eval()
    with torch.no_grad():
        Xa = torch.from_numpy(X_flat).to(DEVICE)
        Aa = torch.from_numpy(aux).to(DEVICE)
        _, mu_q_all, _, _, _, _ = model(Xa, Aa)
        Z = mu_q_all.cpu().numpy()

    # Compute per-class centroids on TRAIN POSITIVES
    Y_tr = Y_all[tr_mask]; Z_tr = Z[tr_mask]
    z_centroids = np.zeros((234, 32), dtype=np.float32)
    centroid_valid = np.zeros(234, dtype=bool)
    MIN_POS = 3
    for c in range(234):
        if Y_tr[:, c].sum() >= MIN_POS:
            z_centroids[c] = Z_tr[Y_tr[:, c] == 1].mean(axis=0)
            centroid_valid[c] = True
    print(f"Valid centroids: {centroid_valid.sum()}/234")

    # Save artifacts. Use only encoder + aux encoder for inference.
    # For deployment we'll skip the prior aux since aux info isn't reliable for test rows
    # (we can use mean aux as default)
    ivae_state = {k: v.cpu() for k, v in model.state_dict().items()
                  if k.startswith("enc") or k.startswith("aux_mlp")}

    torch.save({
        "encoder_state_dict": ivae_state,
        "in_dim": X_flat.shape[1],
        "z_dim": 32,
        "n_aux": n_sites + 1,
        "T_POOL": T_POOL,
        "N_MELS": N_MELS,
    }, OUT / "ivae_encoder.pt")
    print(f"Saved encoder → {OUT / 'ivae_encoder.pt'}")

    np.savez_compressed(OUT / "ivae_mel_stats.npz",
                         mean=train_mean.astype(np.float32),
                         std=train_std.astype(np.float32),
                         sites=np.array(sites),
                         T_POOL=T_POOL, N_MELS=N_MELS)
    print(f"Saved stats → {OUT / 'ivae_mel_stats.npz'}")

    np.savez_compressed(OUT / "ivae_z_centroids.npz",
                         centroids=z_centroids,
                         valid=centroid_valid,
                         primary_labels=np.array(primary))
    print(f"Saved centroids → {OUT / 'ivae_z_centroids.npz'}")

    print(f"\nArtifact sizes:")
    for f in ["ivae_encoder.pt", "ivae_mel_stats.npz", "ivae_z_centroids.npz"]:
        print(f"  {f}: {(OUT / f).stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
