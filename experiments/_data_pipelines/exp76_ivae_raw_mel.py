#!/usr/bin/env python3
"""exp76 — iVAE on RAW MEL spectrogram (time × freq multivariate series).

Per user direction: Perch features are saturated; iVAE on raw mel preserves
acoustic structure and may surface species-level disentangling beyond what
the supervised Perch backbone captures.

Pipeline:
  1. Extract mel-spec for each labeled SS 5-sec window: (T=313, F=128)
  2. Temporal pool to (T=16, F=128) for tractability → flatten to 2048-d
  3. Train iVAE with site one-hot + hour as auxiliary
  4. Per-class kNN species discriminability vs Perch raw baseline
  5. Compare vs exp75 result (iVAE on Perch emb: +0.034 over raw)
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/_data_pipelines/exp43a_outputs"
OUT = ROOT / "experiments/_audits_post_v26/exp76_outputs"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"; SEED = 42
SR = 32000
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000
WINDOW_SEC = 5
T_POOL = 16  # pool to 16 temporal frames

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_meta(fn):
    m = FNAME_RE.match(fn)
    if not m: return None, -1
    return m.group(2), int(m.group(4)[:2])


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


def extract_mel(sc_all, cache_path):
    """Extract pooled mel-spec for each labeled SS row.
    Returns (N, T_POOL, N_MELS) array."""
    if cache_path.exists():
        print(f"loading cache {cache_path}")
        return np.load(cache_path)["mel"]
    print(f"extracting mel for {len(sc_all)} rows...")
    mels = np.zeros((len(sc_all), T_POOL, N_MELS), dtype=np.float32)
    file_cache = {}
    for i, row in sc_all.iterrows():
        if row.filename not in file_cache:
            try:
                wav, sr = sf.read(DATA / "train_soundscapes" / row.filename, dtype="float32")
                if wav.ndim > 1: wav = wav.mean(1)
                if sr != SR:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
                file_cache[row.filename] = wav
            except Exception as e:
                file_cache[row.filename] = np.zeros(SR * 60, dtype=np.float32)
        wav = file_cache[row.filename]
        end_sec = int(row.end_sec)
        start = max(0, (end_sec - 5) * SR)
        end = end_sec * SR
        clip = wav[start:end]
        if len(clip) < SR * 5:
            clip = np.pad(clip, (0, SR * 5 - len(clip)))
        # mel
        mel = librosa.feature.melspectrogram(y=clip, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                              n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)  # (N_MELS, T)
        # pool temporally to T_POOL
        T = mel_db.shape[1]  # ~313
        # average pool to T_POOL frames
        bins = np.linspace(0, T, T_POOL + 1).astype(int)
        pooled = np.zeros((T_POOL, N_MELS), dtype=np.float32)
        for k in range(T_POOL):
            chunk = mel_db[:, bins[k]:bins[k+1]]
            if chunk.size > 0:
                pooled[k] = chunk.mean(axis=1)
        mels[i] = pooled
        if i % 100 == 0: print(f"  {i}/{len(sc_all)}")
    np.savez_compressed(cache_path, mel=mels)
    return mels


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


def main():
    print("Loading SS labels...")
    sc_all, Y_all, primary, l2i = build_ss_data()
    sites = sorted(sc_all.site.unique())
    site_idx = {s: i for i, s in enumerate(sites)}
    n_sites = len(sites)

    # Extract mel
    mels = extract_mel(sc_all, OUT / "mel_cache.npz")
    print(f"mel shape: {mels.shape}")

    # Flatten to (N, T_POOL * N_MELS)
    X_flat = mels.reshape(len(sc_all), -1).astype(np.float32)
    X_flat = (X_flat - X_flat.mean(0)) / (X_flat.std(0) + 1e-6)  # standardize
    print(f"flat input dim: {X_flat.shape[1]}")

    # Aux: site one-hot + hour
    aux = np.zeros((len(sc_all), n_sites + 1), dtype=np.float32)
    for i, r in sc_all.iterrows():
        aux[i, site_idx[r.site]] = 1.0
        aux[i, -1] = r.hour / 24.0

    # Compare to Perch embeddings for baseline
    d = np.load(EXP43A / "perch_ss_all.npz")
    embs = d["emb"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    E = np.zeros((len(sc_all), embs.shape[1]), np.float32)
    for i, rid in enumerate(sc_all.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: E[i] = embs[j]

    tr_mask = (sc_all.split == "train").values
    ev_mask = (sc_all.split == "eval").values

    # Train iVAE
    model = IVAE(in_dim=X_flat.shape[1], z_dim=32, n_aux=n_sites + 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X_tr = torch.from_numpy(X_flat[tr_mask]).to(DEVICE)
    A_tr = torch.from_numpy(aux[tr_mask]).to(DEVICE)
    BETA = 0.05
    print("Training iVAE on raw-mel-pooled (200 ep)...")
    for ep in range(200):
        model.train()
        opt.zero_grad()
        x_recon, mu_q, lv_q, mu_p, lv_p, z = model(X_tr, A_tr)
        recon = F.mse_loss(x_recon, X_tr) * X_tr.shape[1]
        kl_loss = kl(mu_q, lv_q, mu_p, lv_p)
        loss = recon + BETA * kl_loss
        loss.backward(); opt.step()
        if ep % 20 == 0 or ep == 199:
            print(f"  ep {ep:03d}  recon {recon.item():.3f}  kl {kl_loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        Xa = torch.from_numpy(X_flat).to(DEVICE)
        Aa = torch.from_numpy(aux).to(DEVICE)
        _, mu_q_all, _, _, _, _ = model(Xa, Aa)
        Z = mu_q_all.cpu().numpy()
    print(f"Z shape: {Z.shape}")

    # Per-class kNN audit
    print("\n=== Q1: species discriminability comparison (held-out kNN) ===")
    Y_tr = Y_all[tr_mask]; Y_ev = Y_all[ev_mask]
    Z_tr = Z[tr_mask]; Z_ev = Z[ev_mask]
    E_tr = E[tr_mask]; E_ev = E[ev_mask]

    z_aucs = []; perch_aucs = []
    big_diffs = []
    for c in range(Y_tr.shape[1]):
        n_pos_tr = int(Y_tr[:, c].sum()); n_pos_ev = int(Y_ev[:, c].sum())
        if n_pos_tr < 3 or n_pos_ev == 0 or n_pos_ev == len(Y_ev): continue
        z_centroid = Z_tr[Y_tr[:, c] == 1].mean(axis=0, keepdims=True)
        E_centroid = E_tr[Y_tr[:, c] == 1].mean(axis=0, keepdims=True)
        sim_z = cosine_similarity(Z_ev, z_centroid).flatten()
        sim_E = cosine_similarity(E_ev, E_centroid).flatten()
        try:
            auc_z = roc_auc_score(Y_ev[:, c], sim_z)
            auc_perch = roc_auc_score(Y_ev[:, c], sim_E)
            z_aucs.append(auc_z); perch_aucs.append(auc_perch)
            if abs(auc_z - auc_perch) > 0.05:
                big_diffs.append((primary[c], n_pos_tr, n_pos_ev, auc_z, auc_perch))
        except: pass

    print(f"\n  Mean kNN AUC ({len(z_aucs)} eval classes):")
    print(f"    Raw-mel iVAE z (32-d):  {np.mean(z_aucs):.4f}")
    print(f"    Perch raw (1536-d):     {np.mean(perch_aucs):.4f}")
    print(f"    Δ:                      {np.mean(z_aucs) - np.mean(perch_aucs):+.4f}")
    print(f"    (exp75 reference: Perch-emb iVAE z = 0.886, raw = 0.852, Δ +0.034)")

    print("\n  Classes with |Δ| > 0.05:")
    for cls, ntp, nev, az, ap in sorted(big_diffs, key=lambda x: -(x[3] - x[4])):
        print(f"    {cls:<14}  n_pos_tr={ntp:3d} n_pos_ev={nev:3d}  z={az:.3f}  perch={ap:.3f}  Δ={az-ap:+.3f}")

    np.savez_compressed(OUT / "z_raw_mel.npz", Z=Z, sites=np.array(sites))
    print(f"\nSaved → {OUT}/z_raw_mel.npz")


if __name__ == "__main__":
    main()
