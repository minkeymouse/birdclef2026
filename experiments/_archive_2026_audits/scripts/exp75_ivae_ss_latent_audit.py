#!/usr/bin/env python3
"""exp75 — iVAE on SS embeddings: examine latent space for species/site/time
structure. Revisits exp43e with explicit per-class diagnostics + site-conditioned
auxiliary variable.

Inputs:
  - Perch 1536-d embeddings on 10,658 unlabeled SS rows (exp43a cache)
  - 617 labeled SS train rows + 122 held-out eval rows
  - Auxiliary: site (categorical), hour-of-day (continuous)

iVAE setup (auxiliary-conditioned VAE):
  encoder: emb (1536) → z (32-d)
  decoder: z + aux → emb_recon
  loss: MSE(emb_recon, emb) + KL(q(z|emb), p(z|aux))

Then probe species-discriminability in z-space:
  - kNN classifier on labeled rows
  - PCA visualization
  - Mutual information(z, site) vs MI(z, species)
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/_data_pipelines/exp43a_outputs"
OUT = ROOT / "experiments/_audits_post_v26/exp75_outputs"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"
SEED = 42
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_meta(fn):
    m = FNAME_RE.match(fn)
    if not m: return None, -1
    return m.group(2), int(m.group(4)[:2])  # site, hour


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


def align_43a(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    embs = d["emb"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    E = np.zeros((len(df), embs.shape[1]), np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: E[i] = embs[j]
    return E


class IVAE(nn.Module):
    """Auxiliary-conditioned VAE.
    p(z|u) is Gaussian with site/hour-conditioned mean.
    """
    def __init__(self, in_dim=1536, z_dim=32, n_sites=10, hidden=256):
        super().__init__()
        # encoder
        self.enc_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 2 * z_dim))
        # auxiliary embedder: site one-hot + hour scalar → z_dim
        self.aux_mlp = nn.Sequential(
            nn.Linear(n_sites + 1, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * z_dim))
        # decoder
        self.dec_mlp = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, in_dim))
        self.z_dim = z_dim

    def forward(self, x, aux):
        # encoder posterior
        h = self.enc_mlp(x)
        mu_q, logvar_q = h.chunk(2, dim=-1)
        # prior conditional on aux
        h_a = self.aux_mlp(aux)
        mu_p, logvar_p = h_a.chunk(2, dim=-1)
        # reparam
        std_q = (0.5 * logvar_q).exp()
        z = mu_q + std_q * torch.randn_like(mu_q)
        # decode
        x_recon = self.dec_mlp(z)
        return x_recon, mu_q, logvar_q, mu_p, logvar_p, z


def kl_div(mu_q, lv_q, mu_p, lv_p):
    return 0.5 * (lv_p - lv_q - 1 + (lv_q - lv_p).exp() + (mu_q - mu_p).pow(2) * (-lv_p).exp()).sum(-1).mean()


def main():
    print("Loading...")
    sc_all, Y_all, primary, l2i = build_ss_data()
    E = align_43a(sc_all)
    print(f"E shape {E.shape} (labeled SS rows × Perch dim)")

    # Build aux: site one-hot + hour normalized
    sites = sorted(sc_all.site.unique())
    site_idx = {s: i for i, s in enumerate(sites)}
    n_sites = len(sites)
    print(f"sites: {sites}")

    aux = np.zeros((len(sc_all), n_sites + 1), dtype=np.float32)
    for i, r in sc_all.iterrows():
        si = site_idx[r.site]
        aux[i, si] = 1.0
        aux[i, -1] = r.hour / 24.0

    # Use labeled rows for training (audit), eval on held-out
    tr_mask = (sc_all.split == "train").values
    ev_mask = (sc_all.split == "eval").values

    # Train iVAE on the train SS rows (617)
    model = IVAE(in_dim=E.shape[1], z_dim=32, n_sites=n_sites).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    X_tr = torch.from_numpy(E[tr_mask]).to(DEVICE)
    A_tr = torch.from_numpy(aux[tr_mask]).to(DEVICE)
    print(f"Train: {X_tr.shape[0]} rows")

    print("Training iVAE 200 epochs...")
    BETA = 0.05
    for ep in range(200):
        model.train()
        opt.zero_grad()
        x_recon, mu_q, lv_q, mu_p, lv_p, z = model(X_tr, A_tr)
        recon_loss = F.mse_loss(x_recon, X_tr, reduction="mean") * X_tr.shape[1]  # sum over dim ~equiv
        kl = kl_div(mu_q, lv_q, mu_p, lv_p)
        loss = recon_loss + BETA * kl
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == 199:
            print(f"  ep {ep:03d}  recon {recon_loss.item():.3f}  kl {kl.item():.3f}")

    # Extract latent representations
    model.eval()
    with torch.no_grad():
        X_all_t = torch.from_numpy(E).to(DEVICE)
        A_all_t = torch.from_numpy(aux).to(DEVICE)
        _, mu_q_all, _, _, _, _ = model(X_all_t, A_all_t)
        Z = mu_q_all.cpu().numpy()
    print(f"Z shape: {Z.shape}")

    # ─── Audit Q1: species clustering in latent space (kNN classify) ───
    print("\n=== Q1: species discriminability in z-space (kNN) ===")
    # Per-class: positive vs negative cosine similarity in z-space
    from sklearn.metrics.pairwise import cosine_similarity
    Y_tr = Y_all[tr_mask]; Y_ev = Y_all[ev_mask]
    Z_tr = Z[tr_mask]; Z_ev = Z[ev_mask]

    # Compute per-class centroids in z-space (mean of training positives)
    # Then kNN-style score on eval rows
    print(f"  {'class':<14} {'taxon':<10} {'n_pos_tr':>8} {'AUC_z':>6} {'AUC_perch':>10}")
    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    tax_arr = np.array([lbl2tax.get(p, "?") for p in primary])

    z_aucs = []; perch_aucs = []
    for c in range(Y_tr.shape[1]):
        n_pos_tr = int(Y_tr[:, c].sum())
        n_pos_ev = int(Y_ev[:, c].sum())
        if n_pos_tr < 3 or n_pos_ev == 0 or n_pos_ev == len(Y_ev): continue
        # centroid: mean z of train positives
        z_centroid = Z_tr[Y_tr[:, c] == 1].mean(axis=0, keepdims=True)
        # cosine similarity from each eval row to centroid
        sim_z = cosine_similarity(Z_ev, z_centroid).flatten()
        try: auc_z = roc_auc_score(Y_ev[:, c], sim_z)
        except: continue
        # baseline: same with Perch raw
        E_centroid = E[tr_mask][Y_tr[:, c] == 1].mean(axis=0, keepdims=True)
        sim_E = cosine_similarity(E[ev_mask], E_centroid).flatten()
        try: auc_perch = roc_auc_score(Y_ev[:, c], sim_E)
        except: continue
        z_aucs.append(auc_z); perch_aucs.append(auc_perch)
        if abs(auc_z - auc_perch) > 0.05:
            print(f"  {primary[c]:<14} {tax_arr[c]:<10} {n_pos_tr:>8} {auc_z:>6.3f} {auc_perch:>10.3f}")
    print(f"\n  Mean kNN AUC (held-out eval, {len(z_aucs)} classes):")
    print(f"    z-space (iVAE):  {np.mean(z_aucs):.4f}")
    print(f"    Perch raw:       {np.mean(perch_aucs):.4f}")
    print(f"    Δ:               {np.mean(z_aucs) - np.mean(perch_aucs):+.4f}")

    # ─── Q2: site dependence of z (good iVAE: lower than raw) ───
    print("\n=== Q2: site dependence of z (lower = better disentangled) ===")
    from sklearn.linear_model import LogisticRegression
    # Site classification AUC on z vs E
    site_y = np.array([site_idx[s] for s in sc_all.site])
    # multiclass — use macro logloss / accuracy
    from sklearn.model_selection import cross_val_score
    clf_z = LogisticRegression(max_iter=200, C=1.0)
    clf_E = LogisticRegression(max_iter=200, C=1.0)
    acc_z = cross_val_score(clf_z, Z, site_y, cv=3, scoring='accuracy').mean()
    acc_E = cross_val_score(clf_E, E, site_y, cv=3, scoring='accuracy').mean()
    print(f"  Site classification accuracy (3-fold CV):")
    print(f"    z-space: {acc_z:.4f}")
    print(f"    Perch:   {acc_E:.4f}")
    print(f"    (lower z-acc = better site disentangling)")

    # ─── Q3: PCA scatter sample ───
    print("\n=== Q3: PCA-2D structure of z ===")
    pca_z = PCA(n_components=2).fit_transform(Z)
    pca_E = PCA(n_components=2).fit_transform(E)
    # Just report std spreads
    print(f"  PCA-2D std on z:     [{pca_z[:,0].std():.3f}, {pca_z[:,1].std():.3f}]")
    print(f"  PCA-2D std on Perch: [{pca_E[:,0].std():.3f}, {pca_E[:,1].std():.3f}]")

    # Save z + meta for further analysis
    np.savez_compressed(OUT / "ivae_z.npz",
                         Z=Z, aux=aux, sites=np.array(sites), site_y=site_y)
    print(f"\nSaved → {OUT}/ivae_z.npz")


if __name__ == "__main__":
    main()
