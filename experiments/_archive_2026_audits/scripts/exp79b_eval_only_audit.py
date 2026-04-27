#!/usr/bin/env python3
"""exp79b — held-out eval-only audit. Re-train iVAE on TRAIN-split-only and
build centroids from TRAIN-positives only, then evaluate disagreement signal
on the 11-file held-out (122 eval rows). This is the honest generalization
test.

Compares 3 conditions:
  - in-dist (pilot): iVAE+centroids fit on full 739 rows. AUC inflated.
  - train-fit: iVAE+centroids fit on 617 train rows. AUC reported on 122 eval.
  - perch baseline: same eval rows, just Perch sigmoid taxon-max.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP76 = ROOT / "experiments/_audits_post_v26/exp76_outputs"
EXP43A = ROOT / "experiments/_data_pipelines/exp43a_outputs"
OUT = ROOT / "experiments/_audits_post_v26/exp79_outputs"
OUT.mkdir(exist_ok=True, parents=True)
SEED = 42; DEVICE = "cuda"
N_CLS = 234
T_POOL = 16; N_MELS = 128

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
        h = self.enc(x); mu_q, lv_q = h.chunk(2, dim=-1)
        h_a = self.aux_mlp(aux); mu_p, lv_p = h_a.chunk(2, dim=-1)
        z = mu_q + (0.5 * lv_q).exp() * torch.randn_like(mu_q)
        return self.dec(z), mu_q, lv_q, mu_p, lv_p, z
    def encode(self, x):
        h = self.enc(x); mu, _ = h.chunk(2, dim=-1); return mu


def kl(mu_q, lv_q, mu_p, lv_p):
    return 0.5 * (lv_p - lv_q - 1 + (lv_q - lv_p).exp() + (mu_q - mu_p).pow(2) * (-lv_p).exp()).sum(-1).mean()


def main():
    print("=== exp79b: eval-only generalization audit ===\n")
    sc_g, Y, primary, l2i = build_ss_data()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    label2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([label2tax.get(p, "?") for p in primary])

    mel = np.load(EXP76 / "mel_cache.npz")["mel"]   # (739, 16, 128)
    X = mel.reshape(len(sc_g), -1).astype(np.float32)

    tr_mask = (sc_g.split == "train").values
    ev_mask = (sc_g.split == "eval").values
    print(f"Train rows: {tr_mask.sum()}, Eval rows: {ev_mask.sum()}")

    # Build aux (site one-hot + hour normalized)
    sites = sorted(sc_g.site.unique())
    s2i = {s: i for i, s in enumerate(sites)}
    n_sites = len(sites)
    aux = np.zeros((len(sc_g), n_sites + 1), dtype=np.float32)
    for i, r in sc_g.iterrows():
        aux[i, s2i[r.site]] = 1.0
        aux[i, -1] = r.hour / 24.0

    # TRAIN-only standardize (this is what exp78 also did)
    train_mean = X[tr_mask].mean(0)
    train_std = X[tr_mask].std(0) + 1e-6
    X = (X - train_mean) / train_std

    # Train iVAE on TRAIN ONLY
    model = IVAE(in_dim=X.shape[1], z_dim=32, n_aux=n_sites + 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X[tr_mask]).to(DEVICE)
    At = torch.from_numpy(aux[tr_mask]).to(DEVICE)
    BETA = 0.05
    print("Training iVAE on train-only (200 ep)...")
    for ep in range(200):
        model.train(); opt.zero_grad()
        x_recon, mu_q, lv_q, mu_p, lv_p, z = model(Xt, At)
        recon = F.mse_loss(x_recon, Xt) * Xt.shape[1]
        kld = kl(mu_q, lv_q, mu_p, lv_p)
        (recon + BETA * kld).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        Xall = torch.from_numpy(X).to(DEVICE)
        Z = model.encode(Xall).cpu().numpy()
    print(f"Z: {Z.shape}")

    # Per-class centroids from TRAIN POSITIVES only
    Y_tr = Y[tr_mask]; Z_tr = Z[tr_mask]
    centroids = np.zeros((N_CLS, 32), dtype=np.float32)
    cv = np.zeros(N_CLS, dtype=bool)
    MIN_POS = 3
    for c in range(N_CLS):
        if Y_tr[:, c].sum() >= MIN_POS:
            centroids[c] = Z_tr[Y_tr[:, c] == 1].mean(0)
            cv[c] = True
    print(f"Valid centroids: {cv.sum()}/{N_CLS}")

    # cos sim
    z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    cos = z_norm @ c_norm.T
    cos[:, ~cv] = -np.inf  # mask invalid

    # Perch
    perch_emb = np.load(EXP43A / "perch_ss_all.npz")
    perch_scores = perch_emb["scores"]
    perch_meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(perch_meta["row_id"].values)}
    P_sc = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    for i, rid in enumerate(sc_g.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: P_sc[i] = perch_scores[j]
    perch_prob = 1.0 / (1.0 + np.exp(-P_sc))

    # === Eval-only metrics ===
    Y_ev = Y[ev_mask]
    cos_ev = cos[ev_mask]
    perch_ev = perch_prob[ev_mask]
    gt_taxa_ev = []
    for i in range(len(sc_g)):
        if not ev_mask[i]: continue
        idx = np.where(Y[i] == 1)[0]
        gt_taxa_ev.append(set(species_taxon[idx]) if len(idx) > 0 else set())
    print(f"Eval rows: {len(gt_taxa_ev)}\n")

    print("=== Per-taxon detection AUC on EVAL HOLD-OUT ===")
    print(f"  {'taxon':<12} {'n_eval_pos':>10} {'Perch_max':>10} {'iVAE_max':>10}  {'Δ':>7}")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        gt_t = np.array([t in s for s in gt_taxa_ev])
        n_pos = gt_t.sum()
        if n_pos < 3 or n_pos > len(gt_t) - 3:
            print(f"  {t:<12} {n_pos:>10}  (degenerate)"); continue
        idx = np.where(species_taxon == t)[0]
        valid_idx = np.where(cv & (species_taxon == t))[0]
        p_score = perch_ev[:, idx].max(axis=1)
        i_score = cos_ev[:, valid_idx].max(axis=1) if len(valid_idx) > 0 else np.full(len(gt_t), -np.inf)
        try:
            p_auc = roc_auc_score(gt_t, p_score)
            i_auc = roc_auc_score(gt_t, i_score) if np.isfinite(i_score).all() else np.nan
            delta = i_auc - p_auc if np.isfinite(i_auc) else np.nan
            print(f"  {t:<12} {n_pos:>10} {p_auc:>10.4f} {i_auc:>10.4f}  {delta:>+7.4f}")
        except Exception as e:
            print(f"  {t:<12} error: {e}")

    # === Disagreement analysis on EVAL ===
    p_top = perch_ev.argmax(axis=1)
    p_top_taxon = species_taxon[p_top]
    i_top = cos_ev.argmax(axis=1)
    i_top_taxon = species_taxon[i_top]
    p_aves = (p_top_taxon == "Aves")
    i_aves = (i_top_taxon == "Aves")

    print("\n=== Disagreement quadrants on EVAL HOLD-OUT (Perch_top1 × iVAE_top1) ===")
    for pa in [True, False]:
        for ia in [True, False]:
            mask = (p_aves == pa) & (i_aves == ia)
            n = mask.sum()
            if n == 0: continue
            gt_aves = sum(1 for i, gt in enumerate(gt_taxa_ev) if mask[i] and "Aves" in gt)
            gt_nonaves = sum(1 for i, gt in enumerate(gt_taxa_ev) if mask[i] and any(t != "Aves" for t in gt))
            quad = f"P={'A' if pa else 'X'}_iV={'A' if ia else 'X'}"
            print(f"  {quad:<10} n={n:>4}  GT_has_Aves={gt_aves} ({100*gt_aves/n:.0f}%)  GT_has_nonAves={gt_nonaves} ({100*gt_nonaves/n:.0f}%)")

    # === Q3 on EVAL: among Perch=Aves, iVAE=non-Aves, does iVAE pick the right taxon? ===
    print("\n=== EVAL Q3: Perch=Aves, iVAE=non-Aves taxon match rate ===")
    mask = p_aves & (~i_aves)
    print(f"  rows: {mask.sum()}/{len(gt_taxa_ev)}")
    if mask.sum() > 0:
        for tt in ["Amphibia", "Insecta", "Mammalia", "Reptilia"]:
            sel = mask & (i_top_taxon == tt)
            if sel.sum() == 0: continue
            gt_match = sum(1 for i in np.where(sel)[0] if tt in gt_taxa_ev[i])
            print(f"    iVAE→{tt:<10} {sel.sum():>3} rows, GT contains {tt}: {gt_match} ({100*gt_match/sel.sum():.0f}%)")

    print("\n=== exp79b done ===\n")


if __name__ == "__main__":
    main()
