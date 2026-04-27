#!/usr/bin/env python3
"""exp79d — Generate concrete audit candidate list.

Filter: Perch_top1 = Aves AND iVAE_top1 = Insecta sonotype AND cos > 0.6.
Output: 30 candidates spread across sites, with file path + start_time so
the user can listen and judge whether they're real insects.

Also includes per-row Perch confidence on iVAE-predicted-species (should be
near zero — Perch can't see Insecta) for completeness.
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
SR = 32000; N_WIN = 12; T_POOL = 16; N_MELS = 128
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
    print("=== exp79d: audit candidate generation ===\n")

    # Load existing probe results
    df = pd.read_parquet(OUT / "unlabeled_probe.parquet")
    print(f"loaded {len(df)} probe rows")

    # Need Perch top1 — get from the perch_ss_all
    perch_emb = np.load(EXP43A / "perch_ss_all.npz")
    perch_meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    l2t = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([l2t.get(p, "?") for p in primary])

    # Build row_id for our probe rows
    def make_rid(r):
        fn = r.filename.replace(".ogg", "")
        return f"{fn}_{int(r.end)}"
    df["row_id"] = df.apply(make_rid, axis=1)

    # Align Perch
    rid2i = {r: i for i, r in enumerate(perch_meta["row_id"].values)}
    matched_idx = np.array([rid2i.get(rid, -1) for rid in df.row_id.values])
    has_perch = matched_idx >= 0
    print(f"Perch alignment: {has_perch.sum()}/{len(df)} rows matched")

    perch_scores = perch_emb["scores"]
    P = np.zeros((len(df), 234), dtype=np.float32)
    P[has_perch] = perch_scores[matched_idx[has_perch]]
    perch_prob = 1.0 / (1.0 + np.exp(-P))

    df["perch_top1"] = [primary[i] for i in perch_prob.argmax(axis=1)]
    df["perch_top1_taxon"] = [species_taxon[i] for i in perch_prob.argmax(axis=1)]
    df["perch_top1_prob"] = perch_prob.max(axis=1)
    # Perch prob on iVAE-predicted species (should be near-zero for unmapped Insecta)
    sp2i = {p: i for i, p in enumerate(primary)}
    df["perch_on_ivae_sp"] = [perch_prob[i, sp2i[df.iloc[i].insecta_max_lbl]] for i in range(len(df))]

    # === Filter: high-confidence Insecta-disagreement ===
    mask = (
        (df.ivae_top1_taxon == "Insecta") &
        (df.perch_top1_taxon == "Aves") &
        (df.insecta_max_cos > 0.55)
    )
    cand = df[mask].sort_values("insecta_max_cos", ascending=False).reset_index(drop=True)
    print(f"\nDisagreement candidates (iVAE=Insecta & Perch=Aves & cos>0.55): {len(cand)}")
    print(f"site distribution of candidates:")
    print(cand.site.value_counts().to_string())
    print(f"\npredicted Insecta species:")
    print(cand.insecta_max_lbl.value_counts().head(10).to_string())

    # Pick 30 audit candidates: top per-site (max 5 each), highest cos
    parts = []
    for s, g in cand.groupby("site"):
        parts.append(g.nlargest(5, "insecta_max_cos"))
    audit = pd.concat(parts, ignore_index=True).nlargest(30, "insecta_max_cos")
    audit_cols = ["filename", "site", "hour", "start", "end",
                  "insecta_max_lbl", "insecta_max_cos",
                  "perch_top1", "perch_top1_prob", "perch_on_ivae_sp"]
    audit[audit_cols].to_csv(OUT / "audit30.csv", index=False)
    print(f"\nSaved audit30.csv:")
    print(audit[audit_cols].to_string(index=False))

    # === Cluster the disagreement set in z-space ===
    if len(cand) >= 20:
        from sklearn.cluster import KMeans
        Z_cand = np.stack([df[df.row_id == r.row_id].iloc[0].z for _, r in cand.iterrows()]) if False else None
        # easier: re-compute z from full df.z column (already saved)
        Z_full = np.stack(df.z.values)
        Z_cand = Z_full[mask.values]
        K = min(8, len(Z_cand) // 5)
        if K >= 2:
            km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(Z_cand)
            cand_cl = cand.copy()
            cand_cl["cluster"] = km.labels_
            print(f"\n=== KMeans k={K} on disagreement set ({len(Z_cand)} rows) ===")
            for k in range(K):
                sub = cand_cl[cand_cl.cluster == k]
                top_sp = sub.insecta_max_lbl.value_counts().head(3).to_dict()
                top_site = sub.site.value_counts().head(3).to_dict()
                print(f"  cluster {k:2d}: n={len(sub):3d}  cos_mean={sub.insecta_max_cos.mean():.3f}  "
                      f"top_sp={top_sp}  top_site={top_site}")

    # === Sanity: how does Perch on iVAE-predicted-species compare to Perch top1 prob? ===
    print("\n=== Sanity: Perch can't see Insecta sonotypes ===")
    print(f"  mean perch_prob on iVAE-predicted-Insecta-sp: {df.perch_on_ivae_sp.mean():.4f}")
    print(f"  mean perch_top1_prob:                          {df.perch_top1_prob.mean():.4f}")


if __name__ == "__main__":
    main()
