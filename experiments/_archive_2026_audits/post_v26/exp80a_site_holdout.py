#!/usr/bin/env python3
"""exp80a — Site holdout validation. Train iVAE WITHOUT a known Insecta site,
test detection AUC on that held-out site.

Decisive test: if held-out S19 Insecta detection AUC stays high → species
acoustic signal. If drops to random / sub-random → site fingerprint.

Result: S19 Insecta AUC = 0.073. Site shortcut confirmed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, aux_matrix, load_labeled_mel,
                        N_CLS, EXP80, TAXA)
from _lib.ivae import train_full, encode_all, DEVICE

OUT = EXP80
OUT.mkdir(exist_ok=True, parents=True)
HOLDOUT_SITES = ["S08", "S15", "S19", "S23"]


def eval_one(holdout_site: str, X, Y, sc_g, sp_taxon, sites):
    aux, _ = aux_matrix(sc_g, sites=sites)
    is_holdout = (sc_g.site.values == holdout_site)
    tr_mask = ~is_holdout
    print(f"  holdout={holdout_site}  train={tr_mask.sum()}  holdout={is_holdout.sum()}", flush=True)

    model, mu, sd = train_full(X, aux, tr_mask, z_dim=32, hidden=512, epochs=200,
                                beta=0.05, verbose_every=0)
    Z = encode_all(model, X, mu, sd)

    # Centroids from train-only positives
    Y_tr = Y[tr_mask]; Z_tr = Z[tr_mask]
    centroids = np.zeros((N_CLS, 32), dtype=np.float32)
    cv = np.zeros(N_CLS, dtype=bool)
    for c in range(N_CLS):
        if Y_tr[:, c].sum() >= 3:
            centroids[c] = Z_tr[Y_tr[:, c] == 1].mean(0); cv[c] = True

    z_n = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    c_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    cos = z_n @ c_n.T
    cos[:, ~cv] = -np.inf

    out = {"holdout": holdout_site, "n_holdout_rows": int(is_holdout.sum())}
    for tax in TAXA:
        gt = (Y[:, sp_taxon == tax].sum(axis=1) > 0).astype(np.uint8)
        n_pos = int(gt[is_holdout].sum())
        out[f"npos_{tax}"] = n_pos
        if n_pos < 3 or n_pos == is_holdout.sum():
            out[f"auc_{tax}"] = np.nan; continue
        valid = np.where(cv & (sp_taxon == tax))[0]
        out[f"n_centroids_{tax}"] = int(len(valid))
        if len(valid) == 0:
            out[f"auc_{tax}"] = np.nan; continue
        score = cos[is_holdout][:, valid].max(axis=1)
        try: out[f"auc_{tax}"] = float(roc_auc_score(gt[is_holdout], score))
        except Exception: out[f"auc_{tax}"] = np.nan
    return out


def main():
    print("=== exp80a: site holdout validation ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    mel = load_labeled_mel()
    X = mel.reshape(len(sc_g), -1).astype(np.float32)
    sites = sorted(sc_g.site.unique())
    print(f"sites: {sites}", flush=True)

    rows = []
    for hs in HOLDOUT_SITES:
        print(f"\n--- HOLDOUT={hs} ---", flush=True)
        r = eval_one(hs, X, Y, sc_g, sp_taxon, sites)
        print(f"  result: {r}", flush=True)
        rows.append(r)

    df = pd.DataFrame(rows)
    cols = ["holdout"] + sorted([c for c in df.columns if c != "holdout"])
    print("\n=== Summary ===")
    print(df[cols].to_string(index=False))
    df.to_csv(OUT / "site_holdout.csv", index=False)
    print(f"\nSaved → {OUT / 'site_holdout.csv'}")


if __name__ == "__main__":
    main()
