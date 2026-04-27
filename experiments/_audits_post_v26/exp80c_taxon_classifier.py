#!/usr/bin/env python3
"""exp80c — Taxon-level classifier comparison (Perch vs iVAE-z vs concat).

Tests whether iVAE z adds any information to Perch emb for taxon detection.
Reports per-taxon AUC on 122 same-site eval rows + LOSO site CV.

Result: same-site eval shows iVAE neutral or slight harm; LOSO site CV
shows Insecta detection collapses to 0.20-0.30 AUC for ALL feature sets.
Confirms site-shortcut origin of any Insecta detection signal we have.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, get_taxon_y, load_labeled_mel,
                        load_perch_emb_labeled, EXP80, MW, TAXA)
from _lib.ivae import IVAE, DEVICE
from _lib.eval_metrics import gpu_mlp_binary_auc

import torch


def encode_smallpool_z(mel_lab):
    ck = torch.load(MW / "ivae_encoder.pt", map_location=DEVICE, weights_only=False)
    stats = np.load(MW / "ivae_mel_stats.npz")
    enc = IVAE(int(ck["in_dim"]), int(ck["z_dim"]), int(ck["n_aux"])).to(DEVICE).eval()
    enc.load_state_dict(ck["encoder_state_dict"], strict=False)
    X = mel_lab.reshape(len(mel_lab), -1).astype(np.float32)
    X = (X - stats["mean"].astype(np.float32)) / stats["std"].astype(np.float32)
    with torch.no_grad():
        return enc.encode(torch.from_numpy(X).to(DEVICE)).cpu().numpy()


def main():
    print("=== exp80c: taxon classifier comparison ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    P = load_perch_emb_labeled()
    print(f"Perch emb: {P.shape}", flush=True)

    Z_small = encode_smallpool_z(load_labeled_mel())
    print(f"iVAE z small: {Z_small.shape}", flush=True)

    big_path = EXP80 / "bigpool_z.npz"
    Z_big = np.load(big_path)["Z_lab"] if big_path.exists() else None
    print(f"iVAE z big: {Z_big.shape if Z_big is not None else None}", flush=True)

    tr = sc_g.split.values == "train"; ev = sc_g.split.values == "eval"
    print(f"train={tr.sum()} eval={ev.sum()}\n", flush=True)

    feats = {
        "Perch_only(1536)": P,
        "iVAE_small(32)":   Z_small,
        "Perch+iVAE_small": np.concatenate([P, Z_small], axis=1),
    }
    if Z_big is not None:
        feats["iVAE_big(32)"]    = Z_big
        feats["Perch+iVAE_big"]  = np.concatenate([P, Z_big], axis=1)

    def std_(X_tr, X_full):
        return ((X_full - X_tr.mean(0)) / (X_tr.std(0) + 1e-6)).astype(np.float32)

    print("=== Per-taxon AUC on 122 held-out eval (GPU MLP h=64, 300ep) ===")
    hdr = "  {:<24} {:>8} {:>8} {:>8} {:>8} {:>8}".format("feat", *TAXA)
    print(hdr, flush=True)
    for fname, X in feats.items():
        Xs = std_(X[tr], X)
        cells = []
        for t in TAXA:
            y = get_taxon_y(Y, sp_taxon, t)
            auc = gpu_mlp_binary_auc(Xs[tr], y[tr], Xs[ev], y[ev])
            cells.append(f"{auc:.4f}" if not np.isnan(auc) else "  --  ")
        print("  {:<24} ".format(fname) + " ".join(f"{c:>8}" for c in cells), flush=True)

    print("\n=== LOSO site CV (mean AUC across sites with ≥2 pos) — 2 key feats ===", flush=True)
    sites = sorted(sc_g.site.unique())
    LOSO_FEATS = {"Perch_only(1536)": P}
    if Z_big is not None:
        LOSO_FEATS["Perch+iVAE_big"] = np.concatenate([P, Z_big], axis=1)
    print(f"  {'taxon':<10} {'feat':<24} {'mean_AUC':>9} {'n_sites':>8}", flush=True)
    for t in TAXA:
        y = get_taxon_y(Y, sp_taxon, t)
        for fname, X in LOSO_FEATS.items():
            aucs = []
            for s in sites:
                ho = sc_g.site.values == s
                tr_m = ~ho
                if y[ho].sum() < 2 or y[ho].sum() == ho.sum() or y[tr_m].sum() < 5: continue
                Xs = std_(X[tr_m], X)
                a = gpu_mlp_binary_auc(Xs[tr_m], y[tr_m], Xs[ho], y[ho], epochs=200)
                if not np.isnan(a): aucs.append(a)
            if aucs:
                print(f"  {t:<10} {fname:<24} {np.mean(aucs):>9.4f} {len(aucs):>8}", flush=True)


if __name__ == "__main__":
    main()
