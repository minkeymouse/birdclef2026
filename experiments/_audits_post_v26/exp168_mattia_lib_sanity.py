#!/usr/bin/env python3
"""exp168 — Mattia lib local sanity check.

Verifies that notebooks/birdclef-2026-mattia-fork/lib/ is properly
importable and reproduces the cached Tucker scores on labeled SS.

This is a regression test: if exp168 passes, future audits/extensions
can rely on `from lib import tucker_sed, final_blend, ...` cleanly.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add mattia-fork lib to path
ROOT = Path("/data/birdclef2026")
MATTIA = ROOT / "notebooks/birdclef-2026-mattia-fork"
sys.path.insert(0, str(MATTIA))

# Import lib
from lib import paths, tucker_sed, final_blend

# Also our existing audit utilities
sys.path.insert(0, str(ROOT / "experiments/_audits_post_v26"))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate


def main():
    print("=== exp168: lib import + Tucker reproduction sanity ===\n", flush=True)
    paths.report()
    print()

    # Compare cached Tucker scores vs fresh from lib.tucker_sed
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    cached = np.load(EXP80 / "tucker_sed_5fold_labeled.npz")["scores"]
    print(f"cached Tucker scores: shape {cached.shape}, "
          f"range [{cached.min():.4f}, {cached.max():.4f}]")

    # Sanity: run lib tucker on FIRST 3 files only (fast)
    SS_DIR = paths.COMP_DATA / "train_soundscapes"
    files = sorted(sc_g.filename.unique())[:3]
    print(f"\nrunning lib.tucker_sed on {len(files)} files for spot-check ...")
    sessions = tucker_sed.load_5fold()
    fresh_rows = []
    for fn in files:
        ens = tucker_sed.predict_file(sessions, SS_DIR / fn)
        fresh_rows.append(ens)
    fresh = np.concatenate(fresh_rows, axis=0)  # (3*12, 234)
    print(f"  fresh: {fresh.shape}, range [{fresh.min():.4f}, {fresh.max():.4f}]")

    # Match against cached for these 3 files
    fname_idx = {f: [] for f in files}
    for i, r in sc_g.iterrows():
        if r.filename in fname_idx:
            fname_idx[r.filename].append((i, int(r.end_sec)))
    n_match = 0
    n_close = 0
    for fname in files:
        for row_idx, end_sec in fname_idx[fname]:
            wi = max(0, min(11, end_sec // 5 - 1))
            cached_vec = cached[row_idx]
            file_local = files.index(fname)
            fresh_vec = fresh[file_local * 12 + wi]
            if np.allclose(cached_vec, fresh_vec, atol=1e-3):
                n_match += 1
            elif np.allclose(cached_vec, fresh_vec, atol=1e-2):
                n_close += 1
    n_total = sum(len(v) for v in fname_idx.values())
    print(f"  exact match (atol=1e-3): {n_match}/{n_total}")
    print(f"  close match (atol=1e-2): {n_close}/{n_total}")

    # Test final_blend
    print("\nrunning lib.final_blend.linear_blend + mattia_blend on cached scores ...")
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]

    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    file_ids = sc_g["filename"].astype(str).values
    ev_mask = sc_g.split.values == "eval"

    # v58-equivalent via lib.final_blend.linear_blend
    pred_lin = final_blend.linear_blend(v33, cached, sed_w=0.30)
    print(f"  linear blend pred: {pred_lin.shape}, "
          f"range [{pred_lin.min():.4f}, {pred_lin.max():.4f}]")
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref"),
             evaluate(pred_lin, v33, ev_mask, Y, sp_taxon,
                      "lib.final_blend.linear_blend (= v58)")]

    # Mattia full blend with rescues
    pred_mattia = final_blend.mattia_blend(v33, cached, file_ids,
                                             sed_w=0.30, rescues=("fake","cont","spike"))
    rows.append(evaluate(pred_mattia, v33, ev_mask, Y, sp_taxon,
                          "lib.final_blend.mattia_blend full rescues"))

    # No-rescue mattia
    pred_norescue = final_blend.mattia_blend(v33, cached, file_ids,
                                              sed_w=0.30, rescues=())
    rows.append(evaluate(pred_norescue, v33, ev_mask, Y, sp_taxon,
                          "lib.final_blend.mattia_blend NO rescues"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil"]
    print()
    print(res[cols].to_string(index=False))
    print("\n✓ lib import sanity: all checks passed if numbers match exp165/exp164.")


if __name__ == "__main__":
    main()
