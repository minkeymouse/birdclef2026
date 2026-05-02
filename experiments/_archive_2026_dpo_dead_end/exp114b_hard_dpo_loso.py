#!/usr/bin/env python3
"""exp114b — LOSO validation of Hard-mined DPO β=1.0."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        ROOT, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from exp106_pnew_hybrid import build_perch_init
from exp113_pnew3_dpo import train_bce_reference
from exp114_hard_dpo import mine_hard_pairs, train_hard_dpo


def main():
    print("=== exp114b: LOSO validation of Hard-DPO β=1.0 ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0

    W_init, b_init, _ = build_perch_init()

    sites_arr = sc_g.site.values
    unique_sites = sorted(set(sites_arr))

    print(f"  {'holdout':<8} {'n':>4} {'BCE':>8} {'HardDPO':>9} {'Δ':>8} {'Aves':>8} {'Amphib':>8} {'Insecta':>8} {'Mam':>8}")

    bce_results = []
    dpo_results = []
    for ho_site in unique_sites:
        ho_mask = sites_arr == ho_site
        if ho_mask.sum() < 5: continue
        if (Y[ho_mask].sum(axis=0) > 0).sum() < 1: continue

        keep_mask = ~ho_mask
        X_train = np.concatenate([ta_emb[valid], perch_emb_ss[keep_mask]], axis=0)
        Y_train = np.concatenate([Y_ta[valid], Y[keep_mask].astype(np.float32)], axis=0)
        src_w = np.concatenate([np.ones(valid.sum()), np.full(keep_mask.sum(), 5.0)])
        Y_eval = Y[ho_mask].astype(np.float32)

        # 1. BCE reference
        ref_model, ref_macro = train_bce_reference(
            X_train, Y_train, src_w, perch_emb_ss[ho_mask], Y_eval,
            W_init, b_init, n_epochs=12
        )

        # 2. Hard-mine pairs
        triplets = mine_hard_pairs(ref_model, X_train, Y_train, margin=0.5, max_pairs_per_row=20)

        # 3. Hard-DPO with β=1.0
        best_dpo, ev_pred, _, best_ep, _ = train_hard_dpo(
            X_train, Y_train, src_w, perch_emb_ss[ho_mask], Y_eval,
            W_init, b_init, ref_model, triplets, beta=1.0, n_epochs=6, verbose=False
        )

        pt = per_taxon_macro(Y_eval, ev_pred, sp_taxon)
        a = pt.get("Aves", float("nan"))
        amp = pt.get("Amphibia", float("nan"))
        i = pt.get("Insecta", float("nan"))
        m = pt.get("Mammalia", float("nan"))
        a_s = f"{a:.4f}" if not np.isnan(a) else "  --  "
        amp_s = f"{amp:.4f}" if not np.isnan(amp) else "  --  "
        i_s = f"{i:.4f}" if not np.isnan(i) else "  --  "
        m_s = f"{m:.4f}" if not np.isnan(m) else "  --  "
        print(f"  {ho_site:<8} {ho_mask.sum():>4} {ref_macro:>8.4f} {best_dpo:>9.4f} "
              f"{best_dpo-ref_macro:>+8.4f} {a_s:>8} {amp_s:>8} {i_s:>8} {m_s:>8} (n_trips={len(triplets):>5})", flush=True)

        bce_results.append({"site": ho_site, "macro": ref_macro})
        dpo_results.append({"site": ho_site, "macro": best_dpo, **{k: pt.get(k, float('nan')) for k in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}})

    bce_mean = np.mean([r["macro"] for r in bce_results])
    dpo_mean = np.mean([r["macro"] for r in dpo_results])
    print(f"\n  Mean LOSO macro:")
    print(f"    BCE ref:        {bce_mean:.4f}")
    print(f"    Hard-DPO β=1.0: {dpo_mean:.4f}")
    print(f"    Δ:              {dpo_mean - bce_mean:+.4f}")

    print(f"\n  Per-taxon LOSO mean (Hard-DPO):")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        vals = [r[t] for r in dpo_results if not np.isnan(r[t])]
        if vals:
            print(f"    {t}: {np.mean(vals):.4f} (n={len(vals)})")


if __name__ == "__main__":
    main()
