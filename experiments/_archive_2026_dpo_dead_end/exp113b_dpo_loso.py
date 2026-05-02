#!/usr/bin/env python3
"""exp113b — LOSO-site validation of DPO β=1.0 (best from exp113).

For each holdout site:
  1. Train BCE reference on TA + (N-1 sites SS)
  2. Train DPO policy from reference, β=1.0, 1 epoch (best in exp113)
  3. Eval policy on holdout site

Compare to P_NEW3 BCE LOSO mean (0.767 from exp104b).
"""
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
from exp113_pnew3_dpo import train_bce_reference, train_dpo


def main():
    print("=== exp113b: LOSO validation of DPO β=1.0 ===\n", flush=True)

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

    print(f"  {'holdout':<8} {'n':>4} {'BCE ref':>9} {'DPO best':>10} {'Δ':>8} {'Aves':>8} {'Amphib':>8} {'Insecta':>8} {'Mam':>6}")

    results = []
    for ho_site in unique_sites:
        ho_mask = sites_arr == ho_site
        if ho_mask.sum() < 5: continue
        if (Y[ho_mask].sum(axis=0) > 0).sum() < 1: continue

        keep_mask = ~ho_mask
        X_train = np.concatenate([ta_emb[valid], perch_emb_ss[keep_mask]], axis=0)
        Y_train = np.concatenate([Y_ta[valid], Y[keep_mask].astype(np.float32)], axis=0)
        src_w = np.concatenate([np.ones(valid.sum()), np.full(keep_mask.sum(), 5.0)])
        Y_eval = Y[ho_mask].astype(np.float32)

        # Train BCE reference
        ref_model, ref_macro = train_bce_reference(
            X_train, Y_train, src_w, perch_emb_ss[ho_mask], Y_eval,
            W_init, b_init, n_epochs=12
        )

        # Train DPO policy
        best_dpo, ev_pred, _, best_ep, _ = train_dpo(
            X_train, Y_train, src_w, perch_emb_ss[ho_mask], Y_eval,
            W_init, b_init, ref_model, beta=1.0, n_epochs=5, verbose=False
        )

        pt = per_taxon_macro(Y_eval, ev_pred, sp_taxon)
        a = pt.get("Aves", float("nan"))
        amp = pt.get("Amphibia", float("nan"))
        i = pt.get("Insecta", float("nan"))
        m = pt.get("Mammalia", float("nan"))
        a_str = f"{a:.4f}" if not np.isnan(a) else "  --  "
        amp_str = f"{amp:.4f}" if not np.isnan(amp) else "  --  "
        i_str = f"{i:.4f}" if not np.isnan(i) else "  --  "
        m_str = f"{m:.4f}" if not np.isnan(m) else "  --  "
        print(f"  {ho_site:<8} {ho_mask.sum():>4} {ref_macro:>9.4f} {best_dpo:>10.4f} "
              f"{best_dpo-ref_macro:>+8.4f} {a_str:>8} {amp_str:>8} {i_str:>8} {m_str:>6}", flush=True)

        results.append({
            "site": ho_site, "n": int(ho_mask.sum()),
            "bce_macro": ref_macro, "dpo_macro": best_dpo,
            "Aves": a, "Amphibia": amp, "Insecta": i, "Mammalia": m,
        })

    # Summary
    bce_mean = np.mean([r["bce_macro"] for r in results])
    dpo_mean = np.mean([r["dpo_macro"] for r in results])
    print(f"\n  Mean LOSO macro:")
    print(f"    BCE ref:    {bce_mean:.4f}")
    print(f"    DPO β=1.0:  {dpo_mean:.4f}")
    print(f"    Δ:          {dpo_mean - bce_mean:+.4f}")

    print(f"\n  Per-taxon LOSO mean:")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
        vals = [r[t] for r in results if not np.isnan(r[t])]
        if vals:
            print(f"    {t}: {np.mean(vals):.4f} (n={len(vals)})")


if __name__ == "__main__":
    main()
