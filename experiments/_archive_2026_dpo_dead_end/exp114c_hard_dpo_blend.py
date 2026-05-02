#!/usr/bin/env python3
"""exp114c — Build Hard-DPO LOSO predictions for full SS, blend test with v33."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate
from exp106_pnew_hybrid import build_perch_init
from exp113_pnew3_dpo import train_bce_reference
from exp114_hard_dpo import mine_hard_pairs, train_hard_dpo


def main():
    print("=== exp114c: Hard-DPO blend with v33 audit ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()
    perch_prob_ss = load_perch_scores_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]

    base = 0.7 * perch_prob_ss + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb_ss, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0
    W_init, b_init, _ = build_perch_init()

    sites_arr = sc_g.site.values

    # Build LOSO Hard-DPO predictions for full SS
    print("Building LOSO Hard-DPO predictions for all 739 SS rows...")
    hd_pred = np.zeros((len(perch_emb_ss), N_CLS), dtype=np.float32)
    for ho_site in sorted(set(sites_arr)):
        ho_mask = sites_arr == ho_site
        if ho_mask.sum() < 5: continue
        keep_mask = ~ho_mask
        X_train = np.concatenate([ta_emb[valid], perch_emb_ss[keep_mask]], axis=0)
        Y_train = np.concatenate([Y_ta[valid], Y[keep_mask].astype(np.float32)], axis=0)
        src_w = np.concatenate([np.ones(valid.sum()), np.full(keep_mask.sum(), 5.0)])

        ref_model, _ = train_bce_reference(
            X_train, Y_train, src_w, perch_emb_ss[ho_mask],
            Y[ho_mask].astype(np.float32), W_init, b_init, n_epochs=12
        )
        triplets = mine_hard_pairs(ref_model, X_train, Y_train, margin=0.5, max_pairs_per_row=20)
        _, ev_pred, _, _, _ = train_hard_dpo(
            X_train, Y_train, src_w, perch_emb_ss[ho_mask],
            Y[ho_mask].astype(np.float32), W_init, b_init, ref_model, triplets,
            beta=1.0, n_epochs=6, verbose=False
        )
        hd_pred[ho_mask] = ev_pred
        print(f"  {ho_site}: n={ho_mask.sum()} done", flush=True)

    np.savez_compressed(EXP80 / "p_new3_hard_dpo_predictions.npz",
                         predictions=hd_pred.astype(np.float32))
    print("  Saved → p_new3_hard_dpo_predictions.npz")

    # Blend audit on 122 eval
    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]
    rows.append(evaluate(hd_pred, v33, ev_mask, Y, sp_taxon, "Hard-DPO alone"))

    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        P = (1 - w) * v33 + w * hd_pred
        P = np.clip(P, 0, 1).astype(np.float32)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"v33 + Hard-DPO w={w}"))

    # Compare to existing P_NEW3 hybrid (BCE) blend
    pnew3_bce = np.load(EXP80 / "p_new2_hybrid_predictions.npz")["predictions"]
    rows.append(evaluate(pnew3_bce, v33, ev_mask, Y, sp_taxon, "P_NEW3 BCE alone (leaky)"))
    for w in [0.10, 0.20]:
        P = (1 - w) * v33 + w * pnew3_bce
        P = np.clip(P, 0, 1).astype(np.float32)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"v33 + P_NEW3 BCE w={w} (leaky)"))

    # 3-way: v33 + small exp50 + Hard-DPO
    for w_hd in [0.05, 0.10, 0.15]:
        P = (1 - w_hd - 0.05) * v33 + 0.05 * exp50 + w_hd * hd_pred
        P = np.clip(P, 0, 1).astype(np.float32)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"3-way v33+0.05exp50+{w_hd}HD"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== Blend audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
