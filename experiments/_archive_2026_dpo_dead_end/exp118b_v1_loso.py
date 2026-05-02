#!/usr/bin/env python3
"""exp118b — LOSO validation of V1 (rare 20x) and V3 (wrong (r,c) 10x)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        ROOT, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from exp106_pnew_hybrid import build_perch_init, PerchHybrid
from exp118_targeted_boosting import train_pnew3_with_loss_mods


def main():
    print("=== exp118b: LOSO of V1 (rare 20x) + V3 (wrong-rc 10x) vs BCE ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb = load_perch_emb_labeled()

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0

    W_init, b_init, _ = build_perch_init()
    sp_taxon_arr = np.array(sp_taxon)

    # V1 per-class weights (rare 20x)
    pcw_v1 = np.ones(N_CLS, dtype=np.float32)
    pcw_v1[np.isin(sp_taxon_arr, ["Mammalia", "Reptilia", "Insecta"])] = 20.0
    pcw_v1[sp_taxon_arr == "Amphibia"] = 5.0

    sites_arr = sc_g.site.values
    unique_sites = sorted(set(sites_arr))

    print(f"  {'holdout':<6} {'n':>4} {'BCE':>8} {'V1':>8} {'V3':>8} {'V1Δ':>8} {'V3Δ':>8} | "
          f"{'V1 Insecta':>10} {'V1 Mam':>8}")

    bce_list = []
    v1_list = []
    v3_list = []
    v1_pt = {t: [] for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
    v3_pt = {t: [] for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}

    for ho_site in unique_sites:
        ho_mask = sites_arr == ho_site
        if ho_mask.sum() < 5: continue
        if (Y[ho_mask].sum(axis=0) > 0).sum() < 1: continue

        keep_mask = ~ho_mask
        X_train = np.concatenate([ta_emb[valid], perch_emb[keep_mask]], axis=0)
        Y_train = np.concatenate([Y_ta[valid], Y[keep_mask].astype(np.float32)], axis=0)
        src_w = np.concatenate([np.ones(valid.sum()), np.full(keep_mask.sum(), 5.0)])
        Y_eval = Y[ho_mask].astype(np.float32)

        # BCE baseline
        macro_bce, ev_pred_bce, state_bce = train_pnew3_with_loss_mods(
            X_train, Y_train, src_w, perch_emb[ho_mask], Y_eval,
            W_init, b_init, n_epochs=12
        )

        # V1: rare 20x
        macro_v1, ev_pred_v1, _ = train_pnew3_with_loss_mods(
            X_train, Y_train, src_w, perch_emb[ho_mask], Y_eval,
            W_init, b_init, per_class_w=pcw_v1, n_epochs=12
        )

        # V3: wrong (r, c) 10x — based on BCE predictions on training data
        base_model = PerchHybrid(W_init, b_init).cuda()
        base_model.load_state_dict(state_bce)
        base_model.eval()
        rc_mask = np.ones((len(X_train), N_CLS), dtype=np.float32)
        X_t = torch.from_numpy(X_train.astype(np.float32)).cuda()
        BATCH = 2048
        with torch.no_grad():
            for s in range(0, len(X_t), BATCH):
                x = X_t[s:s+BATCH]
                probs = torch.sigmoid(base_model(x)).cpu().numpy()
                y_b = Y_train[s:s+BATCH]
                for i in range(len(x)):
                    wrong_pos = (y_b[i] > 0) & (probs[i] < 0.3)
                    if wrong_pos.any():
                        rc_mask[s + i, wrong_pos] = 10.0

        macro_v3, ev_pred_v3, _ = train_pnew3_with_loss_mods(
            X_train, Y_train, src_w, perch_emb[ho_mask], Y_eval,
            W_init, b_init, per_rc_mask=rc_mask, n_epochs=12
        )

        pt_v1 = per_taxon_macro(Y_eval, ev_pred_v1, sp_taxon)
        pt_v3 = per_taxon_macro(Y_eval, ev_pred_v3, sp_taxon)

        v1_ins = pt_v1.get("Insecta", float("nan"))
        v1_mam = pt_v1.get("Mammalia", float("nan"))
        v1_ins_s = f"{v1_ins:.4f}" if not np.isnan(v1_ins) else "  --  "
        v1_mam_s = f"{v1_mam:.4f}" if not np.isnan(v1_mam) else "  --  "

        print(f"  {ho_site:<6} {ho_mask.sum():>4} {macro_bce:>8.4f} {macro_v1:>8.4f} {macro_v3:>8.4f} "
              f"{macro_v1-macro_bce:>+8.4f} {macro_v3-macro_bce:>+8.4f} | {v1_ins_s:>10} {v1_mam_s:>8}", flush=True)

        bce_list.append(macro_bce)
        v1_list.append(macro_v1)
        v3_list.append(macro_v3)
        for t in v1_pt:
            v = pt_v1.get(t, float('nan'))
            if not np.isnan(v): v1_pt[t].append(v)
            v = pt_v3.get(t, float('nan'))
            if not np.isnan(v): v3_pt[t].append(v)

    print(f"\n  Mean LOSO macro:")
    print(f"    BCE:  {np.mean(bce_list):.4f}")
    print(f"    V1:   {np.mean(v1_list):.4f}  Δ {np.mean(v1_list) - np.mean(bce_list):+.4f}")
    print(f"    V3:   {np.mean(v3_list):.4f}  Δ {np.mean(v3_list) - np.mean(bce_list):+.4f}")

    print(f"\n  V1 per-taxon LOSO:")
    for t, vals in v1_pt.items():
        if vals: print(f"    {t}: {np.mean(vals):.4f} (n={len(vals)})")
    print(f"\n  V3 per-taxon LOSO:")
    for t, vals in v3_pt.items():
        if vals: print(f"    {t}: {np.mean(vals):.4f} (n={len(vals)})")


if __name__ == "__main__":
    main()
