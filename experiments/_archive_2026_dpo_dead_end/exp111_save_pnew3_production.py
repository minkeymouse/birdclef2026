#!/usr/bin/env python3
"""exp111 — Train P_NEW3 production model on TA + ALL SS, save to model-weights.

Production model: trained on every labeled row we have (no LOSO holdout).
Saves state_dict for the Hybrid (frozen Perch-init + trainable correction MLP)
plus a small JSON config for the loader in the inference notebook.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        ROOT, N_CLS)
from exp106_pnew_hybrid import build_perch_init, train_hybrid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MW = ROOT / "model-weights"


def main():
    print("=== exp111: Train P_NEW3 production model + save ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_ss = load_perch_emb_labeled()

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0

    # Train on TA + ALL SS (production: no holdout)
    X_train = np.concatenate([ta_emb[valid], perch_emb_ss], axis=0)
    Y_train = np.concatenate([Y_ta[valid], Y.astype(np.float32)], axis=0)
    src_w = np.concatenate([np.ones(valid.sum()), np.full(len(perch_emb_ss), 5.0)])
    print(f"  Production train: TA {valid.sum()} + SS {len(perch_emb_ss)} = {len(X_train)} rows")
    print(f"  Total positives: {int(Y_train.sum())}")

    W_init, b_init, mapped_idx = build_perch_init()

    # Train (use SS as monitor — for tracking only, not selection since this is production)
    ev_mask = sc_g.split.values == "eval"
    Y_ss_ev = Y[ev_mask].astype(np.float32)
    print(f"\n  Training (20 epochs, lr=1e-3, monitor on 122 eval rows)...", flush=True)
    best_auc, best_pred, init_macro, best_ep, model = train_hybrid(
        X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=20, verbose=True
    )
    print(f"\n  Best monitor macro: {best_auc:.4f} @ ep {best_ep}")

    # Save the LAST trained state (production = train on all data, take last state)
    # NOTE: train_hybrid returns the model in last-epoch state, with internal "best_pred"
    # tracking only being used for the returned numbers. For production we simply save
    # what's currently in the model.
    out_path = MW / "p_new3_hybrid.pt"
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    config = {
        "in_dim": 1536,
        "hidden": 768,
        "n_classes": N_CLS,
        "dropout": 0.3,
        "trained_on": "TA(35549) + labeled_SS(739), 20 epochs, lr=1e-3 cosine",
        "perch_init_npz": "perch_head_extracted.npz",
        "monitor_macro": float(best_auc),
        "best_ep": int(best_ep),
    }
    torch.save({
        "state_dict": state_dict,
        "config": config,
    }, out_path)
    print(f"\n  Saved → {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.2f} MB")

    # Sanity check: reload + inference test
    print("\n  Sanity check: reload + inference on full SS...")
    from exp106_pnew_hybrid import PerchHybrid
    m2 = PerchHybrid(W_init, b_init).to(DEVICE)
    ckpt = torch.load(out_path, map_location=DEVICE, weights_only=False)
    m2.load_state_dict(ckpt["state_dict"])
    m2.eval()
    with torch.no_grad():
        emb_t = torch.from_numpy(perch_emb_ss).to(DEVICE)
        logits = m2(emb_t)
        prob = torch.sigmoid(logits).cpu().numpy()
    from _lib.eval_metrics import macro_auc
    macro_full, _ = macro_auc(Y.astype(np.float32), prob)
    macro_ev, _ = macro_auc(Y_ss_ev, prob[ev_mask])
    print(f"  Reloaded macro on full SS (leaky): {macro_full:.4f}")
    print(f"  Reloaded macro on 122 eval (leaky): {macro_ev:.4f}")


if __name__ == "__main__":
    main()
