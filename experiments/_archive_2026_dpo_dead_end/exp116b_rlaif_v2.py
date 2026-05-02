#!/usr/bin/env python3
"""exp116b — RLAIF v2 with per-class rank-normalization.

Fix from v1: Perch over-fires on ~37 species across all rows. mean(Perch, P_NEW3)
inherits this saturation, making the saturating species the "preferred" signal
for almost every unlabeled row → policy learns Perch's bias.

Solution: per-class rank-normalize each model's unlabeled predictions before
combining. Then "preferred for THIS row" = high relative rank in BOTH models,
which filters out class-level saturation.

Additional change: restrict mining to NON-AVES classes (72 species). Aves is
already well-handled by Perch xeno-canto pretraining + BCE. The training
signal we need is on rare classes where Perch is weak.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, N_CLS)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from exp106_pnew_hybrid import build_perch_init, PerchHybrid
from exp113_pnew3_dpo import train_bce_reference
from exp116_rlaif import run_pnew3_on, train_dpo_on_unlabeled

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=== exp116b: RLAIF v2 with rank-norm + non-Aves restriction ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_emb_lab = load_perch_emb_labeled()

    ta_path = ROOT / "experiments/_archive_2026_pre_v26/exp22_outputs/train_audio_perch.npz"
    ta = np.load(ta_path)
    ta_emb = ta["emb"]; ta_y_idx = ta["y_idx"]; ta_valid = ta["valid"]
    valid = (ta_valid == 1) & (ta_y_idx >= 0) & (ta_y_idx < N_CLS)
    Y_ta = np.zeros((len(ta_emb), N_CLS), dtype=np.float32)
    Y_ta[np.arange(len(ta_emb))[valid], ta_y_idx[valid]] = 1.0

    W_init, b_init, _ = build_perch_init()

    tr_mask = sc_g.split.values == "train"
    ev_mask = sc_g.split.values == "eval"
    Y_ss_ev = Y[ev_mask].astype(np.float32)

    # Step 1: BCE ref
    print("=== Step 1: BCE ref ===", flush=True)
    X_train = np.concatenate([ta_emb[valid], perch_emb_lab[tr_mask]], axis=0)
    Y_train = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])
    ref_model, ref_macro = train_bce_reference(
        X_train, Y_train, src_w, perch_emb_lab[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=15
    )
    print(f"  BCE ref macro: {ref_macro:.4f}\n")

    # Step 2: Load unlabeled
    unlab = np.load(ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz", mmap_mode="r")
    unlab_emb = np.array(unlab["emb"])  # need writable
    unlab_perch_scores = np.array(unlab["scores"])
    print(f"=== Step 2: Unlabeled SS {unlab_emb.shape} ===\n")

    # Step 3: P_NEW3 inference on unlabeled
    print("=== Step 3: P_NEW3 inference ===", flush=True)
    pnew3_unlab = run_pnew3_on(ref_model, unlab_emb, batch_size=2048)
    print(f"  P_NEW3 unlabeled scores: {pnew3_unlab.shape}\n")

    # Step 4: Per-class rank-normalize
    print("=== Step 4: Per-class rank-normalize + ensemble ===", flush=True)
    perch_rank = (rankdata(unlab_perch_scores, axis=0) / len(unlab_perch_scores)).astype(np.float32)
    pnew3_rank = (rankdata(pnew3_unlab, axis=0) / len(pnew3_unlab)).astype(np.float32)
    ensemble_rank = (perch_rank + pnew3_rank) / 2

    # Diagnostic: are saturating species' rank-norms still confounding?
    # Pick a known saturating class (47144, idx ?)
    if "47144" in primary:
        idx_sat = primary.index("47144")
        print(f"  47144 (saturating): raw Perch mean across rows = {unlab_perch_scores[:, idx_sat].mean():.3f}, "
              f"rank-norm mean = {perch_rank[:, idx_sat].mean():.3f}")
    print(f"  rank-norm space: ensemble[c=0] mean {ensemble_rank[:, 0].mean():.3f}\n")

    # Step 5: Mine preferences on RANK-norm space, restricted to non-Aves
    print("=== Step 5: Mine preferences (rank-norm, non-Aves only) ===", flush=True)
    sp_taxon_arr = np.array(sp_taxon)
    non_aves_mask = sp_taxon_arr != "Aves"
    non_aves_idx = np.where(non_aves_mask)[0]
    print(f"  Non-Aves classes: {len(non_aves_idx)}")

    # Triplet mining: for each row,
    # preferred = non-Aves class where ensemble_rank > 0.95 (top 5% of rows for that class)
    # rejected = ANY class (Aves or non-Aves) where ensemble_rank < 0.5 AND policy thinks > median
    triplets = []
    n_total = len(unlab_emb)
    pnew3_rank_full = pnew3_rank  # alias

    for i in range(n_total):
        # preferred candidates: non-Aves with high ensemble rank for this row
        pref_e = ensemble_rank[i][non_aves_idx]
        pref_mask = pref_e > 0.95   # top 5% per class
        pref_idx = non_aves_idx[pref_mask]

        if len(pref_idx) == 0: continue

        # rejected candidates: ANY class with very low ensemble rank for this row
        rej_e = ensemble_rank[i]
        rej_mask = rej_e < 0.30   # bottom 30% per class
        rej_idx = np.where(rej_mask)[0]

        # Cap pairs per row
        for p in pref_idx[:3]:
            for q in np.random.choice(rej_idx, min(5, len(rej_idx)), replace=False):
                triplets.append((i, int(p), int(q)))

        if (i + 1) % 25000 == 0:
            print(f"    {i+1}/{n_total} rows, {len(triplets):,} triplets", flush=True)

    print(f"\n  Total triplets: {len(triplets):,}")
    if len(triplets) > 1_500_000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(triplets), 1_500_000, replace=False)
        triplets = [triplets[i] for i in idx]
        print(f"  Sub-sampled: {len(triplets):,}")

    # Distribution
    from collections import Counter
    pref_classes = [t[1] for t in triplets]
    rej_classes = [t[2] for t in triplets]
    top_pref = Counter(pref_classes).most_common(10)
    top_rej = Counter(rej_classes).most_common(10)
    print(f"\n  Top preferred (non-Aves): {[(primary[c], n) for c, n in top_pref]}")
    print(f"  Top rejected: {[(primary[c], sp_taxon[c], n) for c, n in top_rej]}")

    # Step 6: DPO sweep
    print("\n=== Step 6: DPO training ===", flush=True)
    results = [{"variant": "BCE ref", "macro": ref_macro,
                 **{t: float('nan') for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}}]
    pt_ref = per_taxon_macro(Y_ss_ev,
                              torch.sigmoid(ref_model(torch.from_numpy(perch_emb_lab[ev_mask]).to(DEVICE))).detach().cpu().numpy(),
                              sp_taxon)
    results[0].update({k: pt_ref.get(k, float('nan')) for k in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]})

    for beta in [0.3, 1.0, 3.0, 10.0]:
        print(f"\n--- β={beta} ---")
        best, ev_pred, _, best_ep, _ = train_dpo_on_unlabeled(
            unlab_emb, triplets, perch_emb_lab[ev_mask], Y_ss_ev,
            W_init, b_init, ref_model, beta=beta, n_epochs=3, verbose=True
        )
        pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
        results.append({
            "variant": f"RLAIF β={beta}", "macro": best,
            **{t: pt.get(t, float('nan')) for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
        })

    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
