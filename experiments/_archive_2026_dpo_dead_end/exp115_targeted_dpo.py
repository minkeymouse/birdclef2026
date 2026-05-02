#!/usr/bin/env python3
"""exp115 — Targeted hard-DPO variants.

6 strategies for restricting hard-pair mining:
  1. all_classes     — baseline (= exp114)
  2. rare_only       — true positive in non-Aves (72 classes)
  3. unmapped_only   — true positive in 31 Perch-unmapped species
  4. cross_taxon     — pairs where true_pos.taxon ≠ false_pos.taxon
  5. within_taxon    — pairs where true_pos.taxon == false_pos.taxon (sister)
  6. mammalia_only   — true positive in 8 Mammalia species

Hypothesis: rare-class / unmapped targeting concentrates capacity on the
species where (a) Perch alone has no prior, (b) BCE relies entirely on
sparse labels. DPO ranking signal there is most informative.

Aves (162/234) get very high coverage already from Perch xeno-canto pretraining;
hard pairs in Aves teach SITE-CONDITIONAL subtle features → poor transfer.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        ROOT, N_CLS, DATA)
from _lib.eval_metrics import macro_auc, per_taxon_macro
from exp106_pnew_hybrid import build_perch_init, PerchHybrid
from exp113_pnew3_dpo import train_bce_reference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERCH_DIR = ROOT / "perch_v2"


def get_perch_mapped_set():
    """Return set of class indices that ARE mapped to Perch's 14k."""
    tax = pd.read_csv(DATA / "taxonomy.csv")
    species = sorted(tax["primary_label"].astype(str).tolist())
    sp2idx = {s: i for i, s in enumerate(species)}
    sci2pl = dict(zip(tax["scientific_name"], tax["primary_label"]))
    perch_labels = open(PERCH_DIR / "assets" / "labels.csv").read().strip().split("\n")
    mapped = set()
    for pname in perch_labels:
        if pname in sci2pl and sci2pl[pname] in sp2idx:
            mapped.add(sp2idx[sci2pl[pname]])
    return mapped


def mine_hard_pairs_targeted(model, X_train, Y_train, sp_taxon, mapped_set,
                                target="all_classes", margin=0.5,
                                max_pairs_per_row=20, batch_size=512):
    """Mine hard (row, true_pos, false_pos) triplets with restriction."""
    model.eval()
    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    Y_t = torch.from_numpy(Y_train.astype(np.float32))
    n = len(X_t)
    sp_taxon_arr = np.array(sp_taxon)
    triplets = []

    # Pre-compute restriction sets
    pos_allowed = np.ones(N_CLS, dtype=bool)
    if target == "rare_only":
        pos_allowed = sp_taxon_arr != "Aves"
    elif target == "unmapped_only":
        pos_allowed = np.array([i not in mapped_set for i in range(N_CLS)])
    elif target == "mammalia_only":
        pos_allowed = sp_taxon_arr == "Mammalia"
    # For cross/within taxon, need both species' taxons
    same_taxon = sp_taxon_arr[:, None] == sp_taxon_arr[None, :]  # (234, 234) bool

    with torch.no_grad():
        for s in range(0, n, batch_size):
            x_b = X_t[s:s+batch_size]
            logits_b = model(x_b).cpu()
            y_b = Y_t[s:s+batch_size]
            B = len(x_b)
            for i in range(B):
                pos_idx = (y_b[i] > 0).nonzero(as_tuple=False).squeeze(-1)
                neg_idx = (y_b[i] == 0).nonzero(as_tuple=False).squeeze(-1)
                if len(pos_idx) == 0 or len(neg_idx) == 0: continue

                # Filter pos_idx by pos_allowed
                pos_filter = pos_allowed[pos_idx.numpy()]
                if not pos_filter.any(): continue
                pos_idx = pos_idx[torch.from_numpy(pos_filter)]
                if len(pos_idx) == 0: continue

                pos_logits = logits_b[i, pos_idx]
                neg_logits = logits_b[i, neg_idx]
                diffs = neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1)  # (P, N)
                hard_mask = diffs > -margin

                # Apply taxon restriction
                if target in ("cross_taxon", "within_taxon"):
                    tax_match = same_taxon[pos_idx.numpy()][:, neg_idx.numpy()]
                    if target == "cross_taxon":
                        hard_mask = hard_mask & torch.from_numpy(~tax_match)
                    else:
                        hard_mask = hard_mask & torch.from_numpy(tax_match)

                hard_p, hard_n = torch.where(hard_mask)
                if len(hard_p) == 0: continue
                violations = diffs[hard_p, hard_n]
                top = torch.argsort(violations, descending=True)[:max_pairs_per_row]
                for k in top.tolist():
                    triplets.append((s + i, int(pos_idx[hard_p[k]]), int(neg_idx[hard_n[k]])))
    return triplets


def train_dpo_on_triplets(X_train, Y_train, src_weight, X_eval, Y_eval, W_init, b_init,
                            ref_model, hard_triplets, beta=1.0, n_epochs=6, lr=5e-4, verbose=False):
    """DPO training on pre-mined triplets (same as exp114 train_hard_dpo)."""
    if len(hard_triplets) == 0:
        return float('nan'), None, float('nan'), -1

    policy = PerchHybrid(W_init, b_init).to(DEVICE)
    policy.load_state_dict(ref_model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/10)

    X_t = torch.from_numpy(X_train.astype(np.float32)).to(DEVICE)
    Xev_t = torch.from_numpy(X_eval.astype(np.float32)).to(DEVICE)
    rows_t = torch.tensor([t[0] for t in hard_triplets], dtype=torch.long, device=DEVICE)
    pos_t = torch.tensor([t[1] for t in hard_triplets], dtype=torch.long, device=DEVICE)
    neg_t = torch.tensor([t[2] for t in hard_triplets], dtype=torch.long, device=DEVICE)
    n_triplets = len(hard_triplets)

    BATCH = 1024
    best_auc = 0.0; best_pred = None; best_ep = -1

    policy.eval()
    with torch.no_grad():
        ev_pred0 = torch.sigmoid(policy(Xev_t)).cpu().numpy()
    macro0, _ = macro_auc(Y_eval, ev_pred0)
    best_auc = macro0; best_pred = ev_pred0; best_ep = -1
    if verbose: print(f"  ep -1 (init)  macro {macro0:.4f}")

    for ep in range(n_epochs):
        policy.train()
        perm = torch.randperm(n_triplets, device=DEVICE)
        ep_loss = 0.0; nb = 0
        for s in range(0, n_triplets, BATCH):
            idx = perm[s:s+BATCH]
            r = rows_t[idx]; p_cls = pos_t[idx]; n_cls = neg_t[idx]
            x = X_t[r]
            logits_p = policy(x)
            with torch.no_grad():
                logits_r = ref_model(x)
            log_p_pos = F.logsigmoid(logits_p.gather(1, p_cls.unsqueeze(1)).squeeze(1))
            log_p_neg = F.logsigmoid(logits_p.gather(1, n_cls.unsqueeze(1)).squeeze(1))
            log_r_pos = F.logsigmoid(logits_r.gather(1, p_cls.unsqueeze(1)).squeeze(1))
            log_r_neg = F.logsigmoid(logits_r.gather(1, n_cls.unsqueeze(1)).squeeze(1))
            margin = beta * ((log_p_pos - log_r_pos) - (log_p_neg - log_r_neg))
            loss = -F.logsigmoid(margin).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        policy.eval()
        with torch.no_grad():
            ev_pred = torch.sigmoid(policy(Xev_t)).cpu().numpy()
        macro, _ = macro_auc(Y_eval, ev_pred)
        if macro > best_auc:
            best_auc = macro; best_pred = ev_pred; best_ep = ep
        if verbose:
            print(f"  ep {ep:02d}  loss {ep_loss/nb:.4f}  macro {macro:.4f}")

    return best_auc, best_pred, macro0, best_ep


def main():
    print("=== exp115: Targeted DPO variants ===\n", flush=True)

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

    tr_mask = sc_g.split.values == "train"
    ev_mask = sc_g.split.values == "eval"
    Y_ss_ev = Y[ev_mask].astype(np.float32)

    X_train = np.concatenate([ta_emb[valid], perch_emb_ss[tr_mask]], axis=0)
    Y_train = np.concatenate([Y_ta[valid], Y[tr_mask].astype(np.float32)], axis=0)
    src_w = np.concatenate([np.ones(valid.sum()), np.full(tr_mask.sum(), 5.0)])

    print(f"  Train: TA {valid.sum()} + SS {tr_mask.sum()} = {len(X_train)}\n")

    # Step 1: BCE reference
    print("=== BCE reference ===", flush=True)
    ref_model, ref_macro = train_bce_reference(
        X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
        W_init, b_init, n_epochs=15
    )
    print(f"  BCE macro: {ref_macro:.4f}\n")

    pt_ref = per_taxon_macro(Y_ss_ev,
                              torch.sigmoid(ref_model(torch.from_numpy(perch_emb_ss[ev_mask]).to(DEVICE))).detach().cpu().numpy(),
                              sp_taxon)
    print(f"  Ref per-taxon: Aves {pt_ref['Aves']:.4f}, Amphib {pt_ref['Amphibia']:.4f}, "
          f"Insecta {pt_ref['Insecta']:.4f}, Mam {pt_ref['Mammalia']:.4f}, Rept {pt_ref['Reptilia']:.4f}\n")

    mapped_set = get_perch_mapped_set()
    print(f"  Perch-mapped classes: {len(mapped_set)}/{N_CLS}\n")

    # Step 2: Test all variants
    variants = [
        "all_classes",
        "rare_only",
        "unmapped_only",
        "cross_taxon",
        "within_taxon",
        "mammalia_only",
    ]

    results = [{"variant": "BCE ref", "n_triplets": 0, "macro": ref_macro,
                 **{t: pt_ref.get(t, float('nan')) for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}}]

    saved_preds = {"BCE ref": torch.sigmoid(ref_model(torch.from_numpy(perch_emb_ss[ev_mask]).to(DEVICE))).detach().cpu().numpy()}

    for variant in variants:
        print(f"=== Variant: {variant} ===", flush=True)
        triplets = mine_hard_pairs_targeted(
            ref_model, X_train, Y_train, sp_taxon, mapped_set,
            target=variant, margin=0.5, max_pairs_per_row=20
        )
        n_unique_rows = len(set(t[0] for t in triplets))
        n_pos_classes = len(set(t[1] for t in triplets))
        n_neg_classes = len(set(t[2] for t in triplets))
        print(f"  Mined: {len(triplets)} triplets, {n_unique_rows} rows, "
              f"{n_pos_classes} pos classes, {n_neg_classes} neg classes")

        if len(triplets) < 100:
            print(f"  Too few triplets, skipping DPO")
            results.append({"variant": variant, "n_triplets": len(triplets), "macro": ref_macro,
                              **{t: pt_ref.get(t, float('nan')) for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}})
            continue

        best, ev_pred, _, best_ep = train_dpo_on_triplets(
            X_train, Y_train, src_w, perch_emb_ss[ev_mask], Y_ss_ev,
            W_init, b_init, ref_model, triplets, beta=1.0, n_epochs=6, verbose=False
        )
        pt = per_taxon_macro(Y_ss_ev, ev_pred, sp_taxon)
        rec = {"variant": variant, "n_triplets": len(triplets), "macro": best,
                "best_ep": best_ep,
                **{t: pt.get(t, float('nan')) for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}}
        results.append(rec)
        saved_preds[variant] = ev_pred
        print(f"  best macro {best:.4f} @ ep{best_ep}, "
              f"Δvs ref {best-ref_macro:+.4f}, "
              f"Aves {pt.get('Aves', float('nan')):.4f} Amphib {pt.get('Amphibia', float('nan')):.4f} "
              f"Insecta {pt.get('Insecta', float('nan')):.4f} Mam {pt.get('Mammalia', float('nan')):.4f}\n", flush=True)

    df = pd.DataFrame(results)
    print("\n=== Summary (122 eval) ===")
    print(df.sort_values("macro", ascending=False).to_string(index=False))

    # Save best non-baseline variant for blend testing
    df_filtered = df[df.variant != "BCE ref"]
    if len(df_filtered) > 0:
        best_var = df_filtered.loc[df_filtered.macro.idxmax(), "variant"]
        print(f"\nBest variant: {best_var}")


if __name__ == "__main__":
    main()
