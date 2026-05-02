#!/usr/bin/env python3
"""exp86 — Identify cross-site rare classes that are good specialist targets.

Criteria:
  - n_train_audio ≥ 30 (cross-site source data exists)
  - n_labeled_SS_sites ≥ 2 (multi-site labeled, less site shortcut risk)
  - v33 OOF per-class AUC on labeled SS < 0.75 (room to improve)
  - n_pos in labeled-SS eval ≥ 3 (measurable AUC)
  - taxon = Aves (mapped, in Perch vocab — avoid 27 sonotype trap)

Outputs candidate list with per-class AUC, n_train_audio, n_sites,
external clip count.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, ROOT, DATA, N_CLS)
from _lib.eval_metrics import per_class_auc

# Reuse v33 builder + exp50/exp59/etc cached
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend


def get_cached(name):
    return np.load(EXP80 / name)["scores"]


def main():
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_cached("exp50_scores_labeled.npz")

    # Build v33 baseline on all rows
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # Per-class AUC on EVAL split only
    ev_mask = sc_g.split.values == "eval"
    Y_ev = Y[ev_mask]
    P_ev = v33[ev_mask]
    aucs = per_class_auc(Y_ev, P_ev)
    # Per-class on TRAIN split (for OOF-ish view)
    tr_mask = sc_g.split.values == "train"
    Y_tr = Y[tr_mask]
    P_tr = v33[tr_mask]
    aucs_tr = per_class_auc(Y_tr, P_tr)

    # n_train_audio counts (from train.csv)
    train_csv = pd.read_csv(DATA / "train.csv")
    n_train_audio = train_csv["primary_label"].astype(str).value_counts()

    # Site count per class in labeled SS
    site_count_per_class = {}
    for c in range(N_CLS):
        sites = set()
        for i in np.where(Y[:, c] == 1)[0]:
            sites.add(sc_g.iloc[i].site)
        site_count_per_class[c] = len(sites)

    # External clip count per species
    ext_dir = ROOT / "data" / "external"
    ext_count_per_class = {}
    for c, lbl in enumerate(primary):
        sp_dir = ext_dir / str(lbl)
        if not sp_dir.exists():
            ext_count_per_class[c] = 0
            continue
        ext_count_per_class[c] = len(list(sp_dir.glob("*.ogg")))

    rows = []
    for c, lbl in enumerate(primary):
        if sp_taxon[c] != "Aves": continue   # focus on mapped Aves
        n_ev_pos = int(Y_ev[:, c].sum())
        n_tr_pos = int(Y_tr[:, c].sum())
        if n_ev_pos < 1: continue
        rows.append({
            "label": lbl, "taxon": sp_taxon[c],
            "n_train_audio": int(n_train_audio.get(lbl, 0)),
            "n_labeled_ss_pos": n_ev_pos + n_tr_pos,
            "n_eval_pos": n_ev_pos,
            "n_sites_in_ss": site_count_per_class[c],
            "v33_AUC_eval": aucs.get(c, np.nan),
            "v33_AUC_train": aucs_tr.get(c, np.nan),
            "n_ext_clips": ext_count_per_class[c],
        })
    df = pd.DataFrame(rows)
    print(f"Total mapped Aves with eval positives: {len(df)}")

    # Filter: cross-site rare candidates
    cand = df[
        (df.n_train_audio >= 30) &
        (df.n_sites_in_ss >= 2) &
        (df.v33_AUC_eval < 0.75) &
        (df.n_eval_pos >= 3)
    ].copy()
    cand = cand.sort_values("v33_AUC_eval")

    print(f"\n=== Cross-site rare Aves candidates (≥30 train_audio + ≥2 SS sites + AUC<0.75 + ≥3 eval pos) ===")
    print(cand[["label", "n_train_audio", "n_sites_in_ss", "n_labeled_ss_pos",
                 "n_eval_pos", "v33_AUC_eval", "v33_AUC_train", "n_ext_clips"]].to_string(index=False))

    # Looser cut: AUC < 0.85 OR sites ≥ 3
    cand2 = df[
        (df.n_train_audio >= 50) &
        (df.n_sites_in_ss >= 2) &
        (df.v33_AUC_eval < 0.85) &
        (df.n_eval_pos >= 3)
    ].copy().sort_values("v33_AUC_eval")
    print(f"\n=== Looser candidates (n_TA≥50, AUC<0.85): {len(cand2)} ===")
    print(cand2[["label", "n_train_audio", "n_sites_in_ss",
                  "n_eval_pos", "v33_AUC_eval", "n_ext_clips"]].to_string(index=False))

    # Save
    df.to_csv(EXP80 / "exp86_aves_class_audit.csv", index=False)
    cand.to_csv(EXP80 / "exp86_specialist_targets.csv", index=False)
    print(f"\nSaved → {EXP80}/exp86_specialist_targets.csv")


if __name__ == "__main__":
    main()
