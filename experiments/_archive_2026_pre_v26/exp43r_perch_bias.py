#!/usr/bin/env python3
"""exp43r — Perch inductive bias on Perch-unseen species.

User insight: 31/234 BirdCLEF species are NOT in Perch's 14,795-class training
(mostly 25 Insecta sonotypes 47158son01..25 + 3 Amphibia + 2 Mammalia + 1 Reptilia).
Perch embeddings for windows containing these species should be systematically
biased — mapped to the closest known species region.  iVDFM/iVAE on Perch
features cannot fix this because the same bias sits in the representation.

Tests:
  T1 — Teacher score distribution: unseen-only labeled windows vs seen-only
       (expected: unseen-only max score should be much lower)
  T2 — Nearest-neighbor "confusion target": for each unseen species c, which
       MAPPED species do its windows' 10-NN (in Perch space) most often belong
       to?  Systematic confusion → bias confirmed.
  T3 — Clusterability under bias: impostor-detection AUC (exp43o style) for
       unseen species windows vs seen.  Expected: centroid-based filters fail
       for unseen because teacher never produces consistent c-positive groups.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
OUT = ROOT / "experiments/exp43r_outputs"
OUT.mkdir(exist_ok=True)


def load_data():
    tax = pd.read_csv(DATA / "taxonomy.csv")
    perch_sci = set(open(ROOT / "perch_v2/assets/labels.csv").read().strip().split("\n"))
    tax["in_perch"] = tax["scientific_name"].isin(perch_sci)
    sci2label = dict(zip(tax["scientific_name"], tax["primary_label"]))
    mapped_labels = set(tax[tax.in_perch]["primary_label"].astype(str).tolist())
    unmapped_labels = set(tax[~tax.in_perch]["primary_label"].astype(str).tolist())

    # Perch embeddings + scores
    emb = np.load(EXP43A / "perch_ss_all.npz")["emb"].astype(np.float32)
    scores = np.load(EXP43A / "perch_ss_all.npz")["scores"].astype(np.float32)
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")

    # Labeled SS windows
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    rid2lbls = {r.row_id: r.lbls for _,r in sc_g.iterrows()}

    mask = np.zeros(len(meta), dtype=bool)
    Y = np.zeros((len(meta), len(primary)), dtype=np.uint8)
    for i,rid in enumerate(meta["row_id"].values):
        if rid in rid2lbls:
            mask[i] = True
            for l in rid2lbls[rid]:
                if l in l2i: Y[i, l2i[l]] = 1
    return emb, scores, meta, mask, Y, primary, tax, mapped_labels, unmapped_labels


def main():
    emb, scores, meta, mask, Y, primary, tax, mapped, unmapped = load_data()
    label_to_class = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    label_to_in_perch = dict(zip(tax["primary_label"].astype(str), tax["in_perch"]))

    # Columns per species: is-it-mapped?
    mapped_cols = np.array([label_to_in_perch.get(p, False) for p in primary])
    unmapped_cols = ~mapped_cols
    print(f"Classes: {mapped_cols.sum()} mapped, {unmapped_cols.sum()} unmapped")

    Y_lab = Y[mask]
    scores_lab = scores[mask]
    emb_lab = emb[mask]
    n_labeled = len(Y_lab)

    # ─── Classify each labeled window ─────────────────────────────────
    has_mapped = (Y_lab[:, mapped_cols].sum(1) > 0)
    has_unmapped = (Y_lab[:, unmapped_cols].sum(1) > 0)

    only_mapped = has_mapped & ~has_unmapped
    only_unmapped = ~has_mapped & has_unmapped
    both = has_mapped & has_unmapped
    print(f"\nLabeled SS windows ({n_labeled} total):")
    print(f"  only-mapped species  : {only_mapped.sum()}")
    print(f"  only-unmapped species: {only_unmapped.sum()}")
    print(f"  both mixed           : {both.sum()}")

    # ─── T1: Teacher score distribution by window group ────────────────
    print("\n[T1] Teacher max-score distribution by window group")
    groups = {"only_mapped": only_mapped, "only_unmapped": only_unmapped, "both": both}
    t1 = {}
    for name, m in groups.items():
        if m.sum() == 0: continue
        max_per_win = scores_lab[m].max(1)
        # Also max restricted to mapped cols vs unmapped cols
        max_on_mapped = scores_lab[m][:, mapped_cols].max(1) if mapped_cols.any() else np.array([np.nan])
        max_on_unmapped = scores_lab[m][:, unmapped_cols].max(1) if unmapped_cols.any() else np.array([np.nan])
        t1[name] = {
            "n": int(m.sum()),
            "max_any_mean": float(max_per_win.mean()),
            "max_any_q50": float(np.median(max_per_win)),
            "max_any_q90": float(np.quantile(max_per_win, 0.9)),
            "max_on_mapped_cols_mean": float(max_on_mapped.mean()),
            "max_on_unmapped_cols_mean": float(max_on_unmapped.mean()),
        }
        print(f"  {name:<18} n={int(m.sum()):3d}  "
              f"max_any[mean/q50/q90]={max_per_win.mean():.3f}/{np.median(max_per_win):.3f}/{np.quantile(max_per_win,0.9):.3f}  "
              f"on_mapped={max_on_mapped.mean():.3f}  on_unmapped={max_on_unmapped.mean():.3f}")

    # ─── T2: For each unmapped species, which mapped species are its 10-NN? ──
    print("\n[T2] Unmapped species → confusion targets (mapped) via 10-NN")
    print("  For each unmapped species c with labeled windows, report top-3 mapped species")
    print("  that appear most often in those windows' Perch-space 10-NN.")

    # Normalize embeddings for cosine kNN
    from sklearn.neighbors import NearestNeighbors
    emb_n = emb_lab / (np.linalg.norm(emb_lab, axis=1, keepdims=True) + 1e-8)
    nn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(emb_n)
    _, I = nn.kneighbors(emb_n)

    unmap_idx_to_analyze = np.where(unmapped_cols)[0]
    files_lab = meta["filename"].values[mask]
    t2 = []
    for c in unmap_idx_to_analyze:
        pos_idx = np.where(Y_lab[:, c] == 1)[0]
        if len(pos_idx) < 3: continue
        # Collect neighbors (exclude self + same-file) from each positive window
        neigh_labels = Counter()
        for pi in pos_idx:
            for j in I[pi][1:]:
                if files_lab[j] == files_lab[pi]: continue
                classes_in_j = np.where(Y_lab[j] == 1)[0]
                for cj in classes_in_j:
                    if mapped_cols[cj]:        # only count mapped species as "bias targets"
                        neigh_labels[int(cj)] += 1
        top3 = neigh_labels.most_common(3)
        t2.append({
            "unmapped_species_label": primary[c],
            "scientific_name": tax.loc[tax["primary_label"].astype(str) == primary[c], "scientific_name"].values[0] if (tax["primary_label"].astype(str) == primary[c]).any() else "?",
            "class_name": label_to_class.get(primary[c], "?"),
            "n_positive_labeled_windows": int(len(pos_idx)),
            "top3_mapped_neighbors": [
                {"species": primary[t[0]], "sci": tax.loc[tax["primary_label"].astype(str) == primary[t[0]], "scientific_name"].values[0] if (tax["primary_label"].astype(str) == primary[t[0]]).any() else "?", "count": t[1]}
                for t in top3
            ]
        })
    for r in t2:
        targets = " | ".join(f"{t['sci'][:30]}({t['count']})" for t in r["top3_mapped_neighbors"])
        print(f"  {r['scientific_name'][:25]:<25} ({r['class_name']:<8}, n={r['n_positive_labeled_windows']:2d}) → {targets}")

    # ─── T3: Impostor detection AUC for seen vs unseen species groups ────
    print("\n[T3] Impostor detection AUC (centroid distance, τ=0.3) by seen/unseen")
    # For each class, compute AUC using raw Perch pooled (as in exp43o)
    from sklearn.metrics import roc_auc_score
    def aucs_for_cols(col_mask_name, col_mask):
        aucs = []
        for c in np.where(col_mask)[0]:
            teacher_hi = scores_lab[:, c] > 0.3
            is_pos = (Y_lab[:, c] == 1)
            tp_idx = np.where(teacher_hi & is_pos)[0]
            imp_idx = np.where(teacher_hi & ~is_pos)[0]
            if len(tp_idx) < 5 or len(imp_idx) < 3: continue
            tp_mat = emb_lab[tp_idx]
            labels = np.concatenate([np.zeros(len(tp_idx)), np.ones(len(imp_idx))])
            dists = []
            for i in range(len(tp_idx)):
                mask_loo = np.ones(len(tp_idx), bool); mask_loo[i] = False
                cent = tp_mat[mask_loo].mean(0, keepdims=True)
                d = 1.0 - (emb_lab[tp_idx[i]] / (np.linalg.norm(emb_lab[tp_idx[i]])+1e-8)) @ (cent / (np.linalg.norm(cent)+1e-8)).T
                dists.append(float(np.asarray(d).ravel()[0]))
            for j in imp_idx:
                cent = tp_mat.mean(0, keepdims=True)
                d = 1.0 - (emb_lab[j] / (np.linalg.norm(emb_lab[j])+1e-8)) @ (cent / (np.linalg.norm(cent)+1e-8)).T
                dists.append(float(np.asarray(d).ravel()[0]))
            try:
                auc = roc_auc_score(labels, dists)
                aucs.append({"class_idx": int(c), "label": primary[c], "auc": float(auc),
                             "n_tp": len(tp_idx), "n_imp": len(imp_idx)})
            except Exception:
                pass
        return aucs

    mapped_aucs = aucs_for_cols("mapped", mapped_cols)
    unmapped_aucs = aucs_for_cols("unmapped", unmapped_cols)
    def summ(lst):
        if not lst: return {}
        a = [x["auc"] for x in lst]
        return {"n_classes_eval": len(lst), "mean_auc": float(np.mean(a)),
                "median_auc": float(np.median(a)), "frac_above_0.7": float(np.mean([x>0.7 for x in a]))}

    print(f"  mapped classes  : {summ(mapped_aucs)}")
    print(f"  unmapped classes: {summ(unmapped_aucs)}")
    if unmapped_aucs:
        print(f"\n  Per-unmapped detail:")
        for r in sorted(unmapped_aucs, key=lambda x: -x["auc"])[:10]:
            print(f"    {r['label']:<10} AUC={r['auc']:.3f}  n_tp={r['n_tp']} n_imp={r['n_imp']}")

    # Save
    out = {
        "mapped_count": int(mapped_cols.sum()),
        "unmapped_count": int(unmapped_cols.sum()),
        "window_groups": {g: int(m.sum()) for g, m in groups.items()},
        "T1_teacher_scores": t1,
        "T2_confusion_targets_unmapped": t2,
        "T3_impostor_AUC_by_group": {
            "mapped_summary": summ(mapped_aucs),
            "unmapped_summary": summ(unmapped_aucs),
            "unmapped_detail": unmapped_aucs,
        },
    }
    with open(OUT / "perch_bias.json", "w") as fp:
        json.dump(out, fp, indent=2, default=float)
    print(f"\nSaved → {OUT}/perch_bias.json")


if __name__ == "__main__":
    main()
