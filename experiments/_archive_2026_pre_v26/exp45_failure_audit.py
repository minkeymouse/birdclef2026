#!/usr/bin/env python3
"""exp45 — Pipeline failure-mode audit on 11 held-out labeled SS files.

Question: beyond the 27 double-blind species (known broken), which OTHER
species does our v12-style pipeline fail on, and what patterns explain it?

Inputs we have:
  - exp43a_outputs/perch_ss_all.npz : Perch raw scores (127896, 234)
  - experiments/exp29_outputs/val_scores.npz : SED29 preds (708 rows, old split)
  - experiments/exp41f_outputs/val_scores_full.npz : SED41f preds (708, old split)
  - data/birdclef-2026/{taxonomy,train,train_soundscapes_labels}.csv

Problem: exp29/41f are on OLD 59-file cache, not new 66-file split.
The 11 held-out (from 66) partially overlaps with the 59.  We use the rows
that exist in both (by row_id).

Analyses:
  T1. Per-class AUC on 11 held-out (only mapped 207 meaningful; 27 unmapped stays ~0.5)
  T2. Stratify by (taxon, n_train_audio, n_labeled_SS_positives, site_diversity)
  T3. Bottom-20% species: confusion pairs (which other species does model predict instead)
  T4. Perch vs SED29 agreement on per-class basis (Pearson) — low agreement = diversity opportunity
  T5. Per-file & per-site error concentration
  T6. Multi-label vs single-label accuracy
  T7. Compare Perch-alone vs SED29-alone vs blend — which species benefits from blend?

Output: experiments/exp45_outputs/audit.json + summary.md
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP41F = ROOT / "experiments/exp41f_outputs"
EXP28 = ROOT / "experiments/exp28_outputs"
OUT = ROOT / "experiments/exp45_outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
EVAL_N_FILES = 11


def build_eval_frame():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["site"] = sc_g["filename"].str.extract(r"_(S\d+)_")
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)

    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique())
    rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)

    l2i = {c: i for i, c in enumerate(primary)}
    Y = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["lbls"]):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_eval, Y, primary


def align_perch_scores(sc_eval):
    """Perch raw scores (logits) from exp43a, aligned to eval rows."""
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    scores = np.load(EXP43A / "perch_ss_all.npz")["scores"]
    out = np.zeros((len(sc_eval), scores.shape[1]), dtype=np.float32)
    missing = 0
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = scores[j]
        else: missing += 1
    return out, missing


def try_align_sed(sc_eval, sed_npz_path, old_n_rows=708):
    """SED preds (old 59-file cache). Rows match 59 labeled files, need overlap."""
    if not sed_npz_path.exists():
        return None, 0, "npz missing"
    d = np.load(sed_npz_path)
    preds = d["preds"].astype(np.float32)
    # We need to know the row_ids of the old 59-file cache.
    old_meta = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(old_meta) != preds.shape[0]:
        return None, 0, f"shape {preds.shape} vs meta {len(old_meta)}"
    old_rid2i = {r: i for i, r in enumerate(old_meta["row_id"].values)}
    out = np.full((len(sc_eval), preds.shape[1]), np.nan, dtype=np.float32)
    hit = 0
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = old_rid2i.get(rid, -1)
        if j >= 0:
            out[i] = preds[j]; hit += 1
    return out, hit, "ok"


def zs(X, axis=0):
    m = X.mean(axis=axis, keepdims=True); s = X.std(axis=axis, keepdims=True) + 1e-8
    return (X - m) / s


def per_class_auc(Y, P):
    """Return dict c → {auc, n_pos, n_neg}. Skip classes all-pos or all-neg."""
    out = {}
    for c in range(Y.shape[1]):
        y = Y[:, c].astype(int)
        p = P[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
        if not np.isfinite(p).all(): continue
        try:
            out[c] = {"auc": float(roc_auc_score(y, p)),
                      "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
        except Exception: pass
    return out


def stratify(aucs, tax, primary, train_audio_counts, labeled_ss_positives, species_site_count):
    """Table of per-class metrics + structural features."""
    rows = []
    for c, v in aucs.items():
        lbl = str(primary[c])
        tax_row = tax[tax["primary_label"].astype(str) == lbl]
        class_name = tax_row["class_name"].iloc[0] if len(tax_row) else "?"
        in_perch = bool(tax_row["in_perch"].iloc[0]) if len(tax_row) else False
        rows.append({
            "class_idx": c, "label": lbl, "taxon": class_name, "in_perch": in_perch,
            "auc": v["auc"], "n_pos_eval": v["n_pos"], "n_neg_eval": v["n_neg"],
            "n_train_audio": int(train_audio_counts.get(lbl, 0)),
            "n_labeled_ss_pos": int(labeled_ss_positives.get(lbl, 0)),
            "n_sites": int(species_site_count.get(lbl, 0)),
        })
    return pd.DataFrame(rows)


def confusion_pairs(Y, P, bottom_idx, primary, k=3):
    """For each bottom species c, find top-k OTHER species that model confuses it with."""
    out = {}
    for c in bottom_idx:
        pos_rows = np.where(Y[:, c] == 1)[0]
        if len(pos_rows) == 0: continue
        # rank of c among all classes on positive rows
        other_higher = Counter()
        for r in pos_rows:
            order = np.argsort(-P[r])
            for cc in order:
                if cc == c: break
                other_higher[cc] += 1
        top = other_higher.most_common(k)
        out[str(primary[c])] = [
            {"confused_with": str(primary[cc]), "times_ranked_higher": n, "out_of": int(len(pos_rows))}
            for cc, n in top
        ]
    return out


def main():
    print("=" * 60)
    print("exp45 — Failure Audit")
    print("=" * 60)

    sc_eval, Y, primary = build_eval_frame()
    print(f"\nEval frame: {len(sc_eval)} rows across {sc_eval['filename'].nunique()} files, {Y.shape[1]} classes")
    print(f"Eval sites: {sorted(sc_eval['site'].unique())}")

    # Metadata features
    tax = pd.read_csv(DATA / "taxonomy.csv")
    perch_sci = set(open(ROOT / "perch_v2/assets/labels.csv").read().strip().split("\n"))
    tax["in_perch"] = tax["scientific_name"].isin(perch_sci)
    train_audio_counts = pd.read_csv(DATA / "train.csv")["primary_label"].astype(str).value_counts().to_dict()
    sc_full = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sc_full["site"] = sc_full["filename"].str.extract(r"_(S\d+)_")
    labeled_ss_positives = Counter()
    species_sites = {}
    for _, r in sc_full.iterrows():
        labels = [t.strip() for t in str(r.primary_label).split(";") if t.strip()]
        for l in labels:
            labeled_ss_positives[l] += 1
            species_sites.setdefault(l, set()).add(r.site)
    species_site_count = {k: len(v) for k, v in species_sites.items()}

    # Perch scores
    perch, missing = align_perch_scores(sc_eval)
    print(f"\nPerch aligned: {len(sc_eval) - missing}/{len(sc_eval)} rows")
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    perch_prob = sigmoid(perch)

    # SED29
    sed29, sed29_hit, sed29_msg = try_align_sed(sc_eval, EXP29 / "val_scores.npz")
    print(f"SED29: {sed29_msg}, aligned {sed29_hit} rows")

    # SED41f
    sed41f, sed41f_hit, sed41f_msg = try_align_sed(sc_eval, EXP41F / "val_scores_full.npz")
    print(f"SED41f: {sed41f_msg}, aligned {sed41f_hit} rows")

    # v12 analog: 0.80*Perch + 0.20*SED29 (z-blend) + Gauss sigma=0.5
    if sed29 is not None and sed29_hit > 0:
        # Fill missing rows of SED29 with 0 (no info) for blending
        sed29_filled = np.nan_to_num(sed29, nan=0.0)
        v12_z = 0.80 * zs(perch_prob) + 0.20 * zs(sed29_filled)
    else:
        print("WARNING: SED29 unavailable, using Perch-only as v12 analog.")
        v12_z = zs(perch_prob)
    # Gauss smooth across windows within each file (file-grouped)
    v12_smoothed = np.zeros_like(v12_z)
    for fname in sc_eval["filename"].unique():
        mask = (sc_eval["filename"] == fname).values
        block = v12_z[mask]
        for c in range(block.shape[1]):
            v12_smoothed[mask, c] = gaussian_filter1d(block[:, c], sigma=0.5, mode="nearest")

    # ========== T1: Per-class AUC =========
    print("\n[T1] Per-class AUC comparison")
    aucs_perch = per_class_auc(Y, perch_prob)
    aucs_v12 = per_class_auc(Y, v12_smoothed)
    aucs_sed29 = per_class_auc(Y, np.nan_to_num(sed29, nan=0.0)) if sed29 is not None else {}
    aucs_sed41f = per_class_auc(Y, np.nan_to_num(sed41f, nan=0.0)) if sed41f is not None else {}

    def macro(d, keys=None):
        if keys: d = {k: d[k] for k in keys if k in d}
        return float(np.mean([v["auc"] for v in d.values()])) if d else 0.0

    all_keys = set(aucs_v12.keys()) | set(aucs_perch.keys())
    print(f"  n_eval classes: {len(all_keys)}")
    print(f"  macro: Perch={macro(aucs_perch):.4f}  v12={macro(aucs_v12):.4f}  "
          f"SED29={macro(aucs_sed29):.4f}  SED41f={macro(aucs_sed41f):.4f}")

    # ========== T2: Stratify =========
    df = stratify(aucs_v12, tax, primary, train_audio_counts, labeled_ss_positives, species_site_count)
    df = df.sort_values("auc").reset_index(drop=True)

    print("\n[T2] Stratification by taxon:")
    for tx, g in df.groupby("taxon"):
        print(f"  {tx:<10}  n={len(g):3d}  mean_auc={g.auc.mean():.4f}  median={g.auc.median():.4f}  min={g.auc.min():.4f}")

    print("\n  Stratification by in_perch:")
    for ip, g in df.groupby("in_perch"):
        print(f"  in_perch={ip}  n={len(g):3d}  mean_auc={g.auc.mean():.4f}  min={g.auc.min():.4f}")

    print("\n  Stratification by train_audio count bucket:")
    df["ta_bucket"] = pd.cut(df["n_train_audio"], bins=[-1, 0, 10, 50, 200, 10000], labels=["0", "1-10", "11-50", "51-200", "200+"])
    for b, g in df.groupby("ta_bucket", observed=True):
        print(f"  ta={str(b):<8}  n={len(g):3d}  mean_auc={g.auc.mean():.4f}")

    print("\n  Stratification by labeled_SS_positives:")
    df["lss_bucket"] = pd.cut(df["n_labeled_ss_pos"], bins=[-1, 0, 5, 20, 100, 100000], labels=["0", "1-5", "6-20", "21-100", "100+"])
    for b, g in df.groupby("lss_bucket", observed=True):
        print(f"  lss={str(b):<8}  n={len(g):3d}  mean_auc={g.auc.mean():.4f}")

    print("\n  Stratification by n_sites (labeled SS site diversity):")
    df["site_bucket"] = pd.cut(df["n_sites"], bins=[-1, 0, 1, 2, 3, 10], labels=["0", "1", "2", "3", "4+"])
    for b, g in df.groupby("site_bucket", observed=True):
        print(f"  sites={str(b):<8}  n={len(g):3d}  mean_auc={g.auc.mean():.4f}")

    # ========== T3: Bottom 20% confusion =========
    bottom_n = max(1, len(df) // 5)
    print(f"\n[T3] Bottom-20% species ({bottom_n} classes), confusion pairs:")
    bottom = df.head(bottom_n)
    bottom_idx = bottom["class_idx"].tolist()
    print(f"  Bottom AUC range: [{bottom.auc.min():.3f}, {bottom.auc.max():.3f}]")
    print(f"  Taxa: {bottom['taxon'].value_counts().to_dict()}")
    print(f"  in_perch: {bottom['in_perch'].value_counts().to_dict()}")
    print(f"  n_sites: {bottom['n_sites'].value_counts().to_dict()}")
    confusion = confusion_pairs(Y, v12_smoothed, bottom_idx, primary, k=3)
    for sp, pairs in list(confusion.items())[:10]:
        line = " | ".join(f"{p['confused_with']}({p['times_ranked_higher']}/{p['out_of']})" for p in pairs)
        auc_sp = df.loc[df["label"] == sp, "auc"].iloc[0]
        taxon_sp = df.loc[df["label"] == sp, "taxon"].iloc[0]
        print(f"    {sp:<12} AUC={auc_sp:.3f} ({taxon_sp:<8}) → {line}")

    # ========== T4: Perch vs SED29 per-class agreement =========
    print("\n[T4] Perch vs SED29 per-class correlation (low = diversity opportunity):")
    if sed29 is not None:
        correlations = {}
        for c in aucs_v12:
            if c in aucs_perch and c in aucs_sed29:
                try:
                    r, _ = pearsonr(perch_prob[:, c], sed29[:, c])
                    correlations[c] = r
                except Exception:
                    pass
        if correlations:
            rs = np.array(list(correlations.values()))
            print(f"  Pearson distribution: mean={rs.mean():+.3f}  median={np.median(rs):+.3f}  q10={np.quantile(rs,.1):+.3f}")
            print(f"  Low-corr classes (< 0.1 correlation, SED29 adds independent signal):")
            low = sorted(correlations.items(), key=lambda x: x[1])[:10]
            for c, r in low:
                auc_c = aucs_v12[c]["auc"]
                sed29_auc = aucs_sed29[c]["auc"]
                perch_auc = aucs_perch[c]["auc"]
                print(f"    {str(primary[c]):<12}  Perch={perch_auc:.3f}  SED29={sed29_auc:.3f}  v12={auc_c:.3f}  Pearson={r:+.3f}")

    # ========== T5: Per-file error concentration =========
    print("\n[T5] Per-file macro AUC:")
    for fname in sc_eval["filename"].unique():
        mask = (sc_eval["filename"] == fname).values
        Y_f = Y[mask]; P_f = v12_smoothed[mask]
        per = per_class_auc(Y_f, P_f)
        n_cls_with = sum(1 for v in per.values() if v["n_pos"] > 0)
        macro_f = float(np.mean([v["auc"] for v in per.values()])) if per else 0.0
        site = sc_eval.loc[sc_eval["filename"] == fname, "site"].iloc[0]
        n_pos_total = Y_f.sum()
        print(f"  {fname[:40]}... site={site}  n_cls={n_cls_with:3d}  macro_auc={macro_f:.4f}  n_pos={n_pos_total}")

    # ========== T6: Multi-label analysis =========
    print("\n[T6] Per-window accuracy breakdown:")
    n_per_row = Y.sum(1)
    print(f"  Rows: total {len(Y)}, with 0 pos {(n_per_row==0).sum()}, 1 pos {(n_per_row==1).sum()}, "
          f"2 pos {(n_per_row==2).sum()}, 3+ pos {(n_per_row>=3).sum()}")

    # ========== T7: Contribution analysis (Perch vs SED29 vs blend) =========
    print("\n[T7] Where does v12 blend beat Perch-alone?")
    if sed29 is not None:
        deltas = []
        for c in aucs_v12:
            if c in aucs_perch:
                d = aucs_v12[c]["auc"] - aucs_perch[c]["auc"]
                deltas.append((c, d))
        wins = sum(1 for _, d in deltas if d > 0.01)
        losses = sum(1 for _, d in deltas if d < -0.01)
        ties = len(deltas) - wins - losses
        print(f"  v12 vs Perch alone: wins {wins}, ties {ties}, losses {losses} of {len(deltas)} classes")
        print(f"  mean Δ = {np.mean([d for _, d in deltas]):+.4f}")
        # Top gain classes
        top_gains = sorted(deltas, key=lambda x: -x[1])[:10]
        print(f"\n  Top-10 blend-helps species:")
        for c, d in top_gains:
            print(f"    {str(primary[c]):<12}  Δ={d:+.4f}  (Perch {aucs_perch[c]['auc']:.3f} → v12 {aucs_v12[c]['auc']:.3f})")

    # Save artifacts
    df.to_csv(OUT / "per_class.csv", index=False)
    with open(OUT / "audit.json", "w") as fp:
        json.dump({
            "macro": {"Perch": macro(aucs_perch), "v12": macro(aucs_v12),
                      "SED29": macro(aucs_sed29), "SED41f": macro(aucs_sed41f)},
            "by_taxon": df.groupby("taxon")["auc"].agg(["count","mean","median","min"]).to_dict(),
            "bottom_20pct_confusion": confusion,
        }, fp, indent=2, default=float)
    print(f"\nSaved: {OUT}/per_class.csv, {OUT}/audit.json")
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    # Top takeaways
    mapped = df[df.in_perch]
    unmapped = df[~df.in_perch]
    print(f"Mapped (207): mean AUC = {mapped.auc.mean():.4f}, "
          f"bottom 10% = {mapped.auc.quantile(0.1):.4f}")
    if len(unmapped):
        print(f"Unmapped (27): mean AUC = {unmapped.auc.mean():.4f}, "
              f"bottom 10% = {unmapped.auc.quantile(0.1):.4f}")
    print(f"\nClasses with AUC < 0.5 (worse than random): {(df.auc < 0.5).sum()}")
    print(f"Classes with AUC < 0.7: {(df.auc < 0.7).sum()}")
    print(f"Classes with AUC > 0.95: {(df.auc > 0.95).sum()}")


if __name__ == "__main__":
    main()
