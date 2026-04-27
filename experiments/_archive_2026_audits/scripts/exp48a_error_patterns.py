#!/usr/bin/env python3
"""exp48a — New error-pattern deep dive on 11-file held-out.

Explores patterns NOT already covered by exp43r / exp45:
  1. per-SITE error distribution (does v12 fail per-site?)
  2. hour-of-day patterns (diurnal/nocturnal species misprediction)
  3. prediction confidence distribution on errors (confident-wrong vs low-signal)
  4. sibling-Aves confusion pairs with highest co-prediction (for exp48d)
  5. frequency-band signature per misclassified species
  6. within-file temporal coherence (does pred flip unreasonably between windows?)
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
OUT = ROOT / "experiments/exp48_outputs"
OUT.mkdir(exist_ok=True)
SEED = 42; EVAL_N = 11

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")


def parse_site_hour(fn):
    m = FNAME_RE.match(fn)
    if not m: return None, None
    site = m.group(2)
    hour = int(m.group(4)[:2])
    return site, hour


def build_eval():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g[["site", "hour"]] = sc_g["filename"].apply(lambda f: pd.Series(parse_site_hour(f)))
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N])
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_eval), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_eval["lbls"]):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_eval, Y, primary


def align_43a(sc_eval):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(sc_eval), scs.shape[1]), np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]
    return S


def align_old(sc_eval, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(sc_eval), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s

def gauss_pf(scores, sc_eval, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in sc_eval["filename"].unique():
        m = (sc_eval["filename"] == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out


def main():
    sc_eval, Y, primary = build_eval()
    print(f"Eval: {len(sc_eval)} rows × {Y.shape[1]} classes; sites {sc_eval.site.unique()}")

    S_perch = align_43a(sc_eval)
    perch_prob = sigmoid(S_perch)
    S_sed29 = align_old(sc_eval, EXP29 / "val_scores.npz")

    # v12 config
    zP = zs(perch_prob); z29 = zs(np.nan_to_num(S_sed29, nan=0))
    v12_raw = 0.8*zP + 0.2*z29
    v12_smoothed = gauss_pf(v12_raw, sc_eval, 0.5)
    v12_prob = sigmoid(v12_smoothed)
    print(f"v12 pipeline rebuilt: shape {v12_prob.shape}")

    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    ta = pd.read_csv(DATA / "train.csv").groupby("primary_label").size().to_dict()
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])
    n_ta = np.array([ta.get(p, 0) for p in primary])

    evaluable_cls = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    print(f"Evaluable classes: {len(evaluable_cls)}")

    # Per-class AUC on v12
    per_class = {}
    for c in evaluable_cls:
        try: per_class[c] = roc_auc_score(Y[:, c], v12_prob[:, c])
        except Exception: pass

    # ─────────────────────── PATTERN 1: per-SITE ───────────────────────
    print("\n=== PATTERN 1: per-site error distribution ===")
    site_stats = {}
    for site, grp in sc_eval.groupby("site"):
        idx = grp.index.values
        Y_s = Y[idx]; P_s = v12_prob[idx]
        cls_in_site = [c for c in evaluable_cls if 0 < Y_s[:, c].sum() < len(Y_s)]
        aucs = []
        for c in cls_in_site:
            try: aucs.append(roc_auc_score(Y_s[:, c], P_s[:, c]))
            except Exception: pass
        if aucs:
            site_stats[site] = {
                "n_rows": len(idx), "n_eval_cls": len(cls_in_site),
                "macro": float(np.mean(aucs)),
                "p_pos_per_row": float(Y_s.sum(axis=1).mean()),
            }
    for s, v in sorted(site_stats.items(), key=lambda kv: kv[1]["macro"]):
        print(f"  {s}  rows={v['n_rows']:3d}  n_cls={v['n_eval_cls']:2d}  macro={v['macro']:.3f}  avg_pos/row={v['p_pos_per_row']:.2f}")

    # ─────────────────────── PATTERN 2: hour-of-day ───────────────────────
    print("\n=== PATTERN 2: hour-of-day (diurnal/nocturnal) ===")
    for hour_bin in [("00-04", (0, 4)), ("04-08", (4, 8)), ("08-12", (8, 12)),
                     ("12-16", (12, 16)), ("16-20", (16, 20)), ("20-24", (20, 24))]:
        name, (lo, hi) = hour_bin
        mask = ((sc_eval.hour >= lo) & (sc_eval.hour < hi)).values
        if mask.sum() == 0:
            print(f"  {name}  n=0"); continue
        Y_h = Y[mask]; P_h = v12_prob[mask]
        aucs = []
        for c in evaluable_cls:
            if 0 < Y_h[:, c].sum() < len(Y_h):
                try: aucs.append(roc_auc_score(Y_h[:, c], P_h[:, c]))
                except Exception: pass
        if aucs:
            print(f"  {name}  n={mask.sum():3d}  n_cls={len(aucs):2d}  macro={np.mean(aucs):.3f}  avg_pos/row={Y_h.sum(axis=1).mean():.2f}")

    # ─────────────────────── PATTERN 3: confidence distribution on errors ───────────────────────
    print("\n=== PATTERN 3: error confidence distribution ===")
    # For each eval class, separate (a) positive rows ranked where (prob), (b) negative rows' prob dist
    for c in evaluable_cls:
        y = Y[:, c]; p = v12_prob[:, c]
        if per_class.get(c, 0.5) < 0.5:
            pos = p[y == 1]; neg = p[y == 0]
            if len(pos) > 0 and len(neg) > 0:
                neg_top10 = np.sort(neg)[-min(10, len(neg)):].mean()
                pos_median = np.median(pos)
                lbl = primary[c]; t = species_taxon[c]
                print(f"  {lbl:<12} ({t:<8}) auc={per_class[c]:.3f}  pos_median={pos_median:.3f}  neg_top10={neg_top10:.3f}")

    # ─────────────────────── PATTERN 4: sibling Aves confusion pairs ───────────────────────
    print("\n=== PATTERN 4: Aves confusion pairs (v12 predicts B high when truth is A) ===")
    # For each Aves class with AUC < 0.7, find which Aves classes get confidently predicted on its positive rows
    aves_cls = set(c for c in evaluable_cls if species_taxon[c] == "Aves")
    pair_counts = defaultdict(int)
    for c in aves_cls:
        y = Y[:, c]
        if y.sum() == 0: continue
        auc_c = per_class.get(c)
        if auc_c is None or auc_c >= 0.7: continue
        pos_rows = np.where(y == 1)[0]
        for r in pos_rows:
            # Find top-3 Aves predictions that are NOT c
            scores_r = v12_prob[r].copy(); scores_r[c] = -1
            aves_idx = [i for i in range(len(primary)) if species_taxon[i] == "Aves"]
            aves_scores = scores_r[aves_idx]
            top3 = np.argsort(aves_scores)[-3:]
            for ti in top3:
                confused = aves_idx[ti]
                pair_counts[(c, confused)] += 1
    top_pairs = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:15]
    for (a, b), cnt in top_pairs:
        print(f"  true={primary[a]:<12} → confused with {primary[b]:<12}  ({cnt}x on pos rows)")

    # ─────────────────────── PATTERN 5: non-Aves confusion (extended from exp43r) ───────────────────────
    print("\n=== PATTERN 5: non-Aves → Aves confusion verification ===")
    for c in evaluable_cls:
        if species_taxon[c] in ("Aves", "?"): continue
        y = Y[:, c]
        if y.sum() == 0: continue
        auc_c = per_class.get(c)
        if auc_c is None: continue
        # Top Aves predictions on positive rows
        pos_rows = np.where(y == 1)[0]
        scores = v12_prob[pos_rows].mean(axis=0)
        aves_idx = [i for i in range(len(primary)) if species_taxon[i] == "Aves"]
        top = sorted(aves_idx, key=lambda i: -scores[i])[:3]
        lbl = primary[c]; t = species_taxon[c]
        print(f"  {lbl:<12} ({t:<8}) auc={auc_c:.3f}  n_pos={int(y.sum())}  top3_Aves_pred={[primary[i] for i in top]}")

    # ─────────────────────── PATTERN 6: within-file temporal flip ───────────────────────
    print("\n=== PATTERN 6: within-file prediction volatility ===")
    # For each class, compute SD of predictions within the same file vs across files
    vol_by_cls = []
    for c in evaluable_cls:
        within_sd = []; across_sd = []
        for fn, g in sc_eval.groupby("filename"):
            idx = g.index.values
            if len(idx) < 3: continue
            within_sd.append(v12_prob[idx, c].std())
        if within_sd:
            vol_by_cls.append((c, float(np.mean(within_sd))))
    vol_by_cls.sort(key=lambda x: -x[1])
    print("  Top-10 most volatile within file (v12):")
    for c, v in vol_by_cls[:10]:
        auc_c = per_class.get(c, float("nan"))
        print(f"  {primary[c]:<12} ({species_taxon[c]:<8}) within_file_SD={v:.3f}  auc={auc_c:.3f}")
    print("  Bottom-10 (most stable — confident always-on or always-off):")
    for c, v in vol_by_cls[-10:]:
        auc_c = per_class.get(c, float("nan"))
        print(f"  {primary[c]:<12} ({species_taxon[c]:<8}) within_file_SD={v:.3f}  auc={auc_c:.3f}")

    # ─────────────────────── PATTERN 7: per-class prediction saturation ───────────────────────
    print("\n=== PATTERN 7: saturation (median prediction across all windows) ===")
    # Classes with median prob very high (always on = false positives) or always low (never fires)
    med_probs = v12_prob.mean(axis=0)
    print(f"  Overall median pred: min {med_probs.min():.4f}  max {med_probs.max():.4f}  mean {med_probs.mean():.4f}")
    high_firing = np.where(med_probs > 0.1)[0]
    print(f"  Always-high (mean pred > 0.1, n={len(high_firing)}):")
    for c in sorted(high_firing, key=lambda i: -med_probs[i])[:10]:
        auc_c = per_class.get(c, float("nan"))
        n_pos = Y[:, c].sum()
        print(f"    {primary[c]:<12} ({species_taxon[c]:<8}) mean_pred={med_probs[c]:.3f}  n_pos={n_pos}  auc={auc_c:.3f}")

    # Save summary
    with open(OUT / "error_patterns.json", "w") as f:
        json.dump({
            "site_stats": site_stats,
            "top_aves_confusion_pairs": [
                {"true": primary[a], "confused_with": primary[b], "count": cnt}
                for (a, b), cnt in top_pairs
            ],
            "n_eval_classes": len(evaluable_cls),
            "n_rows": len(sc_eval),
        }, f, indent=2, default=float)
    print(f"\nSaved → {OUT}/error_patterns.json")


if __name__ == "__main__":
    main()
