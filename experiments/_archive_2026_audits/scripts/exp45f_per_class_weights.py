#!/usr/bin/env python3
"""exp45f — Per-class optimal blend weights.

Question: for each class, which teacher (Perch / SED29 / SED41f) is strongest?
Given answer, compute the ORACLE per-class blend — an upper bound on what a
per-class stacker could achieve. Also identify which species have high
diversity between teachers (= ensemble headroom) vs which are consensus
strong/weak.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP41F = ROOT / "experiments/exp41f_outputs"
OUT = ROOT / "experiments/exp45f_outputs"
OUT.mkdir(exist_ok=True)
SEED = 42; EVAL_N = 11


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


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x,-20,20)))


def auc_safe(y, p):
    y = y.astype(int)
    if y.sum() == 0 or y.sum() == len(y): return None
    if not np.isfinite(p).all(): return None
    try: return float(roc_auc_score(y, p))
    except Exception: return None


def main():
    sc_eval, Y, primary = build_eval()
    perch = align_43a(sc_eval)
    perch_prob = sigmoid(perch)
    sed29 = align_old(sc_eval, EXP29 / "val_scores.npz")
    sed41f = align_old(sc_eval, EXP41F / "val_scores_full.npz")
    print(f"Shapes: perch {perch_prob.shape}  sed29 {None if sed29 is None else sed29.shape}  sed41f {None if sed41f is None else sed41f.shape}")

    # Taxonomy info
    tax = pd.read_csv(DATA / "taxonomy.csv")
    ta = pd.read_csv(DATA / "train.csv").groupby("primary_label").size()
    label_info = {}
    for p in primary:
        row = tax[tax["primary_label"].astype(str) == p]
        label_info[p] = {
            "taxon": row["class_name"].iloc[0] if len(row) else "?",
            "n_ta": int(ta.get(p, 0)),
        }

    # Per-class AUC with each teacher
    rows = []
    for c in range(Y.shape[1]):
        y = Y[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
        p_p = auc_safe(y, perch_prob[:, c])
        p_29 = auc_safe(y, sed29[:, c]) if sed29 is not None else None
        p_41 = auc_safe(y, sed41f[:, c]) if sed41f is not None else None
        if p_p is None: continue
        # Find best single teacher
        teacher_aucs = {"P": p_p, "S29": p_29, "S41f": p_41}
        teacher_aucs = {k: v for k, v in teacher_aucs.items() if v is not None}
        best_t = max(teacher_aucs, key=teacher_aucs.get)
        # Optimal 2-way blend between P and best SED (grid sweep alpha)
        best_2way_auc, best_alpha = p_p, 0.0
        if p_41 is not None and p_p is not None:
            for a in np.arange(0.0, 1.05, 0.05):
                blend = (1 - a) * perch_prob[:, c] + a * sed41f[:, c]
                auc_b = auc_safe(y, blend)
                if auc_b is not None and auc_b > best_2way_auc:
                    best_2way_auc, best_alpha = auc_b, a
        rows.append({
            "class_idx": c, "label": primary[c],
            "taxon": label_info[primary[c]]["taxon"],
            "n_ta": label_info[primary[c]]["n_ta"],
            "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
            "auc_Perch": p_p, "auc_SED29": p_29, "auc_SED41f": p_41,
            "best_teacher": best_t, "best_single_auc": teacher_aucs[best_t],
            "opt_P_S41f_alpha": best_alpha,
            "opt_P_S41f_auc": best_2way_auc,
            "oracle_minus_perch": best_2way_auc - p_p,
        })
    df = pd.DataFrame(rows).sort_values("oracle_minus_perch", ascending=False)

    # Summary
    print(f"\n[{len(df)} eval classes]")
    print(f"Macro per teacher:")
    print(f"  Perch:  {df['auc_Perch'].mean():.4f}")
    print(f"  SED29:  {df['auc_SED29'].mean():.4f}")
    print(f"  SED41f: {df['auc_SED41f'].mean():.4f}")
    print(f"  Oracle 2-way (P, S41f): {df['opt_P_S41f_auc'].mean():.4f}  (hypothetical ceiling)")

    print(f"\nBest teacher distribution (per-class winner):")
    print(df['best_teacher'].value_counts())

    print(f"\nOptimal α (P, S41f) distribution:")
    for lo, hi in [(0.0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        sub = df[(df["opt_P_S41f_alpha"] >= lo) & (df["opt_P_S41f_alpha"] < hi)]
        print(f"  α ∈ [{lo:.2f}, {hi:.2f})  n={len(sub)}  oracle_mean={sub['opt_P_S41f_auc'].mean():.3f}")

    print(f"\n[Top-10 where SED41f beats Perch by the most]")
    for _, r in df.head(10).iterrows():
        print(f"  {r.label:<12} ({r.taxon:<9}) n_ta={r.n_ta:3d} n_pos={r.n_pos:3d}  "
              f"Perch {r.auc_Perch:.3f}  SED41f {r.auc_SED41f:.3f}  opt@α={r.opt_P_S41f_alpha:.2f}")

    print(f"\n[Where Perch beats SED41f (top-5 Perch advantage)]")
    df_pa = df.sort_values("auc_Perch", ascending=False)
    df_pa = df_pa[df_pa["auc_Perch"] > df_pa["auc_SED41f"]]
    for _, r in df_pa.head(5).iterrows():
        print(f"  {r.label:<12} ({r.taxon:<9}) Perch {r.auc_Perch:.3f}  SED41f {r.auc_SED41f:.3f}  diff {r.auc_Perch - r.auc_SED41f:+.3f}")

    # Per-taxon oracle vs v12 (0.8P + 0.2S29 + Gauss omitted for simplicity)
    print(f"\n[Per-taxon breakdown]")
    print(f"  {'taxon':<10}  {'n':>4}  {'P':>6}  {'S29':>6}  {'S41f':>6}  {'oracle':>7}")
    for t, g in df.groupby("taxon"):
        print(f"  {t:<10}  {len(g):>4}  {g.auc_Perch.mean():.3f}  {g.auc_SED29.mean():.3f}  {g.auc_SED41f.mean():.3f}  {g.opt_P_S41f_auc.mean():.3f}")

    df.to_csv(OUT / "per_class_weights.csv", index=False)
    print(f"\nSaved → {OUT}/per_class_weights.csv")


if __name__ == "__main__":
    main()
