#!/usr/bin/env python3
"""exp48d — Confusion-cluster rewrite rule.

exp48a Pattern 5 finding: several rare species (Insecta / Amphibia / Mammalia)
systematically trigger the SAME Aves triplet. Exploit this by rewriting:
  - If confusion-cluster Aves are co-firing on a window AND not expected at site,
    then boost the rare species that produces this cluster.

Defined clusters (from exp48a Pattern 5):
  C1 = {grhtan1, greant1, compot1}       → sonotypes 17, 21, 22, 23
  C2 = {bcwfin2, rutjac1, tattin1}       → Amphibia 25073, 67107, 326272; son11, 24
  C3 = {sobtyr1, chvcon1, flawar1}       → son10, son25
  C4 = {compau, osprey, bbwduc/greani1}  → Mammalia 74113, Amphibia 22967/22973

Rule: cluster_score(c) = min(v12_prob[aves in c])   # all three must fire
      boost(target) = base[target] + alpha * cluster_score   OR
                     = base[target] * (1 + beta * cluster_score)

Cross-validate on 11-file eval.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
OUT = ROOT / "experiments/exp48_outputs"
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
    return sc_eval, Y, primary, l2i


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


def per_class_auc(Y, P):
    ev = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    return {c: float(roc_auc_score(Y[:, c], P[:, c])) for c in ev
            if np.isfinite(P[:, c]).all()}


# Clusters: (confusion_Aves_trigger, rare_species_targets)
CLUSTERS = [
    ("C1_S08_insect_HF", ["grhtan1", "greant1", "compot1"],
     ["47158son17", "47158son21", "47158son22", "47158son23"]),
    ("C2_Amphibia+son", ["bcwfin2", "rutjac1", "tattin1"],
     ["25073", "67107", "326272", "47158son11", "47158son24"]),
    ("C3_son10_25",     ["sobtyr1", "chvcon1", "flawar1"],
     ["47158son10", "47158son25"]),
    ("C4_Mammalia",     ["compau", "osprey", "yebcar"],
     ["74113"]),
]


def main():
    sc_eval, Y, primary, l2i = build_eval()
    S_perch = align_43a(sc_eval)
    S_sed29 = align_old(sc_eval, EXP29 / "val_scores.npz")
    zP = zs(sigmoid(S_perch)); z29 = zs(np.nan_to_num(S_sed29, nan=0))
    v12_prob = sigmoid(gauss_pf(0.8*zP + 0.2*z29, sc_eval, 0.5))
    base_aucs = per_class_auc(Y, v12_prob)
    base_macro = np.mean(list(base_aucs.values()))
    print(f"v12 base macro: {base_macro:.4f}")

    # Compute per-row cluster scores
    cluster_scores = {}
    for name, trig, target in CLUSTERS:
        trig_idx = [l2i[t] for t in trig if t in l2i]
        if not trig_idx:
            cluster_scores[name] = np.zeros(len(sc_eval), dtype=np.float32)
            continue
        # min (all triggers firing) — strong
        min_score = v12_prob[:, trig_idx].min(axis=1)
        # mean (lenient)
        mean_score = v12_prob[:, trig_idx].mean(axis=1)
        cluster_scores[name] = {"min": min_score, "mean": mean_score}
        print(f"\n[{name}] triggers={trig}  targets={target}")
        print(f"  cluster mean score range: [{mean_score.min():.3f}, {mean_score.max():.3f}], "
              f"median={np.median(mean_score):.3f}")
        # Check validity: do rows with cluster_score high actually contain the target species?
        any_target_y = Y[:, [l2i[t] for t in target if t in l2i]].sum(axis=1) > 0
        if any_target_y.any():
            for agg, sc_arr in [("min", min_score), ("mean", mean_score)]:
                try:
                    det_auc = roc_auc_score(any_target_y.astype(int), sc_arr)
                    print(f"  {agg}-agg cluster-score as detector for any-target: AUC={det_auc:.3f}")
                except Exception: pass

    # Apply rewrite: final[:, target] = base[:, target] * (1 + alpha * cluster_score)
    print("\n=== Rewrite overlay tests ===")
    for agg in ["min", "mean"]:
        print(f"\n[aggregator: {agg}]")
        for alpha in [0.25, 0.5, 1.0, 2.0, 4.0]:
            p_new = v12_prob.copy()
            for name, trig, target in CLUSTERS:
                sc_arr = cluster_scores[name][agg]
                tgt_idx = [l2i[t] for t in target if t in l2i]
                if not tgt_idx: continue
                p_new[:, tgt_idx] = p_new[:, tgt_idx] * (1 + alpha * sc_arr[:, None])
            aucs = per_class_auc(Y, p_new)
            all_tgt = [l2i[t] for _, _, tgt in CLUSTERS for t in tgt if t in l2i]
            in_eval = [c for c in all_tgt if c in aucs]
            m_tgt_base = np.mean([base_aucs[c] for c in in_eval])
            m_tgt_new = np.mean([aucs[c] for c in in_eval])
            m_all = np.mean([aucs[c] for c in base_aucs if c in aucs])
            print(f"  alpha={alpha}  targets macro {m_tgt_base:.3f} → {m_tgt_new:.3f}  Δ={m_tgt_new-m_tgt_base:+.3f}  overall {m_all:.4f} Δ{m_all-base_macro:+.4f}")

    # Suppress Aves triggers too: reduce confusion-Aves where cluster is active
    print("\n=== Two-way: boost target + suppress trigger ===")
    for alpha in [1.0, 2.0]:
        for suppress in [0.0, 0.3, 0.5, 0.7]:
            p_new = v12_prob.copy()
            for name, trig, target in CLUSTERS:
                sc_arr = cluster_scores[name]["mean"]
                tgt_idx = [l2i[t] for t in target if t in l2i]
                trig_idx = [l2i[t] for t in trig if t in l2i]
                if tgt_idx:
                    p_new[:, tgt_idx] = p_new[:, tgt_idx] * (1 + alpha * sc_arr[:, None])
                if trig_idx:
                    p_new[:, trig_idx] = p_new[:, trig_idx] * (1 - suppress * sc_arr[:, None])
            aucs = per_class_auc(Y, p_new)
            m_all = np.mean([aucs[c] for c in base_aucs if c in aucs])
            print(f"  alpha={alpha}  suppress={suppress:.1f}  overall {m_all:.4f}  Δ{m_all-base_macro:+.4f}")

    # Per-target detail at best alpha=1 mean
    print("\n=== Per-target AUC change (alpha=1.0 mean, no suppress) ===")
    p_new = v12_prob.copy()
    for name, trig, target in CLUSTERS:
        sc_arr = cluster_scores[name]["mean"]
        tgt_idx = [l2i[t] for t in target if t in l2i]
        if not tgt_idx: continue
        p_new[:, tgt_idx] = p_new[:, tgt_idx] * (1 + 1.0 * sc_arr[:, None])
    aucs = per_class_auc(Y, p_new)
    for name, trig, target in CLUSTERS:
        print(f"\n  {name} (targets: {target})")
        for t in target:
            c = l2i.get(t)
            if c is None: continue
            b = base_aucs.get(c); n = aucs.get(c)
            if b is None or n is None: continue
            print(f"    {t:<12}  {b:.3f} → {n:.3f}  Δ{n-b:+.3f}")


if __name__ == "__main__":
    main()
