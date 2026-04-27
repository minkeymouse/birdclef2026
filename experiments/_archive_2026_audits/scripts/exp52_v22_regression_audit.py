#!/usr/bin/env python3
"""exp52 — Root-cause audit of v22 LB regression (0.929 → 0.913).

Goal: localize WHICH class/step lost ranking quality. Since LB metric skips
zero-positive classes and is dominated by common Aves, any rank damage on
common species translates to LB loss.

Audits:
  (A) Per-class AUC change v12 → v22 on TRAIN 55 labeled SS (where we have
      more positives than the 11-eval). Cluster by taxon, n_positives, n_train_audio.
  (B) Isolate per-lever damage: v12+site only, v12+cluster only, v12+gate only.
  (C) Check whether common Aves in cluster TRIGGER sets are hurt.
  (D) Rank-breakage detector: for each row, does the ordering of top-k species
      change between v12 and v22? (AUC-invariant per-class but cross-class matters).
  (E) Estimate which common classes drive LB using high-pos proxy from
      train_soundscapes_labels.csv.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP45A = ROOT / "experiments/exp45a_outputs"
OUT = ROOT / "experiments/exp52_outputs"
OUT.mkdir(exist_ok=True)
SEED = 42; DEVICE = "cuda"
FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_site(fn):
    m = FNAME_RE.match(fn); return m.group(2) if m else None


def build_all():
    """Build eval data for ALL 66 labeled SS files (train 55 + eval 11)."""
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g["site"] = sc_g["filename"].apply(parse_site)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g.filename.unique()); rng.shuffle(files)
    # mark split
    sc_g["split"] = ["eval" if f in set(files[:11]) else "train" for f in sc_g.filename]
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g.lbls):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_g, Y, primary, l2i


def align_43a(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]; embs = d["emb"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(df), scs.shape[1]), np.float32)
    E = np.zeros((len(df), embs.shape[1]), np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]; E[i] = embs[j]
    return S, E


def align_old(df, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(df), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(df.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s
def gauss_pf(scores, df, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in df["filename"].unique():
        m = (df["filename"] == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out

def per_class_auc(Y, P):
    ev = [c for c in range(Y.shape[1]) if 0 < Y[:, c].sum() < len(Y)]
    return {c: float(roc_auc_score(Y[:, c], P[:, c])) for c in ev
            if np.isfinite(P[:, c]).all()}


class TaxonHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=256, n_taxa=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(hidden, n_taxa))
    def forward(self, x): return self.net(x)


def apply_v9_gate(probs, embs):
    ck = torch.load(EXP45A / "taxon_head.pt", map_location=DEVICE, weights_only=False)
    m = TaxonHead().to(DEVICE); m.load_state_dict(ck["state_dict"]); m.eval()
    sp2tx = np.asarray(ck["species_to_taxon"], dtype=np.int64)
    with torch.no_grad():
        tp = torch.sigmoid(m(torch.from_numpy(embs).to(DEVICE))).cpu().numpy()
    return probs * np.clip(tp[:, sp2tx] + 0.1, 0, 1)


def main():
    sc_all, Y_all, primary, l2i = build_all()
    print(f"ALL labeled SS: {len(sc_all)} rows, {len(sc_all.filename.unique())} files")
    print(f"  train={len(sc_all[sc_all.split=='train'])}, eval={len(sc_all[sc_all.split=='eval'])}")

    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])
    ta_cnt = pd.read_csv(DATA / "train.csv").groupby("primary_label").size()

    # Base preds on all 66 files
    S_perch, E_perch = align_43a(sc_all)
    perch_prob = sigmoid(S_perch)
    S29 = np.nan_to_num(align_old(sc_all, EXP29 / "val_scores.npz"), nan=0)

    zP = zs(perch_prob); z29 = zs(S29)
    v12_raw = 0.8 * zP + 0.2 * z29
    v12_prob = sigmoid(gauss_pf(v12_raw, sc_all, 0.5))

    # Site prior from TRAIN 55 only (matching what we submitted)
    tr_mask = (sc_all.split == "train").values
    sc_tr = sc_all[tr_mask].reset_index(drop=True)
    Y_tr = Y_all[tr_mask]; v12_tr = v12_prob[tr_mask]
    sites = sorted(sc_all.site.unique())
    site_idx = {s: i for i, s in enumerate(sites)}
    sp = np.zeros((len(sites), len(primary)), dtype=np.float32)
    for site, grp in sc_tr.groupby("site"):
        si = site_idx[site]
        cnt = np.zeros(len(primary), dtype=np.float32)
        for _, r in grp.iterrows():
            for l in r.lbls:
                if l in l2i: cnt[l2i[l]] += 1
        sp[si] = cnt / (cnt.max() + 1e-8)
    site_vec = np.ones((len(sc_all), len(primary)), dtype=np.float32)
    for i, r in sc_all.iterrows():
        si = site_idx.get(r.site)
        if si is not None: site_vec[i] = sp[si]

    # Cluster map from TRAIN only
    aves_idx = np.array([c for c in range(len(primary)) if species_taxon[c] == "Aves"])
    cluster_map = {}
    for c in range(len(primary)):
        if species_taxon[c] in ("Aves", "?"): continue
        if Y_tr[:, c].sum() < 3: continue
        pos = np.where(Y_tr[:, c] == 1)[0]
        mp = v12_tr[pos][:, aves_idx].mean(axis=0)
        cluster_map[c] = aves_idx[np.argsort(mp)[-3:]].tolist()

    # Build configs (same as v22 submission)
    v12_g = apply_v9_gate(v12_prob, E_perch)    # v12 + V9 gate

    # Site prior applied
    v12_gs = v12_g * (0.75 * site_vec + 0.25)

    # Cluster rewrite
    v22 = v12_gs.copy()
    for tc, trig in cluster_map.items():
        sc_arr = v22[:, trig].min(axis=1)
        v22[:, tc] = v22[:, tc] * (1 + 2.0 * sc_arr)
    v22 = np.clip(v22, 0, 1)

    # Per-class analysis on ALL 66 files
    aucs_v12 = per_class_auc(Y_all, v12_prob)
    aucs_v12g = per_class_auc(Y_all, v12_g)
    aucs_v22 = per_class_auc(Y_all, v22)
    print(f"\nPer-class evaluable (66 files): v12 {len(aucs_v12)}, v22 {len(aucs_v22)}")

    common = set(aucs_v12) & set(aucs_v22)
    v12_macro = np.mean([aucs_v12[c] for c in common])
    v22_macro = np.mean([aucs_v22[c] for c in common])
    v12g_macro = np.mean([aucs_v12g[c] for c in common])
    print(f"\nOn ALL 66 files over {len(common)} common classes:")
    print(f"  v12 macro:       {v12_macro:.4f}")
    print(f"  v12+gate macro:  {v12g_macro:.4f}  Δ {v12g_macro-v12_macro:+.4f}")
    print(f"  v22 macro:       {v22_macro:.4f}  Δ {v22_macro-v12_macro:+.4f}")

    # ── Audit A: classes where v22 < v12 (LB-damaging) ──
    print(f"\n=== AUDIT A: classes where v22 < v12 (LB loss candidates) ===")
    deltas = sorted([(c, aucs_v22[c] - aucs_v12[c]) for c in common], key=lambda x: x[1])
    print(f"  Top-15 BIGGEST DROPS:")
    print(f"  {'class':<14} {'taxon':<10} {'n_pos':>6} {'n_ta':>5}  {'v12':>6} {'v22':>6} {'Δ':>7}")
    for c, d in deltas[:15]:
        print(f"  {primary[c]:<14} {species_taxon[c]:<10} {int(Y_all[:, c].sum()):>6} "
              f"{int(ta_cnt.get(primary[c], 0)):>5}  {aucs_v12[c]:.3f}  {aucs_v22[c]:.3f}  {d:+.3f}")

    print(f"\n  Top-15 BIGGEST GAINS (confirms what worked):")
    for c, d in deltas[-15:][::-1]:
        print(f"  {primary[c]:<14} {species_taxon[c]:<10} {int(Y_all[:, c].sum()):>6} "
              f"{int(ta_cnt.get(primary[c], 0)):>5}  {aucs_v12[c]:.3f}  {aucs_v22[c]:.3f}  {d:+.3f}")

    # ── Audit B: isolate each lever ──
    print(f"\n=== AUDIT B: per-lever damage (each vs v12) ===")
    # site only
    v12_s_only = v12_prob * (0.75 * site_vec + 0.25)
    # cluster only (derive cluster from v12_prob)
    v12_c_only = v12_prob.copy()
    for tc, trig in cluster_map.items():
        sc_arr = v12_c_only[:, trig].min(axis=1)
        v12_c_only[:, tc] = v12_c_only[:, tc] * (1 + 2.0 * sc_arr)
    # gate only (already have v12_g)

    print(f"  {'config':<30}  {'macro':>6}  {'Δ':>7}  {'Aves_Δ':>8}  {'worst_class':<16}  {'worst_Δ':>7}")
    for name, pred in [("v12 + gate only", v12_g),
                       ("v12 + site only", v12_s_only),
                       ("v12 + cluster only", v12_c_only),
                       ("v22 (all three)", v22)]:
        aucs = per_class_auc(Y_all, pred)
        c_cm = set(aucs) & set(aucs_v12)
        macro = np.mean([aucs[c] for c in c_cm])
        aves_cls = [c for c in c_cm if species_taxon[c] == "Aves"]
        aves_delta = np.mean([aucs[c] - aucs_v12[c] for c in aves_cls])
        worst = min([(c, aucs[c] - aucs_v12[c]) for c in c_cm], key=lambda x: x[1])
        print(f"  {name:<30}  {macro:.4f}  {macro-v12_macro:+.4f}  {aves_delta:+.4f}  "
              f"{primary[worst[0]]:<16}  {worst[1]:+.3f}")

    # ── Audit C: are cluster TRIGGER Aves hurt? ──
    print(f"\n=== AUDIT C: cluster trigger Aves classes — collateral damage? ===")
    trigger_aves = set()
    for trig in cluster_map.values():
        trigger_aves.update(trig)
    print(f"  {len(trigger_aves)} distinct Aves appear as triggers in cluster_map")
    print(f"  Of {len(aves_idx)} total Aves, {len([c for c in aves_idx if c in aucs_v12 and c in aucs_v22])} are evaluable")
    ta_in_eval = [c for c in trigger_aves if c in aucs_v12 and c in aucs_v22]
    non_ta_aves = [c for c in aves_idx if c not in trigger_aves and c in aucs_v12 and c in aucs_v22]
    if ta_in_eval:
        ta_delta = np.mean([aucs_v22[c] - aucs_v12[c] for c in ta_in_eval])
        print(f"  Trigger Aves    n={len(ta_in_eval):2d}  mean Δ AUC v22-v12: {ta_delta:+.4f}")
    if non_ta_aves:
        non_delta = np.mean([aucs_v22[c] - aucs_v12[c] for c in non_ta_aves])
        print(f"  Non-trigger Aves n={len(non_ta_aves):2d}  mean Δ AUC v22-v12: {non_delta:+.4f}")

    # ── Audit D: Common (high-pos) classes — most LB-relevant ──
    print(f"\n=== AUDIT D: sort damage by 'LB relevance' (n_pos + Aves taxon) ===")
    # LB evaluable = classes with positives in full test. Proxy: use n_pos from our 66 files.
    # Higher n_pos = more likely to be LB-evaluable.
    n_pos = Y_all.sum(axis=0)
    common_cls = [c for c in aucs_v12 if c in aucs_v22]
    # Proxy LB impact score: classes with >=10 positives and moderate AUC (room to move)
    lb_proxy_cls = [c for c in common_cls if n_pos[c] >= 10]
    print(f"  Classes with n_pos >= 10 (LB-likely evaluable): {len(lb_proxy_cls)}")
    p10_delta = np.mean([aucs_v22[c] - aucs_v12[c] for c in lb_proxy_cls])
    print(f"  Mean AUC Δ on these: {p10_delta:+.4f}")
    # Break down by taxon
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        sub = [c for c in lb_proxy_cls if species_taxon[c] == t]
        if sub:
            d = np.mean([aucs_v22[c] - aucs_v12[c] for c in sub])
            print(f"    {t:<10}  n={len(sub):2d}  Δ {d:+.4f}")

    # ── Audit E: biggest LB-proxy damage classes ──
    print(f"\n=== AUDIT E: worst LB-proxy class drops (n_pos >= 10) ===")
    lb_deltas = sorted([(c, aucs_v22[c] - aucs_v12[c]) for c in lb_proxy_cls], key=lambda x: x[1])
    print(f"  {'class':<14} {'taxon':<10} {'n_pos':>6} {'n_ta':>5}  {'v12':>6} {'v22':>6} {'Δ':>7}  {'in_trig?':<8}")
    for c, d in lb_deltas[:15]:
        in_trig = "YES" if c in trigger_aves else "no"
        print(f"  {primary[c]:<14} {species_taxon[c]:<10} {int(n_pos[c]):>6} "
              f"{int(ta_cnt.get(primary[c], 0)):>5}  {aucs_v12[c]:.3f}  {aucs_v22[c]:.3f}  "
              f"{d:+.3f}  {in_trig}")

    # ── Audit F: Spearman correlation of v12 vs v22 predictions ──
    print(f"\n=== AUDIT F: Spearman(v12, v22) per-row vs per-class ===")
    # For each row, compute rank correlation of 234-vector between v12 and v22
    sp_r_per_row = []
    for i in range(len(sc_all)):
        r, _ = spearmanr(v12_prob[i], v22[i])
        if np.isfinite(r): sp_r_per_row.append(r)
    print(f"  Per-row Spearman (v12 vs v22): mean={np.mean(sp_r_per_row):.3f}  min={np.min(sp_r_per_row):.3f}")
    # Per-class
    sp_r_per_cls = []
    for c in range(len(primary)):
        r, _ = spearmanr(v12_prob[:, c], v22[:, c])
        if np.isfinite(r): sp_r_per_cls.append(r)
    print(f"  Per-class Spearman (v12 vs v22): mean={np.mean(sp_r_per_cls):.3f}  min={np.min(sp_r_per_cls):.3f}")

    # Save CSV
    rows = []
    for c in aucs_v12:
        rows.append({
            "class": primary[c], "taxon": species_taxon[c],
            "n_pos_ss66": int(n_pos[c]), "n_train_audio": int(ta_cnt.get(primary[c], 0)),
            "v12_auc": aucs_v12.get(c),
            "v12_gate_auc": aucs_v12g.get(c),
            "v22_auc": aucs_v22.get(c),
            "delta_v22_v12": (aucs_v22.get(c, np.nan) - aucs_v12.get(c)) if c in aucs_v22 else np.nan,
            "in_cluster_trigger": c in trigger_aves,
            "has_cluster_target": c in cluster_map,
        })
    pd.DataFrame(rows).to_csv(OUT / "52_per_class_audit.csv", index=False)
    print(f"\nSaved → {OUT}/52_per_class_audit.csv")


if __name__ == "__main__":
    main()
