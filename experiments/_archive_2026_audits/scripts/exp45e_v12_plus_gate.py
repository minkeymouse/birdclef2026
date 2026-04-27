#!/usr/bin/env python3
"""exp45e — Does v12 base + taxon gate beat v17 base + taxon gate?
Also test v17 base + taxon gate (our v20 submission config) locally."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP41F = ROOT / "experiments/exp41f_outputs"
EXP45A = ROOT / "experiments/exp45a_outputs"
OUT = ROOT / "experiments/exp45e_outputs"
OUT.mkdir(exist_ok=True)
SEED = 42; EVAL_N = 11
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]


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
    scs, embs = d["scores"], d["emb"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(sc_eval), scs.shape[1]), np.float32)
    E = np.zeros((len(sc_eval), embs.shape[1]), np.float32)
    for i, rid in enumerate(sc_eval["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]; E[i] = embs[j]
    return S, E


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
    for fname in sc_eval["filename"].unique():
        m = (sc_eval["filename"] == fname).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out


def per_class_auc(Y, P):
    out = {}
    for c in range(Y.shape[1]):
        y = Y[:, c].astype(int); p = P[:, c]
        if y.sum() == 0 or y.sum() == len(y): continue
        if not np.isfinite(p).all(): continue
        try: out[c] = float(roc_auc_score(y, p))
        except Exception: pass
    return out

def macro(d): return float(np.mean(list(d.values()))) if d else 0.0


class TaxonHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=256, n_taxa=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                                 nn.Linear(hidden, n_taxa))
    def forward(self, x): return self.net(x)


def main():
    sc_eval, Y, primary = build_eval()
    S_perch, E_perch = align_43a(sc_eval)
    S_sed29 = align_old(sc_eval, EXP29 / "val_scores.npz")
    S_sed41f = align_old(sc_eval, EXP41F / "val_scores_full.npz")

    perch_prob = sigmoid(S_perch)

    # Taxon head
    ckpt = torch.load(EXP45A / "taxon_head.pt", map_location="cuda", weights_only=False)
    m = TaxonHead().cuda(); m.load_state_dict(ckpt["state_dict"]); m.eval()
    species_to_taxon = np.asarray(ckpt["species_to_taxon"], dtype=np.int64)
    with torch.no_grad():
        tprob = torch.sigmoid(m(torch.from_numpy(E_perch).cuda())).cpu().numpy()
    tprob_sp = tprob[:, species_to_taxon]    # (N, 234)
    gate = np.clip(tprob_sp + 0.1, 0.0, 1.0)

    # Define variants
    zP = zs(perch_prob); z29 = zs(S_sed29); z41 = zs(S_sed41f)

    def eval_macro(P, name):
        aucs = per_class_auc(Y, P)
        m_ = macro(aucs)
        print(f"  {name:<50}  macro={m_:.4f}  ({len(aucs)} cls)")
        return m_, aucs

    print("\n[Baselines and overlay comparison on 40 classes]")

    # v12 base
    v12_raw = 0.80*zP + 0.20*z29
    v12_s = gauss_pf(v12_raw, sc_eval, 0.5)
    r0, a0 = eval_macro(v12_s, "v12 base (0.8P + 0.2S29 + Gauss)")
    r1, a1 = eval_macro(v12_s * gate, "v12 base × V9 gate")

    # v17 base
    v17_raw = 0.80*zP + 0.20*z41
    v17_s = gauss_pf(v17_raw, sc_eval, 0.5)
    r2, a2 = eval_macro(v17_s, "v17 base (0.8P + 0.2S41f + Gauss)")
    r3, a3 = eval_macro(v17_s * gate, "v17 base × V9 gate (= v20 submission)")

    # 3-way (v13/v15 style)
    v3way = 0.70*zP + 0.15*z29 + 0.15*z41
    v3way_s = gauss_pf(v3way, sc_eval, 0.5)
    r4, a4 = eval_macro(v3way_s, "3-way (0.7P + 0.15S29 + 0.15S41f + Gauss)")
    r5, a5 = eval_macro(v3way_s * gate, "3-way × V9 gate")

    # Perch only + gate (no SED)
    r6, a6 = eval_macro(perch_prob, "Perch only")
    r7, a7 = eval_macro(perch_prob * gate, "Perch × V9 gate")

    print("\n[Per-taxon Δ from V9 gate applied to each base]")
    bases = {"v12": (v12_s, a0, a1), "v17": (v17_s, a2, a3), "3-way": (v3way_s, a4, a5), "Perch": (perch_prob, a6, a7)}
    for name, (_, base_aucs, gated_aucs) in bases.items():
        print(f"\n  Base: {name}")
        for tidx, tname in enumerate(TAXA):
            cols = [c for c in range(len(primary)) if species_to_taxon[c] == tidx]
            b_sub = {c: base_aucs[c] for c in cols if c in base_aucs}
            g_sub = {c: gated_aucs[c] for c in cols if c in gated_aucs}
            if not b_sub: continue
            print(f"    {tname:<10} n={len(b_sub):2d}  base={macro(b_sub):.3f} → gated={macro(g_sub):.3f}  Δ={macro(g_sub)-macro(b_sub):+.3f}")

    # Bottom-8 tracking for v12+gate (the proposed next submission)
    bottom8 = ["516975", "67107", "326272", "bafcur1", "74113", "25073", "116570", "47158son11"]
    print(f"\n[Bottom-8 under v12 base × V9 gate (proposed next config)]")
    for lbl in bottom8:
        if lbl not in primary: continue
        c = primary.index(lbl)
        b = a0.get(c, float('nan')); g = a1.get(c, float('nan'))
        taxon = TAXA[species_to_taxon[c]]
        if not np.isnan(b) and not np.isnan(g):
            print(f"  {lbl:<12} ({taxon:<9}) {b:.3f} → {g:.3f}  Δ={g-b:+.3f}")


if __name__ == "__main__":
    main()
