#!/usr/bin/env python3
"""exp45c — Soft taxon gating variants.

exp45a (multiplicative `base * taxon_prob`) works for species where taxon is
detectable (516975 Mico 0.5→0.98) but HURTS species where taxon head itself
fails (74113 0.37→0.04, 47158son11 0.5→0.39).

Explore softer gating variants that bound the suppression:
  V1. multiplicative (baseline, = exp45a)     : base * tprob
  V2. linear blend                            : 0.5 * base + 0.5 * (base * tprob)
  V3. geometric (sqrt)                         : base * sqrt(tprob)
  V4. floor at 0.5 (no complete suppression)  : base * (0.5 + 0.5 * tprob)
  V5. additive boost (only upweight)           : base * (1 + tprob) / 2
  V6. max (OR-semantics)                       : max(base, base * tprob)
  V7. threshold (hard passthrough)             : if tprob > 0.3 use base else base * tprob

Uses the SAME exp45a taxon head (no retrain, just re-evaluate).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP22 = ROOT / "experiments/exp22_outputs"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP45A = ROOT / "experiments/exp45a_outputs"
OUT = ROOT / "experiments/exp45c_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
SEED = 42
EVAL_N_FILES = 11
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]


def build_mappings():
    tax = pd.read_csv(DATA / "taxonomy.csv")
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    label_to_class = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    taxon_to_idx = {t: i for i, t in enumerate(TAXA)}
    species_to_taxon = np.array([
        taxon_to_idx.get(label_to_class.get(p, "Aves"), 0) for p in primary
    ], dtype=np.int64)
    return primary, species_to_taxon


def load_labeled_ss(primary):
    scores = np.load(EXP43A / "perch_ss_all.npz")
    ss_emb = scores["emb"].astype(np.float32)
    ss_logits = scores["scores"].astype(np.float32)
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    l2i = {p: i for i, p in enumerate(primary)}
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique())
    rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    sub = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    emb_rows, Y, scrs = [], [], []
    for _, r in sub.iterrows():
        if r.row_id not in rid2i: continue
        emb_rows.append(ss_emb[rid2i[r.row_id]])
        scrs.append(ss_logits[rid2i[r.row_id]])
        y = np.zeros(len(primary), dtype=np.float32)
        for l in r.lbls:
            if l in l2i: y[l2i[l]] = 1.0
        Y.append(y)
    return np.stack(emb_rows), np.stack(Y), np.stack(scrs)


class TaxonHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=256, n_taxa=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, n_taxa),
        )
    def forward(self, x): return self.net(x)


@torch.no_grad()
def predict_taxon(model, emb, batch=512):
    model.eval()
    out = np.zeros((len(emb), 5), dtype=np.float32)
    for i in range(0, len(emb), batch):
        x = torch.from_numpy(emb[i:i+batch]).to(DEVICE)
        out[i:i+batch] = torch.sigmoid(model(x)).cpu().numpy()
    return out


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


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


def main():
    primary, species_to_taxon = build_mappings()
    ss_ev_emb, ss_ev_Y, ss_ev_scores = load_labeled_ss(primary)

    # Load exp45a taxon head
    ckpt = torch.load(EXP45A / "taxon_head.pt", map_location=DEVICE, weights_only=False)
    model = TaxonHead().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    tprobs = predict_taxon(model, ss_ev_emb)        # (N, 5)
    tprobs_per_sp = tprobs[:, species_to_taxon]     # (N, 234)

    baseline = sigmoid(ss_ev_scores)

    variants = {
        "V0_baseline":                  baseline,
        "V1_mult (exp45a)":             baseline * tprobs_per_sp,
        "V2_blend_50_50":               0.5 * baseline + 0.5 * baseline * tprobs_per_sp,
        "V3_sqrt":                      baseline * np.sqrt(tprobs_per_sp),
        "V4_floor_0.5":                 baseline * (0.5 + 0.5 * tprobs_per_sp),
        "V5_one_plus_over2":            baseline * (1 + tprobs_per_sp) / 2,
        "V6_max":                       np.maximum(baseline, baseline * tprobs_per_sp),
        "V7_thresh_0.3":                np.where(tprobs_per_sp > 0.3, baseline, baseline * tprobs_per_sp),
        "V8_pow_0.25":                  baseline * tprobs_per_sp ** 0.25,
        "V9_add_offset_0.1":            baseline * np.clip(tprobs_per_sp + 0.1, 0, 1),
    }

    bottom8 = ["516975", "67107", "326272", "bafcur1", "74113", "25073", "116570", "47158son11"]

    print(f"{'variant':<24}  {'macro':>7}  {'Δ':>7}  |  " +
          "  ".join(f"{l[:8]:>8}" for l in bottom8))
    base_macro = None
    for name, P in variants.items():
        aucs = per_class_auc(ss_ev_Y, P)
        m = macro(aucs)
        if name == "V0_baseline": base_macro = m
        delta = m - base_macro if base_macro is not None else 0
        bot_aucs = []
        for lbl in bottom8:
            if lbl in primary:
                c = primary.index(lbl)
                bot_aucs.append(aucs.get(c, float("nan")))
            else:
                bot_aucs.append(float("nan"))
        bot_str = "  ".join(f"{v:>8.3f}" if not np.isnan(v) else "   nan " for v in bot_aucs)
        print(f"{name:<24}  {m:.4f}  {delta:+.4f}  |  {bot_str}")

    # Per-taxon breakdown for best variant
    best_name = max(variants, key=lambda k: macro(per_class_auc(ss_ev_Y, variants[k])) if k != "V0_baseline" else 0)
    P = variants[best_name]; aucs = per_class_auc(ss_ev_Y, P)
    print(f"\nBest variant: {best_name}  macro={macro(aucs):.4f}")
    print(f"Per-taxon:")
    for tidx, tname in enumerate(TAXA):
        cols = [c for c in range(len(primary)) if species_to_taxon[c] == tidx]
        sub = {c: aucs[c] for c in cols if c in aucs}
        base_aucs = per_class_auc(ss_ev_Y, variants["V0_baseline"])
        base_sub = {c: base_aucs[c] for c in cols if c in base_aucs}
        print(f"  {tname:<10}  n={len(sub):2d}  base={macro(base_sub):.3f}  best={macro(sub):.3f}  Δ={macro(sub)-macro(base_sub):+.3f}")

    # Save
    with open(OUT / "results.json", "w") as fp:
        results = {}
        for name, P in variants.items():
            aucs = per_class_auc(ss_ev_Y, P)
            results[name] = {"macro": macro(aucs)}
        json.dump(results, fp, indent=2, default=float)
    print(f"\nSaved → {OUT}/results.json")


if __name__ == "__main__":
    main()
