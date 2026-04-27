#!/usr/bin/env python3
"""exp45a — Taxon-gated hierarchical classifier on Perch features.

Motivation (exp45 audit):
  Perch confidently classifies non-Aves taxa (Mammalia, Reptilia, Amphibia)
  as bird species. 4 of 8 bottom offenders are non-Aves species predicted
  as Aves. Hypothesis: gating species score by a learned taxon-presence
  probability can suppress these cross-taxon mispredictions.

Design (simplest viable):
  final_score[c] = sigmoid(perch_species_logit[c]) * sigmoid(taxon_head(c)_prob)
  where taxon_head is trained from scratch on Perch 1536-d features to
  multi-label predict {Aves, Amphibia, Insecta, Mammalia, Reptilia}.

Data:
  train_audio: 35,549 clips with single primary_label → single taxon
  labeled SS 55 files (exp38 split): 617 windows with multi-species → multi-taxon
  eval 11 files: 122 windows

  Perch caches:
    exp22_outputs/train_audio_perch.npz  (TF-CPU embs, 1536-d)
    exp43a_outputs/perch_ss_all.npz      (ONNX-GPU embs, 1536-d; slight drift ok for probe)

Architecture:
  Linear(1536, 256) → GELU → Dropout(0.2) → Linear(256, 5)
  Light (~400K params) to avoid overfit on ~36k samples.

Eval:
  - macro AUC on 40 evaluable classes
  - Bottom-8 offender tracking (specifically non-Aves mislabeled as Aves)
  - Compare (a) raw Perch score vs (b) taxon-gated Perch score
"""
from __future__ import annotations
import json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP22 = ROOT / "experiments/exp22_outputs"
EXP43A = ROOT / "experiments/exp43a_outputs"
OUT = ROOT / "experiments/exp45a_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"
SEED = 42
EVAL_N_FILES = 11
EPOCHS = 40
BATCH = 256
LR = 1e-3
WD = 1e-4
HIDDEN = 256
DROPOUT = 0.2
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_mappings():
    tax = pd.read_csv(DATA / "taxonomy.csv")
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    # species idx → taxon idx
    label_to_class = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    taxon_to_idx = {t: i for i, t in enumerate(TAXA)}
    species_to_taxon = np.array([
        taxon_to_idx.get(label_to_class.get(p, "Aves"), 0) for p in primary
    ], dtype=np.int64)
    print(f"Species→taxon map: {dict(zip(TAXA, np.bincount(species_to_taxon, minlength=5)))}")
    return primary, species_to_taxon


def load_train_audio():
    """train_audio Perch embs + species-level labels. Each clip: 1 primary species → 1 taxon."""
    d = np.load(EXP22 / "train_audio_perch.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32)         # (35549, 1536)
    y_idx = d["y_idx"].astype(np.int64)        # species index
    valid = d["valid"].astype(bool)
    emb = emb[valid]; y_idx = y_idx[valid]
    return emb, y_idx


def load_labeled_ss(primary):
    """Labeled SS Perch embs (ONNX) + multi-label targets, split into train/eval by files."""
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

    # Same 55/11 split as exp38/44c
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique())
    rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    train_files = set(files[EVAL_N_FILES:])

    # Map row_id → position in SS emb array
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}

    def rows_for(file_set):
        sub = sc_g[sc_g.filename.isin(file_set)].reset_index(drop=True)
        emb_rows, Y = [], []
        for _, r in sub.iterrows():
            if r.row_id not in rid2i: continue
            e = ss_emb[rid2i[r.row_id]]
            y = np.zeros(len(primary), dtype=np.float32)
            for l in r.lbls:
                if l in l2i: y[l2i[l]] = 1.0
            emb_rows.append(e); Y.append(y)
        return np.stack(emb_rows), np.stack(Y), sub

    train_emb, train_Y, _ = rows_for(train_files)
    eval_emb, eval_Y, eval_sub = rows_for(eval_files)
    # Also need raw Perch scores for eval rows (for computing final gated output)
    eval_scores = np.stack([ss_logits[rid2i[r]] for r in eval_sub["row_id"].values if r in rid2i])
    return train_emb, train_Y, eval_emb, eval_Y, eval_scores, eval_sub


class TaxonHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=HIDDEN, n_taxa=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, n_taxa),
        )
    def forward(self, x): return self.net(x)


def taxon_target_from_species(species_multihot, species_to_taxon, n_taxa=5):
    """Convert (N, 234) species multihot → (N, 5) taxon multihot via aggregation."""
    N = len(species_multihot)
    T = np.zeros((N, n_taxa), dtype=np.float32)
    for c in range(species_multihot.shape[1]):
        if species_multihot[:, c].sum() == 0: continue
        T[:, species_to_taxon[c]] = np.maximum(T[:, species_to_taxon[c]], species_multihot[:, c])
    return T


def train_taxon_head(train_emb, train_taxon_Y, epochs=EPOCHS, batch=BATCH):
    set_seed(SEED)
    ds = TensorDataset(torch.from_numpy(train_emb), torch.from_numpy(train_taxon_Y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    model = TaxonHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for x, y in dl:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:02d}  loss {tot/n:.4f}")
    return model


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


def evaluate(eval_Y, eval_scores, taxon_probs, primary, species_to_taxon):
    """Compare baseline Perch sigmoid vs taxon-gated Perch sigmoid."""
    baseline = sigmoid(eval_scores)
    taxon_per_species = taxon_probs[:, species_to_taxon]     # (N, 234)
    gated = baseline * taxon_per_species

    auc_base = per_class_auc(eval_Y, baseline)
    auc_gated = per_class_auc(eval_Y, gated)

    # Per-taxon breakdown
    tax_breakdown = {}
    for tidx, tname in enumerate(TAXA):
        cols = [c for c in range(len(primary)) if species_to_taxon[c] == tidx]
        base_sub = {c: auc_base[c] for c in cols if c in auc_base}
        gated_sub = {c: auc_gated[c] for c in cols if c in auc_gated}
        tax_breakdown[tname] = {
            "n_eval": len(base_sub),
            "base_mean": macro(base_sub),
            "gated_mean": macro(gated_sub),
            "delta": macro(gated_sub) - macro(base_sub),
        }
    return {
        "macro_baseline": macro(auc_base),
        "macro_gated": macro(auc_gated),
        "n_eval_classes": len(auc_base),
        "per_taxon": tax_breakdown,
        "per_class_base": auc_base,
        "per_class_gated": auc_gated,
    }


def main():
    set_seed(SEED)
    primary, species_to_taxon = build_mappings()

    # ── train_audio ──
    ta_emb, ta_y_idx = load_train_audio()
    print(f"train_audio: {ta_emb.shape}")
    # Single primary species → single-species multihot for consistency
    ta_species_multihot = np.zeros((len(ta_emb), len(primary)), dtype=np.float32)
    ta_species_multihot[np.arange(len(ta_emb)), ta_y_idx] = 1.0
    ta_taxon_Y = taxon_target_from_species(ta_species_multihot, species_to_taxon)
    print(f"  train_audio taxon distribution: {ta_taxon_Y.sum(0)}")

    # ── labeled SS ──
    ss_tr_emb, ss_tr_Y, ss_ev_emb, ss_ev_Y, ss_ev_scores, ss_ev_sub = load_labeled_ss(primary)
    ss_tr_taxon_Y = taxon_target_from_species(ss_tr_Y, species_to_taxon)
    print(f"labeled SS train: {ss_tr_emb.shape}  eval: {ss_ev_emb.shape}")
    print(f"  SS-train taxon distribution: {ss_tr_taxon_Y.sum(0)}")

    # ── combined training data ──
    all_emb = np.concatenate([ta_emb, ss_tr_emb], axis=0)
    all_taxon_Y = np.concatenate([ta_taxon_Y, ss_tr_taxon_Y], axis=0)
    print(f"\nCombined training: {all_emb.shape[0]} samples (train_audio {ta_emb.shape[0]} + SS {ss_tr_emb.shape[0]})")
    print(f"Taxon frequencies: {dict(zip(TAXA, all_taxon_Y.sum(0).astype(int)))}")

    # ── train taxon head ──
    print("\nTraining taxon head...")
    t0 = time.time()
    model = train_taxon_head(all_emb, all_taxon_Y, epochs=EPOCHS)
    print(f"trained in {time.time()-t0:.0f}s")

    # ── predict on eval ──
    eval_taxon_probs = predict_taxon(model, ss_ev_emb)
    print(f"\nEval taxon prob mean per taxon: {dict(zip(TAXA, eval_taxon_probs.mean(0).round(3)))}")

    # ── evaluate ──
    result = evaluate(ss_ev_Y, ss_ev_scores, eval_taxon_probs, primary, species_to_taxon)
    print(f"\nMacro AUC:")
    print(f"  baseline (raw Perch sigmoid): {result['macro_baseline']:.4f}  (n_classes={result['n_eval_classes']})")
    print(f"  taxon-gated:                  {result['macro_gated']:.4f}")
    print(f"  Δ = {result['macro_gated'] - result['macro_baseline']:+.4f}")

    print(f"\nPer-taxon breakdown:")
    for t, v in result["per_taxon"].items():
        print(f"  {t:<10}  n={v['n_eval']:2d}  base={v['base_mean']:.4f}  gated={v['gated_mean']:.4f}  Δ={v['delta']:+.4f}")

    # ── Bottom-8 offenders tracking (from exp45) ──
    bottom8 = ["516975", "67107", "326272", "bafcur1", "74113", "25073", "116570", "47158son11"]
    print(f"\nBottom-8 offenders (AUC baseline → gated):")
    for lbl in bottom8:
        if lbl not in primary: continue
        c = primary.index(lbl)
        bb = result["per_class_base"].get(c, float("nan"))
        bg = result["per_class_gated"].get(c, float("nan"))
        if not np.isnan(bb) and not np.isnan(bg):
            taxon = TAXA[species_to_taxon[c]]
            print(f"  {lbl:<12} ({taxon:<9})  {bb:.3f} → {bg:.3f}  Δ={bg-bb:+.3f}")

    # Save
    torch.save({"state_dict": model.state_dict(), "species_to_taxon": species_to_taxon.tolist(),
                "TAXA": TAXA}, OUT / "taxon_head.pt")
    with open(OUT / "results.json", "w") as fp:
        dumpable = {**{k: v for k, v in result.items() if k not in ("per_class_base", "per_class_gated")}}
        dumpable["per_class_base_sample"] = {str(primary[c]): v for c, v in list(result["per_class_base"].items())[:30]}
        dumpable["per_class_gated_sample"] = {str(primary[c]): v for c, v in list(result["per_class_gated"].items())[:30]}
        json.dump(dumpable, fp, indent=2, default=float)
    print(f"\nSaved → {OUT}/")


if __name__ == "__main__":
    main()
