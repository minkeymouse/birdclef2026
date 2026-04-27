#!/usr/bin/env python3
"""exp45b — Class-balanced taxon head.

exp45a issue: Training taxa frequencies skewed (Aves 96.8%, Reptilia 0.03%).
Taxon head over-suppresses minority taxa → Mammalia AUC dropped and 74113
went from 0.375 to 0.042.

Fix: per-taxon pos_weight in BCE loss.
  pos_weight[t] = N_total / (5 * N_pos_taxon_t)

Everything else identical to exp45a.
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
OUT = ROOT / "experiments/exp45b_outputs"
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
POS_WEIGHT_CAP = 100.0   # cap to avoid Reptilia pos_weight exploding (10 samples)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_mappings():
    tax = pd.read_csv(DATA / "taxonomy.csv")
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    label_to_class = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    taxon_to_idx = {t: i for i, t in enumerate(TAXA)}
    species_to_taxon = np.array([
        taxon_to_idx.get(label_to_class.get(p, "Aves"), 0) for p in primary
    ], dtype=np.int64)
    return primary, species_to_taxon


def load_train_audio():
    d = np.load(EXP22 / "train_audio_perch.npz", allow_pickle=True)
    emb = d["emb"].astype(np.float32)
    y_idx = d["y_idx"].astype(np.int64)
    valid = d["valid"].astype(bool)
    return emb[valid], y_idx[valid]


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
    train_files = set(files[EVAL_N_FILES:])
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    def rows_for(file_set):
        sub = sc_g[sc_g.filename.isin(file_set)].reset_index(drop=True)
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
    tr_emb, tr_Y, _ = rows_for(train_files)
    ev_emb, ev_Y, ev_scores = rows_for(eval_files)
    return tr_emb, tr_Y, ev_emb, ev_Y, ev_scores


class TaxonHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=HIDDEN, n_taxa=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, n_taxa),
        )
    def forward(self, x): return self.net(x)


def taxon_target_from_species(species_multihot, species_to_taxon, n_taxa=5):
    N = len(species_multihot)
    T = np.zeros((N, n_taxa), dtype=np.float32)
    for c in range(species_multihot.shape[1]):
        if species_multihot[:, c].sum() == 0: continue
        T[:, species_to_taxon[c]] = np.maximum(T[:, species_to_taxon[c]], species_multihot[:, c])
    return T


def train_taxon_head(emb, taxon_Y, pos_weight, epochs=EPOCHS, batch=BATCH):
    set_seed(SEED)
    ds = TensorDataset(torch.from_numpy(emb), torch.from_numpy(taxon_Y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    model = TaxonHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)
    model.train()
    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for x, y in dl:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
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


def main():
    set_seed(SEED)
    primary, species_to_taxon = build_mappings()

    ta_emb, ta_y_idx = load_train_audio()
    ta_species_multihot = np.zeros((len(ta_emb), len(primary)), dtype=np.float32)
    ta_species_multihot[np.arange(len(ta_emb)), ta_y_idx] = 1.0
    ta_taxon_Y = taxon_target_from_species(ta_species_multihot, species_to_taxon)

    ss_tr_emb, ss_tr_Y, ss_ev_emb, ss_ev_Y, ss_ev_scores = load_labeled_ss(primary)
    ss_tr_taxon_Y = taxon_target_from_species(ss_tr_Y, species_to_taxon)

    all_emb = np.concatenate([ta_emb, ss_tr_emb], axis=0)
    all_taxon_Y = np.concatenate([ta_taxon_Y, ss_tr_taxon_Y], axis=0)

    # ── compute pos_weight ──
    N = all_taxon_Y.shape[0]
    n_pos_per_taxon = all_taxon_Y.sum(0)
    pos_weight = np.clip(N / (5 * np.maximum(n_pos_per_taxon, 1)), 1.0, POS_WEIGHT_CAP)
    print(f"Taxon positives: {dict(zip(TAXA, n_pos_per_taxon.astype(int)))}")
    print(f"pos_weight (capped at {POS_WEIGHT_CAP}): {dict(zip(TAXA, pos_weight.round(2)))}")

    print("\nTraining balanced taxon head...")
    t0 = time.time()
    model = train_taxon_head(all_emb, all_taxon_Y, pos_weight, epochs=EPOCHS)
    print(f"trained in {time.time()-t0:.0f}s")

    eval_taxon_probs = predict_taxon(model, ss_ev_emb)
    print(f"\nEval taxon prob mean: {dict(zip(TAXA, eval_taxon_probs.mean(0).round(3)))}")
    print(f"Eval taxon prob q90:  {dict(zip(TAXA, np.quantile(eval_taxon_probs, 0.9, axis=0).round(3)))}")

    # ── evaluate ──
    baseline = sigmoid(ss_ev_scores)
    taxon_per_species = eval_taxon_probs[:, species_to_taxon]
    gated = baseline * taxon_per_species

    auc_base = per_class_auc(ss_ev_Y, baseline)
    auc_gated = per_class_auc(ss_ev_Y, gated)

    print(f"\nMacro AUC (40 eval cls):")
    print(f"  baseline:    {macro(auc_base):.4f}")
    print(f"  taxon-gated: {macro(auc_gated):.4f}")
    print(f"  Δ = {macro(auc_gated) - macro(auc_base):+.4f}")

    print(f"\nPer-taxon:")
    for tidx, tname in enumerate(TAXA):
        cols = [c for c in range(len(primary)) if species_to_taxon[c] == tidx]
        base_sub = {c: auc_base[c] for c in cols if c in auc_base}
        gated_sub = {c: auc_gated[c] for c in cols if c in auc_gated}
        print(f"  {tname:<10}  n={len(base_sub):2d}  base={macro(base_sub):.4f}  gated={macro(gated_sub):.4f}  Δ={macro(gated_sub) - macro(base_sub):+.4f}")

    # Bottom-8 tracking
    bottom8 = ["516975", "67107", "326272", "bafcur1", "74113", "25073", "116570", "47158son11"]
    print(f"\nBottom-8 (base → gated):")
    for lbl in bottom8:
        if lbl not in primary: continue
        c = primary.index(lbl)
        bb = auc_base.get(c, float("nan")); bg = auc_gated.get(c, float("nan"))
        if not np.isnan(bb) and not np.isnan(bg):
            taxon = TAXA[species_to_taxon[c]]
            print(f"  {lbl:<12} ({taxon:<9})  {bb:.3f} → {bg:.3f}  Δ={bg-bb:+.3f}")

    # compare to exp45a (without balance)
    print(f"\n=== exp45a vs exp45b ===")
    print(f"Macro:      0.745 (exp45a unbalanced) → {macro(auc_gated):.4f} (exp45b balanced)")

    torch.save({"state_dict": model.state_dict(), "species_to_taxon": species_to_taxon.tolist(),
                "TAXA": TAXA, "pos_weight": pos_weight.tolist()}, OUT / "taxon_head_balanced.pt")
    with open(OUT / "results.json", "w") as fp:
        json.dump({"macro_base": macro(auc_base), "macro_gated": macro(auc_gated),
                   "pos_weight": pos_weight.tolist(),
                   "eval_taxon_mean": eval_taxon_probs.mean(0).tolist()}, fp, indent=2, default=float)


if __name__ == "__main__":
    main()
