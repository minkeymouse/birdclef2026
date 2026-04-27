#!/usr/bin/env python3
"""exp49c — Retrain V9 taxon gate with 2025+2026 train_audio + labeled SS pool.

Builds Aves vs Amphibia vs Insecta vs Mammalia vs Reptilia 5-way classifier
on Perch 1536-d embeddings. Expanded data via 2025 train_audio (28k clips,
extracted in exp49b). The 2026 V9 gate had limited Mammalia data (n=8);
2025 adds +9 Mammalia + 34 Amphibia + 17 Insecta → better class balance.

Evaluation:
  - 5-way test-set accuracy (held-out 10% of combined pool)
  - Sigmoid pseudo-multi-label AUC (each taxon vs rest)
  - Apply as taxon gate on 11-file held-out labeled SS, compare to V9
"""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path("/data/birdclef2026")
DATA26 = ROOT / "data/birdclef-2026"
DATA25 = ROOT / "data/birdclef-2025"
EXP22 = ROOT / "experiments/exp22_outputs"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP49 = ROOT / "experiments/exp49_outputs"
OUT = ROOT / "experiments/exp49_outputs"
OUT.mkdir(exist_ok=True)

DEVICE = "cuda"; SEED = 42
EVAL_N_FILES = 11; EPOCHS = 30; BATCH = 256; LR = 1e-3; WD = 1e-4
HIDDEN = 256; DROPOUT = 0.2
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class TaxonHead(nn.Module):
    def __init__(self, in_dim=1536, hidden=HIDDEN, n_taxa=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, n_taxa),
        )
    def forward(self, x): return self.net(x)


def build_2026_train_audio():
    """Load 2026 Perch train_audio embs + per-clip taxon label."""
    d = np.load(EXP22 / "train_audio_perch.npz")
    embs = d["emb"]               # (N, 1536)
    y_idx = d["y_idx"]            # class index per clip
    valid = d["valid"]            # bool mask
    # Convert y_idx → primary_label string
    primary = pd.read_csv(DATA26 / "sample_submission.csv").columns[1:].tolist()
    y_lbl = np.array([primary[i] if 0 <= i < len(primary) else "?" for i in y_idx])
    # Apply valid mask
    embs = embs[valid]; y_lbl = y_lbl[valid]
    print(f"  2026 TA embs: {embs.shape}  valid clips: {len(y_lbl)}")
    return embs.astype(np.float32), y_lbl.astype(str)


def build_2025_train_audio():
    p = EXP49 / "train_audio_2025_perch.npz"
    if not p.exists():
        print(f"WARNING: {p} not found")
        return None, None
    d = np.load(p, allow_pickle=True)
    return d["embs"].astype(np.float32), np.asarray(d["primary_label"]).astype(str)


def build_2026_labeled_ss():
    """Load 2026 labeled SS Perch embs (from exp43a)."""
    d = np.load(EXP43A / "perch_ss_all.npz")
    embs = d["emb"]  # (N, 1536)
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    # Load labels
    sc = pd.read_csv(DATA26 / "train_soundscapes_labels.csv").drop_duplicates()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    # Filter train files (not eval)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g.filename.unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    sc_train = sc_g[~sc_g.filename.isin(eval_files)].reset_index(drop=True)
    # Align
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    rows_embs = []; rows_taxa = []
    tax = pd.read_csv(DATA26 / "taxonomy.csv")
    lbl2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    for _, r in sc_train.iterrows():
        j = rid2i.get(r.row_id, -1)
        if j < 0: continue
        # Each window can have multiple taxa — multi-label
        taxa_present = set()
        for l in r.lbls:
            t = lbl2tax.get(str(l))
            if t: taxa_present.add(t)
        if not taxa_present: continue
        rows_embs.append(embs[j])
        rows_taxa.append(taxa_present)
    print(f"  2026 labeled SS: {len(rows_embs)} windows (multi-label)")
    return (np.stack(rows_embs).astype(np.float32) if rows_embs else np.zeros((0, 1536), dtype=np.float32)), rows_taxa


def main():
    set_seed(SEED)
    tax26 = pd.read_csv(DATA26 / "taxonomy.csv")
    tax25 = pd.read_csv(DATA25 / "taxonomy.csv")

    # Build 2026 single-label train_audio
    ta26_embs, ta26_lbls = build_2026_train_audio()
    lbl2tax26 = dict(zip(tax26.primary_label.astype(str), tax26.class_name))
    ta26_taxa = np.array([lbl2tax26.get(str(l), "?") for l in ta26_lbls])

    # Build 2025 single-label train_audio
    ta25_embs, ta25_lbls = build_2025_train_audio()
    if ta25_embs is not None:
        lbl2tax25 = dict(zip(tax25.primary_label.astype(str), tax25.class_name))
        ta25_taxa = np.array([lbl2tax25.get(str(l), "?") for l in ta25_lbls])
    else:
        ta25_embs = np.zeros((0, 1536), dtype=np.float32); ta25_taxa = np.array([])

    # Build 2026 labeled SS (multi-label)
    ss26_embs, ss26_taxa_sets = build_2026_labeled_ss()

    # Combine single-label (one-hot)
    tax_idx = {t: i for i, t in enumerate(TAXA)}
    def make_y_single(taxa_arr):
        Y = np.zeros((len(taxa_arr), 5), dtype=np.float32)
        for i, t in enumerate(taxa_arr):
            if t in tax_idx: Y[i, tax_idx[t]] = 1.0
        return Y
    def make_y_multi(taxa_sets):
        Y = np.zeros((len(taxa_sets), 5), dtype=np.float32)
        for i, ts in enumerate(taxa_sets):
            for t in ts:
                if t in tax_idx: Y[i, tax_idx[t]] = 1.0
        return Y

    ta26_Y = make_y_single(ta26_taxa)
    ta25_Y = make_y_single(ta25_taxa)
    ss26_Y = make_y_multi(ss26_taxa_sets)

    # Filter unknown taxa
    def keep_valid(embs, Y):
        m = Y.sum(axis=1) > 0
        return embs[m], Y[m]
    ta26_embs, ta26_Y = keep_valid(ta26_embs, ta26_Y)
    ta25_embs, ta25_Y = keep_valid(ta25_embs, ta25_Y)
    ss26_embs, ss26_Y = keep_valid(ss26_embs, ss26_Y)

    X = np.concatenate([ta26_embs, ta25_embs, ss26_embs], axis=0)
    Y = np.concatenate([ta26_Y, ta25_Y, ss26_Y], axis=0)
    print(f"\nCombined pool: {X.shape[0]} samples  (TA26 {len(ta26_Y)}, TA25 {len(ta25_Y)}, SS26 {len(ss26_Y)})")
    print("Per-taxon positives:", dict(zip(TAXA, Y.sum(axis=0).astype(int))))

    # Split 90/10
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    n_val = int(len(X) * 0.1)
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_vl, Y_vl = X[val_idx], Y[val_idx]
    print(f"Train {len(X_tr)}  Val {len(X_vl)}")

    # Class weights for BCE (inverse frequency)
    pos_freq = Y_tr.sum(axis=0) / len(Y_tr)
    pos_weight = torch.from_numpy((1.0 / (pos_freq + 1e-6)).clip(max=100)).float().to(DEVICE)
    print(f"pos_weight: {dict(zip(TAXA, pos_weight.cpu().numpy().round(2)))}")

    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    ds_vl = TensorDataset(torch.from_numpy(X_vl), torch.from_numpy(Y_vl))
    loader_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=2)
    loader_vl = DataLoader(ds_vl, batch_size=BATCH, shuffle=False, num_workers=2)

    model = TaxonHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_auc = 0; patience = 0
    hist = []

    for ep in range(1, EPOCHS + 1):
        model.train(); t_loss = 0; t_n = 0
        for x, y in loader_tr:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item() * x.size(0); t_n += x.size(0)
        sched.step()
        model.eval(); all_p = []; all_y = []
        with torch.no_grad():
            for x, y in loader_vl:
                x = x.to(DEVICE)
                p = torch.sigmoid(model(x)).cpu().numpy()
                all_p.append(p); all_y.append(y.numpy())
        P = np.concatenate(all_p); Yv = np.concatenate(all_y)
        aucs = []
        for t in range(5):
            if Yv[:, t].sum() == 0: continue
            aucs.append(roc_auc_score(Yv[:, t], P[:, t]))
        macro_auc = np.mean(aucs)
        # Per-taxon
        per_t_auc = []
        for t in range(5):
            if Yv[:, t].sum() == 0: per_t_auc.append(float("nan")); continue
            per_t_auc.append(roc_auc_score(Yv[:, t], P[:, t]))
        hist.append({"epoch": ep, "loss": t_loss / t_n, "macro_auc": macro_auc,
                     **{f"auc_{TAXA[t]}": per_t_auc[t] for t in range(5)}})
        print(f"  ep {ep:02d}  loss {t_loss/t_n:.4f}  macro_AUC {macro_auc:.4f}  "
              f"per-taxon: {[f'{v:.2f}' for v in per_t_auc]}", flush=True)
        if macro_auc > best_auc:
            best_auc = macro_auc; patience = 0
            # Build species_to_taxon using 2026 taxonomy
            primary26 = pd.read_csv(DATA26 / "sample_submission.csv").columns[1:].tolist()
            lbl2tax_ = dict(zip(tax26.primary_label.astype(str), tax26.class_name))
            species_to_taxon = np.array([tax_idx.get(lbl2tax_.get(p, "Aves"), 0) for p in primary26], dtype=np.int64)
            torch.save({
                "state_dict": model.state_dict(),
                "species_to_taxon": species_to_taxon,
                "TAXA": TAXA,
                "epoch": ep, "val_macro_auc": macro_auc,
                "source": "2025+2026 taxon pool (exp49c)",
            }, OUT / "taxon_head_v49.pt")
        else:
            patience += 1
            if patience >= 8: break
    print(f"\nBest macro_AUC: {best_auc:.4f}")
    with open(OUT / "exp49c_history.json", "w") as f:
        json.dump({"history": hist, "best_auc": best_auc}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
