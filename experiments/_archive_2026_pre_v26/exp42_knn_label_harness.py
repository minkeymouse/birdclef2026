"""
exp42 — Phase B1 validation: does site-centering improve Perch-embedding kNN
label agreement on labeled SS?

Premise test: two SS windows that are acoustically similar should share labels.
Site shortcut (exp25: site-classifier acc 0.999) may dominate raw Perch cosine.
If site-centering lifts kNN agreement@k meaningfully over raw Perch, the
label-error / pseudo-confidence mechanism is worth building out.

Inputs:
  experiments/exp21_outputs/perch_cache/full_perch_arrays.npz   (emb, scores)
  experiments/exp21_outputs/perch_cache/full_perch_meta.parquet (row_id, filename, site, hour_utc)
  data/birdclef-2026/train_soundscapes_labels.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/data/birdclef2026")
CACHE_NPZ = ROOT / "experiments/exp21_outputs/perch_cache/full_perch_arrays.npz"
CACHE_META = ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet"
LABELS_CSV = ROOT / "data/birdclef-2026/train_soundscapes_labels.csv"
OUT_DIR = ROOT / "experiments/exp42_outputs"
OUT_DIR.mkdir(exist_ok=True)

K_VALUES = [5, 10, 20]

# ─── Load cache + labels, join on row_id ──────────────────────────────────────
arr = np.load(CACHE_NPZ)
emb = arr["emb"].astype(np.float32)          # (N, 1536)
meta = pd.read_parquet(CACHE_META)           # N rows
assert len(meta) == emb.shape[0]

labels = pd.read_csv(LABELS_CSV).drop_duplicates()
# row_id in meta looks like "BC2026_Train_0001_S08_20250606_030007_5"
# labels has filename + start/end as "00:00:00" / "00:00:05"
def to_end_sec(s):
    h, m, sec = map(int, s.split(":"))
    return h * 3600 + m * 60 + sec
labels["end_sec"] = labels["end"].map(to_end_sec)
labels["row_id"] = labels["filename"].str.replace(".ogg", "", regex=False) + "_" + labels["end_sec"].astype(str)

joined = meta.merge(labels[["row_id", "primary_label"]], on="row_id", how="inner")
print(f"cached windows : {len(meta)}")
print(f"label rows     : {len(labels)}")
print(f"join rows      : {len(joined)}")

# align emb to joined order
pos = pd.Series(range(len(meta)), index=meta["row_id"].values)
idx = pos.loc[joined["row_id"].values].values
emb = emb[idx]
meta = joined.reset_index(drop=True)
meta["label_set"] = meta["primary_label"].str.split(";").apply(lambda xs: frozenset(x.strip() for x in xs))

print(f"usable rows    : {len(meta)}")
print(f"sites          : {meta['site'].value_counts().to_dict()}")

# ─── Normalize embeddings + build centered variants ──────────────────────────
def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n

raw = l2(emb)
# site centering: subtract mean embedding per site, then renorm
site_mean = meta.groupby("site").apply(lambda g: emb[g.index.values].mean(0)).to_dict()
site_vec = np.stack([site_mean[s] for s in meta["site"].values])
site_c = l2(emb - site_vec)
# file centering: subtract mean embedding per file (very strong; may kill signal)
file_mean = meta.groupby("filename").apply(lambda g: emb[g.index.values].mean(0)).to_dict()
file_vec = np.stack([file_mean[f] for f in meta["filename"].values])
file_c = l2(emb - file_vec)

# ─── kNN agreement with same-file exclusion ──────────────────────────────────
def agreement_at_k(X, labels_list, files, sites, k, cross_site=False):
    """For each window, find k nearest neighbors EXCLUDING same-file windows,
       optionally excluding same-site, compute fraction sharing ≥1 label."""
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(n, k + 400), metric="cosine").fit(X)
    _, I = nn.kneighbors(X)
    agree = np.zeros(n)
    coverage = np.zeros(n)
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        neigh = []
        for j in I[i]:
            if j == i or files[j] == files[i]:
                continue
            if cross_site and sites[j] == sites[i]:
                continue
            neigh.append(j)
            if len(neigh) >= k:
                break
        if not neigh:
            continue
        q_labels = labels_list[i]
        hits = sum(1 for j in neigh if labels_list[j] & q_labels)
        agree[i] = hits / len(neigh)
        coverage[i] = 1 if len(neigh) >= k else len(neigh) / k
        used[i] = True
    if used.sum() == 0:
        return 0.0, 0.0, 0
    return agree[used].mean(), coverage[used].mean(), used.sum()

labels_list = meta["label_set"].tolist()
files = meta["filename"].values
sites = meta["site"].values

print("\n" + "=" * 78)
print(f"{'method':<22} {'k=5':>8} {'k=10':>8} {'k=20':>8}  | cross-site k=5/10/20  (n)")
print("=" * 78)
results = {}
for name, X in [("raw Perch", raw),
                ("site-centered", site_c),
                ("file-centered", file_c)]:
    row = {"method": name}
    line = f"{name:<22} "
    for k in K_VALUES:
        a, _, _ = agreement_at_k(X, labels_list, files, sites, k, cross_site=False)
        row[f"agree@{k}"] = a
        line += f"{a:.4f}  "
    line += "| "
    for k in K_VALUES:
        a, _, n_used = agreement_at_k(X, labels_list, files, sites, k, cross_site=True)
        row[f"xsite@{k}"] = a
        row[f"xsite_n@{k}"] = int(n_used)
        line += f"{a:.4f} "
    line += f" (n={row['xsite_n@5']}/{row['xsite_n@10']}/{row['xsite_n@20']})"
    print(line)
    results[name] = row

# random baseline: if labels co-occur at rate p, a random pair shares ≥1 with prob p
# compute pairwise expected agreement (lower bound)
n = len(labels_list)
label_ser = meta["label_set"]
# sample 2000 random pairs (different files)
rng = np.random.default_rng(0)
hits = 0; total = 0
for _ in range(5000):
    i, j = rng.integers(0, n, size=2)
    if files[i] == files[j]:
        continue
    if labels_list[i] & labels_list[j]:
        hits += 1
    total += 1
print(f"\nrandom-pair baseline (diff-file): {hits/total:.4f}  (n={total})")

# save
import json
with open(OUT_DIR / "results.json", "w") as f:
    json.dump({"results": results, "random_baseline": hits/total, "n_rows": n}, f, indent=2, default=str)
print(f"\nsaved → {OUT_DIR}/results.json")
