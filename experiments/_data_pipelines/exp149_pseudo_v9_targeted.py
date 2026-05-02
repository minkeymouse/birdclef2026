"""exp149 — Pseudo v9: targeted to multi-site under-rep classes only.

Filter:
  - Only classes in "dead-zone" identified in exp145 (multi-site, v33 random)
  - v33 score > 0.6 OR exp50 score > 0.6 (high single-model agreement)
  - 2+ different sites with high pred (cross-site coverage)
  - Per-class cap 2000 (avoid imbalance)
  - Per-site cap 500 (force diversity)
  - DROP if class is well-handled by v33 (already AUC > 0.7 on labeled SS train)

Target: 25 multi-site moderate-data Amphibia + 8 Mammalia + 14 multi-site Insecta = 47 target classes.
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import re
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import DATA, N_CLS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all_lbl = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
mask_tr = (ss_all_lbl.split == "train").values
ss_all_lbl["site"] = ss_all_lbl["filename"].str.extract(r"(S\d+)")[0].apply(lambda x: f"S{int(x[1:]):02d}")

# Compute v33 per-class AUC on labeled SS train (using v33 cached scores)
perch_lab = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/perch_prob_labeled.npz")["prob"]
exp50_lab = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/exp50_scores_labeled.npz")["scores"]
v33_lab = 0.7 * perch_lab + 0.3 * exp50_lab

Y = np.zeros((len(ss_all_lbl), N_CLS), dtype=np.float32)
for i, lbls in enumerate(ss_all_lbl.lbls):
    for lbl in lbls:
        if lbl in l2i: Y[i, l2i[lbl]] = 1.0

# Per-class AUC on TRAIN portion (to classify "dead-zone" vs "well-handled")
v33_train_auc = {}
for c in range(N_CLS):
    yc = Y[mask_tr, c]
    if 0 < yc.sum() < len(yc):
        pc = v33_lab[mask_tr, c]
        if pc.std() > 1e-9:
            v33_train_auc[c] = roc_auc_score(yc, pc)

# Identify dead-zone classes: multi-site labeled SS pos AND v33 train AUC < 0.65
class_sites = {}
for c in range(N_CLS):
    cls = PRIMARY_LABELS[c]
    rows = [i for i, lbls in enumerate(ss_all_lbl.lbls) if cls in lbls]
    sites = sorted(set(ss_all_lbl.iloc[rows].site)) if rows else []
    class_sites[c] = sites

target_classes = []
tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
for c in range(N_CLS):
    cls = PRIMARY_LABELS[c]
    n_sites = len(class_sites[c])
    if n_sites < 2: continue  # need cross-site supervision
    auc = v33_train_auc.get(c, 1.0)
    if auc > 0.7: continue  # already well-handled
    target_classes.append(c)
    print(f"  TARGET: {cls:>14s} ({tax.get(cls, '?')[:5]:5s}) sites={class_sites[c]} v33_train_auc={auc:.3f}")

print(f"\nTotal target classes: {len(target_classes)}")
print(f"Per-taxon target:")
target_taxa = [tax.get(PRIMARY_LABELS[c], '?') for c in target_classes]
for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
    print(f"  {t:10s}: {target_taxa.count(t)}")


# Now build pseudo for these classes ONLY
print("\n[2/3] Loading v33 unlabeled scores for pseudo build")
z = np.load(ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz", allow_pickle=True)
v33_unlab = z["v33"].astype(np.float32)
filenames_unlab = z["filenames"]
end_secs_unlab = z["end_secs"]
sites_unlab = np.array([f"S{int(re.search(r'S(\d+)', f).group(1)):02d}" for f in filenames_unlab])

# Also load exp50 unlabeled scores if available
exp50_unlab_path = ROOT/"experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
if exp50_unlab_path.exists():
    z2 = np.load(exp50_unlab_path)
    exp50_unlab = z2["scores"].astype(np.float32)
    print(f"  exp50 unlab loaded: {exp50_unlab.shape}")
else:
    exp50_unlab = None
    print(f"  exp50 unlab not available")

THRESH = 0.55
PER_CLASS_CAP = 1500
PER_SITE_PER_CLASS_CAP = 250

print(f"\n[3/3] Building pseudo (thresh {THRESH}, per_cls cap {PER_CLASS_CAP}, per_site_cls cap {PER_SITE_PER_CLASS_CAP})")
def secs_to_hms(s):
    h = s // 3600; m = (s % 3600) // 60; sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

entries = []
for c in target_classes:
    cls_label = PRIMARY_LABELS[c]
    # Combined criterion: v33 > 0.55 OR exp50 > 0.6 (looser to capture Amph signal)
    score = v33_unlab[:, c]
    if exp50_unlab is not None and exp50_unlab.shape[1] == N_CLS:
        score = np.maximum(score, exp50_unlab[:, c])
    rows_high = np.where(score > THRESH)[0]
    if len(rows_high) == 0: continue
    # Site distribution
    site_groups = {}
    for r in rows_high:
        site_groups.setdefault(sites_unlab[r], []).append(r)
    if len(site_groups) < 2: continue  # need cross-site
    cls_added = 0
    for site, rs in site_groups.items():
        rs_sorted = sorted(rs, key=lambda r: -score[r])[:PER_SITE_PER_CLASS_CAP]
        for r in rs_sorted:
            es = int(end_secs_unlab[r])
            ss = max(0, es - 5)
            entries.append({
                "filename": filenames_unlab[r],
                "start": secs_to_hms(ss),
                "end": secs_to_hms(es),
                "primary_label": cls_label,
            })
            cls_added += 1
            if cls_added >= PER_CLASS_CAP: break
        if cls_added >= PER_CLASS_CAP: break

df = pd.DataFrame(entries)
print(f"\nTotal entries: {len(df)}")
print(f"Per-class:")
print(df.primary_label.value_counts().head(20))
print(f"\nPer-taxon:")
df["taxon"] = df["primary_label"].map(tax)
print(df.taxon.value_counts())

out_path = DATA / "pseudo_soundscapes_labels_v9.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved {out_path} ({out_path.stat().st_size/1e3:.1f} KB)")
