"""exp147b — Build site-stratified pseudo v8.

Strategy: pseudo entry (file, end_sec, class) kept ONLY if:
  1. v33 score > 0.50 (high confidence)
  2. Class appears with score > 0.40 in 2+ DIFFERENT sites in unlabeled SS
  3. Total count for class capped at 5000 (avoid Aves dominance)
  4. Per site cap to enforce diversity

Compare with v3 (351k entries, single-site pollution) and v7 (241k, external-verified).
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import re
import warnings; warnings.filterwarnings("ignore")

from experiments._data_pipelines._shared.data import build_primaries
from experiments._data_pipelines._shared.constants import DATA, N_CLS

PRIMARY_LABELS, l2i = build_primaries()

z = np.load(ROOT / "experiments/_data_pipelines/exp126_outputs/v33_unlabeled_scores.npz", allow_pickle=True)
v33 = z["v33"].astype(np.float32)  # (N_rows, 234)
filenames = z["filenames"]
end_secs = z["end_secs"]
print(f"v33 unlabeled: {v33.shape}")

def site_of(fn):
    m = re.search(r"S(\d+)", fn); return f"S{int(m.group(1)):02d}" if m else "??"

sites = np.array([site_of(f) for f in filenames])
print("Sites:", sorted(set(sites)))
print("Files per site:")
for s, n in pd.Series(sites).value_counts().sort_index().items():
    print(f"  {s}: {n}")

THRESH_HIGH = 0.50
THRESH_LOW = 0.40

print(f"\n[1/4] Building per-class site-coverage map (threshold {THRESH_LOW})")
# For each class, count distinct sites where v33 score > THRESH_LOW
class_site_count = {}
for c in range(N_CLS):
    high_rows = v33[:, c] > THRESH_LOW
    if high_rows.sum() == 0:
        class_site_count[c] = (0, [])
        continue
    sites_with = sorted(set(sites[high_rows]))
    class_site_count[c] = (len(sites_with), sites_with)

# Categorize classes
print("\nClass site coverage distribution:")
bins = {0: 0, 1: 0, 2: 0, "3+": 0}
for c, (n, _) in class_site_count.items():
    if n == 0: bins[0] += 1
    elif n == 1: bins[1] += 1
    elif n == 2: bins[2] += 1
    else: bins["3+"] += 1
for k, v in bins.items(): print(f"  {k} sites: {v} classes")

# For pseudo v8, only keep classes that have high pred in 2+ sites
multi_site_classes = [c for c, (n, _) in class_site_count.items() if n >= 2]
print(f"\nMulti-site (≥2) classes: {len(multi_site_classes)}/{N_CLS}")

# Build pseudo entries
print(f"\n[2/4] Building pseudo entries (v33 > {THRESH_HIGH}, class in multi-site list)")
PER_CLASS_CAP = 5000
PER_SITE_PER_CLASS_CAP = 1000
entries = []
for c in multi_site_classes:
    cls_label = PRIMARY_LABELS[c]
    rows_high = np.where(v33[:, c] > THRESH_HIGH)[0]
    if len(rows_high) == 0: continue
    # Site-balanced sampling
    site_groups = {}
    for r in rows_high:
        site_groups.setdefault(sites[r], []).append(r)
    total = 0
    for site, rs in site_groups.items():
        n_take = min(len(rs), PER_SITE_PER_CLASS_CAP)
        # rank by score desc
        rs_sorted = sorted(rs, key=lambda r: -v33[r, c])[:n_take]
        for r in rs_sorted:
            entries.append({
                "filename": filenames[r],
                "start": f"00:{end_secs[r]-5:02d}:00" if end_secs[r] >= 5 else "00:00:00",
                "end": f"00:{end_secs[r]:02d}:00",
                "primary_label": cls_label,
            })
            total += 1
    if total > PER_CLASS_CAP:
        # Trim to PER_CLASS_CAP randomly
        cls_entries = [e for e in entries if e["primary_label"] == cls_label]
        keep_idx = np.random.RandomState(0).choice(len(cls_entries), PER_CLASS_CAP, replace=False)
        keep_set = set(keep_idx.tolist())
        # Replace by keeping only selected
        # (less elegant but simple)
        new_entries = []
        cls_seen = 0
        for e in entries:
            if e["primary_label"] != cls_label:
                new_entries.append(e)
            else:
                if cls_seen in keep_set:
                    new_entries.append(e)
                cls_seen += 1
        entries = new_entries

print(f"  Total pseudo entries: {len(entries)}")

# Format start time properly
def secs_to_hms(s):
    h = s // 3600; m = (s % 3600) // 60; sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"
for e in entries:
    end_s = int(e["end"].replace(":", "")[-4:-2]) * 1 if e["end"].count(":") == 2 else 0
# Better: rebuild start/end properly using end_secs
entries2 = []
for c in multi_site_classes:
    cls_label = PRIMARY_LABELS[c]
    rows_high = np.where(v33[:, c] > THRESH_HIGH)[0]
    if len(rows_high) == 0: continue
    site_groups = {}
    for r in rows_high:
        site_groups.setdefault(sites[r], []).append(r)
    cls_added = 0
    for site, rs in site_groups.items():
        rs_sorted = sorted(rs, key=lambda r: -v33[r, c])[:PER_SITE_PER_CLASS_CAP]
        for r in rs_sorted:
            es = int(end_secs[r])
            ss = max(0, es - 5)
            entries2.append({
                "filename": filenames[r],
                "start": secs_to_hms(ss),
                "end": secs_to_hms(es),
                "primary_label": cls_label,
            })
            cls_added += 1
            if cls_added >= PER_CLASS_CAP: break
        if cls_added >= PER_CLASS_CAP: break

df = pd.DataFrame(entries2)
print(f"\n[3/4] DataFrame: {len(df)} entries")
print(f"  unique classes: {df.primary_label.nunique()}")
print(f"  per-class counts top:")
print(df.primary_label.value_counts().head(15))

# Per-taxon summary
tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
df["taxon"] = df["primary_label"].map(tax)
print(f"\nPer-taxon entries:")
print(df.taxon.value_counts())

# Save
out_path = DATA / "pseudo_soundscapes_labels_v8.csv"
df.to_csv(out_path, index=False)
print(f"\n[4/4] Saved {out_path}")
print(f"  size: {out_path.stat().st_size/1e6:.1f} MB")
