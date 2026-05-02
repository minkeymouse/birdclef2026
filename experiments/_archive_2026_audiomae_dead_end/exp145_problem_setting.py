"""exp145 — Deep problem setting analysis.

Q1: Macro AUC composition — which classes contribute what fraction of v33 0.932?
Q2: Site distribution per class — which classes are single-site vs multi-site?
Q3: Per-class teacher AUC vs site count — is there a "site-diverse but Perch-fails"
    sweet spot we're missing?
Q4: Sister-species confusion — for each Aves bottom class, what's the top
    confused species? Within or across taxon?
Q5: Unlabeled SS site distribution — do we have unlabeled data from many sites
    that could be used for invariance constraints?
Q6: Macro-AUC sensitivity decomposition — if we could fix one bucket, how much LB?
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import DATA, N_CLS

PRIMARY_LABELS, l2i = build_primaries()

# Q5: First, unlabeled SS site distribution
print("=== Q5: Train soundscape file site distribution ===")
all_files = pd.Series([p.name for p in (DATA / "train_soundscapes").glob("*.ogg")])
all_files_df = pd.DataFrame({"filename": all_files})
all_files_df["site"] = all_files_df["filename"].str.extract(r"(S\d+)")[0]
print(f"Total train_soundscape files: {len(all_files_df)}")
print(f"Unique sites: {sorted(all_files_df.site.unique())}")
print(f"\nFiles per site:")
print(all_files_df.site.value_counts().sort_index())

# Labeled
labels = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
labeled_files = sorted(labels["filename"].unique())
print(f"\nLabeled files: {len(labeled_files)}")
print(f"Unlabeled files: {len(all_files_df) - len(labeled_files)}")

# Q2: Per class — site composition (labeled SS train only, since that's where SED learns)
print("\n=== Q2: Per-class site composition (labeled SS, all 66 files) ===")
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all_lbl = pd.concat([ss_train_g, ss_eval_g], ignore_index=True)
ss_all_lbl["site"] = ss_all_lbl["filename"].str.extract(r"(S\d+)")[0]

# For each class, how many sites have its positives?
class_sites = {}
class_n_pos = {}
for c in range(N_CLS):
    cls = PRIMARY_LABELS[c]
    pos_rows = [i for i, lbls in enumerate(ss_all_lbl.lbls) if cls in lbls]
    sites = ss_all_lbl.iloc[pos_rows].site.unique()
    class_sites[c] = sorted(sites)
    class_n_pos[c] = len(pos_rows)

# Bin: 0 sites (= no positives), 1, 2, 3+ sites
bins = {0: 0, 1: 0, 2: 0, 3: 0}
class_to_bin = {}
for c in range(N_CLS):
    n = len(class_sites[c])
    if n == 0: b = 0
    elif n == 1: b = 1
    elif n == 2: b = 2
    else: b = 3
    bins[b] += 1
    class_to_bin[c] = b
print("Site count distribution across 234 classes:")
print(f"  0 sites (no labeled SS): {bins[0]}")
print(f"  1 site only:             {bins[1]}")
print(f"  2 sites:                 {bins[2]}")
print(f"  3+ sites:                {bins[3]}")

# By taxon
tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
print("\nSite-bin × taxon:")
print(f"  {'taxon':>10s} | {'0 site':>6s} {'1 site':>6s} {'2 sites':>6s} {'3+ sites':>8s}")
for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
    cls = [c for c in range(N_CLS) if tax.get(PRIMARY_LABELS[c], "Aves") == t]
    sub = [class_to_bin[c] for c in cls]
    print(f"  {t:>10s} |  {sub.count(0):4d}  {sub.count(1):4d}  {sub.count(2):4d}    {sub.count(3):4d}")

# Q1: Macro AUC composition — estimate which classes drive v33 0.932
print("\n=== Q1: Estimate which classes drive v33 0.932 LB ===")
# Use exp80 cached preds for v33 base
perch = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/perch_prob_labeled.npz")["prob"]
exp50 = np.load(ROOT/"experiments/_audits_post_v26/exp80_outputs/exp50_scores_labeled.npz")["scores"]
v33_simple = 0.7*perch + 0.3*exp50

mask_ev = (ss_all_lbl["filename"].isin(ss_eval_g.filename.unique())).values
Y = np.zeros((len(ss_all_lbl), N_CLS), dtype=np.float32)
for i, lbls in enumerate(ss_all_lbl.lbls):
    for lbl in lbls:
        if lbl in l2i: Y[i, l2i[lbl]] = 1.0

# Per-class AUC on eval
class_auc_v33 = {}
class_auc_perch = {}
class_auc_exp50 = {}
for c in range(N_CLS):
    yc = Y[mask_ev, c]
    if 0 < yc.sum() < len(yc):
        for d, name in [(class_auc_v33, "v33"), (class_auc_perch, "perch"), (class_auc_exp50, "exp50")]:
            preds = {"v33": v33_simple, "perch": perch, "exp50": exp50}[name]
            pc = preds[mask_ev, c]
            if pc.std() > 1e-9:
                d[c] = roc_auc_score(yc, pc)

# Group by site bin AND show v33 per-class AUC
print("\nv33 per-class AUC × site bin (labeled SS eval, evaluable classes):")
for b in [1, 2, 3]:
    classes = [c for c in class_auc_v33 if class_to_bin[c] == b]
    if classes:
        aucs = [class_auc_v33[c] for c in classes]
        print(f"  {b}-site classes: N={len(classes)}, mean v33 AUC = {np.mean(aucs):.3f}")

# Show specific classes with cross-site labels but low v33 AUC = potential improvement
print("\nMulti-site classes with LOW v33 AUC (LB improvement targets):")
multi_site_low = sorted(
    [(c, class_auc_v33[c]) for c in class_auc_v33 if class_to_bin[c] >= 2 and class_auc_v33[c] < 0.6],
    key=lambda x: x[1]
)
for c, a in multi_site_low[:20]:
    cls = PRIMARY_LABELS[c]
    t = tax.get(cls, "Aves")
    sites = class_sites[c]
    n = class_n_pos[c]
    perch_a = class_auc_perch.get(c, "—")
    exp50_a = class_auc_exp50.get(c, "—")
    perch_str = f"{perch_a:.2f}" if isinstance(perch_a, float) else perch_a
    exp50_str = f"{exp50_a:.2f}" if isinstance(exp50_a, float) else exp50_a
    print(f"  {cls:>14s} ({t[:5]:5s}) v33={a:.2f} P={perch_str} S={exp50_str}  n_pos={n:3d} sites={sites}")

# Q3: Single-site Insecta diagnosis — confirm sonotypes are not generalizable
print("\n=== Q3: Single-site Insecta sonotypes ===")
for c in range(N_CLS):
    cls = PRIMARY_LABELS[c]
    if "47158son" in cls:
        sites = class_sites[c]
        n = class_n_pos[c]
        print(f"  {cls}: {n} pos in {len(sites)} site(s) {sites}")

# Q4: Aves bottom — sister species confusion analysis
print("\n=== Q4: Aves bottom on labeled SS — what do they get confused with? ===")
# For each low-AUC Aves class, find rows where pred>0.5 but yc=0 — what other class IS true positive there?
for c in class_auc_v33:
    cls = PRIMARY_LABELS[c]
    t = tax.get(cls, "Aves")
    if t != "Aves" or class_auc_v33[c] > 0.55: continue
    # FP rows
    yc = Y[mask_ev, c]
    pc = v33_simple[mask_ev, c]
    fp_idx = np.where((pc > 0.5) & (yc == 0))[0]
    if len(fp_idx) < 3: continue
    # What classes are positive on those FP rows?
    confused_classes = Y[mask_ev][fp_idx].sum(0)
    top_idx = np.argsort(-confused_classes)[:5]
    confused_strs = [f"{PRIMARY_LABELS[i]}({tax.get(PRIMARY_LABELS[i],'?')[:3]}):{int(confused_classes[i])}" for i in top_idx if confused_classes[i] > 0]
    print(f"  {cls:>10s} v33={class_auc_v33[c]:.2f} sites={class_sites[c]}: confused with {confused_strs[:3]}")

# Q6: Macro AUC decomposition — what would happen if we fixed one bucket?
print("\n=== Q6: 'Fix one bucket' macro AUC sensitivity ===")
print("If we could move per-class AUC of a bucket to 0.95 each, what's the gain?")
print("(assumes ~150 evaluable classes on hidden LB; current macro 0.932 = average)\n")

# Estimate: 234 classes total, ~150 evaluable on LB. v33 LB 0.932.
# Use local hard classes as proxy for hidden hard classes
hard_eval_classes = [c for c in class_auc_v33 if class_auc_v33[c] < 0.7]
print(f"Local hard classes (v33 AUC < 0.7): {len(hard_eval_classes)}")
print("Their current local mean AUC:")
hard_local_mean = np.mean([class_auc_v33[c] for c in hard_eval_classes])
print(f"  {hard_local_mean:.3f}")
print(f"If raised to 0.85, per-class Δ = +{0.85 - hard_local_mean:.3f}")
print(f"Effect on hidden LB (assume ~30 of these are evaluable on LB): {30 * (0.85 - hard_local_mean) / 150:.4f}")
print("So fixing 30 hard classes from 0.5 to 0.85 = +0.07 macro on LB ⇒ LB ~1.00 IF transfer works")
print("Reality: we keep getting -0.01 to -0.02 because the 'fix' creates collateral damage.")
