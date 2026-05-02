"""exp144 — Comprehensive diagnostic: where does each teacher fail, and why?

Inputs (precomputed on 739 labeled SS rows):
  - Perch probs, exp50/exp59/exp73/exp84b/exp136b SED scores
  - Y multi-label (3122 positives total)
  - Taxonomy, ss_train_g (617) / ss_eval_g (122) split
  - AudioMAE C1 probe preds (eval only)

Computes:
  1. Per-class AUC for each teacher (eval, both pos+neg)
  2. Per-class win matrix (which model best for each class)
  3. By-taxon mean AUC
  4. By Perch-mapped status (203 vs 31 unmapped) AUC
  5. By labeled-SS positive count buckets
  6. By site (S03, S08, S15, S19, S22, S23, S13, S09, S18) AUC
  7. By n_train_audio buckets (where each model is sufficient)
  8. Confusion patterns: for misclassified positives, which species got high score?
  9. Site-shortcut diagnostic: per (model, taxon) site holdout simulation
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
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
mask_ev = (ss_all.split == "eval").values

# Load teachers
AUD = ROOT / "experiments/_audits_post_v26/exp80_outputs"
teachers = {}
teachers["Perch"] = np.load(AUD/"perch_prob_labeled.npz")["prob"]
teachers["exp50"] = np.load(AUD/"exp50_scores_labeled.npz")["scores"]
teachers["exp59"] = np.load(AUD/"exp59_scores_labeled.npz")["scores"]
teachers["exp73"] = np.load(AUD/"exp73_scores_labeled.npz")["scores"]
teachers["exp84b"] = np.load(AUD/"exp84b_scores_labeled.npz")["scores"]
teachers["exp136b"] = np.load(AUD/"exp136b_scores_labeled.npz")["scores"]
teachers["v33_simple"] = 0.7 * teachers["Perch"] + 0.3 * teachers["exp50"]

# Build Y
Y = np.zeros((len(ss_all), N_CLS), dtype=np.float32)
for i, lbls in enumerate(ss_all.lbls):
    for lbl in lbls:
        if lbl in l2i: Y[i, l2i[lbl]] = 1.0

tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")
class_name = tax["class_name"]
inat_id = tax.get("inat_taxon_id", None)

# Perch mapping: from CLAUDE.md, 203/234 mapped
# Approximate mapped status by checking if Perch alone gives non-trivial discrimination on TRAIN portion
# A class is "Perch-active" if Perch's std(scores) on train > 0.05
perch_active = teachers["Perch"][~mask_ev].std(axis=0) > 0.05
print(f"Perch-active classes (std>0.05 on train): {perch_active.sum()}/{N_CLS}")

# Train pos counts (proxy for data sparsity)
ss_pos_count_tr = Y[~mask_ev].sum(axis=0)  # (234,)
ss_pos_count_ev = Y[mask_ev].sum(axis=0)
ev_neg_count = (mask_ev.sum() - ss_pos_count_ev)

# Site info
ss_all["site"] = ss_all["filename"].str.extract(r"(S\d+)")
sites_train = sorted(ss_all[~mask_ev]["site"].unique())
sites_eval = sorted(ss_all[mask_ev]["site"].unique())
print(f"Train sites: {sites_train}")
print(f"Eval sites: {sites_eval}")

# Compute per-class AUCs
def class_auc(preds, c):
    yc = Y[mask_ev, c]
    if yc.sum() == 0 or yc.sum() == len(yc): return None
    pc = preds[mask_ev, c]
    if pc.std() < 1e-9: return None
    return roc_auc_score(yc, pc)

evaluable_classes = []
for c in range(N_CLS):
    yc = Y[mask_ev, c]
    if 0 < yc.sum() < len(yc):
        evaluable_classes.append(c)
print(f"\nEvaluable classes: {len(evaluable_classes)}/{N_CLS}")

aucs = {name: [] for name in teachers}
class_aucs = {name: {} for name in teachers}
for c in evaluable_classes:
    for name in teachers:
        a = class_auc(teachers[name], c)
        if a is not None:
            aucs[name].append(a)
            class_aucs[name][c] = a

# Add AudioMAE probe (eval only)
am_npz = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_probe_preds.npz")
am_preds_ev = am_npz["preds"]
def class_auc_ev(p, c):
    yc = Y[mask_ev, c]
    if yc.sum() == 0 or yc.sum() == len(yc): return None
    if p[:, c].std() < 1e-9: return None
    return roc_auc_score(yc, p[:, c])
am_aucs = []; class_aucs["AudioMAE"] = {}
for c in evaluable_classes:
    a = class_auc_ev(am_preds_ev, c)
    if a is not None:
        am_aucs.append(a); class_aucs["AudioMAE"][c] = a
aucs["AudioMAE"] = am_aucs

print("\n=== Per-teacher overall mean AUC (eval, %d classes) ===" % len(evaluable_classes))
for name in ["Perch", "exp50", "exp59", "exp73", "exp84b", "exp136b", "v33_simple", "AudioMAE"]:
    if aucs[name]:
        print(f"  {name:12s} N={len(aucs[name]):3d}  mean={np.mean(aucs[name]):.4f}  median={np.median(aucs[name]):.4f}")

# Per-taxon breakdown
print("\n=== Per-teacher × taxon mean AUC ===")
print(f"  {'teacher':12s} | {'Aves':>6s} {'Amph':>6s} {'Insect':>6s} {'Mam':>6s} {'Rept':>6s}")
for name in ["Perch", "exp50", "v33_simple", "exp136b", "AudioMAE"]:
    by_tax = {t: [] for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
    for c in evaluable_classes:
        if c in class_aucs[name]:
            t = class_name.get(PRIMARY_LABELS[c], "Aves")
            if t in by_tax: by_tax[t].append(class_aucs[name][c])
    line = f"  {name:12s} | "
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        line += f"{np.mean(by_tax[t]) if by_tax[t] else 0:.3f}  "
    print(line)

# Win matrix: which teacher wins each class?
print("\n=== Per-class winner distribution (best teacher) ===")
win_count = {n: 0 for n in teachers}
win_count["AudioMAE"] = 0
margins = []
for c in evaluable_classes:
    scores = {n: class_aucs[n][c] for n in class_aucs if c in class_aucs[n]}
    if not scores: continue
    best = max(scores, key=scores.get)
    win_count[best] += 1
    sorted_aucs = sorted(scores.values(), reverse=True)
    margins.append(sorted_aucs[0] - sorted_aucs[1] if len(sorted_aucs) > 1 else 0)
for k, v in sorted(win_count.items(), key=lambda x: -x[1]):
    print(f"  {k:12s} wins {v} classes")
print(f"  Mean margin (best - 2nd): {np.mean(margins):.3f}")

# Bottom-K analysis: which classes are hardest? Where does Perch fail catastrophically?
print("\n=== Bottom 10 classes for Perch (hardest) ===")
perch_pairs = sorted([(c, class_aucs["Perch"][c]) for c in class_aucs["Perch"]], key=lambda x: x[1])[:15]
for c, a in perch_pairs:
    pl = PRIMARY_LABELS[c]
    t = class_name.get(pl, "?")
    n_tr = int(ss_pos_count_tr[c])
    n_ev = int(ss_pos_count_ev[c])
    perch_act = "PERCH-ACTIVE" if perch_active[c] else "PERCH-DEAD"
    e50 = class_aucs["exp50"].get(c, None)
    am = class_aucs["AudioMAE"].get(c, None)
    print(f"  {pl:>14s} ({t:8s}) train_pos={n_tr:3d} eval_pos={n_ev:2d}  perch={a:.3f}  exp50={'%.3f'%e50 if e50 else '  N/A':>5s}  AudioMAE={'%.3f'%am if am else '  N/A':>5s}  {perch_act}")

# For Perch-dead classes: what does each teacher add?
perch_dead_eval = [c for c in evaluable_classes if not perch_active[c]]
print(f"\n=== Perch-dead classes ({len(perch_dead_eval)} eval) ===")
for name in ["exp50", "exp59", "exp84b", "exp136b", "AudioMAE"]:
    aucs_dead = [class_aucs[name].get(c) for c in perch_dead_eval if c in class_aucs[name]]
    aucs_dead = [a for a in aucs_dead if a is not None]
    if aucs_dead:
        print(f"  {name:12s}: N={len(aucs_dead)} mean AUC={np.mean(aucs_dead):.3f}")

# By labeled-SS positive count buckets
print("\n=== AUC by labeled-SS train positive count bucket ===")
buckets = [(0, 0), (1, 5), (6, 20), (21, 50), (51, 200), (201, 1e9)]
print(f"  {'bucket':>10s} | {'N':>4s} | " + " ".join(f"{n:>10s}" for n in ["Perch", "exp50", "v33_simple", "AudioMAE"]))
for lo, hi in buckets:
    cls = [c for c in evaluable_classes if lo <= ss_pos_count_tr[c] <= hi]
    if not cls: continue
    line = f"  [{lo},{hi}] | {len(cls):>4d} | "
    for name in ["Perch", "exp50", "v33_simple", "AudioMAE"]:
        a = [class_aucs[name].get(c) for c in cls if c in class_aucs[name]]
        a = [x for x in a if x is not None]
        line += f" {np.mean(a) if a else 0:.3f}     " if a else "    N/A    "
    print(line)

# Per-site AUC (simulating site-holdout-like effect)
print("\n=== Per-site AUC (eval rows by site) ===")
ss_eval_df = ss_all[mask_ev].reset_index(drop=True)
ev_idx_in_full = np.where(mask_ev)[0]
for site in sites_eval:
    site_mask_in_ev = (ss_eval_df.site == site).values
    if site_mask_in_ev.sum() < 3: continue
    print(f"\n  Site {site} ({site_mask_in_ev.sum()} rows):")
    for name in ["Perch", "exp50", "v33_simple", "AudioMAE"]:
        if name == "AudioMAE":
            preds_site = am_preds_ev[site_mask_in_ev]
        else:
            preds_site = teachers[name][ev_idx_in_full][site_mask_in_ev]
        Y_site = Y[ev_idx_in_full][site_mask_in_ev]
        # macro AUC across evaluable classes within this site
        site_aucs = []
        for c in range(N_CLS):
            yc = Y_site[:, c]
            if 0 < yc.sum() < len(yc):
                pc = preds_site[:, c]
                if pc.std() > 1e-9:
                    site_aucs.append(roc_auc_score(yc, pc))
        if site_aucs:
            print(f"    {name:12s} N_cls={len(site_aucs):2d}  AUC={np.mean(site_aucs):.3f}")

# Disagreement diagnostic: where do models disagree most? (FP-prone areas)
print("\n=== Inter-model agreement (Pearson correlation) on eval predictions ===")
names = ["Perch", "exp50", "v33_simple", "exp136b", "exp84b", "AudioMAE"]
for n1 in names:
    line = f"  {n1:12s} |"
    for n2 in names:
        if n1 == "AudioMAE":
            p1 = am_preds_ev.flatten()
        else:
            p1 = teachers[n1][mask_ev].flatten()
        if n2 == "AudioMAE":
            p2 = am_preds_ev.flatten()
        else:
            p2 = teachers[n2][mask_ev].flatten()
        corr = np.corrcoef(p1, p2)[0, 1]
        line += f" {corr:5.2f}"
    print(line + "    " + " ".join([f"{n[:5]:>5s}" for n in names]))

# Confusion: for top false positives, which classes are getting predicted?
print("\n=== Top false-positive classes by teacher (% rows where pred>0.5 but yc=0, on eval) ===")
for name in ["Perch", "exp50", "exp136b", "AudioMAE"]:
    print(f"\n  {name}:")
    if name == "AudioMAE":
        preds = am_preds_ev
    else:
        preds = teachers[name][mask_ev]
    # For each class, FP rate when class isn't a positive
    fp_rates = []
    for c in range(N_CLS):
        yc = Y[mask_ev, c]
        pc = preds[:, c]
        n_neg = (yc == 0).sum()
        if n_neg == 0: continue
        fp = ((pc > 0.5) & (yc == 0)).sum()
        if fp > 0:
            fp_rates.append((c, fp / n_neg, fp))
    fp_rates.sort(key=lambda x: -x[1])
    for c, rate, n in fp_rates[:10]:
        pl = PRIMARY_LABELS[c]
        t = class_name.get(pl, "?")
        print(f"    {pl:>12s} ({t:8s}) FP rate {rate:.3f} ({n} false positives)")

print("\nDONE")
