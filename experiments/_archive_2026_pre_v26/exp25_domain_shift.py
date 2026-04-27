#!/usr/bin/env python3
"""
exp25 — domain-shift / site-bias diagnostics.

Research goal (informed by exp21): the v1 pipeline OOF drops from 0.977 in-sample
to 0.488 across-site. Hypothesis: Perch embeddings carry strong site-specific
information, allowing the prior fusion to memorize, but they don't transfer
across sites.

Diagnostics performed:
  D1: Adversarial site classifier — train LogReg on Perch emb to predict site.
      If accuracy >> chance, the embedding is site-confounded.
  D2: Per-site OOF macro-AUC — leave-one-site-out, identify hardest sites.
  D3: Per-site class occurrence — which texture taxa are site-specific?
  D4: Embedding clustering — silhouette of site clusters in PCA space.
  D5: Class-AUC variance across folds — which classes generalize worst?

Output: written conclusions for exp30+ design (does iVAE / domain adaptation help?).
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
OUT = ROOT / "experiments" / "exp25_outputs"
OUT.mkdir(parents=True, exist_ok=True)
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"


def per_class_auc(y_true, y_score):
    aucs = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        if 0 < y_true[:, j].sum() < len(y_true):
            try: aucs[j] = roc_auc_score(y_true[:, j], y_score[:, j])
            except ValueError: pass
    return aucs


def main():
    t0 = time.time()
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    primary = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary)}
    n_classes = len(primary)

    fre = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")
    def parse_lbls(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_clean = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
                .apply(lambda s: sorted({lbl for x in s for lbl in parse_lbls(x)}))
                .reset_index(name="label_list"))
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (sc_clean["filename"].str.replace(".ogg", "", regex=False)
                          + "_" + sc_clean["end_sec"].astype(str))
    Y_SC = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, labs in enumerate(sc_clean["label_list"]):
        for lbl in labs:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    meta_full = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    arr = np.load(EXP21 / "full_perch_arrays.npz")
    scores_full = arr["scores"].astype(np.float32)
    emb_full = arr["emb"].astype(np.float32)
    sites_full = meta_full["site"].to_numpy()
    hours_full = meta_full["hour_utc"].to_numpy()
    sc_idx = sc_clean.set_index("row_id")
    Y_FULL = np.stack([Y_SC[sc_idx.index.get_loc(rid)] for rid in meta_full["row_id"]])

    cn_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    n_sites = len(np.unique(sites_full))
    print(f"Eval rows: {len(emb_full)}, unique sites: {n_sites}, unique hours: {len(np.unique(hours_full))}")

    # ───── D1: Adversarial site classifier ─────
    print("\n=== D1: Adversarial site classifier ===")
    site_to_idx = {s: i for i, s in enumerate(sorted(np.unique(sites_full)))}
    y_site = np.array([site_to_idx[s] for s in sites_full])
    chance = 1.0 / n_sites
    print(f"Chance accuracy: {chance:.3f}")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    for tr, va in skf.split(emb_full, y_site):
        sc = StandardScaler(); Et = sc.fit_transform(emb_full[tr]); Ev = sc.transform(emb_full[va])
        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        try:
            clf.fit(Et, y_site[tr])
            pred = clf.predict(Ev)
            accs.append(accuracy_score(y_site[va], pred))
        except Exception as e:
            print(f"  fold failed: {e}")
    site_acc = float(np.mean(accs))
    print(f"Site classifier accuracy (CV): {site_acc:.3f} (chance={chance:.3f}, ratio={site_acc/chance:.1f}x)")

    # Hour classifier (sanity)
    y_hour = hours_full.astype(int)
    valid = y_hour >= 0
    n_hours = len(np.unique(y_hour[valid]))
    chance_h = 1.0 / n_hours
    accs_h = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(emb_full[valid], y_hour[valid]):
        Et = StandardScaler().fit_transform(emb_full[valid][tr])
        Ev = StandardScaler().fit(emb_full[valid][tr]).transform(emb_full[valid][va])
        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
        try:
            clf.fit(Et, y_hour[valid][tr])
            accs_h.append(accuracy_score(y_hour[valid][va], clf.predict(Ev)))
        except Exception: pass
    hour_acc = float(np.mean(accs_h)) if accs_h else float("nan")
    print(f"Hour classifier accuracy (CV): {hour_acc:.3f} (chance={chance_h:.3f}, ratio={hour_acc/chance_h:.1f}x)")

    # ───── D2: Per-site leave-one-out OOF macro-AUC (raw Perch only) ─────
    print("\n=== D2: Per-site leave-one-out OOF AUC (raw Perch) ===")
    per_site_auc = {}
    for s in sorted(np.unique(sites_full)):
        va = np.where(sites_full == s)[0]
        if len(va) < 5: continue
        keep = Y_FULL[va].sum(0) > 0
        if keep.sum() == 0: continue
        try:
            auc = roc_auc_score(Y_FULL[va][:, keep], scores_full[va][:, keep], average="macro")
            per_site_auc[str(s)] = float(auc)
        except ValueError: pass
    df_site = pd.DataFrame.from_dict(per_site_auc, orient="index", columns=["auc"]).sort_values("auc")
    print(df_site)
    print(f"  mean: {df_site.auc.mean():.4f}, std: {df_site.auc.std():.4f}, "
          f"min: {df_site.auc.min():.4f}, max: {df_site.auc.max():.4f}")

    # ───── D3: Per-site class occurrence (texture taxa specifically) ─────
    print("\n=== D3: Texture-taxa site-specificity ===")
    texture_idx = np.array([label_to_idx[c] for c in primary
                            if cn_map.get(c) in {"Amphibia", "Insecta"}], dtype=np.int32)
    site_class_count = pd.DataFrame(
        np.zeros((n_sites, len(texture_idx)), dtype=np.int32),
        index=sorted(np.unique(sites_full)),
        columns=[primary[i] for i in texture_idx],
    )
    for s in sorted(np.unique(sites_full)):
        m = sites_full == s
        site_class_count.loc[s] = Y_FULL[m][:, texture_idx].sum(0)
    # For each texture class, compute Gini-like concentration over sites
    cls_site_concentration = {}
    for ci, c in enumerate(site_class_count.columns):
        counts = site_class_count[c].to_numpy()
        if counts.sum() == 0: continue
        p = counts / counts.sum()
        gini = float(1 - (p ** 2).sum())  # 0 = single site, near 1 = uniform
        n_sites_present = int((counts > 0).sum())
        cls_site_concentration[c] = {"n_sites": n_sites_present, "gini": gini, "total_pos": int(counts.sum())}
    df_conc = pd.DataFrame.from_dict(cls_site_concentration, orient="index").sort_values("gini")
    print("Most site-concentrated texture classes (low Gini = single-site):")
    print(df_conc.head(15))

    # ───── D4: Embedding silhouette by site (in PCA space) ─────
    print("\n=== D4: Embedding silhouette by site ===")
    sc = StandardScaler(); E = sc.fit_transform(emb_full)
    pca = PCA(n_components=20); Z = pca.fit_transform(E)
    try:
        sil = silhouette_score(Z, y_site, sample_size=min(2000, len(Z)))
        print(f"  silhouette_score(site, PCA20): {sil:.4f}  "
              f"(>0 = sites form clusters in embedding space)")
    except Exception as e:
        sil = None
        print(f"  silhouette failed: {e}")

    # ───── D5: Class-AUC variance across folds ─────
    print("\n=== D5: Class-AUC variance across GroupKFold-by-site folds ===")
    gkf = GroupKFold(n_splits=5)
    pc_per_fold = []
    for tr, va in gkf.split(scores_full, groups=sites_full):
        pc_per_fold.append(per_class_auc(Y_FULL[va], scores_full[va]))
    pc_mat = np.array(pc_per_fold)  # (5, n_classes)
    pc_mean = np.nanmean(pc_mat, axis=0)
    pc_std = np.nanstd(pc_mat, axis=0)
    df_var = pd.DataFrame({
        "primary": primary,
        "class_name": [cn_map.get(p, "?") for p in primary],
        "n_pos": Y_FULL.sum(0),
        "auc_mean": pc_mean,
        "auc_std": pc_std,
    })
    df_var = df_var[df_var["n_pos"] > 0].copy()
    print("Highest-variance classes (worst generalization):")
    print(df_var.sort_values("auc_std", ascending=False).head(10).round(3).to_string(index=False))
    print("\nMost stable classes (well generalized):")
    print(df_var.sort_values("auc_std").head(10).round(3).to_string(index=False))

    # ───── Save and conclude ─────
    results = {
        "n_eval_rows": int(len(emb_full)),
        "n_sites": n_sites,
        "site_classifier_acc": site_acc,
        "site_chance": chance,
        "site_acc_above_chance_ratio": site_acc / chance,
        "hour_classifier_acc": hour_acc,
        "hour_chance": chance_h,
        "per_site_auc_mean": float(df_site["auc"].mean()),
        "per_site_auc_min": float(df_site["auc"].min()),
        "per_site_auc_max": float(df_site["auc"].max()),
        "per_site_auc": df_site["auc"].to_dict(),
        "embedding_silhouette_site": sil,
        "class_auc_std_mean": float(df_var["auc_std"].mean()),
        "class_auc_std_max": float(df_var["auc_std"].max()),
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    df_site.to_csv(OUT / "per_site_auc.csv")
    df_conc.to_csv(OUT / "texture_site_concentration.csv")
    df_var.to_csv(OUT / "class_auc_variance.csv", index=False)

    # ───── Conclusion ─────
    print("\n" + "=" * 64)
    print("CONCLUSIONS for exp30+ design:")
    print("=" * 64)
    if site_acc / chance > 5:
        print(f"  * Strong site signal in embedding (acc {site_acc:.2f} vs chance {chance:.2f}).")
        print("    → Domain adaptation (iVAE / DANN / CORAL) is justified.")
    else:
        print(f"  * Weak site signal in embedding (acc {site_acc:.2f} vs chance {chance:.2f}).")
        print("    → Site confound is mild; iVAE may be unnecessary.")
    if df_site["auc"].std() > 0.05:
        print(f"  * Per-site AUC varies a lot (std {df_site['auc'].std():.3f}).")
        print("    → Some sites are genuinely harder; per-site tuning could help.")
    if df_var["auc_std"].mean() > 0.1:
        print(f"  * High per-class fold variance (mean std {df_var['auc_std'].mean():.3f}).")
        print("    → Many classes are unstable. Ensembles + better priors needed.")
    print(f"\nWrote results to {OUT}")
    print(f"Wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
