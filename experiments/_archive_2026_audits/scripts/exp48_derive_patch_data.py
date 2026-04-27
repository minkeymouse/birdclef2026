#!/usr/bin/env python3
"""Derive site prior + cluster map for Kaggle notebook patch.

Outputs a JSON file with:
  site_prior: {site_code: [234-len list of P(species|site) normalized [0,1]]}
  cluster_map: {target_idx: [top-3 Aves trigger idx]}

Uses ONLY labeled SS train split (55 files) + train_audio Perch preds on
labeled SS for cluster derivation — same setup as exp48e, fully leak-free.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/exp43a_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
OUT = ROOT / "experiments/exp49_outputs"
OUT.mkdir(exist_ok=True)
SEED = 42; EVAL_N = 11

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_site(fn):
    m = FNAME_RE.match(fn); return m.group(2) if m else None


def build_train_split():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g["site"] = sc_g["filename"].apply(parse_site)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_N])
    sc_train = sc_g[~sc_g.filename.isin(eval_files)].reset_index(drop=True)
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_train), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_train["lbls"]):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_train, Y, primary, l2i


def align_43a(df):
    d = np.load(EXP43A / "perch_ss_all.npz")
    scs = d["scores"]
    meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(meta["row_id"].values)}
    S = np.zeros((len(df), scs.shape[1]), np.float32)
    for i, rid in enumerate(df["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = scs[j]
    return S


def align_old(df, p):
    if not p.exists(): return None
    pred = np.load(p)["preds"].astype(np.float32)
    om = pd.read_parquet(ROOT / "experiments/exp21_outputs/perch_cache/full_perch_meta.parquet")
    if len(om) != pred.shape[0]: return None
    rid2i = {r: i for i, r in enumerate(om["row_id"].values)}
    out = np.full((len(df), pred.shape[1]), np.nan, dtype=np.float32)
    for i, rid in enumerate(df["row_id"].values):
        j = rid2i.get(rid, -1)
        if j >= 0: out[i] = pred[j]
    return np.nan_to_num(out, nan=0.0)


def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
def zs(X): m,s = X.mean(0,keepdims=True), X.std(0,keepdims=True)+1e-8; return (X-m)/s

def gauss_pf(scores, df, sigma=0.5):
    out = np.zeros_like(scores)
    for fn in df["filename"].unique():
        m = (df["filename"] == fn).values
        b = scores[m]
        for c in range(b.shape[1]):
            out[m, c] = gaussian_filter1d(b[:, c], sigma=sigma, mode="nearest")
    return out


def main():
    sc_train, Y_train, primary, l2i = build_train_split()
    print(f"train SS: {len(sc_train)} rows, sites {sorted(sc_train.site.unique())}")

    tax = pd.read_csv(DATA / "taxonomy.csv")
    lbl2tax = dict(zip(tax["primary_label"].astype(str), tax["class_name"]))
    species_taxon = np.array([lbl2tax.get(p, "?") for p in primary])

    # ── Site prior: for each site, P(species at site) / max ──
    sites = sorted(sc_train.site.unique())
    prior = {}
    for site in sites:
        mask = (sc_train.site == site).values
        Y_s = Y_train[mask]
        cnt = Y_s.sum(axis=0).astype(np.float32)
        if cnt.max() > 0:
            prior[site] = (cnt / cnt.max()).tolist()
        else:
            prior[site] = [1.0] * len(primary)
    print(f"Site prior built for {len(prior)} sites")

    # ── Cluster map: for each rare target, top-3 Aves with highest mean
    #     prediction on target's positive rows (v12-style base) ──
    S_p = align_43a(sc_train); P_perch = sigmoid(S_p)
    S29 = np.nan_to_num(align_old(sc_train, EXP29 / "val_scores.npz"), nan=0)
    zP = zs(P_perch); z29 = zs(S29)
    v12_train = sigmoid(gauss_pf(0.8*zP + 0.2*z29, sc_train, 0.5))

    aves_idx = np.array([c for c in range(len(primary)) if species_taxon[c] == "Aves"])
    cluster_map = {}
    for c in range(len(primary)):
        if species_taxon[c] in ("Aves", "?"): continue
        pos = np.where(Y_train[:, c] == 1)[0]
        if len(pos) < 3: continue
        mp = v12_train[pos][:, aves_idx].mean(axis=0)
        top3 = aves_idx[np.argsort(mp)[-3:]].tolist()
        cluster_map[str(c)] = top3
    print(f"Cluster map: {len(cluster_map)} rare targets with top-3 triggers")

    # ── Whitelist: only apply cluster rewrite where it actually helped in leak-free eval ──
    # From exp48e best per-class breakdown, we saw 1491113 had −0.447 damage.
    # Be conservative: exclude any target where the cluster trigger self-correlation
    # is below an empirical threshold on training data.
    whitelist = []
    for c_str, trig in cluster_map.items():
        c = int(c_str)
        pos = np.where(Y_train[:, c] == 1)[0]
        if len(pos) == 0: continue
        neg = np.where(Y_train[:, c] == 0)[0]
        # Cluster-min score on pos vs neg
        cs_pos = v12_train[pos][:, trig].min(axis=1).mean()
        cs_neg = v12_train[neg][:, trig].min(axis=1).mean()
        if cs_pos > cs_neg * 1.3:  # trigger min must be 30% higher on positives
            whitelist.append(c_str)
    print(f"Cluster whitelist: {len(whitelist)} / {len(cluster_map)} targets passed trigger-discriminability check")

    cluster_map_whitelisted = {c: cluster_map[c] for c in whitelist}

    # Save
    out_data = {
        "primary_labels": primary,
        "sites": sites,
        "site_prior": prior,
        "cluster_map": cluster_map,
        "cluster_map_whitelisted": cluster_map_whitelisted,
        "species_taxon": species_taxon.tolist(),
    }
    with open(OUT / "patch_data.json", "w") as f:
        json.dump(out_data, f)
    print(f"Saved → {OUT / 'patch_data.json'}")

    # Quick sanity: print a few entries
    print("\nSample site prior entries (S08, S22):")
    for site in ["S08", "S22"]:
        if site in prior:
            probs = np.array(prior[site])
            top_n = np.argsort(probs)[-10:][::-1]
            print(f"  {site}: top-10 species {[(primary[i], f'{probs[i]:.2f}') for i in top_n]}")
    print("\nSample cluster_map (first 5, whitelisted):")
    for c_str in list(whitelist)[:5]:
        c = int(c_str)
        print(f"  {primary[c]:<12} ({species_taxon[c]:<8}) → {[primary[t] for t in cluster_map[c_str]]}")


if __name__ == "__main__":
    main()
