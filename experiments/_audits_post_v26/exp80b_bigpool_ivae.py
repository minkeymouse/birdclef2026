#!/usr/bin/env python3
"""exp80b — Train iVAE on a much larger pool (labeled + 30k unlabeled SS rows)
to test whether bigger pool dilutes site fingerprint.

Result: Aves AUC +0.10 vs small-pool, but Amphibia/Insecta drop. Bigger
pool dilutes already-site-fingerprint centroid signal. Doesn't fix the
fundamental issue; only Aves benefits from added Aves diversity.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, aux_matrix, load_labeled_mel,
                        DATA, EXP43A, EXP80, N_CLS, SEED, FNAME_RE, TAXA)
from _lib.ivae import train_full, encode_all, DEVICE
from _lib.mel import extract_pool_many

OUT = EXP80
OUT.mkdir(exist_ok=True, parents=True)
N_UNLAB_FILES = 2500


def sample_unlab_files() -> list[str]:
    perch_meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    labeled_files = set(pd.read_csv(DATA / "train_soundscapes_labels.csv").filename.unique())
    unlab_all = [f for f in perch_meta.filename.unique() if f not in labeled_files]
    by_site = {}
    for f in unlab_all:
        m = FNAME_RE.match(f); s = m.group(2) if m else "?"
        by_site.setdefault(s, []).append(f)
    rng = np.random.RandomState(SEED)
    per_site = max(1, N_UNLAB_FILES // len(by_site))
    picks = []
    for s, fs in by_site.items():
        picks.extend(rng.choice(fs, size=min(per_site, len(fs)), replace=False).tolist())
    remain = N_UNLAB_FILES - len(picks)
    if remain > 0:
        extra = list(set(unlab_all) - set(picks))
        if remain < len(extra):
            picks.extend(rng.choice(extra, size=remain, replace=False).tolist())
    return picks[:N_UNLAB_FILES]


def main():
    print("=== exp80b: big-pool iVAE training ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    mel_lab = load_labeled_mel()
    print(f"labeled mel: {mel_lab.shape}", flush=True)

    cache_path = OUT / f"unlab_mel_{N_UNLAB_FILES}.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        mel_unlab = d["mel"]; fnames_unlab = d["fnames"]
        print(f"loaded cached unlab mel: {mel_unlab.shape}", flush=True)
    else:
        picks = sample_unlab_files()
        print(f"sampled {len(picks)} unlabeled files", flush=True)
        ss_dir = DATA / "train_soundscapes"
        mel_unlab, fnames_unlab = extract_pool_many([str(ss_dir / f) for f in picks])
        np.savez_compressed(cache_path, mel=mel_unlab, fnames=np.array(fnames_unlab))
        print(f"cached → {cache_path}", flush=True)

    # site/hour from filename
    sites_unlab = []; hours_unlab = []
    for fn in fnames_unlab:
        m = FNAME_RE.match(fn)
        sites_unlab.append(m.group(2) if m else "?")
        hours_unlab.append(int(m.group(4)[:2]) if m else 0)

    # Combine
    X_lab = mel_lab.reshape(len(sc_g), -1).astype(np.float32)
    X_unl = mel_unlab.reshape(len(mel_unlab), -1).astype(np.float32)
    X_all = np.concatenate([X_lab, X_unl], axis=0)
    is_lab = np.concatenate([np.ones(len(sc_g), bool), np.zeros(len(mel_unlab), bool)])
    is_eval = np.concatenate([sc_g.split.values == "eval", np.zeros(len(mel_unlab), bool)])

    # All-row aux: build site list across both
    sites_all = sorted(set(sc_g.site.tolist() + sites_unlab))
    s2i = {s: i for i, s in enumerate(sites_all)}
    aux = np.zeros((len(X_all), len(sites_all) + 1), dtype=np.float32)
    site_arr_all = np.concatenate([sc_g.site.values, np.array(sites_unlab)])
    hour_arr_all = np.concatenate([sc_g.hour.values, np.array(hours_unlab)])
    for i, s in enumerate(site_arr_all):
        if s in s2i: aux[i, s2i[s]] = 1.0
    aux[:, -1] = hour_arr_all / 24.0

    fit_mask = ~is_eval
    print(f"\ncombined: {X_all.shape}, lab/unl/eval: {is_lab.sum()}/{(~is_lab).sum()}/{is_eval.sum()}",
          flush=True)
    print(f"iVAE fit rows: {fit_mask.sum()}", flush=True)

    print("Training big iVAE (z=32, ep=30, batch=4096)...", flush=True)
    model, mu, sd = train_full(X_all, aux, fit_mask, z_dim=32, hidden=512,
                                epochs=30, batch=4096, beta=0.05, verbose_every=5)
    Z_all = encode_all(model, X_all, mu, sd, batch=4096)
    print(f"Z: {Z_all.shape}", flush=True)
    Z_lab = Z_all[is_lab]

    np.savez_compressed(OUT / "bigpool_z.npz",
                         Z_lab=Z_lab, Z_all=Z_all,
                         is_lab=is_lab, is_eval=is_eval,
                         sites=site_arr_all, hours=hour_arr_all,
                         train_mean=mu, train_std=sd)
    torch.save({"state_dict": model.state_dict(), "z_dim": 32, "n_aux": aux.shape[1],
                "in_dim": X_all.shape[1], "sites": sites_all},
                OUT / "bigpool_ivae.pt")
    print(f"saved → {OUT}/bigpool_z.npz, bigpool_ivae.pt", flush=True)

    # ===== Eval AUC: big-pool iVAE on 122 held-out =====
    tr_mask = sc_g.split.values == "train"
    Y_tr = Y[tr_mask]; Z_tr = Z_lab[tr_mask]
    centroids = np.zeros((N_CLS, 32), dtype=np.float32)
    cv = np.zeros(N_CLS, dtype=bool)
    for c in range(N_CLS):
        if Y_tr[:, c].sum() >= 3:
            centroids[c] = Z_tr[Y_tr[:, c] == 1].mean(0); cv[c] = True

    z_n = Z_lab / (np.linalg.norm(Z_lab, axis=1, keepdims=True) + 1e-8)
    c_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    cos = z_n @ c_n.T
    cos[:, ~cv] = -np.inf

    ev_mask = sc_g.split.values == "eval"
    print("\n=== Per-taxon detection AUC on 122 held-out ===")
    print(f"  {'taxon':<12} {'n_pos':>6} {'AUC':>10}  {'small_ref':>10}")
    refs = {"Aves": 0.763, "Amphibia": 0.709, "Insecta": 0.923,
            "Mammalia": 0.516, "Reptilia": 0.652}
    for t in TAXA:
        gt = (Y[:, sp_taxon == t].sum(axis=1) > 0).astype(np.uint8)
        n = int(gt[ev_mask].sum())
        if n < 3 or n == ev_mask.sum():
            print(f"  {t:<12} {n:>6}    --"); continue
        valid = np.where(cv & (sp_taxon == t))[0]
        if len(valid) == 0:
            print(f"  {t:<12} {n:>6}    -- (no centroid)"); continue
        score = cos[ev_mask][:, valid].max(axis=1)
        try:
            auc = roc_auc_score(gt[ev_mask], score)
            print(f"  {t:<12} {n:>6} {auc:>10.4f}  {refs.get(t, np.nan):>10.4f}  Δ={auc - refs.get(t, np.nan):+.4f}")
        except Exception as e:
            print(f"  {t:<12} error: {e}")


if __name__ == "__main__":
    main()
