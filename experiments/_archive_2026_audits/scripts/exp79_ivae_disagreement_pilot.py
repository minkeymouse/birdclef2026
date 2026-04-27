#!/usr/bin/env python3
"""exp79 — iVAE disagreement-pseudo-label PILOT on labeled SS (739 rows).

Question: when Perch confidently calls a window "Aves species X" but the
ground truth is non-Aves, does iVAE-kNN (cos-sim to 55 valid train-positive
centroids) actually identify the row as non-Aves?

If YES → iVAE-kNN is a reliable disagreement oracle and the user's plan
(expand to 127k unlabeled, find systematic Perch→non-Aves errors, pseudo-
label, retrain SED with 2025-BG) is viable.
If NO → iVAE z is dominated by site/acoustic-environment factors, not
species, and the disagreement signal is noise.

Inputs:
  exp76_outputs/mel_cache.npz     (739, 16, 128)   pooled mel of labeled rows
  exp43a_outputs/perch_ss_all*    Perch scores aligned by row_id
  data/.../train_soundscapes_labels.csv   GT
  model-weights/ivae_*.{pt,npz}   exp78 artifacts (reuse for consistency)

Outputs:
  exp79_outputs/disagreement_pilot.csv (per-row analysis)
  exp79_outputs/confusion_taxon.csv    (GT × Perch × iVAE confusion)
  printed summary
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP76 = ROOT / "experiments/_audits_post_v26/exp76_outputs"
EXP43A = ROOT / "experiments/_data_pipelines/exp43a_outputs"
MW = ROOT / "model-weights"
OUT = ROOT / "experiments/_audits_post_v26/exp79_outputs"
OUT.mkdir(exist_ok=True, parents=True)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")
def parse_meta(fn):
    m = FNAME_RE.match(fn); return (m.group(2), int(m.group(4)[:2])) if m else (None, -1)


def build_ss_data():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g[["site","hour"]] = sc_g.filename.apply(lambda f: pd.Series(parse_meta(f)))
    rng = np.random.RandomState(SEED); files = sorted(sc_g.filename.unique()); rng.shuffle(files)
    eval_files = set(files[:11])
    sc_g["split"] = ["eval" if f in eval_files else "train" for f in sc_g.filename]
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g.lbls):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_g, Y, primary, l2i


class IVAEEnc(nn.Module):
    def __init__(self, in_dim, z_dim, n_aux, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256, 2 * z_dim))
        self.aux_mlp = nn.Sequential(
            nn.Linear(n_aux, 64), nn.GELU(),
            nn.Linear(64, 2 * z_dim))
        self.z_dim = z_dim
    def encode(self, x):
        h = self.enc(x)
        mu_q, _ = h.chunk(2, dim=-1)
        return mu_q


def main():
    print("=== exp79: iVAE disagreement-pseudo-label PILOT ===\n")

    # 1. Load labeled SS data
    sc_g, Y, primary, l2i = build_ss_data()
    print(f"labeled SS rows: {len(sc_g)}, classes: {len(primary)}")

    # 2. Load taxonomy
    tax = pd.read_csv(DATA / "taxonomy.csv")
    label2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([label2tax.get(p, "?") for p in primary])
    print(f"taxon distribution: {pd.Series(species_taxon).value_counts().to_dict()}")

    # 3. Load mel + iVAE artifacts
    mel = np.load(EXP76 / "mel_cache.npz")["mel"]   # (739, 16, 128)
    ck = torch.load(MW / "ivae_encoder.pt", map_location=DEVICE, weights_only=False)
    stats = np.load(MW / "ivae_mel_stats.npz")
    cent_d = np.load(MW / "ivae_z_centroids.npz")
    train_mean = stats["mean"].astype(np.float32)
    train_std = stats["std"].astype(np.float32)
    z_centroids = cent_d["centroids"].astype(np.float32)   # (234, 32)
    cent_valid = cent_d["valid"].astype(bool)              # (234,)
    in_dim = int(ck["in_dim"]); z_dim = int(ck["z_dim"]); n_aux = int(ck["n_aux"])
    print(f"iVAE: in_dim={in_dim} z_dim={z_dim} valid_centroids={cent_valid.sum()}/234")

    # 4. iVAE encode all 739 rows
    enc = IVAEEnc(in_dim, z_dim, n_aux).to(DEVICE).eval()
    enc.load_state_dict(ck["encoder_state_dict"], strict=False)

    X = mel.reshape(len(sc_g), -1).astype(np.float32)
    X = (X - train_mean) / train_std
    with torch.no_grad():
        Z = enc.encode(torch.from_numpy(X).to(DEVICE)).cpu().numpy()
    print(f"Z: {Z.shape}")

    # 5. Load Perch scores aligned with sc_g row_ids
    perch_emb = np.load(EXP43A / "perch_ss_all.npz")
    perch_scores = perch_emb["scores"]                            # (127k, 234) logits
    perch_meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(perch_meta["row_id"].values)}
    P_sc = np.zeros((len(sc_g), 234), dtype=np.float32)
    miss = 0
    for i, rid in enumerate(sc_g.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: P_sc[i] = perch_scores[j]
        else: miss += 1
    print(f"Perch alignment: {len(sc_g) - miss}/{len(sc_g)} rows matched ({miss} missing)")

    perch_prob = 1.0 / (1.0 + np.exp(-P_sc))   # sigmoid

    # 6. Per-row: Perch top-1 species + iVAE-kNN top-1 species
    perch_top1 = perch_prob.argmax(axis=1)
    perch_top1_taxon = species_taxon[perch_top1]
    perch_top1_prob = perch_prob[np.arange(len(sc_g)), perch_top1]

    # iVAE-kNN: cosine sim to valid centroids
    z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    c_norm = z_centroids / (np.linalg.norm(z_centroids, axis=1, keepdims=True) + 1e-8)
    cos = z_norm @ c_norm.T                       # (n, 234)
    # mask invalid centroids
    cos_valid = cos.copy()
    cos_valid[:, ~cent_valid] = -np.inf
    ivae_top1 = cos_valid.argmax(axis=1)
    ivae_top1_taxon = species_taxon[ivae_top1]
    ivae_top1_cos = cos_valid[np.arange(len(sc_g)), ivae_top1]

    # 7. Ground truth taxon per row: union of taxa among positives
    def gt_taxon_set(yi):
        idx = np.where(yi == 1)[0]
        return set(species_taxon[idx]) if len(idx) > 0 else set()
    gt_taxa = [gt_taxon_set(Y[i]) for i in range(len(sc_g))]
    gt_has_nonaves = np.array([("Aves" not in s and len(s) > 0) or any(t != "Aves" for t in s) for s in gt_taxa])
    gt_only_aves = np.array([s == {"Aves"} for s in gt_taxa])
    gt_has_some_nonaves = np.array([any(t != "Aves" for t in s) for s in gt_taxa])

    # 8. Disagreement Q1: rows where GT has non-Aves, Perch top1=Aves
    perch_says_aves = (perch_top1_taxon == "Aves")
    ivae_says_aves = (ivae_top1_taxon == "Aves")

    print("\n=== Q1: Perch hallucinated Aves on a row that has non-Aves GT ===")
    perch_wrong_taxon = perch_says_aves & gt_has_some_nonaves
    print(f"  rows: {perch_wrong_taxon.sum()}/{len(sc_g)} ({100*perch_wrong_taxon.mean():.1f}%)")
    # On those, does iVAE correctly say non-Aves?
    ivae_correct = perch_wrong_taxon & (~ivae_says_aves)
    ivae_also_wrong = perch_wrong_taxon & ivae_says_aves
    print(f"  iVAE correctly says non-Aves: {ivae_correct.sum()} ({100*ivae_correct.sum()/max(perch_wrong_taxon.sum(),1):.1f}%)")
    print(f"  iVAE also says Aves (no signal): {ivae_also_wrong.sum()}")

    # 9. Disagreement Q2: rows where BOTH agree on Aves but GT is pure non-Aves
    print("\n=== Q2: Both Perch and iVAE say Aves (does GT confirm?) ===")
    both_aves = perch_says_aves & ivae_says_aves
    print(f"  rows where both say Aves: {both_aves.sum()}/{len(sc_g)}")
    on_those_gt_has_aves = sum(1 for i in np.where(both_aves)[0] if "Aves" in gt_taxa[i])
    on_those_gt_aves_only = sum(1 for i in np.where(both_aves)[0] if gt_taxa[i] == {"Aves"})
    print(f"    of those, GT contains Aves: {on_those_gt_has_aves}")
    print(f"    of those, GT == only Aves: {on_those_gt_aves_only}")

    # 10. Q3: Perch says Aves, iVAE says non-Aves — which taxon does iVAE prefer? GT validation
    print("\n=== Q3: Perch=Aves, iVAE=non-Aves — does iVAE pick the right non-Aves taxon? ===")
    mask = perch_says_aves & (~ivae_says_aves)
    print(f"  rows: {mask.sum()}")
    if mask.sum() > 0:
        for tax_target in ["Amphibia", "Insecta", "Mammalia", "Reptilia"]:
            sel = mask & (ivae_top1_taxon == tax_target)
            if sel.sum() == 0: continue
            gt_match = sum(1 for i in np.where(sel)[0] if tax_target in gt_taxa[i])
            print(f"    iVAE→{tax_target}: {sel.sum()} rows, GT contains {tax_target}: {gt_match} ({100*gt_match/sel.sum():.1f}%)")

    # 11. Q4: confusion table — GT taxon × Perch top-1 taxon × iVAE top-1 taxon
    rows = []
    for i in range(len(sc_g)):
        gts = gt_taxa[i]
        gt_label = "+".join(sorted(gts)) if gts else "(none)"
        rows.append({
            "row_id": sc_g.row_id.iloc[i],
            "site": sc_g.site.iloc[i],
            "split": sc_g.split.iloc[i],
            "gt_taxa": gt_label,
            "perch_top1": primary[perch_top1[i]],
            "perch_top1_taxon": perch_top1_taxon[i],
            "perch_top1_prob": float(perch_top1_prob[i]),
            "ivae_top1": primary[ivae_top1[i]],
            "ivae_top1_taxon": ivae_top1_taxon[i],
            "ivae_top1_cos": float(ivae_top1_cos[i]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "disagreement_pilot.csv", index=False)
    print(f"\nSaved per-row table → {OUT / 'disagreement_pilot.csv'}")

    # 12. Q5: As a baseline — "iVAE-kNN beats Perch" or NOT on detecting non-Aves?
    # Per non-Aves taxon, AUC of (Perch's "anti-Aves" score) vs (iVAE's matching-taxon centroid score)
    print("\n=== Q5: Detection AUC for non-Aves taxa (per-row binary GT-has-this-taxon) ===")
    print(f"  {'taxon':<12} {'n_pos':>6} {'Perch_taxon_AUC':>16} {'iVAE_taxon_AUC':>15}")
    from sklearn.metrics import roc_auc_score
    for tax_target in ["Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        gt_target = np.array([tax_target in gt_taxa[i] for i in range(len(sc_g))])
        n_pos = gt_target.sum()
        if n_pos < 5: print(f"  {tax_target:<12} {n_pos:>6}  (too few positives)"); continue
        # Perch: max sigmoid prob over species in this taxon
        idx = np.where(species_taxon == tax_target)[0]
        perch_taxon_score = perch_prob[:, idx].max(axis=1)
        # iVAE: max cos-sim over valid centroids in this taxon
        valid_idx = np.where(cent_valid & (species_taxon == tax_target))[0]
        if len(valid_idx) == 0:
            ivae_score = np.zeros(len(sc_g))
        else:
            ivae_score = cos[:, valid_idx].max(axis=1)
        try:
            p_auc = roc_auc_score(gt_target, perch_taxon_score)
            i_auc = roc_auc_score(gt_target, ivae_score)
            print(f"  {tax_target:<12} {n_pos:>6} {p_auc:>16.4f} {i_auc:>15.4f}")
        except Exception as e:
            print(f"  {tax_target:<12} error: {e}")

    # 13. Q6: per-class disagreement AUC for ANY-Aves-vs-not
    gt_anyaves = np.array(["Aves" in gt_taxa[i] for i in range(len(sc_g))])
    perch_aves_score = perch_prob[:, species_taxon == "Aves"].max(axis=1)
    ivae_aves_score = cos[:, cent_valid & (species_taxon == "Aves")].max(axis=1) if (cent_valid & (species_taxon == "Aves")).sum() > 0 else np.zeros(len(sc_g))
    if gt_anyaves.sum() > 5 and gt_anyaves.sum() < len(sc_g) - 5:
        from sklearn.metrics import roc_auc_score
        try:
            print(f"\n=== Q6: Aves-detection AUC (Perch vs iVAE) ===")
            print(f"  Perch Aves-max  AUC: {roc_auc_score(gt_anyaves, perch_aves_score):.4f}")
            print(f"  iVAE  Aves-max  AUC: {roc_auc_score(gt_anyaves, ivae_aves_score):.4f}")
        except Exception as e:
            print(f"AUC error: {e}")

    # 14. Q7: among the 4 disagreement quadrants, what does GT say?
    print("\n=== Q7: 4-quadrant matrix (Perch_says_Aves × iVAE_says_Aves) — by GT ===")
    for p_a in [True, False]:
        for i_a in [True, False]:
            mask = (perch_says_aves == p_a) & (ivae_says_aves == i_a)
            n = mask.sum()
            if n == 0: continue
            # GT contains Aves?
            gt_aves = sum(1 for i in np.where(mask)[0] if "Aves" in gt_taxa[i])
            gt_nonaves = sum(1 for i in np.where(mask)[0] if any(t != "Aves" for t in gt_taxa[i]))
            quad = f"P={'A' if p_a else 'X'}_iV={'A' if i_a else 'X'}"
            print(f"  {quad:<10} n={n:>4}  GT_has_Aves={gt_aves} ({100*gt_aves/n:.0f}%)  GT_has_nonAves={gt_nonaves} ({100*gt_nonaves/n:.0f}%)")

    print("\n=== Pilot done ===\n")


if __name__ == "__main__":
    main()
