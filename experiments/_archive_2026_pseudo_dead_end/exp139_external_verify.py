#!/usr/bin/env python3
"""exp139 — Cross-validate v3 pseudo with external (xeno-canto / iNat) audio.

Pipeline:
  1. Run Perch ONNX on ~1.4k external clips per species
  2. Build per-class external centroid (where ≥5 clips)
  3. For each v3 pseudo (row, class):
     compute cos_sim(unlabeled_row_emb, external_centroid_class)
     drop if sim < tau (likely FP)

External coverage:
  Amphibia 35/35, Mammalia 8/8, Reptilia 1/1, Aves 11/162, Insecta 3/28

Particularly useful for filtering v3 saturating Amphibia (22973/24279/555146)
and Aves compot1 — these have external ground truth.
Sonotypes 25/28 have NO external → no verification possible there.
"""
from __future__ import annotations
import os, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
from tqdm.auto import tqdm

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
EXT_DIR = ROOT / "data/external"
ONNX = Path("/tmp/perch_v2.onnx")
PERCH_UNLAB = ROOT / "experiments/_data_pipelines/exp43a_outputs/perch_ss_all.npz"
EXP125 = ROOT / "experiments/_data_pipelines/exp125_outputs/exp50_unlabeled_scores.npz"
V3_CSV = DATA / "pseudo_soundscapes_labels_v3.csv"
OUT = ROOT / "experiments/_data_pipelines/exp139_outputs"
OUT.mkdir(parents=True, exist_ok=True)
EXT_EMB_NPZ = OUT / "external_perch_emb.npz"
OUT_V6 = DATA / "pseudo_soundscapes_labels_v7.csv"

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_CLS = 234
MIN_CLIPS_FOR_CENTROID = 5
TAU_EXT_SIM = 0.3  # drop if cos_sim < tau (less strict than internal centroid since external is different distribution)


def read_5s(path):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != SR:
            import torchaudio.functional as TF
            import torch
            wav = TF.resample(torch.from_numpy(wav), sr, SR).numpy()
        if len(wav) >= WINDOW_SAMPLES: return wav[:WINDOW_SAMPLES]
        return np.pad(wav, (0, WINDOW_SAMPLES - len(wav))).astype(np.float32)
    except Exception:
        return None


def extract_external_embeddings():
    """Run Perch ONNX-GPU on all external clips. Return list of (species, emb) pairs."""
    if EXT_EMB_NPZ.exists():
        print(f"  Loading cached external embs from {EXT_EMB_NPZ}")
        d = np.load(EXT_EMB_NPZ, allow_pickle=True)
        return d["species"].astype(str), d["emb"]

    print("Loading Perch ONNX (GPU)...")
    sess = ort.InferenceSession(str(ONNX), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    species_dirs = sorted([d for d in EXT_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")])
    species_list = []
    emb_list = []

    BATCH = 32
    audio_buf = np.zeros((BATCH, WINDOW_SAMPLES), dtype=np.float32)
    queue = []  # (species, path)

    def flush():
        if not queue: return
        out = sess.run(["embedding"], {"inputs": audio_buf[:len(queue)]})[0]
        for k, (sp, _) in enumerate(queue):
            species_list.append(sp)
            emb_list.append(out[k].copy())
        queue.clear()

    for sp_dir in tqdm(species_dirs, desc="External clips"):
        sp = sp_dir.name
        for clip in sorted(sp_dir.glob("*")):
            if clip.suffix.lower() not in (".mp3", ".ogg", ".wav", ".m4a"):
                continue
            wav = read_5s(clip)
            if wav is None: continue
            audio_buf[len(queue)] = wav
            queue.append((sp, str(clip)))
            if len(queue) >= BATCH:
                flush()
    flush()

    species_arr = np.array(species_list, dtype="U16")
    emb_arr = np.stack(emb_list).astype(np.float32)
    np.savez_compressed(EXT_EMB_NPZ, species=species_arr, emb=emb_arr)
    print(f"  Saved → {EXT_EMB_NPZ}: {emb_arr.shape}")
    return species_arr, emb_arr


def main():
    print("=== exp139: External cross-validation of v3 pseudo ===\n", flush=True)

    # 1. Extract external embeddings (or load cache)
    print("Step 1: External Perch embeddings")
    ext_species, ext_emb = extract_external_embeddings()
    print(f"  Total external: {len(ext_species)} clips, embedding {ext_emb.shape}")
    print(f"  Species coverage: {len(set(ext_species))}\n")

    # 2. Build per-class centroid
    print("Step 2: Per-class external centroids")
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    sp2idx = {s: i for i, s in enumerate(primary)}
    n_per_species = {sp: int(np.sum(ext_species == sp)) for sp in set(ext_species)}
    centroids = {}
    for sp, n in n_per_species.items():
        if n >= MIN_CLIPS_FOR_CENTROID:
            mask = ext_species == sp
            centroids[sp] = ext_emb[mask].mean(axis=0)
    print(f"  Centroids built for {len(centroids)} species (≥{MIN_CLIPS_FOR_CENTROID} clips)")

    # Print centroid coverage by taxon
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    by_taxon = {}
    for sp in centroids:
        t = sp2tax.get(sp, "?")
        by_taxon[t] = by_taxon.get(t, 0) + 1
    print(f"  Coverage by taxon: {by_taxon}")

    # 3. Load unlabeled SS Perch embeddings
    print("\nStep 3: Load unlabeled SS Perch embeddings")
    perch_unlab = np.load(PERCH_UNLAB, mmap_mode="r")
    perch_emb_unlab = np.array(perch_unlab["emb"])

    # Build (filename, end_sec) → row_idx
    exp50_data = np.load(EXP125, allow_pickle=True)
    filenames_unlab = exp50_data["filenames"].astype(str)
    row_ids_unlab = exp50_data["row_ids"].astype(str)
    import re
    rid_to_idx = {}
    for i, rid in enumerate(row_ids_unlab):
        m = re.search(r"_(\d+)$", rid)
        if m:
            rid_to_idx[(filenames_unlab[i], int(m.group(1)))] = i

    # 4. Cross-check v3 pseudo-positives
    print("\nStep 4: Cross-check v3 pseudo against external centroids")
    df = pd.read_csv(V3_CSV)
    df["end"] = df["end"].astype(int)
    print(f"  v3 entries: {len(df)}")

    # Pre-normalize unlabeled embeddings
    unlab_n = perch_emb_unlab / (np.linalg.norm(perch_emb_unlab, axis=1, keepdims=True) + 1e-9)

    keep_mask = np.ones(len(df), dtype=bool)
    sim_values = np.full(len(df), np.nan, dtype=np.float32)
    has_centroid = np.zeros(len(df), dtype=bool)

    # For each pseudo entry, check vs centroid
    for i in range(len(df)):
        row = df.iloc[i]
        sp = row.primary_label
        if sp not in centroids: continue
        has_centroid[i] = True
        key = (row.filename, int(row.end))
        if key not in rid_to_idx: continue
        r_idx = rid_to_idx[key]
        # cos_sim = u_n · centroid_normed
        cent = centroids[sp]
        cent_n = cent / (np.linalg.norm(cent) + 1e-9)
        sim = float(unlab_n[r_idx] @ cent_n)
        sim_values[i] = sim
        if sim < TAU_EXT_SIM:
            keep_mask[i] = False

    n_with_centroid = int(has_centroid.sum())
    n_dropped = int((~keep_mask).sum())
    print(f"\n  Entries with external centroid: {n_with_centroid}")
    print(f"  Entries WITHOUT centroid (Insecta sonotype, missing Aves): {len(df) - n_with_centroid}")
    print(f"  Dropped (cos_sim < {TAU_EXT_SIM}): {n_dropped}")

    # Per-species drop stats for top-impacted
    print(f"\n  Top-15 species with most drops:")
    df["dropped"] = ~keep_mask
    drops_per_species = df[df.dropped & has_centroid].primary_label.value_counts().head(15)
    for sp, n in drops_per_species.items():
        total_for_sp = (df.primary_label == sp).sum()
        print(f"    {sp:<14} {sp2tax.get(sp,'?'):<10} dropped {n}/{total_for_sp} ({100*n/total_for_sp:.1f}%)")

    # Sim distribution for compot1 (saturating, has external)
    if "compot1" in centroids:
        compot1_sims = sim_values[(df.primary_label == "compot1").values & has_centroid]
        if len(compot1_sims) > 0:
            print(f"\n  compot1 (Aves saturating, 30k entries) sim distribution:")
            print(f"    mean {compot1_sims.mean():.3f}, median {np.median(compot1_sims):.3f}, "
                  f"p10 {np.percentile(compot1_sims, 10):.3f}, p25 {np.percentile(compot1_sims, 25):.3f}, "
                  f"p50 {np.percentile(compot1_sims, 50):.3f}, p75 {np.percentile(compot1_sims, 75):.3f}")

    # Sim distribution for 22973 (saturating Amphibia)
    if "22973" in centroids:
        sims_22973 = sim_values[(df.primary_label == "22973").values & has_centroid]
        if len(sims_22973) > 0:
            print(f"\n  22973 (Amphibia saturating, 16k entries) sim distribution:")
            print(f"    mean {sims_22973.mean():.3f}, p10 {np.percentile(sims_22973, 10):.3f}, "
                  f"p50 {np.percentile(sims_22973, 50):.3f}, p90 {np.percentile(sims_22973, 90):.3f}")

    # 5. Save v6
    df_v6 = df[keep_mask].drop(columns=["dropped"]).copy()
    df_v6["taxon"] = df_v6.primary_label.map(sp2tax).fillna("?")
    print(f"\n=== Final v7 (more external) ===")
    print(f"  v3 → v7: {len(df)} → {len(df_v6)} ({100*(len(df)-len(df_v6))/len(df):.1f}% dropped)")
    print(f"\n  Per-taxon:")
    for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        n_t = (df_v6.taxon == t).sum()
        n_classes = df_v6[df_v6.taxon == t].primary_label.nunique()
        print(f"    {t}: {n_t} entries, {n_classes} classes")

    df_v6.to_csv(OUT_V6, index=False)
    print(f"\n  Saved → {OUT_V6}")


if __name__ == "__main__":
    main()
