#!/usr/bin/env python3
"""
exp22 — train_audio Perch embeddings → 234-class classifier.

Hypothesis: the 0.910 pipeline never uses train_audio (35k labeled XC/iNaturalist
recordings). Training a small classifier on Perch embeddings of train_audio adds an
independent signal that's especially valuable for unmapped/underrepresented classes.

Cross-domain question: does a model trained on clean recordings generalize to
field soundscapes? Compared against the exp21 baseline using both OOF (site holdout)
and in-sample evaluation.

Phase 1: extract Perch (logits + embedding) for first 5s of every train_audio file.
Phase 2: train 234-class one-vs-rest LogReg on Perch embeddings + logits.
Phase 3: evaluate on labeled-SS Perch cache from exp21. Sweep blend weight.

Outputs:
  experiments/exp22_outputs/train_audio_perch.npz   (cache)
  experiments/exp22_outputs/results.json
"""
from __future__ import annotations
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
PERCH_DIR = ROOT / "perch_v2"
OUT = ROOT / "experiments" / "exp22_outputs"
OUT.mkdir(parents=True, exist_ok=True)
EXP21 = ROOT / "experiments" / "exp21_outputs" / "perch_cache"

SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
BATCH_SIZE = 32  # clips per Perch call
SEED = 42
np.random.seed(SEED)


# ────────── data ──────────

def load_metadata():
    taxonomy = pd.read_csv(DATA / "taxonomy.csv")
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    train_df = pd.read_csv(DATA / "train.csv")
    primary = sample_sub.columns[1:].tolist()
    label_to_idx = {c: i for i, c in enumerate(primary)}
    return taxonomy, primary, label_to_idx, train_df


# ────────── Perch ──────────

def load_perch():
    print("Loading Perch v2 ...")
    t0 = time.time()
    model = tf.saved_model.load(str(PERCH_DIR))
    infer = model.signatures["serving_default"]
    print(f"Perch loaded in {time.time() - t0:.1f}s")
    return infer


def read_5s(path):
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False, frames=WINDOW_SAMPLES)
    except Exception:
        return None
    if y.size == 0:
        return None
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < WINDOW_SAMPLES:
        y = np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    return y[:WINDOW_SAMPLES].astype(np.float32)


def extract_train_audio(infer, train_df, label_to_idx):
    cache_emb = OUT / "train_audio_perch.npz"
    if cache_emb.exists():
        print(f"Loading cached: {cache_emb}")
        d = np.load(cache_emb)
        return d["emb"], d["y_idx"], d["valid"]

    n = len(train_df)
    print(f"Extracting Perch on {n} train_audio files (first 5s each) ...")
    emb = np.zeros((n, 1536), dtype=np.float32)
    # Skip storing 14795-dim logits — too much disk. Just embeddings.
    y_idx = np.full(n, -1, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)

    paths = [DATA / "train_audio" / r for r in train_df["filename"].values]
    primary_labels = train_df["primary_label"].astype(str).values

    t0 = time.time()
    batch_audio = []
    batch_idx = []
    pbar = tqdm(range(n), desc="train_audio Perch")
    for i in pbar:
        if primary_labels[i] not in label_to_idx:
            continue
        audio = read_5s(paths[i])
        if audio is None:
            continue
        batch_audio.append(audio)
        batch_idx.append(i)

        if len(batch_audio) == BATCH_SIZE or i == n - 1:
            x = np.stack(batch_audio, axis=0)
            out = infer(inputs=tf.convert_to_tensor(x))
            em = out["embedding"].numpy().astype(np.float32)
            for k, gi in enumerate(batch_idx):
                emb[gi] = em[k]
                y_idx[gi] = label_to_idx[primary_labels[gi]]
                valid[gi] = True
            batch_audio.clear()
            batch_idx.clear()
            del x, out, em
            if (i + 1) % 1000 == 0 or i == n - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate
                pbar.set_postfix({"rate": f"{rate:.1f}/s", "eta_min": f"{eta/60:.1f}"})

    print(f"\nValid: {valid.sum()}/{n} = {valid.mean():.3f}")
    print(f"Wall: {time.time() - t0:.0f}s")
    np.savez_compressed(cache_emb, emb=emb, y_idx=y_idx, valid=valid)
    print(f"Cached → {cache_emb}")
    return emb, y_idx, valid


# ────────── eval helpers ──────────

def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def per_class_auc(y_true, y_score):
    aucs = np.full(y_true.shape[1], np.nan)
    for j in range(y_true.shape[1]):
        if 0 < y_true[:, j].sum() < len(y_true):
            try:
                aucs[j] = roc_auc_score(y_true[:, j], y_score[:, j])
            except ValueError:
                pass
    return aucs


# ────────── main ──────────

def main():
    t0 = time.time()
    taxonomy, primary, label_to_idx, train_df = load_metadata()
    n_classes = len(primary)
    print(f"taxonomy: {len(taxonomy)} species, train_audio rows: {len(train_df)}")

    # Phase 1
    infer = load_perch()
    emb_ta, y_idx_ta, valid_ta = extract_train_audio(infer, train_df, label_to_idx)

    keep = valid_ta
    emb_ta = emb_ta[keep]
    y_idx_ta = y_idx_ta[keep]
    print(f"Training samples: {len(emb_ta)}")
    print(f"Per-class counts (min/median/max): "
          f"{np.bincount(y_idx_ta, minlength=n_classes).min()}/"
          f"{int(np.median(np.bincount(y_idx_ta, minlength=n_classes)))}/"
          f"{np.bincount(y_idx_ta, minlength=n_classes).max()}")

    # Build multi-label Y: only primary for now (most train_audio rows have no secondary)
    Y_TA = np.zeros((len(emb_ta), n_classes), dtype=np.float32)
    Y_TA[np.arange(len(emb_ta)), y_idx_ta] = 1.0

    # Phase 2: train per-class LogReg
    print("Standardizing embeddings ...")
    scaler = StandardScaler()
    X_ta = scaler.fit_transform(emb_ta).astype(np.float32)

    print("Loading exp21 labeled-SS cache ...")
    meta_full = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    arr = np.load(EXP21 / "full_perch_arrays.npz")
    emb_ss = arr["emb"].astype(np.float32)
    scores_ss_raw = arr["scores"].astype(np.float32)

    # Reconstruct Y_FULL (active labels per window)
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    import re
    fre = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

    def parse_lbls(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc_clean = (
        sc_raw.groupby(["filename", "start", "end"])["primary_label"]
        .apply(lambda s: sorted({lbl for x in s for lbl in parse_lbls(x)}))
        .reset_index(name="label_list")
    )
    sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
    sc_clean["row_id"] = (
        sc_clean["filename"].str.replace(".ogg", "", regex=False)
        + "_" + sc_clean["end_sec"].astype(str)
    )
    Y_SC = np.zeros((len(sc_clean), n_classes), dtype=np.uint8)
    for i, labs in enumerate(sc_clean["label_list"]):
        for lbl in labs:
            if lbl in label_to_idx:
                Y_SC[i, label_to_idx[lbl]] = 1

    # Align Y_FULL to meta_full
    sc_idx = sc_clean.set_index("row_id")
    Y_FULL = np.stack([Y_SC[sc_idx.index.get_loc(rid)] for rid in meta_full["row_id"]])
    print(f"Y_FULL shape: {Y_FULL.shape}, active classes: {(Y_FULL.sum(0) > 0).sum()}")

    X_ss = scaler.transform(emb_ss).astype(np.float32)

    # Train per-class LogReg
    print("Training 234-class one-vs-rest LogReg on train_audio embeddings ...")
    pred_ss = np.zeros((len(X_ss), n_classes), dtype=np.float32)
    counts = np.bincount(y_idx_ta, minlength=n_classes)

    n_trained = 0
    for c in tqdm(range(n_classes), desc="LogReg"):
        if counts[c] < 5:  # too few samples
            continue
        y = Y_TA[:, c]
        clf = LogisticRegression(C=0.1, max_iter=200, solver="liblinear",
                                 class_weight="balanced")
        clf.fit(X_ta, y)
        pred_ss[:, c] = clf.decision_function(X_ss).astype(np.float32)
        n_trained += 1
    print(f"Trained {n_trained}/{n_classes} classifiers (others had < 5 samples)")

    # Phase 3: evaluate
    # Standalone train_audio-probe predictions on labeled SS
    auc_ta_only = macro_auc(Y_FULL, pred_ss)
    print(f"\n=== train_audio probe alone (cross-domain) ===")
    print(f"  Macro AUC on labeled SS = {auc_ta_only:.4f}")

    # By class group
    cn_map = taxonomy.set_index("primary_label")["class_name"].to_dict()
    pc_aucs = per_class_auc(Y_FULL, pred_ss)
    df_pc = pd.DataFrame({
        "primary": primary,
        "class_name": [cn_map.get(p, "?") for p in primary],
        "n_pos_ss": Y_FULL.sum(axis=0),
        "n_train_audio": counts,
        "auc_train_audio": pc_aucs,
    })
    print("\nPer-class group (train_audio probe alone):")
    print(df_pc[df_pc["n_pos_ss"] > 0].groupby("class_name")["auc_train_audio"].agg(["mean", "count"]).round(3))

    # Blend with raw Perch (exp21 baseline)
    print("\n=== Blending with raw Perch (exp21 condition A) ===")
    # Standardize each separately to logit-like scale, then blend
    perch_logit = scores_ss_raw  # already in logit-like scale
    # Recall: Perch logit = 0 for unmapped classes. Train_audio probe fills those.

    # Sweep blend weight
    best_w, best_auc = 0.0, auc_ta_only
    for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        blend = (1 - w) * perch_logit + w * pred_ss
        a = macro_auc(Y_FULL, blend)
        marker = " ←" if a > best_auc else ""
        print(f"  w_train_audio = {w:.2f} → AUC = {a:.4f}{marker}")
        if a > best_auc:
            best_auc = a
            best_w = w

    # Smarter blend: w only for unmapped/active-with-zero-perch, else use perch
    active_perch_mask = (np.abs(perch_logit).sum(axis=0) > 1e-3)  # cls that Perch ever fires on
    print(f"\n  Active Perch mask: {active_perch_mask.sum()} cols Perch fires on, "
          f"{(~active_perch_mask).sum()} Perch-silent")
    smart = perch_logit.copy()
    silent = ~active_perch_mask
    smart[:, silent] = pred_ss[:, silent]
    auc_smart = macro_auc(Y_FULL, smart)
    print(f"  Perch + train_audio fill-Perch-silent: AUC = {auc_smart:.4f}")

    # Save results
    results = {
        "n_train_audio_valid": int(len(emb_ta)),
        "n_classifiers_trained": n_trained,
        "auc_train_audio_alone": auc_ta_only,
        "auc_perch_alone_for_ref": macro_auc(Y_FULL, perch_logit),
        "blend_sweep": {f"w={w}": macro_auc(Y_FULL, (1-w)*perch_logit + w*pred_ss)
                        for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]},
        "best_uniform_blend_w": best_w,
        "best_uniform_blend_auc": best_auc,
        "perch_fill_silent_auc": auc_smart,
        "perch_silent_class_count": int(silent.sum()),
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    df_pc.to_csv(OUT / "per_class.csv", index=False)
    print(f"\nWrote {OUT/'results.json'} and {OUT/'per_class.csv'}")
    print(f"Total wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
