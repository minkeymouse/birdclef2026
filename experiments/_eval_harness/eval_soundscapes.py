#!/usr/bin/env python3
"""
eval_soundscapes.py — Evaluate model on labeled train_soundscapes.
Computes macro-AUC matching competition metric (skip classes with no true positives).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import librosa
from pathlib import Path
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"

# ── Config (must match exp2 training) ─────────────────────
SR = 32000
DURATION = 5.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0
FMAX = 16000
IMG_SIZE = 224
BACKBONE = "tf_efficientnet_b0_ns"

# ── Taxonomy ──────────────────────────────────────────────
taxonomy_df = pd.read_csv(DATA / "taxonomy.csv")
SPECIES_LIST = sorted(taxonomy_df["primary_label"].astype(str).tolist())
SPECIES2IDX = {sp: i for i, sp in enumerate(SPECIES_LIST)}
NUM_CLASSES = len(SPECIES_LIST)

# ── Model ─────────────────────────────────────────────────
class BirdModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False,
            in_chans=1, num_classes=0, global_pool="avg"
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def load_models():
    models = []
    for ckpt_path in sorted(WEIGHTS_DIR.glob("exp2_*.pth")):
        print(f"Loading {ckpt_path.name}...")
        model = BirdModel(BACKBONE, NUM_CLASSES)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def compute_mel(wav, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models()
    for m in models:
        m.to(device)
    print(f"Loaded {len(models)} model(s) on {device}")

    # Load labels
    labels_df = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    print(f"Labeled segments: {len(labels_df)}, unique files: {labels_df['filename'].nunique()}")

    # Parse start time to seconds
    def time_to_sec(t):
        parts = t.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    labels_df["start_sec"] = labels_df["start"].apply(time_to_sec)
    labels_df["end_sec"] = labels_df["end"].apply(time_to_sec)

    # Build ground truth and predictions
    all_y_true = []
    all_y_pred = []

    # Group by file to avoid reloading audio
    target_len = int(SR * DURATION)
    for filename, group in tqdm(labels_df.groupby("filename"), desc="Eval"):
        filepath = DATA / "train_soundscapes" / filename
        if not filepath.exists():
            print(f"  skip missing: {filename}")
            continue

        wav, _ = librosa.load(filepath, sr=SR)

        for _, row in group.iterrows():
            start_sample = row["start_sec"] * SR
            end_sample = start_sample + target_len
            chunk = wav[start_sample:end_sample]

            if len(chunk) < target_len:
                chunk = np.pad(chunk, (0, target_len - len(chunk)))

            mel = compute_mel(chunk)
            mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            mel_t = F.interpolate(mel_t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

            with torch.no_grad():
                preds = []
                for model in models:
                    logits = model(mel_t)
                    prob = torch.sigmoid(logits).cpu().numpy()[0]
                    preds.append(prob)
                pred = np.mean(preds, axis=0)

            # Ground truth: multi-label from semicolon-separated taxon IDs
            y_true = np.zeros(NUM_CLASSES, dtype=np.float32)
            for taxon_id in str(row["primary_label"]).split(";"):
                taxon_id = taxon_id.strip()
                if taxon_id in SPECIES2IDX:
                    y_true[SPECIES2IDX[taxon_id]] = 1.0

            all_y_true.append(y_true)
            all_y_pred.append(pred)

    y_true = np.stack(all_y_true)
    y_pred = np.stack(all_y_pred)
    print(f"\nTotal segments evaluated: {len(y_true)}")

    # Competition metric: macro-AUC, skip classes with no positives
    col_has_pos = y_true.sum(axis=0) > 0
    n_scored = col_has_pos.sum()
    print(f"Classes with positives: {n_scored} / {NUM_CLASSES}")

    aucs = []
    class_results = []
    for i in range(NUM_CLASSES):
        if not col_has_pos[i]:
            continue
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
            class_results.append((SPECIES_LIST[i], auc, int(y_true[:, i].sum())))
        except ValueError:
            pass

    macro_auc = np.mean(aucs)
    print(f"\n{'='*50}")
    print(f"Macro AUC (soundscapes): {macro_auc:.4f}")
    print(f"Scored classes: {len(aucs)}")
    print(f"{'='*50}")

    # Per-class breakdown (worst 10)
    class_results.sort(key=lambda x: x[1])
    print(f"\nWorst 10 classes:")
    for sp, auc, n_pos in class_results[:10]:
        print(f"  {sp:>12s}: AUC={auc:.4f} (n_pos={n_pos})")

    # Best 10
    print(f"\nBest 10 classes:")
    for sp, auc, n_pos in class_results[-10:]:
        print(f"  {sp:>12s}: AUC={auc:.4f} (n_pos={n_pos})")

    # By class_name (Aves, Insecta, etc.)
    tax_map = {str(r["primary_label"]): r["class_name"] for _, r in taxonomy_df.iterrows()}
    class_group_aucs = {}
    for sp, auc, n_pos in class_results:
        cn = tax_map.get(sp, "Unknown")
        class_group_aucs.setdefault(cn, []).append(auc)

    print(f"\nAUC by taxon class:")
    for cn, auc_list in sorted(class_group_aucs.items()):
        print(f"  {cn:>12s}: {np.mean(auc_list):.4f} (n={len(auc_list)})")


if __name__ == "__main__":
    main()
