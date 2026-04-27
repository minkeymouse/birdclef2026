#!/usr/bin/env python3
"""
exp10_ensemble.py — Evaluate ensemble + post-processing locally.

Tests different ensemble combinations and post-processing on full_ss eval set.
No training — purely inference-time optimization.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
WEIGHTS_DIR = ROOT / "model-weights"
OUT_DIR = ROOT / "experiments" / "exp10_outputs"
EXP3_DIR = ROOT / "experiments" / "exp3_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

taxonomy_df = pd.read_csv(DATA / "taxonomy.csv")
SPECIES_LIST = sorted(taxonomy_df["primary_label"].astype(str).tolist())
SPECIES2IDX = {sp: i for i, sp in enumerate(SPECIES_LIST)}
NUM_CLASSES = len(SPECIES_LIST)

# Non-bird species for power transform
rare_species_idx = set()
if "class_name" in taxonomy_df.columns:
    non_bird = taxonomy_df[taxonomy_df["class_name"] != "Aves"]["primary_label"].tolist()
    for sp in non_bird:
        if sp in SPECIES2IDX:
            rare_species_idx.add(SPECIES2IDX[sp])

IMG_SIZE = 224
BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"

print(f"Classes: {NUM_CLASSES}, Non-bird: {len(rare_species_idx)}, Device: {DEVICE}")


# ── Model ──────────────────────────────────────────────────
class AttentionHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, num_classes),
            nn.Softmax(dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        framewise = self.classifier(x)
        attn_weights = self.attention(x)
        clipwise = (framewise * attn_weights).sum(dim=1)
        framewise_max = framewise.max(dim=1).values
        return clipwise, framewise_max


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            in_chans=1, num_classes=0, global_pool="",
        )
        with torch.no_grad():
            dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[1]
        self.head = AttentionHead(self.feat_dim, num_classes, dropout=0.3)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.mean(dim=2).permute(0, 2, 1)
        clipwise, framewise_max = self.head(feat)
        return clipwise, framewise_max


# ── Dataset ────────────────────────────────────────────────
class EvalDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel = np.load(row["mel_path"])
        label = row["label"]
        if isinstance(label, str):
            label = np.array(json.loads(label), dtype=np.float32)
        mel_t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel_t = F.interpolate(mel_t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                              mode="bilinear", align_corners=False).squeeze(0)
        return mel_t, torch.tensor(label, dtype=torch.float32)


def load_model(ckpt_path):
    model = BirdSEDModel(BACKBONE, NUM_CLASSES, pretrained=False).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader):
    all_preds = []
    for mels, _ in loader:
        mels = mels.to(DEVICE, non_blocking=True)
        with autocast("cuda"):
            clipwise, _ = model(mels)
        all_preds.append(torch.sigmoid(clipwise).cpu().numpy())
    return np.concatenate(all_preds)


def compute_auc(preds, labels):
    aucs = []
    for c in range(NUM_CLASSES):
        gt = (labels[:, c] > 0.5).astype(int)
        if 0 < gt.sum() < len(gt):
            try:
                aucs.append(roc_auc_score(gt, preds[:, c]))
            except ValueError:
                pass
    return np.mean(aucs) if aucs else 0.0, len(aucs)


def apply_power_transform(preds, exponent=0.7, target_indices=None):
    """Apply power transform to boost low predictions for specified species."""
    result = preds.copy()
    if target_indices is None:
        # Apply to all species
        result = np.power(result + 1e-8, exponent)
    else:
        for idx in target_indices:
            result[:, idx] = np.power(result[:, idx] + 1e-8, exponent)
    return result


def main():
    start_time = time.time()

    # Load eval data
    print("Loading eval data...")
    full_eval = pd.read_parquet(EXP3_DIR / "eval_meta.parquet")
    full_eval["label"] = full_eval["label"].apply(lambda x: np.array(x, dtype=np.float32))
    eval_ds = EvalDataset(full_eval)
    eval_loader = DataLoader(eval_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Get ground truth labels
    all_labels = []
    for _, labels in eval_loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)

    # Load all available models
    model_configs = {
        "exp3_f0": "exp3_sed_fold0_best.pth",
        "exp3_f1": "exp3_sed_fold1_best.pth",
        "exp7_f0": "exp7_st2_fold0_best.pth",
        "exp7_f1": "exp7_st2_fold1_best.pth",
        "exp8_f0": "exp8_focal_fold0_best.pth",
        "exp8_f1": "exp8_focal_fold1_best.pth",
        "exp9_f0": "exp9_aug_fold0_best.pth",
        "exp9_f1": "exp9_aug_fold1_best.pth",
    }

    all_preds = {}
    for name, fname in model_configs.items():
        ckpt = WEIGHTS_DIR / fname
        if not ckpt.exists():
            print(f"  {fname} not found, skipping")
            continue
        print(f"  Loading {fname}...")
        model = load_model(ckpt)
        preds = predict(model, eval_loader)
        all_preds[name] = preds
        del model
        torch.cuda.empty_cache()

    print(f"\nLoaded {len(all_preds)} models\n")

    # ── Test different ensemble combinations ──────────────
    results = {}

    # 1. Individual models
    print("=" * 60)
    print("INDIVIDUAL MODELS")
    print("=" * 60)
    for name, preds in all_preds.items():
        auc, n = compute_auc(preds, all_labels)
        print(f"  {name}: full_ss={auc:.4f} ({n} cls)")
        results[name] = auc

    # 2. Experiment-level ensembles
    print("\n" + "=" * 60)
    print("EXPERIMENT ENSEMBLES (averaging folds)")
    print("=" * 60)
    for exp in ["exp3", "exp7", "exp8", "exp9"]:
        fold_preds = [all_preds[k] for k in all_preds if k.startswith(exp)]
        if fold_preds:
            avg = np.mean(fold_preds, axis=0)
            auc, n = compute_auc(avg, all_labels)
            print(f"  {exp}_ensemble: full_ss={auc:.4f} ({n} cls)")
            results[f"{exp}_ensemble"] = auc

    # 3. Cross-experiment ensembles
    print("\n" + "=" * 60)
    print("CROSS-EXPERIMENT ENSEMBLES")
    print("=" * 60)

    combos = {
        "exp8+exp9": ["exp8_f0", "exp8_f1", "exp9_f0", "exp9_f1"],
        "exp7+exp8+exp9": ["exp7_f0", "exp7_f1", "exp8_f0", "exp8_f1", "exp9_f0", "exp9_f1"],
        "all_exp": list(all_preds.keys()),
    }
    for combo_name, keys in combos.items():
        available = [all_preds[k] for k in keys if k in all_preds]
        if available:
            avg = np.mean(available, axis=0)
            auc, n = compute_auc(avg, all_labels)
            print(f"  {combo_name}: full_ss={auc:.4f} ({n} cls)")
            results[f"ensemble_{combo_name}"] = auc

    # 4. Post-processing: power transform on ensembles
    print("\n" + "=" * 60)
    print("POST-PROCESSING (power transform)")
    print("=" * 60)

    best_ensemble_keys = ["exp7_f0", "exp7_f1", "exp8_f0", "exp8_f1", "exp9_f0", "exp9_f1"]
    best_available = [all_preds[k] for k in best_ensemble_keys if k in all_preds]
    if best_available:
        best_avg = np.mean(best_available, axis=0)

        # Test different power transform settings
        for exp_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
            # All species
            pt_all = apply_power_transform(best_avg, exponent=exp_val)
            auc_all, _ = compute_auc(pt_all, all_labels)

            # Only non-bird species
            pt_rare = apply_power_transform(best_avg, exponent=exp_val, target_indices=rare_species_idx)
            auc_rare, _ = compute_auc(pt_rare, all_labels)

            print(f"  power={exp_val:.1f}: all_species={auc_all:.4f}, non_bird_only={auc_rare:.4f}")
            results[f"pt_all_{exp_val}"] = auc_all
            results[f"pt_rare_{exp_val}"] = auc_rare

    # 5. Weighted ensemble (give more weight to better models)
    print("\n" + "=" * 60)
    print("WEIGHTED ENSEMBLE")
    print("=" * 60)
    if len(all_preds) >= 4:
        # Weight by experiment quality: exp9 > exp8 > exp7 > exp3
        weights = {"exp9": 3.0, "exp8": 2.0, "exp7": 1.0, "exp3": 0.5}
        weighted_sum = np.zeros_like(list(all_preds.values())[0])
        total_weight = 0
        for name, preds in all_preds.items():
            exp = name.split("_")[0]
            w = weights.get(exp, 1.0)
            weighted_sum += preds * w
            total_weight += w
        weighted_avg = weighted_sum / total_weight
        auc, n = compute_auc(weighted_avg, all_labels)
        print(f"  weighted(3:2:1:0.5): full_ss={auc:.4f} ({n} cls)")
        results["weighted_ensemble"] = auc

        # Also test with power transform
        for exp_val in [0.7, 0.8]:
            pt = apply_power_transform(weighted_avg, exponent=exp_val, target_indices=rare_species_idx)
            auc, _ = compute_auc(pt, all_labels)
            print(f"  weighted + pt_rare_{exp_val}: full_ss={auc:.4f}")
            results[f"weighted_pt_rare_{exp_val}"] = auc

    # Summary
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*60}")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, auc in sorted_results[:10]:
        print(f"  {name}: {auc:.4f}")
    print(f"\nTime: {elapsed:.1f} min")

    with open(OUT_DIR / "exp10_results.json", "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
