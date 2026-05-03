"""Validation + post-processing helpers.

Self-contained. metric utilities + temporal smoothing + per-taxon
temperature. Mostly replicas of cells 15-19.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# === Metrics ===

def macro_auc_skip_empty(y_true, y_pred, eps=1e-9):
    """Macro AUC across columns; skip columns with all-positive or
    all-negative ground truth. y_true: (N, C) binary, y_pred: (N, C)."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.sum() < 1 or yt.sum() == len(yt):
            continue
        try:
            aucs.append(roc_auc_score(yt, y_pred[:, c]))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def per_class_auc(y_true, y_pred):
    """Per-class AUC. NaN if undefined. Returns (n_classes,) array."""
    from sklearn.metrics import roc_auc_score
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.sum() < 1 or yt.sum() == len(yt):
            continue
        try:
            out[c] = roc_auc_score(yt, y_pred[:, c])
        except Exception:
            pass
    return out


# === Temporal smoothing (Gauss σ neighbour blend) ===

def gauss_smooth_windows(probs, n_windows=12, sigma=0.5):
    """Gaussian smoothing across n_windows per file.

    Assumes probs.shape[0] is multiple of n_windows.
    """
    if sigma <= 0:
        return probs
    # Build symmetric kernel
    radius = max(1, int(np.ceil(3 * sigma)))
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(offs ** 2) / (2 * sigma * sigma))
    k = (k / k.sum()).astype(np.float32)

    n_files = probs.shape[0] // n_windows
    n_cls = probs.shape[1]
    view = probs.reshape(n_files, n_windows, n_cls)
    out = np.zeros_like(view)
    for t in range(n_windows):
        weight_sum = 0.0
        for i, off in enumerate(offs.astype(int)):
            j = t + int(off)
            if 0 <= j < n_windows:
                out[:, t, :] += k[i] * view[:, j, :]
                weight_sum += k[i]
        if weight_sum > 0:
            out[:, t, :] /= weight_sum
    return out.reshape(probs.shape).astype(np.float32)


# === Per-taxon temperature ===

def build_taxon_temperature(primary_labels, taxonomy_df,
                             texture_taxa=("Amphibia", "Insecta"),
                             T_event=1.10, T_texture=0.95):
    """Per-class temperature vector. Texture taxa (continuous-call
    species) get sharper T<1; event taxa (Aves) get softer T>1.
    """
    class_name_map = taxonomy_df.set_index("primary_label")["class_name"].to_dict()
    n = len(primary_labels)
    temps = np.full(n, T_event, dtype=np.float32)
    for ci, label in enumerate(primary_labels):
        cls = class_name_map.get(label, "Aves")
        if cls in set(texture_taxa):
            temps[ci] = T_texture
    return temps


def apply_temperature(probs, temps):
    """Per-class temperature scaling on logit-equivalent probs."""
    eps = 1e-6
    p = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))
    scaled = logits / temps[None, :]
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)
