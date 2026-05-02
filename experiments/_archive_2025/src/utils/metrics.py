#!/usr/bin/env python3
"""
metrics.py – evaluation helpers for the BirdCLEF-2025 pipeline
==============================================================
All functions accept

    * ``y_true`` – array-like, shape *(N, C)*  
      Binary **or soft-label** ground-truth vectors.  Any value > 0 is treated
      as “positive” for a class when deciding whether that class is *present*.

    * ``y_pred`` – array-like, shape *(N, C)*  
      Model probabilities (0-1).  For thresholded metrics we default to 0.5.

Only classes that actually occur in *y_true* are included in the macro
aggregation – this avoids undefined metrics on empty columns and matches
common practice in multi-label bio-acoustic tasks.

Public helpers
--------------
macro_auc_score            – macro-averaged ROC-AUC  
macro_precision_score      – macro-averaged precision at threshold  
macro_recall_score         – macro-averaged recall at threshold  
macro_average_precision    – macro-averaged AP (area under PR-curve)  
create_pseudo_labels       – threshold → {species: 1.0} dict per chunk
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "macro_auc_score",
    "macro_precision_score",
    "macro_recall_score",
    "macro_average_precision",
    "create_pseudo_labels",
]


# -----------------------------------------------------------------------------#
# Internals                                                                    #
# -----------------------------------------------------------------------------#
def _valid_classes(y_true: np.ndarray) -> np.ndarray:
    """Return indices of classes that appear at least once in *y_true*."""
    return np.where((y_true > 0).sum(axis=0) > 0)[0]


# -----------------------------------------------------------------------------#
# ROC-AUC                                                                      #
# -----------------------------------------------------------------------------#
def macro_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-averaged ROC-AUC, skipping classes with no positives in *y_true*.
    Returns **0.0** when no class is evaluable.
    """
    classes = _valid_classes(y_true)
    if len(classes) == 0:
        return 0.0

    aucs: List[float] = []
    for c in classes:
        try:
            aucs.append(roc_auc_score((y_true[:, c] > 0).astype(int), y_pred[:, c]))
        except ValueError:
            # constant predictions or other degenerate case
            continue
    return float(np.mean(aucs)) if aucs else 0.0


# -----------------------------------------------------------------------------#
# Precision / Recall                                                           #
# -----------------------------------------------------------------------------#
def _binary_preds(y_pred: np.ndarray, threshold: float) -> np.ndarray:
    return (y_pred >= threshold).astype(int)


def macro_precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    """
    Macro-averaged precision (a.k.a. PPV) at the given threshold.
    Undefined classes are skipped.  If **all** remaining classes have
    zero positive predictions, returns 0.0 by convention.
    """
    classes = _valid_classes(y_true)
    if len(classes) == 0:
        return 0.0

    y_bin = _binary_preds(y_pred, threshold)
    precisions: List[float] = []
    for c in classes:
        tp = np.sum((y_true[:, c] > 0) & (y_bin[:, c] == 1))
        pp = np.sum(y_bin[:, c] == 1)
        if pp == 0:
            # no predicted positives – precision undefined (skip)
            continue
        precisions.append(tp / pp)
    return float(np.mean(precisions)) if precisions else 0.0


def macro_recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    """
    Macro-averaged recall (a.k.a. sensitivity) at the given threshold.
    """
    classes = _valid_classes(y_true)
    if len(classes) == 0:
        return 0.0

    y_bin = _binary_preds(y_pred, threshold)
    recalls: List[float] = []
    for c in classes:
        tp = np.sum((y_true[:, c] > 0) & (y_bin[:, c] == 1))
        fn = np.sum((y_true[:, c] > 0) & (y_bin[:, c] == 0))
        denom = tp + fn
        if denom == 0:
            continue
        recalls.append(tp / denom)
    return float(np.mean(recalls)) if recalls else 0.0


# -----------------------------------------------------------------------------#
# Average Precision (area under PR-curve)                                      #
# -----------------------------------------------------------------------------#
def macro_average_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-averaged **Average Precision** (integral of precision-recall curve).
    Useful for ranking evaluations.  Skips classes with no positives.
    """
    classes = _valid_classes(y_true)
    if len(classes) == 0:
        return 0.0

    aps: List[float] = []
    for c in classes:
        try:
            aps.append(
                average_precision_score((y_true[:, c] > 0).astype(int), y_pred[:, c])
            )
        except ValueError:
            continue
    return float(np.mean(aps)) if aps else 0.0


# -----------------------------------------------------------------------------#
# Pseudo-label helper                                                          #
# -----------------------------------------------------------------------------#
def create_pseudo_labels(
    chunk_probs: np.ndarray,
    species_list: Sequence[str],
    *,
    threshold: float = 0.5,
) -> List[Dict[str, float]]:
    """
    Convert ``(T, C)`` probability array to a list of dicts suitable for
    process_pseudo.py. Each dict maps species codes → 1.0 for classes
    whose probability ≥ threshold.
    """
    assert chunk_probs.shape[1] == len(species_list), (
        "`species_list` length must equal num_classes"
    )
    out: List[Dict[str, float]] = []
    for probs in chunk_probs:
        idxs = np.where(probs >= threshold)[0]
        out.append({species_list[i]: 1.0 for i in idxs})
    return out
