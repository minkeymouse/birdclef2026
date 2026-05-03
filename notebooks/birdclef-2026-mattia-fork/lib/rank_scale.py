"""Rank-aware file_max^0.4 scaling + adaptive delta smoothing.

Self-contained. Replicates Mattia's cells 24-25 in callable form.
"""
from __future__ import annotations
import numpy as np


def rank_aware_scale(probs, n_windows=12, power=0.4):
    """out_window = score * file_max^power (per-class).

    Assumes probs.shape[0] is multiple of n_windows. For uneven file
    layouts, use rank_aware_scale_groupby instead.
    """
    n_files = probs.shape[0] // n_windows
    n_cls = probs.shape[1]
    view = probs.reshape(n_files, n_windows, n_cls)
    fmax = view.max(axis=1, keepdims=True)
    scale = np.power(np.maximum(fmax, 1e-9), power)
    return (view * scale).reshape(probs.shape).astype(np.float32)


def rank_aware_scale_groupby(probs, sc_g, power=0.4):
    """Same as rank_aware_scale but groups rows by sc_g['filename'].
    Use for labeled SS where rows-per-file varies.
    """
    out = probs.copy()
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = probs[idx]
        fmax = sub.max(axis=0, keepdims=True)
        scale = np.power(np.maximum(fmax, 1e-9), power)
        out[idx] = sub * scale
    return out.astype(np.float32)


def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    """alpha = base_alpha * (1 - confidence). Confidence = max-prob across
    classes per window. Confident peaks barely change; uncertain ones
    smooth to neighbours.
    """
    n_files = probs.shape[0] // n_windows
    n_cls = probs.shape[1]
    view = probs.reshape(n_files, n_windows, n_cls)
    out = view.copy()
    for t in range(n_windows):
        prev = view[:, max(0, t-1), :]
        nxt = view[:, min(n_windows-1, t+1), :]
        conf = view[:, t, :].max(axis=1, keepdims=True)
        alpha = base_alpha * (1.0 - conf)
        neighbor_avg = 0.5 * (prev + nxt)
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg
    return out.reshape(probs.shape).astype(np.float32)


def adaptive_delta_smooth_groupby(probs, sc_g, base_alpha=0.20):
    """Variable-window-per-file variant for labeled SS."""
    out = probs.copy()
    for fname, idx in sc_g.groupby("filename").indices.items():
        idx_sorted = idx[np.argsort(sc_g.iloc[idx]["end_sec"].values)]
        sub = probs[idx_sorted]
        n = len(sub)
        if n < 2:
            continue
        smoothed = sub.copy()
        for t in range(n):
            prev = sub[max(0, t-1)]
            nxt = sub[min(n-1, t+1)]
            conf = sub[t].max()
            alpha = base_alpha * (1.0 - conf)
            neighbor_avg = 0.5 * (prev + nxt)
            smoothed[t] = (1.0 - alpha) * sub[t] + alpha * neighbor_avg
        out[idx_sorted] = smoothed
    return out.astype(np.float32)


def file_max_blend(probs, sc_g=None, n_windows=12, alpha=0.10):
    """v33-style file-max coherence: out = (1-α) * sub + α * fmax.

    If sc_g is provided, groupby by filename (handles variable rows).
    Otherwise reshape into (n_files, n_windows, n_cls).
    """
    if sc_g is not None:
        out = probs.copy()
        for fname, idx in sc_g.groupby("filename").indices.items():
            sub = probs[idx]
            fmax = sub.max(axis=0, keepdims=True)
            out[idx] = (1 - alpha) * sub + alpha * fmax
        return out.astype(np.float32)
    n_files = probs.shape[0] // n_windows
    n_cls = probs.shape[1]
    view = probs.reshape(n_files, n_windows, n_cls)
    fmax = view.max(axis=1, keepdims=True)
    return ((1 - alpha) * view + alpha * fmax).reshape(probs.shape).astype(np.float32)
