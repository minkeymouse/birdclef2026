"""Final rank-percentile blend + 3 conditional rescue rules.

Self-contained, locally runnable. Replicates Mattia's cell-39 logic
in callable form.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-6

# Mattia's hyperparameters
DEFAULT_SED_W = 0.30

# Rescue rule thresholds
FAKE_ONLY_THR = 0.50
SED_LOW_THR = 0.05
FAKE_ONLY_BLEND = 0.12

PROTO_CONT_RADIUS = 3
PROTO_CONT_DF = 2.0
PROTO_CONT_SCALE = 1.20
PROTO_CONT_RANK_THR = 0.88
PROTO_LOCAL_RANK_THR = 0.75
SED_CONT_LOW_THR = 0.12
PROTO_CONT_BLEND = 0.15

SED_ONLY_RANK_THR = 0.95
FAKE_RANK_LOW_THR = 0.80
SED_ONLY_BLEND = 0.12


def rank_pct(arr):
    """Per-class (column) rank percentile. Shape (N, C)."""
    return pd.DataFrame(arr).rank(axis=0, pct=True).to_numpy(dtype=np.float32)


def t_dist_kernel(radius=PROTO_CONT_RADIUS, df=PROTO_CONT_DF, scale=PROTO_CONT_SCALE):
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = (1.0 + (offs / scale) ** 2 / df) ** (-(df + 1.0) / 2.0)
    return (k / k.sum()).astype(np.float32)


def proto_context_rank(pa, file_ids, radius=PROTO_CONT_RADIUS,
                        df=PROTO_CONT_DF, scale=PROTO_CONT_SCALE):
    """Per-class fat-tail t-distribution kernel applied within each file."""
    kernel = t_dist_kernel(radius, df, scale)
    pa_ctx = pa.copy()
    R = radius
    for fid in pd.unique(file_ids):
        m = file_ids == fid
        x = pa[m]
        if len(x) > 1:
            xp = np.pad(x, ((R, R), (0, 0)), mode="edge")
            pa_ctx[m] = sum(kernel[i] * xp[i:i + len(x)] for i in range(2 * R + 1))
    return pa_ctx


def mattia_blend(pa_raw, pb_raw, file_ids, sed_w=DEFAULT_SED_W,
                  rescues=("fake", "cont", "spike")):
    """Mattia rank-blend + 3 rescue rules.

    pa_raw: streamA probs, shape (N, C)
    pb_raw: streamB probs, shape (N, C)
    file_ids: array-like of length N (for proto_continuity rescue)
    sed_w: weight on streamB
    rescues: subset of ("fake", "cont", "spike"); use () to disable
    """
    pa = np.clip(pa_raw, EPS, 1.0 - EPS)
    pb = np.clip(pb_raw, EPS, 1.0 - EPS)
    xa = rank_pct(pa)
    xb = rank_pct(pb)
    pred = xa * (1.0 - sed_w) + xb * sed_w

    if "fake" in rescues or "cont" in rescues or "spike" in rescues:
        fake_only = (pa > FAKE_ONLY_THR) & (pb < SED_LOW_THR)
    if "cont" in rescues or "spike" in rescues:
        pa_ctx = proto_context_rank(pa, file_ids)
        xctx = rank_pct(pa_ctx)
        proto_cont = ((xctx > PROTO_CONT_RANK_THR) & (xa > PROTO_LOCAL_RANK_THR)
                      & (pb < SED_CONT_LOW_THR) & (~fake_only))

    if "fake" in rescues:
        pred = np.where(fake_only,
                         (1.0 - FAKE_ONLY_BLEND) * pred + FAKE_ONLY_BLEND * xa, pred)
    if "cont" in rescues:
        pred = np.where(proto_cont,
                         (1.0 - PROTO_CONT_BLEND) * pred + PROTO_CONT_BLEND * np.maximum(xa, xctx),
                         pred)
    if "spike" in rescues:
        sed_only = ((xb > SED_ONLY_RANK_THR) & (xa < FAKE_RANK_LOW_THR)
                    & (~fake_only) & (~proto_cont))
        pred = np.where(sed_only,
                         (1.0 - SED_ONLY_BLEND) * pred + SED_ONLY_BLEND * xb, pred)
    return pred


def linear_blend(pa, pb, sed_w=DEFAULT_SED_W):
    """Plain linear blend (v33-style). Use for calibrated streamA cases."""
    return ((1.0 - sed_w) * pa + sed_w * pb).astype(np.float32)
