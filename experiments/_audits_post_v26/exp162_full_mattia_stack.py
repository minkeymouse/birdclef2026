#!/usr/bin/env python3
"""exp162 — Full Mattia 0.943 recipe local audit.

Reproduces the public 0.943 recipe on labeled SS (122 eval rows) to give
an upper bound on the v57 lever's local potential.

Components:
  - streamA = perch_prob (ProtoSSM proxy; we don't have trained ProtoSSM
    offline, but Perch alone gives a reasonable lower bound)
  - streamB = tucker_5fold ensemble (cached from exp160)
  - rank-pct each stream per-class
  - base blend: pred = xa * (1 - SED_W) + xb * SED_W   (SED_W = 0.30)
  - Rescue 1 (fake_only): pa > 0.50 AND pb < 0.05 → blend xa more
  - Rescue 2 (proto continuity): t-dist fat-tail kernel ±3 windows on
    ProtoSSM rank-pct; if context-rank > 0.88 AND local-rank > 0.75 AND
    pb < 0.12 → blend max(xa, xctx)
  - Rescue 3 (sed local spike): xb > 0.95 AND xa < 0.80 → blend xb

Output: macro_d, sp_row, per-taxon Δ for each step.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, EXP80, N_CLS)
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate

# Mattia rescue config
SED_W = 0.30
EPS = 1e-6
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
    return pd.DataFrame(arr).rank(axis=0, pct=True).to_numpy(dtype=np.float32)


def t_dist_kernel(radius, df, scale):
    offs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = (1.0 + (offs / scale) ** 2 / df) ** (-(df + 1.0) / 2.0)
    return (k / k.sum()).astype(np.float32)


def proto_context_rank(pa, file_ids, radius, df, scale):
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


def main():
    print("=== exp162: Full Mattia 0.943 recipe local audit ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]
    tucker_path = EXP80 / "tucker_sed_5fold_labeled.npz"
    tucker = np.load(tucker_path)["scores"]

    # === Reference: linear v33 ===
    base_v33 = 0.7 * perch_prob + 0.3 * exp50
    g_v33 = apply_v9_gate(base_v33, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(g_v33, sc_g, alpha=0.10)

    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 linear ref")]

    # === Reproduce Mattia stack ===
    pa = np.clip(perch_prob, EPS, 1.0 - EPS)  # streamA = ProtoSSM proxy = Perch
    pb = np.clip(tucker, EPS, 1.0 - EPS)      # streamB = Tucker SED ensemble

    xa = rank_pct(pa)
    xb = rank_pct(pb)

    # 0) Base rank blend
    pred_base = xa * (1.0 - SED_W) + xb * SED_W
    rows.append(evaluate(pred_base, v33, ev_mask, Y, sp_taxon,
                          f"M0: rank-blend (SED_W={SED_W})"))

    # File IDs for rescue 2
    file_ids = sc_g["filename"].astype(str).values

    # === Apply rescues stepwise ===
    # 1) fake_only
    fake_only = (pa > FAKE_ONLY_THR) & (pb < SED_LOW_THR)
    pred_r1 = np.where(fake_only,
                        (1.0 - FAKE_ONLY_BLEND) * pred_base + FAKE_ONLY_BLEND * xa,
                        pred_base)
    rows.append(evaluate(pred_r1, v33, ev_mask, Y, sp_taxon,
                          f"M1: + fake_only rescue (n={fake_only.sum()})"))

    # 2) proto continuity (fat-tail kernel)
    pa_ctx = proto_context_rank(pa, file_ids, PROTO_CONT_RADIUS,
                                  PROTO_CONT_DF, PROTO_CONT_SCALE)
    xctx = rank_pct(pa_ctx)
    proto_cont = (
        (xctx > PROTO_CONT_RANK_THR) &
        (xa > PROTO_LOCAL_RANK_THR) &
        (pb < SED_CONT_LOW_THR) &
        (~fake_only)
    )
    pred_r2 = np.where(proto_cont,
                        (1.0 - PROTO_CONT_BLEND) * pred_r1
                        + PROTO_CONT_BLEND * np.maximum(xa, xctx),
                        pred_r1)
    rows.append(evaluate(pred_r2, v33, ev_mask, Y, sp_taxon,
                          f"M2: + proto_continuity (n={proto_cont.sum()})"))

    # 3) sed local spike
    sed_only = (
        (xb > SED_ONLY_RANK_THR) &
        (xa < FAKE_RANK_LOW_THR) &
        (~fake_only) &
        (~proto_cont)
    )
    pred_r3 = np.where(sed_only,
                        (1.0 - SED_ONLY_BLEND) * pred_r2 + SED_ONLY_BLEND * xb,
                        pred_r2)
    rows.append(evaluate(pred_r3, v33, ev_mask, Y, sp_taxon,
                          f"M3: + sed_only_spike (n={sed_only.sum()})"))

    # === Variant: substitute streamA with v33 (so we have full ProtoSSM+rank+rescues) ===
    pa2 = np.clip(v33, EPS, 1.0 - EPS)
    xa2 = rank_pct(pa2)
    pred_v33A = xa2 * (1.0 - SED_W) + xb * SED_W
    rows.append(evaluate(pred_v33A, v33, ev_mask, Y, sp_taxon,
                          "Mvar: v33-as-streamA + rank-blend Tucker (no rescues)"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal",
            "Amphib", "Reptil", "predicted"]
    print("\n=== Audit (in order) ===")
    print(res[cols].to_string(index=False))
    print()
    print("=== Audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
