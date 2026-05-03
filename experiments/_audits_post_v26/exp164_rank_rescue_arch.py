#!/usr/bin/env python3
"""exp164 — Rank-blend + rescue architecture: was it just LINEAR holding us back?

Test if our direction was right but blocked by linear-fusion choice.

5 variants on labeled SS (122 eval rows):
  A) Linear v33 ref          = 0.7 P + 0.3 exp50, V9, file-max, Gauss
  B) v33-as-A + rank-blend exp50 + rescues   (our SED in Mattia arch)
  C) v33-as-A + rank-blend Tucker + rescues  (Mattia SED in our composite-A)
  D) Perch-as-A + rank-blend exp50 + rescues (single SED, raw Perch streamA)
  E) Perch-as-A + rank-blend Tucker + rescues = exp162 (sanity baseline)
  F) 3-stream: Perch + rank(exp50) + rank(Tucker) at split weights + rescues

This isolates whether:
  - rank-blend alone helps (A vs C without rescues)
  - rescues help (C vs C-no-rescues)
  - SED choice matters (B vs C: exp50 vs Tucker as streamB)
  - 3-stream stacks (F vs C / B)
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

EPS = 1e-6
SED_W = 0.30

# Mattia rescue config
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


def mattia_blend(pa_raw, pb_raw, file_ids, sed_w=SED_W, with_rescues=True):
    """Full Mattia recipe: rank-blend pa, pb + 3 rescue rules."""
    pa = np.clip(pa_raw, EPS, 1.0 - EPS)
    pb = np.clip(pb_raw, EPS, 1.0 - EPS)
    xa = rank_pct(pa)
    xb = rank_pct(pb)
    pred = xa * (1.0 - sed_w) + xb * sed_w

    if not with_rescues:
        return pred

    # Rescue 1: fake_only
    fake_only = (pa > FAKE_ONLY_THR) & (pb < SED_LOW_THR)
    pred = np.where(fake_only, (1.0 - FAKE_ONLY_BLEND) * pred + FAKE_ONLY_BLEND * xa, pred)

    # Rescue 2: proto temporal continuity
    pa_ctx = proto_context_rank(pa, file_ids, PROTO_CONT_RADIUS, PROTO_CONT_DF, PROTO_CONT_SCALE)
    xctx = rank_pct(pa_ctx)
    proto_cont = ((xctx > PROTO_CONT_RANK_THR) & (xa > PROTO_LOCAL_RANK_THR)
                  & (pb < SED_CONT_LOW_THR) & (~fake_only))
    pred = np.where(proto_cont,
                     (1.0 - PROTO_CONT_BLEND) * pred + PROTO_CONT_BLEND * np.maximum(xa, xctx),
                     pred)

    # Rescue 3: SED local spike
    sed_only = ((xb > SED_ONLY_RANK_THR) & (xa < FAKE_RANK_LOW_THR)
                & (~fake_only) & (~proto_cont))
    pred = np.where(sed_only, (1.0 - SED_ONLY_BLEND) * pred + SED_ONLY_BLEND * xb, pred)
    return pred


def main():
    print("=== exp164: was linear the bottleneck? ===\n", flush=True)

    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    file_ids = sc_g["filename"].astype(str).values

    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]
    tucker = np.load(EXP80 / "tucker_sed_5fold_labeled.npz")["scores"]

    # Reference: linear v33
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)
    ev_mask = sc_g.split.values == "eval"

    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "A) v33 linear ref")]

    # B) v33-as-streamA + rank-blend exp50 + rescues
    pred_B = mattia_blend(v33, exp50, file_ids, with_rescues=True)
    rows.append(evaluate(pred_B, v33, ev_mask, Y, sp_taxon,
                          "B) v33-as-A + rank-blend exp50 + rescues"))
    pred_B_nr = mattia_blend(v33, exp50, file_ids, with_rescues=False)
    rows.append(evaluate(pred_B_nr, v33, ev_mask, Y, sp_taxon,
                          "B-) v33-as-A + rank-blend exp50 (NO rescues)"))

    # C) v33-as-streamA + rank-blend Tucker + rescues
    pred_C = mattia_blend(v33, tucker, file_ids, with_rescues=True)
    rows.append(evaluate(pred_C, v33, ev_mask, Y, sp_taxon,
                          "C) v33-as-A + rank-blend Tucker + rescues"))
    pred_C_nr = mattia_blend(v33, tucker, file_ids, with_rescues=False)
    rows.append(evaluate(pred_C_nr, v33, ev_mask, Y, sp_taxon,
                          "C-) v33-as-A + rank-blend Tucker (NO rescues)"))

    # D) Perch-as-A + rank-blend exp50 + rescues
    pred_D = mattia_blend(perch_prob, exp50, file_ids, with_rescues=True)
    rows.append(evaluate(pred_D, v33, ev_mask, Y, sp_taxon,
                          "D) Perch-as-A + rank-blend exp50 + rescues"))

    # E) Perch-as-A + rank-blend Tucker + rescues  (= exp162 M3)
    pred_E = mattia_blend(perch_prob, tucker, file_ids, with_rescues=True)
    rows.append(evaluate(pred_E, v33, ev_mask, Y, sp_taxon,
                          "E) Perch-as-A + rank-blend Tucker + rescues"))

    # F) 3-stream: pred = xa * 0.4 + xb * 0.3 + xc * 0.3 then rescues
    # Using v33 as A, exp50 as B, Tucker as C
    pa = np.clip(v33, EPS, 1.0 - EPS)
    pb = np.clip(exp50, EPS, 1.0 - EPS)
    pc = np.clip(tucker, EPS, 1.0 - EPS)
    xa = rank_pct(pa); xb = rank_pct(pb); xc = rank_pct(pc)
    pred_F_base = 0.5 * xa + 0.25 * xb + 0.25 * xc
    rows.append(evaluate(pred_F_base, v33, ev_mask, Y, sp_taxon,
                          "F) 3-stream rank-blend v33+exp50+Tucker (no rescues)"))

    # G) varying SED_W on best variant
    for w in [0.20, 0.30, 0.40]:
        p = mattia_blend(v33, tucker, file_ids, sed_w=w, with_rescues=True)
        rows.append(evaluate(p, v33, ev_mask, Y, sp_taxon,
                              f"C-w{w}) v33-as-A + rank-Tucker W={w} + rescues"))

    # H) Try our exp50 with different SED_W
    for w in [0.20, 0.30, 0.40]:
        p = mattia_blend(v33, exp50, file_ids, sed_w=w, with_rescues=True)
        rows.append(evaluate(p, v33, ev_mask, Y, sp_taxon,
                              f"B-w{w}) v33-as-A + rank-exp50 W={w} + rescues"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal",
            "Amphib", "Reptil", "predicted"]
    print("\n=== Audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
