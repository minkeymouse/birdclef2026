# birdclef-2026-perch-distill — LEGACY (v33 reference, LB 0.932)

**Status (2026-05-03): superseded by `birdclef-2026-mattia-fork` (LB 0.941).**

This kernel is retained for:
- Ablation reference against v33 (the prior production)
- Historical patch record (v34 through v59)
- Reproducibility of the v33 = 0.932 result

**No new LB submissions to this kernel** unless explicitly testing a
v33-anchored ablation. Forward production work uses
`notebooks/birdclef-2026-mattia-fork/`.

## Stack (v33)

```
0.7 × (Perch v2 ONNX + ProtoSSM v4 ensemble)
+ 0.3 × exp50_SED  (HGNet-B0, mel=128, fmax=14000, 2025 BG mixing)
+ V9 5-way taxon gate (Aves / Amphibia / Insecta / Mammalia / Reptilia)
+ Gauss σ=0.5 temporal smoothing
+ file-max α=0.10 coherence prior
```

## Files

- `notebook.ipynb` — production v33 notebook (currently has v59 patch
  applied: linear v33 + Tucker W=0.40). Pushed as kernel
  `ultimatumgame/birdclef-2026-perch-distill` v61. Reset to v33 base
  if needed.
- `kernel-metadata.json` — Kaggle dataset / kernel declarations.
- `patches/` — historical record of every LB-submitted modification.
  Each file is a code block that gets inserted into `notebook.ipynb`
  cell 50 just before `# --- Build submission ---`. Most patches are
  obsolete; kept for paper / memory references.

## Patch history

| Patch | LB submission | LB Δ vs v33 | Status |
|---|---|---|---|
| `exp44c_patch.py` | v18 | −0.013 | regressed; site fingerprint |
| `exp45c_patch.py` | v20-v21 | 0 | neutral; no transfer |
| `exp48_patch.py` | v22 | −0.016 | site prior + cluster; train-SS-fitted |
| `exp78_v34_patch.py` | v34 | −0.013 | mel-iVAE; site fingerprint |
| `exp102_v38_patch.py` | v38 | −0.012 | LR detectors; LOSO 0.97 invalidated |
| `exp111_v43_pnew3_patch.py` | v43 | −0.009 | P_NEW3 hybrid head |
| `exp121_v44_patch.py` | v44 | −0.012 | cross-region BG fine-tune |
| `exp123_v45_patch.py` | v45 | −0.014 | bird-bias penalty |
| `exp136b_v46_patch.py` | v46 | −0.018 | v3 pseudo retrain; circular distill |
| `exp159_v52_patch.py` | (n/a) | — | drafted, not LB-submitted |
| `exp160_v55_tucker_patch.py` | v55 | ≤0 | Tucker linear W=0.10 (undersized) |
| `exp162_v56_rank_blend_patch.py` | v56 | 0 / −0.001 | Tucker rank-blend W=0.30 |
| `exp165_v58_linear_w30_patch.py` | v58 / v59 | ≤+0.009 | Tucker linear W=0.30 / 0.40 |

## What the v55-v59 sequence taught us

**v55**: undersized dose (W=0.10) — local +0.077, LB ≤0.932.
**v56**: rank-blend on calibrated v33 streamA hurts (sp_row 1.0 → 0.40)
— LB 0.931 / 0.932.
**v58 / v59**: linear blend at W=0.30 / 0.40 — local +0.117 / +0.131
but LB still ≤0.941. Tucker single-SED ceiling at 0.941 regardless of
fusion architecture or dose.

The +0.009 LB delta from v33 → v57 is the **Tucker SED** itself, not
the architecture. Linear blend on calibrated v33 streamA gives the
same LB ceiling as Mattia's rank+rescue path; the architecture choice
was orthogonal to the LB outcome in our pipeline.

See `notebooks/birdclef-2026-mattia-fork/README.md` for current
production. See memory `project_v57_mattia_fork_breakthrough.md` and
`project_why_mattia_works.md` for the strategic interpretation.
