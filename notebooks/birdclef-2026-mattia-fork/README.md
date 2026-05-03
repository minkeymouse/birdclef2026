# birdclef-2026-mattia-fork — production (LB 0.941)

Forked from `mattiaangeli/birdclef-2026-0-943-better-blend` (a fork of
`vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling`).
Public-LB-verified at 0.941 in our submission (kernel v1, 2026-05-03).

This is the production kernel for forward LB submissions. The legacy
`birdclef-2026-perch-distill/` kernel (LB 0.932 v33) is retained for
ablation reference only.

## Files

- `notebook.ipynb` — Kaggle code-competition kernel (40 cells, ~1156 KB).
  Pushed as kernel `ultimatumgame/birdclef-2026-mattia-fork`. Edit in
  place; Kaggle versions automatically.
- `kernel-metadata.json` — input dataset / kernel / model declarations.
- `lib/` — local modular extract of the same notebook for offline
  analysis and forward extension. **Not used at Kaggle inference time**;
  Kaggle runs `notebook.ipynb` directly. Use `lib/` to develop new
  components, then port back to the notebook.

## Pipeline at a glance

```
60 s audio → 12 × 5 s windows
  ├── Perch v2 ONNX → 1536-d embedding + 234 logits per window
  │     │
  │     ├── ProtoSSM v4 (state-space + cross-attn, 12 windows)
  │     ├── ResidualSSM second-pass correction
  │     ├── MLP probe on PCA-32/64 Perch embeddings
  │     ├── Per-taxon temperature scaling (T=0.95 / 1.10)
  │     ├── file_max^0.4 rank-aware scaling
  │     ├── adaptive δ smoothing (α = base × (1 − conf))
  │     └── isotonic calibration + per-class F1 threshold
  │     ↓
  │   stream A   →  rank-percentile per class
  │
  └── Tucker bc2026-distilled-sed-public 5-fold ONNX
        │ (mel=256, fmin=20, fmax=16000, per-spec z-score)
        ↓
      stream B   →  rank-percentile per class

  pred = xa × (1 − W_SED) + xb × W_SED            # base rank blend, W=0.30

  + Rescue 1 (fake_only):       proto says yes, sed missed       → boost xa
  + Rescue 2 (proto continuity): ±3 window t-dist context high   → boost max(xa, xctx)
  + Rescue 3 (sed local spike):  sed >0.95 spike, proto low      → boost xb

  → submission.csv
```

## Module map (`lib/`)

**Self-contained, locally importable** (use these for audits and forward work):

| module | role | source cell(s) |
|---|---|---|
| `paths.py` | Kaggle-vs-local path resolution (TUCKER_DIR, PERCH_ONNX, etc) | (new) |
| `tucker_sed.py` | Tucker 5-fold ONNX SED — `load_5fold()`, `predict_file()`, `predict_files()` | 37 |
| `rank_scale.py` | `rank_aware_scale`, `adaptive_delta_smooth`, `file_max_blend` (+ groupby variants for variable-window labeled SS) | 24-25 |
| `helpers.py` | `macro_auc_skip_empty`, `per_class_auc`, `gauss_smooth_windows`, per-taxon temperature | 15-19 |
| `final_blend.py` | `rank_pct`, `mattia_blend` (configurable rescue subset), `linear_blend` | 39 |

Verified by `experiments/_audits_post_v26/exp168_mattia_lib_sanity.py`:
tucker_sed reproduces cached scores 36/36 exact match; final_blend
reproduces v58 (+0.117), mattia full-rescues (+0.087), no-rescues
(+0.078) numbers exactly.

**Source-dump modules** (need Kaggle-cell context; not yet self-contained):

| module | role | source cell(s) |
|---|---|---|
| `config.py` | CFG dict, runtime constants | 5 |
| `data.py` | taxonomy + sample submission + train SS labels | 7 |
| `perch.py` | Perch v2 ONNX loader + window-level inference | 9, 10, 12, 13 |
| `mlp_probe.py` | PCA-Perch MLP probe + isotonic calibration | 21-23 |
| `protossm.py` | LightProtoSSM + ResidualSSM (training) | 27, 29 |
| `pipeline.py` | OOF + full inference orchestration | 30, 31, 33, 35 |

These are textual extracts; running them outside the notebook requires
proper standalone refactor (TODO: extract Kaggle-only paths, fix
imports, isolate state dependencies).

## Forward extension procedure

1. Develop new component in `lib/` (e.g., a 4th teacher stream).
2. Validate locally on labeled SS via `experiments/_audits_post_v26/exp16x_*.py`.
3. Port to `notebook.ipynb` as a new cell or block before the final
   blend. Keep `lib/` and `notebook.ipynb` in sync — edits to one should
   be mirrored in the other.
4. Push kernel via `kaggle kernels push -p notebooks/birdclef-2026-mattia-fork`.
5. Submit via `kaggle competitions submit -c birdclef-2026 -f submission.csv -k ultimatumgame/birdclef-2026-mattia-fork -v <N>`.

## Kaggle inputs

- `birdclef-2026` (competition data)
- `jaejohn/perch-meta` — Perch metadata cache
- `rishikeshjani/perch-onnx-for-birdclef-2026` — Perch ONNX export
- `tuckerarrants/bc2026-distilled-sed-public` — Tucker 5-fold SED ONNX
- Kernel: `ashok205/tf-wheels` — TF 2.20 wheels
- Kernel: `vyankteshdwivedi/birdclef-2026-onnx-perch-sequence-modeling` — base notebook
- Model: `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`

`enable_internet: false` (offline code-competition).

## Forward levers (open candidates)

1. **exp167**: train our own SED with mel=256, fmax=16000, 5-fold CV.
   Add as 3rd or 4th rank-blend stream. Gives orthogonal Insecta signal
   (different training data than Tucker). 20 hr GPU budget.
2. **External non-Aves data** (Watkins, Macaulay, AnuraSet) — train SED
   with explicit non-Aves supervision.
3. **Hyperparameter tuning of existing rescue rules** on labeled SS OOF
   (PROTO_CONT_RANK_THR, SED_ONLY_RANK_THR, etc.) — small expected gain.
4. **Stochastic averaging** — submit Mattia fork multiple times across
   seeds; ensemble outputs (would need multi-version notebook).
