# `_dumps/` — raw cell extracts (NOT importable)

These are raw text dumps from `notebooks/birdclef-2026-mattia-fork/notebook.ipynb`
cells, kept for code-archaeology reference. They reference Kaggle-only paths
(`/kaggle/input/...`), use names defined in earlier cells (e.g. `tf`, `Path`,
`MODEL_DIR`) without imports, and rely on in-kernel state.

**They will NOT import. They are NOT part of the `lib` public API.**

To use any of this code locally you must:
1. Add proper `import` statements
2. Replace Kaggle paths with `from ..paths import ...`
3. Make functions self-contained (move shared state into module-level helpers)

The `lib/__init__.py` does NOT export anything from this directory.

The 5 properly self-contained modules are at `lib/`'s top level:
- `paths.py` — path resolution
- `tucker_sed.py` — Tucker 5-fold ONNX inference
- `rank_scale.py` — rank-aware scaling, adaptive smoothing, file-max blend
- `helpers.py` — metric, Gaussian smoothing, taxon temperature
- `final_blend.py` — rank-pct blend + 3 rescue rules + linear blend

| dump | original cells | what it should become if/when refactored |
|---|---|---|
| `config.py` | 5 | runtime constants (SR, N_WINDOWS, N_CLASSES) — actually mostly already in tucker_sed.py |
| `data.py` | 7 | taxonomy + sample submission loader (small) |
| `perch.py` | 9, 10, 12, 13 | Perch ONNX inference + cache loader (significant work) |
| `mlp_probe.py` | 21, 22, 23 | PCA-Perch MLP probe + isotonic — needs in-kernel-trained sklearn objects |
| `protossm.py` | 27, 28, 29 | LightProtoSSM + ResidualSSM — torch nn.Module, needs trained weights |
| `pipeline.py` | 30, 31, 33, 35 | full inference orchestration — depends on all of the above |
