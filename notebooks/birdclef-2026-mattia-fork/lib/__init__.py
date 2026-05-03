"""Mattia 0.941 pipeline — locally importable production library.

LB-verified at 0.941 (kernel ultimatumgame/birdclef-2026-mattia-fork v1, 2026-05-03).

Public API (5 self-contained, locally importable modules):
  paths        Kaggle-vs-local path resolution
  tucker_sed   Tucker 5-fold ONNX SED inference
  rank_scale   rank-aware file_max scaling + adaptive δ smoothing + file-max blend
  helpers      macro AUC + Gaussian smoothing + per-taxon temperature
  final_blend  rank-pct blend + 3 rescue rules + linear blend

Verified by experiments/_audits_post_v26/exp168_mattia_lib_sanity.py:
tucker_sed reproduces cached scores 36/36 exact match; final_blend
reproduces v58 (+0.117), mattia full-rescues (+0.087), no-rescues
(+0.078) numbers exactly.

Source-only dumps (NOT importable, NOT in public API):
See `_dumps/` for raw cell extracts of config/data/perch/mlp_probe/
protossm/pipeline. Kept for code-archaeology only. Refactor each into
self-contained module before using.
"""
from . import paths
from . import tucker_sed
from . import rank_scale
from . import helpers
from . import final_blend

__all__ = ["paths", "tucker_sed", "rank_scale", "helpers", "final_blend"]
