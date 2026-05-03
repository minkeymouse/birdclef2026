"""Mattia 0.941 pipeline — locally importable production library.

LB-verified at 0.941 (kernel ultimatumgame/birdclef-2026-mattia-fork v1, 2026-05-03).

Self-contained modules (currently):
  paths       Kaggle-vs-local path resolution
  tucker_sed  5-fold ONNX SED ensemble inference
  final_blend rank-pct blend + 3 rescue rules + linear blend

Source-dump modules (need cell-context to run inside notebook; not yet
properly modularized — kept as historical reference):
  config, data, perch, helpers, mlp_probe, rank_scale, protossm, pipeline

For local audits, use `tucker_sed` + `final_blend` + cached scores in
experiments/_audits_post_v26/exp80_outputs/.
"""
from . import paths
from . import tucker_sed
from . import final_blend

__all__ = ["paths", "tucker_sed", "final_blend"]
