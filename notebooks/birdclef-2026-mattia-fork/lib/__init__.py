"""Mattia 0.941 pipeline — locally importable production library.

LB-verified at 0.941 (kernel ultimatumgame/birdclef-2026-mattia-fork v1, 2026-05-03).

Self-contained, locally importable modules:
  paths        Kaggle-vs-local path resolution (TUCKER_DIR, PERCH_ONNX, etc)
  tucker_sed   5-fold ONNX SED ensemble inference
  rank_scale   rank-aware file_max + adaptive δ smoothing + file-max blend
  helpers      macro AUC + Gaussian smoothing + per-taxon temperature
  final_blend  rank-pct blend + 3 rescue rules + linear blend

Source-dump modules (not yet self-contained; need to be re-extracted with
proper imports/paths if you want to use them outside the notebook):
  config, data, perch, mlp_probe, protossm, pipeline

For local audits, use the self-contained modules together with cached
labeled-SS scores at experiments/_audits_post_v26/exp80_outputs/.
"""
from . import paths
from . import tucker_sed
from . import rank_scale
from . import helpers
from . import final_blend

__all__ = ["paths", "tucker_sed", "rank_scale", "helpers", "final_blend"]
