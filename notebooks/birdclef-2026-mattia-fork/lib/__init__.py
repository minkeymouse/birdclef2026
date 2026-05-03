"""Mattia 0.943 pipeline — modularized for local analysis & extension.

Forked from mattiaangeli/birdclef-2026-0-943-better-blend (2026-05-03).
LB-verified at 0.941 in our env (kernel ultimatumgame/birdclef-2026-mattia-fork v1).

Module layout:
  config        SR/window/N_CLASSES/CFG hyperparameters
  data          taxonomy + sample_submission + meta-loading
  perch         Perch ONNX loader + window-level inference cache
  helpers       metric, temporal smoothing, prior table, file-level scaling, per-taxon T
  mlp_probe     PCA-Perch MLP probe + isotonic calibration + per-class threshold
  rank_scale    rank-aware file_max^0.4 scaling + adaptive delta smoothing
  protossm      LightProtoSSM (state-space + cross-attn) + ResidualSSM second-pass
  pipeline      OOF eval + full inference orchestration
  tucker_sed    Tucker bc2026-distilled-sed-public 5-fold ONNX SED
  final_blend   rank-percentile blend + 3 conditional rescue rules
"""
