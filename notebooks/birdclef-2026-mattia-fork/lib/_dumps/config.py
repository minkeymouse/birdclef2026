"""Configuration constants and CFG dict (cells 5)."""
import os, time, warnings
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

_WALL_START = time.time()
 
BASE      = Path("/kaggle/input/competitions/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
WORK_DIR  = Path("/kaggle/working/cache")
WORK_DIR.mkdir(parents=True, exist_ok=True)
 
SR             = 32_000
WINDOW_SEC     = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES   = 60 * SR
N_WINDOWS      = 12          # 12 × 5s = 60s per file
 
CFG = {
    # inference
    "batch_files": 16,

    # local CV
    "oof_n_splits": 5   if MODE == "train" else 3,

    # dry-run
    "dryrun_n_files": 20 if MODE == "train" else 0,

    # train-only flags
    "run_oof": MODE == "train",
    "verbose": MODE == "train",

    # V18 proto_ssm
    "proto_ssm_train": {
        "n_epochs":        80  if MODE == "train" else 40,
        "lr":              8e-4,
        "weight_decay":    1e-3,
        "val_ratio":       0.15,
        "patience":        20  if MODE == "train" else 8,
        "pos_weight_cap":  25.0,
        "distill_weight":  0.15,
        "proto_margin":    0.15,
        "label_smoothing": 0.03,
        "oof_n_splits":    5   if MODE == "train" else 3,
        "mixup_alpha":     0.4,
        "focal_gamma":     2.5,
        "swa_start_frac":  0.65,
        "swa_lr":          4e-4,
        "use_cosine_restart": True,
        "restart_period":  20,
    },
    "residual_ssm": {
        "d_model": 128, "d_state": 16, "n_ssm_layers": 2,
        "dropout": 0.1, "correction_weight": 0.35,
        "n_epochs": 40  if MODE == "train" else 20,
        "lr": 8e-4,
        "patience": 12  if MODE == "train" else 6,
    },
    "mlp_params": {
        "hidden_layer_sizes": (256, 128), "activation": "relu",
        "max_iter": 500  if MODE == "train" else 200,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20  if MODE == "train" else 10,
        "random_state": 42,
        "learning_rate_init": 5e-4,
        "alpha": 0.005,
    },
}
print("✅ V18 CFG loaded")
print(f"  n_epochs={CFG['proto_ssm_train']['n_epochs']}  "
      f"patience={CFG['proto_ssm_train']['patience']}  "
      f"oof_n_splits={CFG['proto_ssm_train']['oof_n_splits']}  "
      f"mlp_max_iter={CFG['mlp_params']['max_iter']}")
 
print("Config ready")
print(f"  run_oof={CFG['run_oof']}  verbose={CFG['verbose']}  dryrun={CFG['dryrun_n_files']}")
