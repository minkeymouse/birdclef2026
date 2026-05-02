"""Shared constants for SED training pipeline."""
from pathlib import Path

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
DATA25 = ROOT / "data" / "birdclef-2025"

SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
N_WINDOWS = 12
FILE_SAMPLES = SR * 60

N_FFT = 2048
HOP = 512
N_MELS = 128
F_MIN, F_MAX = 50, 14000

N_CLS = 234
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]

# Default training hyperparameters (override per experiment)
BATCH_SIZE = 32
LR = 2e-4
WD = 1e-2
EPOCHS = 8
NUM_WORKERS = 4
SEED = 42
EVAL_SS_N_FILES = 11

# Augmentation defaults
MIXUP_ALPHA = 0.5
MIXUP_P = 0.5
BG_MIX_P_AVES = 0.5
BG_MIX_P_NON_AVES = 0.85
BG_ALPHA_LO, BG_ALPHA_HI = 0.3, 0.7
SECONDARY_WEIGHT = 0.3
SPEC_FREQ_MASK = 16
SPEC_TIME_MASK = 40

# Sample weights
TA_WEIGHT = 1.0
SS_LABELED_WEIGHT = 5.0
SS_PSEUDO_WEIGHT = 1.0

BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"

# Common paths
EXP50_CKPT = ROOT / "experiments/_data_pipelines/exp50_outputs/best_ckpt.pt"
BG_PATH = ROOT / "experiments/_data_pipelines/exp49_outputs/bg_quiet_2025.npz"
