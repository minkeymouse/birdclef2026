"""Path configuration — switches Kaggle env vs local env."""
import os
from pathlib import Path

# Detect environment
IS_KAGGLE = Path("/kaggle/input").exists()


def _first(*candidates):
    """Return first existing path among candidates."""
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None


# Tucker bc2026-distilled-sed-public
TUCKER_DIR = _first(
    "/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public",
    "/kaggle/input/bc2026-distilled-sed-public",
    "/data/birdclef2026/model-weights/tucker_sed",
)

# Perch v2 ONNX
PERCH_ONNX = _first(
    "/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx",
    "/kaggle/input/perch-onnx-for-birdclef-2026/perch_v2.onnx",
    "/tmp/perch_v2.onnx",
    "/data/birdclef2026/model-weights/perch_v2.onnx",
)

# Competition data
COMP_DATA = _first(
    "/kaggle/input/competitions/birdclef-2026",
    "/kaggle/input/birdclef-2026",
    "/data/birdclef2026/data/birdclef-2026",
)

# Repository root (local only)
REPO_ROOT = Path("/data/birdclef2026") if not IS_KAGGLE else None

# Cached scores for local audits
EXP80_OUTPUTS = (REPO_ROOT / "experiments/_audits_post_v26/exp80_outputs"
                  if REPO_ROOT else None)


def report():
    """Print path resolution status."""
    print(f"IS_KAGGLE: {IS_KAGGLE}")
    for name, p in [("TUCKER_DIR", TUCKER_DIR), ("PERCH_ONNX", PERCH_ONNX),
                     ("COMP_DATA", COMP_DATA), ("REPO_ROOT", REPO_ROOT),
                     ("EXP80_OUTPUTS", EXP80_OUTPUTS)]:
        status = "OK" if (p is not None and Path(p).exists()) else "MISSING"
        print(f"  {name:15s}: {p}  [{status}]")


if __name__ == "__main__":
    report()
