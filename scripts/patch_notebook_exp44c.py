#!/usr/bin/env python3
"""Patch notebooks/birdclef-2026-perch-distill/notebook.ipynb to insert
exp44c 27-species overlay RIGHT BEFORE `# --- Build submission ---` in cell 50.

Also bumps version comment so kernel push creates a new revision.
Idempotent: checks for marker `# --- exp44c OVERLAY ---` before inserting.
"""
import json
from pathlib import Path

NB_PATH = Path("notebooks/birdclef-2026-perch-distill/notebook.ipynb")
PATCH_SRC = Path("notebooks/birdclef-2026-perch-distill/exp44c_patch.py")
MARKER_BUILD = "# --- Build submission ---"
MARKER_EXP44C_BEGIN = "# --- exp44c OVERLAY BEGIN ---"
MARKER_EXP44C_END = "# --- exp44c OVERLAY END ---"


def main():
    nb = json.loads(NB_PATH.read_text())
    patch_code = PATCH_SRC.read_text()

    # Wrap the patch with markers for idempotence
    wrapped = f"\n{MARKER_EXP44C_BEGIN}\n{patch_code}\n{MARKER_EXP44C_END}\n\n"

    cell50_src = "".join(nb["cells"][50]["source"])

    if MARKER_EXP44C_BEGIN in cell50_src:
        print("Patch already applied. Removing then re-applying (refresh).")
        # Strip existing patch
        start = cell50_src.find(MARKER_EXP44C_BEGIN)
        end = cell50_src.find(MARKER_EXP44C_END)
        if end < 0:
            raise RuntimeError("Malformed existing patch; manual cleanup required.")
        end = end + len(MARKER_EXP44C_END)
        cell50_src = cell50_src[:start] + cell50_src[end:]

    idx = cell50_src.find(MARKER_BUILD)
    if idx < 0:
        raise RuntimeError(f"Marker {MARKER_BUILD!r} not found in cell 50.")

    new_src = cell50_src[:idx] + wrapped + cell50_src[idx:]
    # Convert back to list-of-lines for Jupyter
    nb["cells"][50]["source"] = new_src.splitlines(keepends=True)
    nb["cells"][50]["outputs"] = []
    nb["cells"][50]["execution_count"] = None

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Patched {NB_PATH}")
    print(f"  cell 50 new length: {len(new_src)} chars (added {len(wrapped)} chars)")


if __name__ == "__main__":
    main()
