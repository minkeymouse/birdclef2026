#!/usr/bin/env python3
"""Replace the exp44c OVERLAY block in notebook.ipynb cell 50 with the exp45c
soft taxon gating patch. Idempotent."""
import json
from pathlib import Path

NB = Path("notebooks/birdclef-2026-perch-distill/notebook.ipynb")
PATCH = Path("notebooks/birdclef-2026-perch-distill/exp45c_patch.py")
BEGIN_44C = "# --- exp44c OVERLAY BEGIN ---"
END_44C = "# --- exp44c OVERLAY END ---"
BEGIN_45C = "# --- exp45c GATE BEGIN ---"
END_45C = "# --- exp45c GATE END ---"


def main():
    nb = json.loads(NB.read_text())
    patch = PATCH.read_text()
    cell = "".join(nb["cells"][50]["source"])

    # Remove existing exp44c or exp45c block if present
    for b_mark, e_mark in [(BEGIN_44C, END_44C), (BEGIN_45C, END_45C)]:
        if b_mark in cell:
            s = cell.find(b_mark)
            e = cell.find(e_mark) + len(e_mark)
            cell = cell[:s] + cell[e:]
            print(f"Removed block {b_mark}")

    # Insert new 45c block before "# --- Build submission ---"
    marker = "# --- Build submission ---"
    idx = cell.find(marker)
    if idx < 0:
        raise RuntimeError(f"Marker not found: {marker}")

    wrapped = f"\n{BEGIN_45C}\n{patch}\n{END_45C}\n\n"
    cell = cell[:idx] + wrapped + cell[idx:]

    nb["cells"][50]["source"] = cell.splitlines(keepends=True)
    nb["cells"][50]["outputs"] = []
    nb["cells"][50]["execution_count"] = None
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Patched {NB} with exp45c gate block")


if __name__ == "__main__":
    main()
