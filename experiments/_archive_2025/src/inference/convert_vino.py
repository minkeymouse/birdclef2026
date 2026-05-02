import torch
import timm
import openvino as ov
from openvino.runtime import serialize
from pathlib import Path
import pandas as pd
import numpy as np

# ── 0) Prerequisites ────────────────────────────────────────
# Make sure you have the OpenVINO dev package for model conversion:
#    pip install --upgrade openvino-dev
# ────────────────────────────────────────────────────────────

torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.dtype
])

print("Loading taxonomy data…")
taxonomy_df = pd.read_csv("data/birdclef/taxonomy.csv")
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f"Number of classes: {num_classes}")

CHECKPOINT_DIR = Path("models")
IR_OUT_DIR     = Path("ir_models")
IR_OUT_DIR.mkdir(parents=True, exist_ok=True)  # create with parents if needed

# ── Loop through all checkpoints and convert ────────────────────
for ckpt in CHECKPOINT_DIR.glob("*.pth"):
    fold_name = ckpt.stem  # e.g. 'efficientnet_b0_fold0_best'

    # 1) Instantiate model architecture based on filename
    if "efficientnet_b0" in fold_name:
        arch = "efficientnet_b0.ra_in1k"
    elif "efficientnet_b3" in fold_name:
        arch = "efficientnet_b3.ra2_in1k"
    elif "focal" in fold_name:
        arch = "efficientnet_b0"
    else:
        print(f"Skipping unknown arch in {fold_name}")
        continue

    model = timm.create_model(
        arch,
        pretrained=False,
        in_chans=1,
        num_classes=num_classes
    )
    model.eval()

    # 2) Load and clean state dict
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    cleaned = {}
    for k, v in sd.items():
        # strip common prefixes from keys
        new_k = k.replace("module.", "").replace("backbone.", "")
        cleaned[new_k] = v
    model.load_state_dict(cleaned)

    # 3) Convert to OpenVINO IR
    example_input = torch.randn(1, 1, 256, 256)
    ov_model = ov.convert_model(model, example_input=example_input)  # :contentReference[oaicite:0]{index=0}

    # 4) Serialize IR (writes .xml and .bin)
    xml_path = IR_OUT_DIR / f"{fold_name}.xml"
    bin_path = IR_OUT_DIR / f"{fold_name}.bin"
    serialize(ov_model, str(xml_path), str(bin_path))  # :contentReference[oaicite:1]{index=1}

    print(f"✔️  Saved IR for {fold_name} → {xml_path}, {bin_path}")
