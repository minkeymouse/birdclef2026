#!/usr/bin/env python3
"""
weight_update.py — bump weights for rare-species chunks based on config
"""
import logging
import yaml
import pandas as pd
from pathlib import Path

# ─── Load pipeline configuration ───────────────────────────
project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "process.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# Paths and config sections
paths_cfg     = CFG["paths"]
selection_cfg = CFG["selection"]
labeling_cfg  = CFG["labeling"]

# Convert to Path objects
data_root = Path(paths_cfg["DATA_ROOT"])
meta_csv  = Path(paths_cfg["meta_data"])
train_csv = data_root / "train.csv"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("weight_update")

# ─── Rare species threshold and target weight ───────────────
thresh = selection_cfg.get("rare_species_threshold", 100)
rare_w = labeling_cfg.get("rare_label_weight", 1.0)

# ─── Load and count primary labels ──────────────────────────
train_df = pd.read_csv(train_csv)
counts   = train_df["primary_label"].value_counts()
log.info("Species counts (top 5):\n%s", counts.head())

# Identify rare species
rare_species = counts[counts < thresh].index.tolist()
log.info("Found %d rare species (< %d samples)", len(rare_species), thresh)

# ─── Load metadata and filter original training chunks ───────
meta_df = pd.read_csv(meta_csv)
original_mask = meta_df.get("source", "") == "train_audio"

# Map filename → primary_label
label_map = train_df.set_index("filename")["primary_label"]
meta_df["primary_label"] = meta_df["filename"].map(label_map)

# ─── Bump weight for rare-species chunks ────────────────────
mask = original_mask & meta_df["primary_label"].isin(rare_species)
log.info("Updating %d chunks to weight = %s", mask.sum(), rare_w)
meta_df.loc[mask, "weight"] = rare_w

# ─── Clean up and save ──────────────────────────────────────
meta_df.drop(columns=["primary_label"], inplace=True)
meta_df.to_csv(meta_csv, index=False)
log.info("Saved updated metadata to %s", meta_csv)
