#!/usr/bin/env python3
"""
exp1_eda.py — BirdCLEF 2026 Exploratory Data Analysis

Covers:
1. Taxonomy group breakdown (species count, sample count per class)
2. train_soundscapes_labels analysis (multi-label distribution, soundscape-only species)
3. 2025 vs 2026 species overlap
4. Audio quality / rating distribution by collection source
"""
import ast
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_2026 = ROOT / "data" / "birdclef-2026"
DATA_2025 = ROOT / "data" / "birdclef-2025"
OUT_DIR = ROOT / "experiments" / "exp1_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load 2026 data ────────────────────────────────────────
train_df = pd.read_csv(DATA_2026 / "train.csv")
taxonomy_df = pd.read_csv(DATA_2026 / "taxonomy.csv")
sample_sub = pd.read_csv(DATA_2026 / "sample_submission.csv", nrows=1)
soundscape_labels = pd.read_csv(DATA_2026 / "train_soundscapes_labels.csv")

# ── Load 2025 data ────────────────────────────────────────
train_2025 = pd.read_csv(DATA_2025 / "train.csv")
taxonomy_2025 = pd.read_csv(DATA_2025 / "taxonomy.csv")

# Species lists
species_2026 = set(taxonomy_df["primary_label"].astype(str))
species_2025 = set(taxonomy_2025["primary_label"].astype(str))
submission_species = set(sample_sub.columns) - {"row_id"}

results = {}

# =====================================================================
# 1. Taxonomy Group Breakdown
# =====================================================================
print("=" * 60)
print("1. TAXONOMY GROUP BREAKDOWN")
print("=" * 60)

# Species per class
tax_group = taxonomy_df.groupby("class_name").size().sort_values(ascending=False)
print("\nSpecies count per taxonomic class:")
print(tax_group.to_string())
results["species_per_class"] = tax_group.to_dict()

# train.csv already has class_name column
train_merged = train_df

# Sample count per class
sample_group = train_merged.groupby("class_name").size().sort_values(ascending=False)
print("\nSample count per taxonomic class:")
print(sample_group.to_string())
results["samples_per_class"] = sample_group.to_dict()

# Samples per species, grouped by class
species_sample_counts = train_merged.groupby(["class_name", "primary_label"]).size().reset_index(name="count")
class_stats = species_sample_counts.groupby("class_name")["count"].agg(["mean", "median", "min", "max", "std"])
print("\nPer-species sample statistics by class:")
print(class_stats.to_string())
results["per_species_stats"] = class_stats.to_dict()

# Plot: species count and sample count side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
tax_group.plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Species Count by Taxonomic Class")
axes[0].set_ylabel("Number of Species")
axes[0].tick_params(axis="x", rotation=45)

sample_group.plot(kind="bar", ax=axes[1], color="coral")
axes[1].set_title("Sample Count by Taxonomic Class")
axes[1].set_ylabel("Number of Recordings")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "taxonomy_breakdown.png", dpi=150)
plt.close()

# Plot: per-species sample distribution (log scale)
fig, ax = plt.subplots(figsize=(12, 5))
for cls_name in species_sample_counts["class_name"].unique():
    subset = species_sample_counts[species_sample_counts["class_name"] == cls_name]
    counts_sorted = subset["count"].sort_values(ascending=False).values
    ax.plot(range(len(counts_sorted)), counts_sorted, label=cls_name, marker=".", markersize=3)
ax.set_yscale("log")
ax.set_xlabel("Species rank (within class)")
ax.set_ylabel("Number of recordings (log)")
ax.set_title("Per-Species Sample Distribution by Taxonomic Class")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "species_sample_distribution.png", dpi=150)
plt.close()

# =====================================================================
# 2. train_soundscapes_labels Analysis
# =====================================================================
print("\n" + "=" * 60)
print("2. TRAIN SOUNDSCAPES LABELS ANALYSIS")
print("=" * 60)

# Parse multi-label: primary_label is semicolon-separated
soundscape_labels["species_list"] = soundscape_labels["primary_label"].str.split(";")
soundscape_labels["num_species"] = soundscape_labels["species_list"].apply(len)

print(f"\nTotal labeled segments: {len(soundscape_labels)}")
print(f"Unique soundscape files: {soundscape_labels['filename'].nunique()}")
results["soundscape_total_segments"] = len(soundscape_labels)
results["soundscape_unique_files"] = soundscape_labels["filename"].nunique()

# Multi-label distribution
multi_label_dist = soundscape_labels["num_species"].value_counts().sort_index()
print("\nSpecies per 5-sec segment:")
print(multi_label_dist.to_string())
results["multi_label_distribution"] = multi_label_dist.to_dict()

# All species in soundscape labels
all_soundscape_species = set()
for species_list in soundscape_labels["species_list"]:
    all_soundscape_species.update(species_list)
print(f"\nUnique species in soundscape labels: {len(all_soundscape_species)}")

# Species in train_audio
train_audio_species = set(train_df["primary_label"].astype(str).unique())
print(f"Unique species in train_audio: {len(train_audio_species)}")

# Soundscape-only species (NOT in train_audio)
soundscape_only = all_soundscape_species - train_audio_species
print(f"\nSpecies ONLY in soundscape labels (not in train_audio): {len(soundscape_only)}")
if soundscape_only:
    # Get taxonomy info for these
    soundscape_only_info = taxonomy_df[taxonomy_df["primary_label"].isin(soundscape_only)][
        ["primary_label", "common_name", "class_name"]
    ].sort_values("class_name")
    print(soundscape_only_info.to_string(index=False))
    results["soundscape_only_species"] = soundscape_only_info.to_dict(orient="records")
else:
    results["soundscape_only_species"] = []

# Train-audio-only species (not in soundscape labels)
train_only = train_audio_species - all_soundscape_species
print(f"\nSpecies in train_audio but NOT in soundscape labels: {len(train_only)}")
results["train_audio_only_count"] = len(train_only)

# Submission species not covered by either
uncovered = submission_species - train_audio_species - all_soundscape_species
print(f"Submission species not in any training data: {len(uncovered)}")
if uncovered:
    print(f"  {uncovered}")
results["uncovered_species"] = list(uncovered)

# Frequency of each species in soundscapes
soundscape_species_freq = Counter()
for species_list in soundscape_labels["species_list"]:
    soundscape_species_freq.update(species_list)
soundscape_freq_df = pd.DataFrame(
    soundscape_species_freq.most_common(),
    columns=["species", "segment_count"]
).merge(taxonomy_df[["primary_label", "class_name"]], left_on="species", right_on="primary_label", how="left")

# Plot: multi-label distribution
fig, ax = plt.subplots(figsize=(8, 5))
multi_label_dist.plot(kind="bar", ax=ax, color="steelblue")
ax.set_xlabel("Number of species per 5-sec segment")
ax.set_ylabel("Number of segments")
ax.set_title("Multi-label Distribution in Soundscape Annotations")
plt.tight_layout()
plt.savefig(OUT_DIR / "multilabel_distribution.png", dpi=150)
plt.close()

# Plot: top 30 species in soundscape by frequency, colored by class
top_soundscape = soundscape_freq_df.head(30)
fig, ax = plt.subplots(figsize=(14, 6))
palette = {"Aves": "steelblue", "Insecta": "orange", "Amphibia": "green",
           "Mammalia": "red", "Reptilia": "purple"}
colors = [palette.get(c, "gray") for c in top_soundscape["class_name"]]
ax.barh(range(len(top_soundscape)), top_soundscape["segment_count"].values, color=colors)
ax.set_yticks(range(len(top_soundscape)))
ax.set_yticklabels(top_soundscape["species"].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Number of 5-sec segments")
ax.set_title("Top 30 Species in Soundscape Labels")
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in palette.items()]
ax.legend(handles=legend_elements, loc="lower right")
plt.tight_layout()
plt.savefig(OUT_DIR / "soundscape_top_species.png", dpi=150)
plt.close()

# =====================================================================
# 3. 2025 vs 2026 Species Overlap
# =====================================================================
print("\n" + "=" * 60)
print("3. 2025 vs 2026 SPECIES OVERLAP")
print("=" * 60)

overlap = species_2026 & species_2025
only_2026 = species_2026 - species_2025
only_2025 = species_2025 - species_2026

print(f"\n2025 species: {len(species_2025)}")
print(f"2026 species: {len(species_2026)}")
print(f"Overlap: {len(overlap)}")
print(f"New in 2026: {len(only_2026)}")
print(f"Removed from 2025: {len(only_2025)}")

results["species_overlap"] = {
    "2025_total": len(species_2025),
    "2026_total": len(species_2026),
    "overlap": len(overlap),
    "new_in_2026": len(only_2026),
    "removed_from_2025": len(only_2025),
}

# New in 2026: which classes?
if only_2026:
    new_tax = taxonomy_df[taxonomy_df["primary_label"].isin(only_2026)]
    new_by_class = new_tax.groupby("class_name").size().sort_values(ascending=False)
    print("\nNew species in 2026 by class:")
    print(new_by_class.to_string())
    results["new_2026_by_class"] = new_by_class.to_dict()

# 2025 train_audio samples available for overlapping species
overlap_2025_samples = train_2025[train_2025["primary_label"].isin(overlap)]
print(f"\n2025 train_audio recordings reusable (overlap species): {len(overlap_2025_samples)}")
results["reusable_2025_samples"] = len(overlap_2025_samples)

# Venn-like summary
fig, ax = plt.subplots(figsize=(8, 5))
data = [len(only_2025), len(overlap), len(only_2026)]
labels_bar = ["2025 only", "Overlap", "2026 only"]
colors_bar = ["#4a90d9", "#7cb342", "#e57373"]
ax.bar(labels_bar, data, color=colors_bar)
ax.set_ylabel("Number of Species")
ax.set_title("Species Overlap: BirdCLEF 2025 vs 2026")
for i, v in enumerate(data):
    ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "species_overlap_2025_2026.png", dpi=150)
plt.close()

# =====================================================================
# 4. Audio Quality / Rating Distribution
# =====================================================================
print("\n" + "=" * 60)
print("4. AUDIO QUALITY & RATING DISTRIBUTION")
print("=" * 60)

# Collection source distribution
collection_dist = train_df["collection"].value_counts()
print("\nCollection source distribution:")
print(collection_dist.to_string())
results["collection_distribution"] = collection_dist.to_dict()

# Rating distribution (0 = no rating, typical for iNat)
print("\nRating distribution:")
rating_dist = train_df["rating"].value_counts().sort_index()
print(rating_dist.to_string())
results["rating_distribution"] = rating_dist.to_dict()

# Rating by collection
rating_by_collection = train_df.groupby("collection")["rating"].describe()
print("\nRating stats by collection:")
print(rating_by_collection.to_string())

# Rating by taxonomic class
rating_by_class = train_merged.groupby("class_name")["rating"].agg(["mean", "median", "count"])
print("\nRating by taxonomic class:")
print(rating_by_class.to_string())

# Plot: rating distribution by collection
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (coll, grp) in enumerate(train_df.groupby("collection")):
    grp["rating"].hist(bins=20, ax=axes[i], color="steelblue" if coll == "XC" else "coral")
    axes[i].set_title(f"Rating Distribution — {coll} (n={len(grp)})")
    axes[i].set_xlabel("Rating")
    axes[i].set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "rating_distribution.png", dpi=150)
plt.close()

# Plot: sample count by collection and taxonomic class
fig, ax = plt.subplots(figsize=(10, 5))
ct = train_merged.groupby(["class_name", "collection"]).size().unstack(fill_value=0)
ct.plot(kind="bar", ax=ax)
ax.set_ylabel("Number of Recordings")
ax.set_title("Recordings by Taxonomic Class and Collection Source")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "collection_by_class.png", dpi=150)
plt.close()

# =====================================================================
# Save results summary as JSON
# =====================================================================
# Convert numpy types for JSON serialization
def convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

results_clean = json.loads(json.dumps(results, default=convert))
with open(OUT_DIR / "exp1_results.json", "w") as f:
    json.dump(results_clean, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print(f"All outputs saved to {OUT_DIR}")
print("=" * 60)
