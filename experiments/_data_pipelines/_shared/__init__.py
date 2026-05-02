"""Shared SED training utilities.

Centralizes common code reused across exp50, exp121, exp123, exp129, exp131,
exp132+. Each experiment focuses only on what's NEW (loss, data filter, ...).
"""
from .constants import *
from .audio import load_audio, random_crop, center_crop, get_taxon_array
from .data import (build_primaries, build_ta_combined, build_ss_splits,
                    build_pseudo_ss, TADataset, SSDataset, SSPseudoDataset)
from .model import SEDModel, MelExtractor, SpecAug, SEDHead
from .augment import aggressive_mixup, load_bg_pool
from .train import train_sed_loop
