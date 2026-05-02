"""Augmentation functions: cross-region BG mix + mixup."""
import random
import numpy as np
import torch

from .constants import (CLIP_SAMPLES, MIXUP_ALPHA, MIXUP_P,
                          BG_MIX_P_AVES, BG_MIX_P_NON_AVES,
                          BG_ALPHA_LO, BG_ALPHA_HI, BG_PATH)


def load_bg_pool():
    """Load 2025 SS quiet BG pool. Returns (n_bg, samples) or None."""
    if not BG_PATH.exists():
        return None
    bg = np.load(BG_PATH)
    return bg["windows"]


def aggressive_mixup(x, y, primary_idx, bg_pool, taxon_array,
                      bg_mix_p_aves=BG_MIX_P_AVES,
                      bg_mix_p_non_aves=BG_MIX_P_NON_AVES,
                      mixup_alpha=MIXUP_ALPHA,
                      mixup_p=MIXUP_P,
                      bg_alpha_lo=BG_ALPHA_LO,
                      bg_alpha_hi=BG_ALPHA_HI):
    """Per-row aggressive mixup with cross-region BG blending.

    For non-Aves rows: higher BG_MIX_P (forces site-invariance).
    For Aves rows: standard BG_MIX_P + standard mixup.
    """
    B = x.size(0)
    out_x = x.clone(); out_y = y.clone()
    for i in range(B):
        if int(primary_idx[i]) >= 0:
            cls_taxon = taxon_array[int(primary_idx[i])]
            bg_mix_p = bg_mix_p_non_aves if cls_taxon != "Aves" else bg_mix_p_aves
        else:
            bg_mix_p = bg_mix_p_aves

        use_bg = (bg_pool is not None and random.random() < bg_mix_p)
        if use_bg:
            bg_idx = random.randint(0, len(bg_pool) - 1)
            bg_5s = bg_pool[bg_idx]
            reps = CLIP_SAMPLES // len(bg_5s) + 1
            bg_partner = np.tile(bg_5s, reps)[:CLIP_SAMPLES].astype(np.float32)
            bg_t = torch.from_numpy(bg_partner).to(x.device)
            lam = random.uniform(bg_alpha_lo, bg_alpha_hi)
            out_x[i] = lam * x[i] + (1 - lam) * bg_t
            # Label kept as-is (BG has no labels)
        else:
            if random.random() < mixup_p:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                lam = max(lam, 1 - lam)
                j = random.randint(0, B - 1)
                out_x[i] = lam * x[i] + (1 - lam) * x[j]
                out_y[i] = lam * y[i] + (1 - lam) * y[j]
    return out_x, out_y
