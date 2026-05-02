# ─────────────────────────────────────────────────────────────────────────
# v46 exp136b SED — additive blend on v33
#
# exp136b = exp50 fine-tuned with v3 pseudo-labels (351k entries: 198k Aves +
# 84k Amphibia + 68k Insecta sonotype + 1.3k Mammalia + 210 Reptilia).
#
# Iterative refinement framework:
#   v3 = v2 (acoustic-verified pseudo) + sonotype confusion mapping
#   sonotype pseudo recovered via: exp43r/108 forward signature inverse
#
# Local validation:
#   exp50 baseline val_SS: 0.838
#   exp136b val_SS: 0.907 (+0.069, biggest jump in pipeline)
#   Per-taxon: Aves 0.935, Amphib 0.849, Insecta 0.938, Mam 0.937
#
# v33 + 0.10 * exp136b additive on 122 eval:
#   macro_d +0.038, sp_row 0.997, Aves +0.007, Insecta +0.086, Mam +0.006,
#   Amphib +0.019, Rept +0.044 — all-taxa positive, biggest swing yet.
#
# Profile vs. previous LB attempts:
#   - macro_d 8x bigger than v44/v45 (+0.038 vs +0.005)
#   - Aves Δ +0.007 = same as v33's own LB-positive signature
#   - sp_row 0.997 = safe (between v33's 0.994 and v44's 0.9997)
#
# First time pseudo-label retrain produces this clean profile.
# ─────────────────────────────────────────────────────────────────────────
EXP136B_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp136b_v3_pseudo_sed.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/exp136b_v3_pseudo_sed.pt'),
]
_exp136b_ckpt = next((p for p in EXP136B_CKPT_PATHS if p.exists()), None)
W_EXP136B = 0.10  # additive on top of v33

if _exp136b_ckpt is None:
    print("v46 exp136b: ckpt missing, skipping")
else:
    print(f"\nv46 exp136b: applying with W_EXP136B={W_EXP136B}", flush=True)
    _exp136b_m, _exp136b_v, _exp136b_bb = _load_sed(_exp136b_ckpt)
    print(f"  exp136b ckpt val_SS: {_exp136b_v}, backbone: {_exp136b_bb}")
    _exp136b_scores = _run_sed(_exp136b_m, test_paths, desc='exp136b')
    probs = (1.0 - W_EXP136B) * probs + W_EXP136B * _exp136b_scores
    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
    print(f"  v46 applied. probs range: [{probs.min():.5f}, {probs.max():.5f}]")
    del _exp136b_m, _exp136b_scores
