# ─────────────────────────────────────────────────────────────────────────
# v45 exp123 surgical bird-bias SED — additive blend on v33
#
# exp123 = exp50 fine-tuned with surgical cross-taxon penalty:
#   only false-Aves (negative in row) vs true non-Aves (positive in row)
#   never penalizes truly-positive Aves
#
# Local validation:
#   val_SS: exp50 0.838 → exp121 0.851 → exp123 0.867 (+0.029 vs exp50)
#   v33 + 0.05 * exp123 on 122 eval:
#     macro_d +0.004, sp_row 0.9997, Aves 0.000, Insecta -0.0002,
#     Mam +0.003, Amphib +0.008, Rept +0.049 — all-taxa ≥ 0
#
# Key signature difference from prior attempts:
#   - v44 (exp121 +0.10): Aves +0.008 → LB -0.012
#   - v45 (exp123 +0.05): Aves +0.000 (exactly preserved)
#
# Bird-bias is SITE-INVARIANT (Perch model bias, not site-conditional).
# Penalty fixes only false-positive Aves (saturating ones from exp109)
# without disturbing positive Aves discrimination.
# ─────────────────────────────────────────────────────────────────────────
EXP123_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp123_bird_bias_sed.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/exp123_bird_bias_sed.pt'),
]
_exp123_ckpt = next((p for p in EXP123_CKPT_PATHS if p.exists()), None)
W_EXP123 = 0.05  # conservative additive on top of v33

if _exp123_ckpt is None:
    print("v45 exp123: ckpt missing, skipping")
else:
    print(f"\nv45 exp123: applying with W_EXP123={W_EXP123}", flush=True)
    _exp123_m, _exp123_v, _exp123_bb = _load_sed(_exp123_ckpt)
    print(f"  exp123 ckpt val_SS: {_exp123_v}, backbone: {_exp123_bb}")
    _exp123_scores = _run_sed(_exp123_m, test_paths, desc='exp123')
    probs = (1.0 - W_EXP123) * probs + W_EXP123 * _exp123_scores
    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
    print(f"  v45 applied. probs range: [{probs.min():.5f}, {probs.max():.5f}]")
    del _exp123_m, _exp123_scores
