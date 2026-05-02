# ─────────────────────────────────────────────────────────────────────────
# v52 exp159 SED — additive blend on v33 (multi-region BG retrain)
#
# exp159 = exp50 retrained with NEW BG pool: ~9k quiet 5-sec windows mined
# from xeno-canto train_audio across 151 lat/lon buckets (multi-region),
# replacing the single-Pantanal `bg_quiet_2025.npz` (11.9k windows).
# Same architecture/loss as exp50; only BG_PATH and BG_ALPHA_LO 0.3→0.1
# (Soltero 2025 low-SNR robustness).
#
# Mechanism class — training-time NEW site-invariance source:
#   v24 added 2025 Pantanal BG (1 new site) → +0.001 LB
#   exp159 adds 12+ regions (151 buckets) → predicted larger gain
#
# Local validation on 122 eval rows (epoch ≥3 ckpt):
#   v33 + 0.05 * exp159 additive: macro_d +0.0031, sp_row 0.99967
#     Aves +0.0034 (rare positive!), Insecta -0.0022, Mam +0.0056,
#     Amphib +0.0111, Reptilia -0.049 (single-class regression)
#   Pearson(Perch, exp159) = -0.059 (orthogonal — KEY insight,
#     exp159 adds genuinely new signal not captured by Perch)
#
# Profile vs prior LB attempts:
#   - Mechanism class matches exp50/v24 (training-time new site source)
#   - Aves Δ POSITIVE (most prior levers degrade Aves)
#   - sp_row best of any post-v33 candidate to date
#   - Orthogonal to Perch — strongest argument for blend gain
#
# Risk: Reptilia regression −0.049. Only 1 class, small impact on macro
# but flag if LB regresses by >0.005.
# ─────────────────────────────────────────────────────────────────────────
EXP159_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp159_multiregion_sed.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/exp159_multiregion_sed.pt'),
]
_exp159_ckpt = next((p for p in EXP159_CKPT_PATHS if p.exists()), None)
W_EXP159 = 0.05  # additive on top of v33 (conservative, sp_row best)

if _exp159_ckpt is None:
    print("v52 exp159: ckpt missing, skipping")
else:
    print(f"\nv52 exp159: applying with W_EXP159={W_EXP159}", flush=True)
    _exp159_m, _exp159_v, _exp159_bb = _load_sed(_exp159_ckpt)
    print(f"  exp159 ckpt val_TA: {_exp159_v}, backbone: {_exp159_bb}")
    _exp159_scores = _run_sed(_exp159_m, test_paths, desc='exp159')
    probs = (1.0 - W_EXP159) * probs + W_EXP159 * _exp159_scores
    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
    print(f"  v52 applied. probs range: [{probs.min():.5f}, {probs.max():.5f}]")
    del _exp159_m, _exp159_scores
