# ─────────────────────────────────────────────────────────────────────────
# v44 exp121 aggressive cross-region SED — additive blend on v33
#
# exp121 = exp50 fine-tuned with stronger cross-region BG augmentation
# (BG_MIX_P 0.4→0.5 Aves / 0.85 non-Aves, +13k 2025 train_audio for 41 overlap).
#
# Local validation:
#   val_SS: exp50 0.838 → exp121 0.851 (+0.013)
#   v33 + 0.10 * exp121 additive on 122 eval:
#     macro_d +0.0052, sp_row 0.9997, Aves +0.008, Insecta +0.002,
#     Mam +0.012, Amphib +0.006 — ALL TAXA POSITIVE
#
# Mechanism: v33's own category (universal physics + cross-region invariance).
# Profile matches v33's own LB-positive trajectory:
#   v33 (file-max α=0.10): macro_d +0.003 sp_row 0.994 Aves +0.007 → LB +0.001
#   v44 (v33 + 0.10 * exp121): macro_d +0.005 sp_row 0.9997 Aves +0.008
#
# Same training-time data-diversity mechanism that v24 (exp50 swap) was
# +0.001 LB-positive — exp121 strengthens this mechanism (more 2025 BG).
# ─────────────────────────────────────────────────────────────────────────
EXP121_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp121_aggressive_synth_sed.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/exp121_aggressive_synth_sed.pt'),
]
_exp121_ckpt = next((p for p in EXP121_CKPT_PATHS if p.exists()), None)
W_EXP121 = 0.10  # additive on top of v33 (post file-max coherence state)

if _exp121_ckpt is None:
    print("v44 exp121: ckpt missing, skipping")
else:
    print(f"\nv44 exp121: applying with W_EXP121={W_EXP121}", flush=True)
    _exp121_m, _exp121_v, _exp121_bb = _load_sed(_exp121_ckpt)
    print(f"  exp121 ckpt val_SS: {_exp121_v}, backbone: {_exp121_bb}")
    _exp121_scores = _run_sed(_exp121_m, test_paths, desc='exp121')
    probs = (1.0 - W_EXP121) * probs + W_EXP121 * _exp121_scores
    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
    print(f"  v44 applied. probs range: [{probs.min():.5f}, {probs.max():.5f}]")
    del _exp121_m, _exp121_scores
