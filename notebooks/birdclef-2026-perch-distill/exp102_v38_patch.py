# ─────────────────────────────────────────────────────────────────────────
# v38 LR-correction patch (exp99-102)
#
# Per-(row, class) selective correction using LR detectors trained on
# universal features (file uniformity, model agreement, raw teacher
# scores). LOSO-site CV: FP detector AUC 0.97, FN detector AUC 0.98.
#
# Mechanism:
#   FN_BOOST: where v33[i,c] < 0.5 AND P_FN > threshold → boost
#   FP_SUPPRESS: where v33[i,c] > 0.5 AND P_FP > threshold → suppress
#
# Best class-A (exp101): α_fn=0.3, β_fp=0.1
#   macro Δ +0.0035, sp_row 0.9996, Aves Δ +0.016, S08 unchanged
#
# Applied AFTER v33 file-max coherence and BEFORE exp48 patch.
# ─────────────────────────────────────────────────────────────────────────
import numpy as np

LR_FP_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/lr_fp_detector.npz'),
    Path('/kaggle/input/birdclef2026-model-weights/lr_fp_detector.npz'),
]
LR_FN_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/lr_fn_detector.npz'),
    Path('/kaggle/input/birdclef2026-model-weights/lr_fn_detector.npz'),
]
LR_META_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/lr_correction_meta.npz'),
    Path('/kaggle/input/birdclef2026-model-weights/lr_correction_meta.npz'),
]
_lr_fp_path = next((p for p in LR_FP_PATHS if p.exists()), None)
_lr_fn_path = next((p for p in LR_FN_PATHS if p.exists()), None)
_lr_meta_path = next((p for p in LR_META_PATHS if p.exists()), None)

# v38 hyperparameters (chosen for safety)
ALPHA_FN = 0.3   # FN-boost strength
BETA_FP = 0.1    # FP-suppress strength

if _lr_fp_path is None or _lr_fn_path is None or _lr_meta_path is None:
    print(f"v38 LR-correction: artifacts missing, skipping")
elif sed59_scores is None:
    print(f"v38 LR-correction: exp59 inference missing — skipping (need exp59 for features)")
else:
    print(f"\nv38 LR-correction: applying with α_fn={ALPHA_FN}, β_fp={BETA_FP}", flush=True)
    _lr_fp = np.load(_lr_fp_path)
    _lr_fn = np.load(_lr_fn_path)
    _lr_meta = np.load(_lr_meta_path, allow_pickle=True)
    _candidate = _lr_meta["candidate_classes"]
    print(f"  candidate Aves classes: {len(_candidate)}")

    # Convert probs back to allow modification on the v33 ranking space.
    # We're working with `probs` (post V9 + file-max + post-blend Gauss + threshold).
    # The LR was trained on v33 = (V9 gate × file-max coherence) without temperature/sigmoid.
    # In the notebook, `probs` after the threshold-sharpen step is in the same scale as v33.
    # We apply correction directly on probs.

    # Compute file-level statistics of exp50 (sed50_scores). meta_test["row_id"] is
    # ordered by file; each file has N_WINDOWS contiguous rows.
    _n_files = len(test_paths)
    _file_mean = np.zeros_like(sed50_scores)
    _file_std = np.zeros_like(sed50_scores)
    _exp50_view = sed50_scores.reshape(_n_files, N_WINDOWS, N_CLASSES)
    _fm = _exp50_view.mean(axis=1, keepdims=True)
    _fs = _exp50_view.std(axis=1, keepdims=True)
    _file_mean = np.broadcast_to(_fm, _exp50_view.shape).reshape(-1, N_CLASSES)
    _file_std = np.broadcast_to(_fs, _exp50_view.shape).reshape(-1, N_CLASSES)

    # Perch sigmoid scores for the row-class (perch_prob equivalent)
    _perch_sig = 1.0 / (1.0 + np.exp(-scores_test_raw))   # raw Perch logits → sigmoid

    # Build feature vectors for each (row, c) where c in candidate
    # Feature order: perch_on_c, exp50_on_c, exp59_on_c, perch_sed_disagree,
    #                perch_low_sed_high, file_mean, file_std, file_uniform, v33_on_c
    _v33_for_features = probs.copy()    # current pipeline probability

    def _apply_lr(detector_npz, X):
        # standardize then linear
        mu = detector_npz["scaler_mean"]
        sd = detector_npz["scaler_scale"]
        X_s = (X - mu) / sd
        z = X_s @ detector_npz["coef"] + detector_npz["intercept"]
        return 1.0 / (1.0 + np.exp(-z))

    # For efficiency, vectorize over (row, c) with c in candidate
    _n_rows = probs.shape[0]
    n_correct = 0
    for c in _candidate:
        cidx = int(c)
        # Build (n_rows, 9) feature matrix for this class
        X = np.stack([
            _perch_sig[:, cidx],
            sed50_scores[:, cidx],
            sed59_scores[:, cidx],
            np.abs(_perch_sig[:, cidx] - sed50_scores[:, cidx]),
            np.maximum(0, sed50_scores[:, cidx] - _perch_sig[:, cidx]),
            _file_mean[:, cidx],
            _file_std[:, cidx],
            1.0 - (_file_std[:, cidx] / (_file_mean[:, cidx] + 1e-6)),
            _v33_for_features[:, cidx],
        ], axis=1).astype(np.float32)

        # Apply detector based on v33 threshold
        v_c = _v33_for_features[:, cidx]
        is_high = v_c > 0.5
        is_low = v_c < 0.5

        # FP detector for high-v33 rows
        if is_high.sum() > 0:
            P_FP = _apply_lr(_lr_fp, X[is_high])
            high_idx = np.where(is_high)[0]
            probs[high_idx, cidx] = v_c[high_idx] * (1.0 - BETA_FP * P_FP)
            n_correct += is_high.sum()
        # FN detector for low-v33 rows
        if is_low.sum() > 0:
            P_FN = _apply_lr(_lr_fn, X[is_low])
            low_idx = np.where(is_low)[0]
            probs[low_idx, cidx] = v_c[low_idx] + ALPHA_FN * P_FN * (1.0 - v_c[low_idx])
            n_correct += is_low.sum()

    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
    print(f"  v38 LR-correction applied on {n_correct} (row, class) pairs across {len(_candidate)} classes")
    print(f"  probs range after: [{probs.min():.5f}, {probs.max():.5f}]")
    del _lr_fp, _lr_fn, _lr_meta
