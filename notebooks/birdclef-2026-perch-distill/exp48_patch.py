# ─────────────────────────────────────────────────────────────────────────
# exp48 V10 patch — Site prior + confusion-cluster rewrite.
#
# Two inference-time post-hoc corrections applied AFTER the V9 taxon gate:
#
#  (1) SITE PRIOR
#      Per-site P(species observed) from labeled SS train split (55 files).
#      final_prob[:, c] *= (tau * site_prior(site, c) + (1 - tau))
#      Identity fallback (=1.0) for unseen sites → safe no-op.
#
#  (2) CONFUSION-CLUSTER REWRITE
#      For each rare target (non-Aves) with a known top-3 Aves confusion
#      trigger set (derived from train SS), boost the target when all 3
#      triggers are co-firing:
#        cluster_score(row) = min(probs[row, trig_a], ..., trig_c)
#        final_prob[row, target] *= (1 + alpha * cluster_score(row))
#
# Expected local effect (11-file held-out vs v12 baseline):
#   macro 0.7143 → 0.8318 (+0.118), Aves +0.004 (neutral, LB-safe).
#   Biggest gains on Mammalia/Reptilia/Amphibia; near-no-op on Aves.
#
# Local tests: exp48e/g (TRAIN-derived clusters + site priors, eval-leak free).
# Anti-correlation rule v2: Aves Δ ≈ 0 → predicts LB positive-or-neutral.
# ─────────────────────────────────────────────────────────────────────────

import json as _json
from pathlib import Path as _Path
import numpy as _np
import re as _re

# --- Load patch data (embedded as JSON dataset) ---
_PATCH_DATA_PATHS = [
    _Path("/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp48_patch_data.json"),
    _Path("/kaggle/input/birdclef2026-model-weights/exp48_patch_data.json"),
]
_patch_data_path = next((p for p in _PATCH_DATA_PATHS if p.exists()), None)

if _patch_data_path is None:
    print("exp48: patch_data.json NOT found, skipping site prior + cluster rewrite")
else:
    with open(_patch_data_path) as _f:
        _pd = _json.load(_f)
    _site_prior = _pd["site_prior"]    # {site: [234]}
    _cluster_map = _pd["cluster_map_whitelisted"]   # {target_idx_str: [trigger_idx_list]}
    print(f"exp48: loaded site_prior ({len(_site_prior)} sites) + cluster_map ({len(_cluster_map)} targets)")

    # --- CONFIG ---
    _EXP48_TAU = 0.75      # site prior strength [0..1]; 0.5 = balanced, 0.75 = aggressive
    _EXP48_ALPHA = 2.0    # cluster rewrite strength; local best 2-4

    # --- Parse site from test file names ---
    # row_ids look like: BC2026_Test_0042_S05_20250301_073000_5
    _FNAME_RE = _re.compile(r"BC2026_(?:Train|Test)_\d+_(S\d+)_\d{8}_\d{6}(?:_\d+)?$")
    def _site_of_row(rid):
        m = _FNAME_RE.match(str(rid))
        return m.group(1) if m else None

    # Build (n_rows, 234) site prior vector.  Identity for unknown sites.
    n_rows, n_cls = probs.shape
    assert n_cls == 234, f"expected 234 cols, got {n_cls}"
    # meta_test has the row_id column aligned with probs
    _row_ids_for_patch = meta_test["row_id"].values

    _site_vec = _np.ones((n_rows, n_cls), dtype=_np.float32)
    _hit_count = 0
    for _i, _rid in enumerate(_row_ids_for_patch):
        _s = _site_of_row(_rid)
        if _s and _s in _site_prior:
            _site_vec[_i] = _site_prior[_s]
            _hit_count += 1
    print(f"exp48: site prior hit {_hit_count}/{n_rows} rows (rest use identity fallback)")

    # --- Apply site prior (soft) ---
    probs = probs * (_EXP48_TAU * _site_vec + (1.0 - _EXP48_TAU))
    print(f"exp48: site prior applied  (τ={_EXP48_TAU})  probs range [{probs.min():.5f}, {probs.max():.5f}]")

    # --- Apply cluster rewrite ---
    for _c_str, _trig in _cluster_map.items():
        _c = int(_c_str)
        _trig = _np.asarray(_trig, dtype=_np.int64)
        if _trig.max() >= n_cls or _trig.min() < 0: continue
        _score = probs[:, _trig].min(axis=1)   # all-triggers-firing
        probs[:, _c] = probs[:, _c] * (1.0 + _EXP48_ALPHA * _score)
    # Re-clip
    probs = _np.clip(probs, 0.0, 1.0)
    print(f"exp48: cluster rewrite applied  (α={_EXP48_ALPHA}, {len(_cluster_map)} targets)")
    print(f"  final probs range: [{probs.min():.5f}, {probs.max():.5f}]")
