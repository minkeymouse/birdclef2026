# Experiments — Index (refreshed 2026-05-02)

```
experiments/
├── _data_pipelines/          production scripts + caches the LB pipeline depends on
│   └── _shared/              SED training library (audio, augment, model, train)
├── _audits_post_v26/         live diagnostic experiments + shared `_lib/`
├── _eval_harness/            11-file Val-A_v2 evaluation
├── _archive_2026_audits/     completed audits (paper-recorded, kept for reference)
│   └── post_v26/             exp79–101 audits archived after v33 (April 2026)
├── _archive_2026_dpo_dead_end/        exp104–117 + exp122 RL/DPO variants, all ≤ BCE on LOSO
├── _archive_2026_pseudo_dead_end/     exp125–140 + 147/147b: pseudo v0→v8 iteration chain (circular distillation)
├── _archive_2026_audiomae_dead_end/   exp143–146 AudioMAE + DANN + blend probe (v49/v50 regressed)
├── _archive_2026_pre_v26/    pre-v26 work
├── _archive_2025/            2025 CNN-era code
├── _deprecated/              early collapsed attempts
└── _scratch_logs/            run logs from scripts/run_exp.sh
```

## Running experiments

`scripts/run_exp.sh <python_script>` is the standard entry point:
- forces `python -u` (unbuffered)
- tees to `experiments/_scratch_logs/<name>_<ts>.log`
- writes a `.pid` file next to the log so a Monitor can poll cleanly

```bash
scripts/run_exp.sh experiments/_data_pipelines/exp50_exp47_with_2025bg.py
```

Avoids the bash-pipe-buffer pitfall where `uv run python ... 2>&1 | tail -50` hides progress until process exit.

## `_data_pipelines/`

Scripts that build artifacts loaded by the Kaggle inference notebook (`notebooks/birdclef-2026-perch-distill/`). Caches large, gitignored.

### Active in v33 production (LB 0.932)

| Script | Purpose | Output |
|---|---|---|
| `exp43a_extract_perch_gpu.py` | Perch v2 ONNX-GPU emb + scores on 10,658 unlabeled SS | `exp43a_outputs/perch_ss_all.npz` |
| `exp45a_hierarchical_head.py` | V9 5-way taxon gate on Perch emb | `model-weights/exp45a_taxon_head.pt` |
| `exp49a_extract_2025_quiet_bg.py` | 11,920 quiet 5-sec windows from 2025 SS for site-invariance BG mix | `exp49_outputs/bg_quiet_2025.npz` (6.6 GB) |
| `exp49b_extract_2025_train_audio_perch.py` | Perch on 2025 train_audio (41 overlap species) | `exp49_outputs/train_audio_2025_perch.npz` |
| `exp50_exp47_with_2025bg.py` | **Active SED in v24–v33** — Boredom recipe + 2025 BG mixing | `model-weights/exp50_hgnet_sed.pt` |

### LB-tested teachers (kept for blend audits)

| Script | LB role | Output |
|---|---|---|
| `exp51_27head_with_2025bg.py` | 27-class sonotype head — v18/v19/v28 (regressed) | `model-weights/exp51_27head_sed.pt` |
| `exp76_ivae_raw_mel.py` + `exp78_save_ivae_artifacts.py` | mel-iVAE for v34 (regressed) | `model-weights/ivae_*.pt` |
| `exp83_q2_focal_sed.py` | Q2 focal SED variant | `exp83_q2_outputs/` |
| `exp84_q4_external_expand.py` + `exp84b_q4_retrain.py` | iNat external positive supervision — used at W=0.05 in **v48 (+0.002)** | `exp84b_q4_outputs/` |
| `exp102_save_lr_artifacts.py` | LR FP/FN detectors — v38-v41 (all regressed −0.012 to −0.017) | `model-weights/lr_*_detector.npz` |
| `exp120_extract_2025_ta.py` | 2025 train_audio Perch cache | `exp120_outputs/ta25_perch.npz` |
| `exp121_aggressive_synth.py` | Cross-region BG fine-tune — v44 (regressed −0.012) | `model-weights/exp121_aggressive_synth_sed.pt` |
| `exp123_bird_bias_fix.py` | Surgical bird-bias penalty — v45 (regressed −0.014) | `model-weights/exp123_bird_bias_sed.pt` |

### Pseudo-label ingredients (chain in `_archive_2026_pseudo_dead_end/`)

The full v0→v8 iteration chain (exp125–140 + 147/147b) was archived after
circular-distillation diagnosis (CLAUDE.md "Pseudo-label work"). Only the
LB-tested or deferred-but-referenced ingredients remain here:

| Script | Status |
|---|---|
| `exp136_v3_retrain.py` + `exp136b_v3_simple.py` | v3 pseudo retrain — **v46/v47 ingredient** (LB 0.914/0.916) |
| `exp142_v7_simple.py` | v7 retrain (deferred per CLAUDE.md "v7 not consumed") |
| `exp149_pseudo_v9_targeted.py` + `exp150_v9_targeted_sed.py` | v9 targeted (deferred, didn't run) |

### Blend / stacking audits

| Script | Purpose |
|---|---|
| `exp148_blend_normalisation.py` | per-class z-score / rank blend (hengck23-suggested) |
| `exp151_save_rank_quantiles.py` | rank-quantile artifact for Kaggle |
| `exp152_save_m5_probe.py` | M5 MLP probe save (AudioMAE) |
| `exp153_stacking.py` | LightGBM per-class stacking on Perch + 4 SEDs + AudioMAE M5 |

### Synthetic-data exploration (NEW, 2026-05-02)

DRASDIC (Hoffman 2025) + Soltero (2025) inspired. Goal: site decoupling via multi-region BG diversification.

| Script | Status | Note |
|---|---|---|
| `exp154_synth_bg_diagnostic.py` | ✓ ran | 360 train_audio → 560 windows, 12 buckets, **PSD diversity +38% vs Pantanal pool**. Feasibility confirmed. |
| `exp155_synth_bg_mine_full.py` | deferred | Full extraction (~15 min CPU, ~24 GB). Awaiting go-ahead. Extrapolated ~43k mineable BG windows (3.6× current). |
| `exp156_synth_bg_perch_qc.py` | deferred | Perch top-1 prob < 0.05 filter. Needs Perch ONNX (`/tmp/perch_v2.onnx`). |

### `_shared/` (used by exp50 / exp51 / exp84b / exp121 / exp136b / etc.)

SED training library:

| Module | Exports |
|---|---|
| `audio.py` | mel-spec frontend (32 kHz, n_mels=128, fmin=50, fmax=14000) |
| `augment.py` | `aggressive_mixup` — per-row BG mix + mixup. Aves p=0.5, non-Aves p=0.85, alpha [0.3, 0.7] |
| `constants.py` | SR=32000, CLIP_SEC=20, WINDOW_SEC=5, N_CLS=234, BACKBONE="hgnetv2_b0..." |
| `data.py` / `data_v3.py` | dataset construction + label encoding |
| `model.py` | HGNet-B0 SED head |
| `train.py` | training loop |

## `_audits_post_v26/`

Live diagnostic experiments after v33 production. Most pre-v33 audits archived to `_archive_2026_audits/post_v26/`.

```
_audits_post_v26/
├── _lib/                              shared utility module
├── exp103_perch_head_finetune.py      P_NEW: Perch backbone frozen + 234-class head fine-tune
├── exp106_pnew_hybrid.py              Perch-init linear + correction MLP
├── exp108_failure_forensics.py        FN/FP analysis on 122 SS eval rows
├── exp121b_score_eval.py              score exp121 + v44 blend audit
├── exp123b_score_eval.py              score exp123 + v45 blend audit
├── exp124_freq_band_gate.py           frequency-band physics gate (no learning)
├── exp136b_score_eval.py              score exp136b + v46 blend audit
└── _outputs/                          per-script caches
```

### `_lib/` — shared utilities

Imported by every audit script via:
```python
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import build_ss, species_taxon_array, ...
```

| Module | Exports |
|---|---|
| `data` | `build_ss()` (cached), `species_taxon_array()`, `aux_matrix()`, `load_labeled_mel()`, `load_perch_emb_labeled()` (cached), `load_perch_scores_labeled()` (cached), constants (paths, SEED, N_CLS, TAXA, FNAME_RE) |
| `ivae` | `IVAE` model class, `train_full()`, `encode_all()`, `load_exp78_encoder()`, `kl_div()` |
| `mel` | `make_mel_pool_gpu()`, `extract_pool_one_file()`, `extract_pool_many()` |
| `eval_metrics` | `per_class_auc`, `macro_auc`, `per_taxon_macro`, `per_row_spearman`, `gpu_mlp_binary_auc` |

**Don't duplicate `build_ss` / `IVAE` / mel extraction in new scripts.** Add helpers to `_lib/` instead.

## `_eval_harness/`

| Script | Purpose |
|---|---|
| `eval_soundscapes.py` | 11-file Val-A_v2 evaluation harness |

**Reminder**: Val-A and val_SS are *anti-correlated with LB at margin moves*. See CLAUDE.md "Key invariants".

## Archives

| Folder | Contents |
|---|---|
| `_archive_2026_audits/` | Completed audits before v33 |
| `_archive_2026_audits/post_v26/` | exp79–101 audits archived April 2026 (iVAE/site-fingerprint teardown, LR-correction lever, exhaustive signal hunt) |
| `_archive_2026_dpo_dead_end/` | exp104–117 + exp122: ~10 RL/DPO/SFT variants (all ≤ BCE on LOSO — info-theoretic ceiling) |
| `_archive_2026_pseudo_dead_end/` | exp125–140 + exp147/147b: v0→v8 pseudo-label iteration chain (circular distillation, see CLAUDE.md). v3/v7/v9 artifact-producing scripts retained in `_data_pipelines/`. |
| `_archive_2026_audiomae_dead_end/` | exp143–146 + exp144/145: AudioMAE foundation-swap probe + DANN + blend audits (v49 LB 0.910, v50 LB 0.905). M5 probe saver kept active as `exp152_save_m5_probe.py`. |
| `_archive_2026_pre_v26/` | pre-v26 layer |
| `_archive_2025/` | 2025 CNN-era code |
| `_deprecated/` | Early collapsed attempts |

---

## LB submission timeline (cumulative)

For full mechanism analysis see CLAUDE.md "CURRENT STATE" section.

| v | Config | LB | Δ vs v33 |
|---|---|---|---|
| v12 | 0.8P + 0.2 SED29 + V9 + Gauss baseline | 0.929 | −0.003 |
| v17 | 0.8P + 0.2 SED41f | 0.922 | −0.010 |
| v22 | v12 + V9 + exp48 (site prior + cluster) | 0.913 | −0.019 |
| **v24** | 0.8P + 0.2 exp50 (2025 BG) | **0.930** | −0.002 |
| **v26** | 0.7P + 0.3 exp50 | **0.931** | −0.001 |
| **v33** | v26 + L2c file-max coherence α=0.10 | **0.932** | reference |
| v34 | v33 + mel-iVAE z-kNN (exp78) | 0.916 | −0.016 |
| v36 | 5-way 0.5P + 4 SEDs each 0.125 | 0.915 | −0.017 |
| v38–v41 | per-class LR-correction (LOSO 0.97) | 0.915–0.920 | −0.012 to −0.017 |
| v43 | + 0.10 × P_NEW3 (Perch-init hybrid) | 0.923 | −0.009 |
| v44 | + 0.10 × exp121 cross-region BG | 0.920 | −0.012 |
| v45 | + 0.05 × exp123 bird-bias penalty | 0.918 | −0.014 |
| v46 | + 0.10 × exp136b v3 pseudo retrain | 0.914 | −0.018 |
| v47 | + exp136b W=0.10 non-Aves freeze | 0.916 | −0.016 |
| v48 | + 0.05 × exp84b iNat ext (uniform) | 0.918 | −0.014 |
| v49 | AudioMAE blend foundation-swap | 0.910 | −0.022 |
| v50 | M5 MLP non-Aves freeze | 0.905 | −0.027 |
| v55 | + Tucker 5-fold SED linear W=0.10 | ≤0.932 | 0 (undersized dose) |
| v56 | + Tucker rank-blend W=0.30 (no rescues) | 0.931 / 0.932 | 0 / −0.001 |
| **v57** | **Mattia fork as-is (full stack with rescues)** | **0.941** | **+0.009** |
| v58 | v33 + Tucker linear W=0.30 | ≤0.941 | ≤+0.009 |
| v59 | v33 + Tucker linear W=0.40 | ≤0.941 | ≤+0.009 |

**Production reset 2026-05-03**: v57 = LB 0.941 (kernel `mattia-fork`).
v33 (kernel `perch-distill`) is now the legacy reference. Tucker SED swap
is the +0.009 lever, not the rank+rescue architecture (verified by v58/v59
linear blend reaching same ceiling). 14× macro_d gap between exp50 and
Tucker (exp165 ablation, same architecture, same dose) — our exp50 mel-128
fmax=14000 was missing the 14-16 kHz Insecta band that Tucker mel-256
fmax=16000 captures.

For invariants, failure modes, anti-patterns, and the locally-or-LB-verified-negative list, **see CLAUDE.md** — that is the single source of truth for strategic state. This README is the structural index only.
