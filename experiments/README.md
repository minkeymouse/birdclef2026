# Experiments — Index (refactored 2026-04-26)

```
experiments/
├── _data_pipelines/      production scripts + caches the LB pipeline depends on
├── _audits_post_v26/     live diagnostic experiments + shared `_lib/`
├── _eval_harness/        11-file Val-A_v2 evaluation
├── _archive_2026_audits/ completed audits (paper-recorded, kept for reference)
├── _archive_2026_pre_v26/  pre-v26 work
├── _archive_2025/        2025 CNN-era code
├── _deprecated/          early collapsed attempts
└── _scratch_logs/        run logs from scripts/run_exp.sh
```

## Running experiments

`scripts/run_exp.sh <python_script>` is the standard entry point. It:
- forces `python -u` (unbuffered)
- tees to `experiments/_scratch_logs/<name>_<ts>.log`
- writes a `.pid` file next to the log so a Monitor can poll cleanly

```bash
scripts/run_exp.sh experiments/_audits_post_v26/exp80a_site_holdout.py
# log path is printed first; tail it.
```

For background + Monitor:
```bash
scripts/run_exp.sh experiments/.../exp80a.py &
# Monitor:
#   until [ ! -f <log>.pid ]; do sleep 30; done; tail -50 <log>
```

This avoids the bash-pipe-buffer pitfall where
`uv run python ... 2>&1 | tail -50` hides progress until process exit.

## `_data_pipelines/` — production

These scripts build artifacts the Kaggle inference notebook (`notebooks/birdclef-2026-perch-distill/`) loads. Caches are large and not in git.

| Script | Purpose | Output |
|---|---|---|
| `exp43a_extract_perch_gpu.py` | Perch v2 ONNX-GPU emb + score on 10,658 unlabeled SS | `exp43a_outputs/perch_ss_all.npz` (825 MB) |
| `exp45a_hierarchical_head.py` | V9 5-way taxon gate on Perch emb (used in v20-v33) | `model-weights/exp45a_taxon_head.pt` |
| `exp49a_extract_2025_quiet_bg.py` | 11,920 quiet 5-sec windows from 2025 SS for site-invariance BG mix | `exp49_outputs/` (6.8 GB) |
| `exp49b_extract_2025_train_audio_perch.py` | Perch on 2025 train_audio | `exp49_outputs/` |
| `exp50_exp47_with_2025bg.py` | **Active SED in v24-v33 (LB 0.931+)** — Boredom recipe + 2025 BG | `model-weights/exp50_hgnet_sed.pt` |
| `exp51_27head_with_2025bg.py` | 27-class dedicated head (sonotypes + 2 Amphibia) + 2025 BG | `model-weights/exp51_27head_sed.pt` |
| `exp76_ivae_raw_mel.py` | Build pooled mel cache for 739 labeled SS rows | `exp76_outputs/mel_cache.npz` |
| `exp78_save_ivae_artifacts.py` | Train iVAE encoder + save (used in v34, regressed) | `model-weights/ivae_*.{pt,npz}` |
| `exp71_external_download.py` | xeno-canto / iNat clip download (partial) | `exp73_outputs/` |
| `exp73_finetune_with_external.py` | SED fine-tune with external clips (WIP) | |
| `exp49_50_51_driver.sh` | Convenience driver to retrain exp50/51 | |

## `_audits_post_v26/` — live audits

After v26 (LB 0.931). Currently around iVAE/Perch disagreement signal investigation (definitively negative — see `paper/exp_current.tex` exp79–80 section).

```
_audits_post_v26/
├── _lib/                       shared utility module
├── exp80a_site_holdout.py      site holdout — confirms iVAE Insecta is site fingerprint
├── exp80b_bigpool_ivae.py      30k-pool iVAE — partial improvements only
├── exp80c_taxon_classifier.py  LOSO site CV — same conclusion
├── exp79_outputs/              unlabeled probe + audit candidates
└── exp80_outputs/              bigpool ckpt, perch_emb_labeled cache, site holdout result
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
| `mel` | `make_mel_pool_gpu()`, `extract_pool_one_file()`, `extract_pool_many()` — torchaudio mel pipeline (T_POOL=16, N_MELS=128, matches exp78) |
| `eval_metrics` | `per_class_auc`, `macro_auc`, `per_taxon_macro`, `per_row_spearman`, `gpu_mlp_binary_auc` |

**Don't duplicate `build_ss` / `IVAE` / mel extraction in new scripts.** Add helpers to `_lib/` instead.

## `_eval_harness/`

| Script | Purpose |
|---|---|
| `eval_soundscapes.py` | 11-file Val-A_v2 evaluation harness |

## `_archive_2026_audits/`

42 completed audit scripts + outputs. Findings recorded in `paper/exp_current.tex`. Kept for reproducibility.

## Other archives

`_archive_2026_pre_v26/`, `_archive_2025/`, `_deprecated/` — historical layers; nothing in production depends on them.

---

## LB submission timeline

| v | Config | LB | Δ vs v12 |
|---|---|---|---|
| v12 | 0.8P + 0.2 SED29 + V9 + Gauss              | 0.929 | 0 |
| v17 | 0.8P + 0.2 SED41f                          | 0.922 | −0.007 |
| v18 | v17 + exp44c overlay                       | 0.916 | −0.013 |
| v19 | v17 + exp44g synth                         | 0.907 | −0.022 |
| v20 | v17 + V9 gate                              | 0.929 | 0 |
| v21 | v12 + V9 gate                              | 0.929 | 0 |
| v22 | v12 + V9 + exp48 (site prior + cluster)    | 0.913 | −0.016 |
| v23 | 0.8P + 0.1 S29 + 0.1 exp50                 | 0.928 | −0.001 |
| **v24** | **0.8P + 0.2 exp50**                   | **0.930** | **+0.001** |
| v25 | 0.5P + 0.5 exp50                           | 0.928 | −0.001 |
| **v26** | **0.7P + 0.3 exp50**                   | **0.931** | **+0.002** |
| v28 | v26 + exp51 27-head additive               | 0.929 | −0.002 vs v26 |
| v29 | v26 + per-class OOF routing                | 0.927 | −0.004 vs v26 |
| v32 | v26 + R5 aliveness routing (4 cls)         | 0.930 | −0.001 vs v26 |
| **v33** | **v26 + L2c file-max coherence (α=0.10)** (production) | **0.932** | **+0.003** |
| v34 | v33 + mel-iVAE z-kNN (exp78, w_z=0.05)     | 0.916 | −0.013 |

## Mechanism findings worth re-reading before proposing levers

- **v18/v19/v22/v28/v34 all regressed**: any post-hoc lever fitted to the 55-file labeled SS pool inherits site shortcut.
- **Only successful train-SS-touching mechanism**: exp50 — 2025-BG mixing during training enforces site-invariance, not post-hoc disentanglement.
- **exp79–80 teardown**: site fingerprint dominates iVAE z-space. S19 holdout Insecta AUC = 0.073. LOSO Insecta AUC ≤ 0.30 for every feature set. Data limitation (4 Insecta sites only), not architecture limitation.
- **Path forward**: bring in external positives (xeno-canto, iNat) and retrain SED with multi-region BG mixing. Lever-pulling on 55 files alone is exhausted.
