# CLAUDE.md

Guidance for Claude Code in this repository.

## CURRENT STATE (2026-05-03)

**Production LB: v33 = 0.932** = `0.7 × (Perch+ProtoSSM_v4 ensemble) + 0.3 × exp50_SED + V9 taxon-gate + Gauss σ=0.5 + file-max α=0.10`.

Notebook cells 41 (ProtoSSM v4 train) / 43 (ResidualSSM second-pass) /
48 (score fusion) are part of the v33 pipeline — **ProtoSSM IS in v33**,
contrary to earlier (incorrect) memory entries. The "0.7 × Perch_ONNX"
shorthand referred to the Perch+ProtoSSM ensemble, not raw Perch alone.

**Public 0.943 lever (2026-05-03)**: `mattiaangeli/birdclef-2026-0-943-better-blend`
hits LB 0.943, +0.011 over our 0.932. Same ProtoSSM architecture; gap is:
1. Tucker `bc2026-distilled-sed-public` 5-fold SED ensemble (mel-256,
   fmin-20, fmax-16000, per-spec z-score) replacing our exp50 single.
2. Rank-percentile blend in place of our linear `0.7 P + 0.3 SED`.
3. 3 conditional rescue rules (fake_only, proto_continuity with t-dist
   fat-tail kernel ±3 windows, sed_local_spike).
See memory `project_0943_gap_analysis.md` for component-by-component
breakdown and predicted +Δ per step.

Pre-0.943-discovery lever audits remain valid: 17 distinct post-v33
modifications regressed −0.002 to −0.018. The "lever exhaustion"
language (DELETED 2026-05-03) was wrong — orthogonal levers exist when
we look outside our own attempts (public ProtoSSM stack works).

### LB ladder (post-v42)
| v | Modification | LB Δ vs v33 |
|---|---|---|
| v43 | + 0.10 × P_NEW3 (Perch-init hybrid head) | −0.009 |
| v44 | + 0.10 × exp121 (cross-region BG fine-tune) | −0.012 |
| v45 | + 0.05 × exp123 (surgical bird-bias penalty) | −0.014 |
| v46 | + 0.10 × exp136b (v3 pseudo retrain, eff ≈ 0.19 due to dup) | **−0.018** |
| **v47** | + 0.10 × exp136b non-Aves only (per-class freeze) | −0.016 |
| **v48** | + 0.05 × exp84b (ext iNat positive supervision, uniform) | −0.014 |

**Recent +0.002/step harm-reduction trajectory** (v46 → v47 → v48):
- v46 (exp136b uniform W≈0.19 due to dup): 0.914
- v47 (exp136b W=0.10 non-Aves only): 0.916 — dedup + Aves freeze
- v48 (exp84b W=0.05 uniform): 0.918 — teacher swap + dose halving

Each modification reduces harm by 0.002. This is a **gradient signal**, not a wall. Decomposes as: dedup contributes ≥+0.002, teacher quality (exp84b val_SS 0.861 < exp136b 0.907 = less site-fitting) contributes some fraction of the remaining ~+0.002. The trend implies a productive mechanism in the regression band: **smaller dose + cleaner teacher + Aves preservation each independently move LB up**. Promising follow-up directions include exp84b non-Aves freeze, dose ablation at W=0.025 / 0.01, and 4-way blend at W_PERCH≥0.7 with each non-Perch teacher at small dose.

### Three universal failure modes
Every regression matches one of:
1. **Duplicates existing site-conflated SED signal** (v17 SED41f, v36 4 SEDs all corr 0.97-0.99 with exp50)
2. **Train-SS-fitted structure** (v18/v19/v22/v28/v29/v34/v36/v46 — anything trained or calibrated on the 5-site labeled SS encodes site fingerprint)
3. **Reduces existing site-invariance** (v25/v36 dropped W_PERCH below 0.7)

### Key invariants (do NOT violate)
- **W_PERCH ≥ 0.7** in any blend
- **Higher val_SS ⇔ worse LB** (overwhelming evidence: exp50 0.838→+0.001, exp121 0.851→−0.012, exp123 0.867→−0.014, exp136b 0.907→−0.018)
- **Same-site eval (val_SS, fair-11-file Val-A) is reliably ANTI-correlated with LB.** Never use as primary acceptance signal.
- **LOSO-site CV at 0.97 is also invalidated** (exp99-100 → v38-v40 all regressed −0.012 to −0.017)
- **All SED variants Pearson 0.97-0.99 with each other** (recipe / arch / external-data tweaks share most signal). Diversity gain is at residual level.

### How v33 0.932 was built (3 independent +0.001 site-invariance steps)
| Step | Change | Mechanism added | LB |
|---|---|---|---|
| v12 | 0.8 P + 0.2 SED29 | Perch xeno-canto multi-region only | 0.929 |
| v24 | SED29 → exp50 | exp50 trained with 2025 Colombia BG mixing | 0.930 |
| v26 | weight 0.2 → 0.3 | optimal mixing of two site-invariance sources | 0.931 |
| v33 | + file-max coherence α=0.10 | universal physics decoupled from site | 0.932 |

## Pseudo-label work (current state)

Goal: extract supervision from unlabeled SS to strengthen SED beyond labeled-55-file ceiling. Iterative refinement framework `exp126 → exp130b → exp134 → exp136b → exp139`:

| ver | definition | rows | result |
|---|---|---|---|
| v0 | v33 > 0.5 + ensemble agree | ~35k | Aves-saturated, no rare taxa |
| v2 | + TA acoustic centroid filter | 277k | Insecta = 0 |
| **v3** | + confusion-signature inverse mapping for 31 unmapped species | 351k (Insecta 67k, Mam 1.3k, Rept 210) | exp136b → val_SS **0.907** but **LB 0.914 (−0.018)** |
| v7 | v3 + expanded external (2,067 clips → 1,588 embedded) Perch-centroid filter cos<0.3 drop | 241k (31% drop) | NOT trained — exp142 GPU blocked |

**Why v3/v7 is structurally broken**:
- Pseudo source IS v33 → circular distillation
- Confusion-mapping encodes 5-site labeled SS acoustic fingerprint
- exp136b learned site-fingerprint aggressively (val_SS +0.069 over exp50) → catastrophic OOD
- Aves loss × 162 cls > Insecta gain × 28 cls = net macro loss

**v7 is sitting in `data/birdclef-2026/pseudo_soundscapes_labels_v7.csv` but not consumed.** Same circular-distillation issue applies; no LB submit recommended without a fundamentally new pseudo source.

## Why no further loss/recipe variation will help (info-theoretic ceiling)

We tested ~10 RL/DPO/SFT variants (exp112-119, exp122). All ≤ BCE on LOSO. End-to-end audio DPO (exp122) was *worse* than BCE on val_SS. Reason: DPO/RL's advantage requires reward richer than supervised labels. Our reward IS the binary species labels. BCE is essentially Bayes-optimal here. Hard mining ≈ aggressive BCE; multi-label "rejected = random negative" introduces false-negative noise.

**Implication: the bottleneck is data + fusion, not the loss optimization.** Only paths that add genuinely new information can break v33:
- **Public 0.943 stack** (highest-confidence lever 2026-05-03): Tucker
  5-fold SED + rank-percentile blend + 3 rescue rules. Expected
  closure +0.011 over 4 LB submissions (v55 → v57+).
- Multi-region external (10× Mammalia/Reptilia from non-Pantanal sites)
- A different foundation model (BirdNET / AudioMAE / BEATs / Perch v3)
- DANN site-adversarial training with unlabeled SS (explicit invariance constraint)
- Earlier-year SS labels (2023, 2024 if obtainable)
- **Synthetic data via domain randomization** (DRASDIC + Soltero 2025): expand BG pool from single-site Pantanal to multi-region (xeno-canto silent windows) — exp154 confirmed +38% PSD diversity, exp159 retrain showed 9k multi-region pool but final ckpt converged to Pearson 0.989 with exp50 (lost orthogonality at convergence). Drop in favor of higher-confidence public stack.

## LB submission discipline (MUST READ)

Kaggle is **hard rate-limited (5 submissions/day)**. Each comp re-run ~1 hour wall clock. Every submission is a precious final-validation slot, never a search tool.

**Strict rule**: local positive → eligible for LB. local negative → rejected. local unmeasured → measure first.

**The val_SS / Val-A trap**: same-site eval is anti-correlated with LB on this pipeline. Don't accept "val_SS up" as a green light. Falsifiable bar for novel candidates: must add a *new source of site-invariance* not duplicated by Perch-multi-region, exp50-2025-BG, or file-max physics.

**Anti-patterns to reject**:
- "Top scorer X uses Y, let's add Y" — verify on our local first
- "Maybe LB differs from local" without a specific testable mechanism
- Mixing locally-verified vs guessed candidates in one bundle

**Already locally-or-LB-verified negative — do NOT re-try**:
- Kalman smoothing on logits/preds (exp30, exp54c)
- Per-class rank/z-score/temperature normalization (AUC-invariant — math no-op)
- Platt/Isotonic calibration (exp34, exp29 OOF — sparse positives)
- MLP/temporal-stack probes (exp33)
- ConvNeXtV2-tiny / B2+ backbone / EffNet-NS / ConvNeXt-small SED (exp34b/37b/40)
- Heavy regularization SED retrain (exp36)
- HGNet BG-mix-only SED (exp31)
- Per-file centering / weak priors (exp26)
- Linear site/file-mean centering of Perch (exp42 — removes species signal too)
- r2 same-teacher pseudo iteration (exp41h)
- iVAE/iVDFM on Perch features for pseudo refinement (exp43e/k/l/n/o)
- mel-iVAE z-kNN train-SS centroid (exp78 → v34 LB −0.016)
- 27-class dedicated head additive (v18/v19/v28 all regressed)
- Post-hoc multiplicative lookups derived from train SS (v22 site prior + cluster — LB −0.016)
- Per-class LR-fit / threshold rule (v38-v41 all regressed −0.012 to −0.017)
- file-max α scaled beyond 0.10 (v42 α=0.20 → −0.014; sharp peak at α=0.10)
- W_PERCH < 0.7 (v25, v36)
- DPO / focal / soft-AUC / contrastive losses (exp83/112-119/122)
- Bottom-up audit and per-class lever (exp95-100) — LOSO 0.97 doesn't transfer

## Local compute discipline

**GPU-first on RTX 5090 32GB.** Kaggle's 90-min CPU limit applies ONLY to the final notebook.

**Perch v2 extraction must use ONNX + CUDAExecutionProvider:**
- `perch_v2/` (TF SavedModel) is XLA-compiled CPU-only and refuses to run on GPU
- ONNX at `/tmp/perch_v2.onnx` (Kaggle dataset `rishikeshjani/perch-onnx-for-birdclef-2026`) → 10× speedup
- Use `onnxruntime-gpu`. Never copy `CUDA_VISIBLE_DEVICES=""` from exp21 (was correct only for the SavedModel path)
- ONNX vs TF SavedModel embeddings differ slightly (float precision). Submission notebook uses ONNX → keep local extraction ONNX too. TF-cached `exp21_outputs/perch_cache/` may drift; revalidate if reused.

**Resolved 2026-05-02**: GPU driver mismatch fixed (NVML + kernel both 580.142). 32 GB free.

**Long-running jobs MUST survive shell/tmux death.** `Bash run_in_background: true` only protects within the active Claude session — when tmux dies, child python processes get SIGHUP and die mid-run. Use the robust pattern instead:
```bash
LOG=/data/birdclef2026/experiments/_scratch_logs/<name>_$(date +%Y%m%d_%H%M%S).log
nohup setsid uv run python -u <script> > "$LOG" 2>&1 < /dev/null &
disown
```
`setsid` creates a new session (detaches from tmux), `nohup` ignores SIGHUP, `< /dev/null` detaches stdin, `disown` removes from shell job control. Survives tmux kill, ssh disconnect, Claude session restart. Recover by checking output files / log tail rather than relying on in-memory PID. (Incident 2026-05-03: tmux died mid-exp155, lost 35-min of progress; recovered with pattern above.)

## Project layout

- ~~`src/`~~ — archived to `experiments/_archive_2025/src/` (2025 mel-spec pipeline, unused by current Perch-distill flow)
- `experiments/` — numbered experiment scripts. **`experiments/README.md` is the canonical index** — read before adding new exps
- `experiments/_archive_2025/`, `_archive_2026_*` — historical layers. Useful as negative-result reference (e.g., `_archive_2026_dpo_dead_end/` records ~10 RL/DPO variants that confirmed BCE is Bayes-optimal here).
- `experiments/_data_pipelines/` — pseudo build, external download, refinement
- `experiments/_audits_post_v26/` — post-v33 audits and ablations
- `notebooks/` — Kaggle submission notebooks. Edit in place; **never create new dirs per variant**
- `paper/` — CLEF 2026 working note (LaTeX). `experiments.tex` stale past exp39
- `data/birdclef-2026/` — current comp data
- `data/birdclef-2025/` — prior year (Pantanal, reusable as BG)
- `model-weights/` — local-only, gitignored. Kaggle dataset slug: `ultimatumgame/birdclef2026-model-weights`
- `OVERVIEW.md`, `WINNING_SOLUTION_2025.md` — comp + prior-year context

### Data layout
```
data/birdclef-2026/
  train.csv                       # primary_label, secondary_labels, rating, filename, collection
  taxonomy.csv                    # 234 species; class_name ∈ {Aves, Amphibia, Mammalia, Insecta, Reptilia}
  train_audio/                    # short clips from xeno-canto / iNat
  train_soundscapes/              # 1-min field recordings (10,658 files)
  train_soundscapes_labels.csv    # semicolon-separated multi-label per 5-sec window. **Has 739 dup rows — drop_duplicates()**
  test_soundscapes/               # populated only at submission re-run
  sample_submission.csv           # row_id={filename}_{end_time}
  pseudo_soundscapes_labels_v3.csv, v7.csv  # local pseudo-label artifacts
```

Audio: 32 kHz ogg, 5-sec windows, predictions per window. Both 2025 and 2026 cover Brazilian Pantanal (combinable as BG).

### Active notebooks
- **`birdclef-2026-perch-distill`** — production. v33 = LB 0.932. TF-cache(train) + ONNX(test) + exp50 blend + V9 gate + Gauss + file-max
- `birdclef-2026-test-submission` — Perch v2 sandbox
- `birdclef-2026-exp20-submission` — frozen LB 0.910 baseline
- `birdclef-2026-exp20-to-exp30` — R1 recipe kernel

## Kaggle submission flow (code competition)

Kaggle re-runs the notebook against hidden test for scoring. Dev run output is NOT what gets scored.

```bash
uv run kaggle datasets version -p model-weights/ -m "msg" -r zip
uv run kaggle kernels push -p notebooks/<dir>
uv run kaggle kernels status ultimatumgame/<slug>
uv run kaggle competitions submit -c birdclef-2026 -f submission.csv \
    -k ultimatumgame/<slug> -v <VERSION> -m "msg"
uv run kaggle competitions submissions -c birdclef-2026 --csv | head -5
```

**Auth**: `~/.kaggle/kaggle.json` (legacy) OR `~/.kaggle/access_token` (new bearer). If 401, ask user to refresh "Create New API Token" at kaggle.com/account.

**Pre-submit checklist (mandatory)**:
1. `grep TTA_SHIFTS notebook.py` → must be `[0]` or removed (TTA = 3× Perch = guaranteed timeout)
2. Dev log shows `Cache loaded from` (NOT `Cache saved`) — cache mount path `/kaggle/input/datasets/jaejohn/perch-meta`
3. Dev wall ≤ **180s** for 20-file dry-run (comp scaling factor ~30×; budget 90 min)
4. No secondary CNN/distillation in same kernel as Perch
5. `submission.csv` shape = (N × 12, 235) = 234 classes + row_id

Comp re-run logs only on Kaggle web UI. If `publicScore` empty → check web UI.

**Notebook discipline**: edit existing notebook in place; Kaggle versions automatically. Never create new slugs per variant. Per-version patches live in `notebooks/birdclef-2026-perch-distill/patches/` (one .py per LB submission, kept as record).

## Commands

```bash
uv add <pkg>
uv run <script.py>
uv run kaggle ...
bash scripts/<exp>.sh
```

## Eval regimes

- **Val-A** (seen-site, file-stratified) — matches actual test regime. Primary local metric historically, but now known to anti-correlate with LB on margin moves.
- **Val-B** (GroupKFold-by-site) — hedge for unseen-site shakeup. Required sanity check.
- **val_SS** — labeled-55/11-split eval. Strongly anti-correlates with LB at margin (see invariants above).
- Treat all local metrics as **necessary but not sufficient**. New mechanism > local Δ.

## Working-note paper

CLEF 2026 working-note deadline: **2026-06-17**. Comp deadline: **2026-06-03**.

Live thesis (post negative-result pivot): "Hidden-distribution covariate shift defeats labeled-site fitting in cross-region bioacoustic monitoring." Built on:
- v33 0.932 robust ceiling decomposition (3 site-invariance sources)
- val_SS↔LB inverse correlation (4 data points)
- LOSO-site CV invalidation (5 LB submissions)
- Pseudo-label circular-distillation negative (v46)
- ~10 RL/DPO variants ≤ BCE (info-theoretic interpretation)
- 6-test refutation of iVAE/iVDFM on Perch features (publishable negative)

`paper/experiments.tex` (formal draft) stops at v12; needs author port of v17–v50 + pseudo-circularity + AudioMAE + info-theoretic ceiling. Header note added 2026-05-02 marking gap and pointing to `paper/exp_current.tex` (running notes, last touched 2026-04-27, covers through v36 era).

## Top-scorer reference (digested 2026-04-20)

- hengck23: `lb 0.92+ = perch2 + proxy class + temporal + PCA`; we already have all four
- Boredom (1st 2025): 0.947+ single EffNet, train_audio + labeled SS only, 13 min
- Salman (18th): 0.922 single SED B0, 20s + raw mixup + CE(clip+frame max)
- Pseudo on unlabeled SS is risky (Don Mathis failed; top 5 skipped) — confirmed by our v46

## Kaggle config

- Username: `ultimatumgame`
- Model weights dataset: `ultimatumgame/birdclef2026-model-weights`
