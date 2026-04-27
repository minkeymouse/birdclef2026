# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CURRENT STATE (2026-04-24, end of day)

**LB ceiling: 0.929** after 11 submitted attempts (v12-v22). v20 and v21 also hit 0.929 via taxon gate. **v22 regressed to 0.913** — the exp48 site prior + cluster rewrite did NOT transfer.

**v22 post-mortem (exp52 audit)**: On ALL 66 labeled SS files, v22 showed massive local gains (macro 0.7224 → 0.9098, +0.188), including on common Aves (trsowl 0.58 → 0.91, litnig1 0.50 → 0.88) — across-the-board, all taxa up. Yet LB dropped −0.016. Per-row Spearman(v12, v22) = 0.745 → predictions got 25%+ reshuffled. **Root cause**: site prior and cluster map were derived from train SS distribution (S03-S23). On hidden test distribution, the trigger-Aves→rare-target correlation pattern doesn't hold → cluster rewrite boosts rare species on wrong rows → false positives on LB evaluable rare classes + common class ranking disrupted by cascade.

**Structural lesson**: post-hoc multiplicative levers that depend on train-SS-derived lookup tables systematically fail LB transfer regardless of local validation result. Even when ALL local metrics (macro, Aves-subset, common-class-weighted) show improvement.

**Lever classification updated:**
- Post-hoc multiplicative on train-SS lookup → ✗ (v18, v19, v22 all failed)
- Post-hoc multiplicative neutral (taxon gate) → ≈ (v20, v21 neutral at 0.929)
- Base-level blend modification → ≈ (v14-v17 all −0.003 to −0.007)
- Completely orthogonal to post-hoc multiply: unverified (exp50 Perch-independent teacher blend)

**NEW teachers trained (exp49-51, 2025-04-24):**
- **exp49c** taxon_head_v49 (5-way Aves/Amphibia/Insecta/Mammalia/Reptilia classifier on Perch embs, retrained with 2025+2026 pool). Val macro AUC 0.9842 vs exp45a's 0.96-0.97, Mammalia +0.03.
- **exp50** SED (Boredom-recipe HGNet-B0 + 2025 soundscapes as BG mix source for site invariance). Val_TA 0.9882 (vs exp47 0.9866), val_SS 0.8378 (vs exp47 0.7858, **+0.052 rare-class improvement**). Saved to `exp50_outputs/best_ckpt.pt`.
- **exp51** 27-class head with 2025 BG. Val 0.8696 (between exp44c 0.848 and exp44g 0.884). Modest improvement over pure exp44c but under exp44g which had v19 LB regression.

**exp53/54 adaptive lever exploration (5-scheme ablation):**

| Scheme | macro Δ | sp_row | Aves Δ |
|---|---|---|---|
| 53a conf-weighted blend (β=0.1) | −0.045 | 0.995 | −0.002 |
| 53b ensemble disagreement smoothing | −0.062 | 0.996 | ≈0 |
| 53c per-class quantile normalization | 0 (AUC-invariant, expected) | 0.93 | 0 |
| 53d per-file bias removal | **−0.19** | **0.56** | −0.14 |
| 53e SSM EMA on embeddings | +0.0001 | 1.0 | +0.0003 |
| 54a exp50 teacher swap w=0.2 | +0.199 | 0.952 | +0.179 |
| 54a exp47 teacher swap w=0.2 | +0.189 | 0.951 | +0.179 |
| **54b 3-way P 0.80 s29 0.10 s50 0.10** | **+0.163** | **0.987** | +0.131 |
| 54b 3-way P 0.80 s29 0.15 s50 0.05 | +0.127 | **0.996** | +0.077 |
| 54c Kalman LDS on logits (q=0.3) | +0.002 | 0.97 | −0.003 |
| 54d per-class adaptive gain | ≈0 | 0.98 | ≈0 |
| 54e teacher-agreement filter (th=0.5) | +0.188 | 0.911 | +0.142 |
| 54f MC mask perturbation | ≈0 | 0.999 | ≈0 |

Takeaways:
- **Post-hoc schemes that depend only on test data self-properties (53a-f) are structurally safe (high Spearman) but ineffective.** Adaptive confidence weighting, disagreement smoothing, per-class quantile, SSM on embeddings — all are near-no-op on macro. Simply no rare-class signal lives in test-self-properties that we can extract without external information.
- **Per-file bias removal (53d)** reshuffles predictions dramatically (Spearman 0.56) and tanks macro (−0.19) — priors-OFF-like effect, hard to control.
- **Teacher blend modifications (54a/b)** are the only non-multiplicative intervention with significant local Δ. sp_row 0.98+ configurations exist.

**Promising v23 candidate (untested on LB):** `0.8*Perch + 0.1*SED29 + 0.1*exp50 + Gauss` — local macro +0.163, Spearman 0.987 (minimal ranking disturbance). Tests whether exp50's Perch-independence breaks the past blend-failure pattern (v14/v15/v17 were all −0.003 to −0.007 vs v12).

**LB submission timeline 2026-04-24/25 (v22-v32):**

| v | Config | LB | vs v26 |
|---|---|---|---|
| v22 | site prior + cluster rewrite | 0.913 | (regression vs v12 baseline) |
| v23 | 0.8P + 0.1 SED29 + 0.1 exp50 | 0.928 | |
| v24 | 0.8P + 0.2 exp50 | 0.930 | |
| v25 | 0.5P + 0.5 exp50 | 0.928 | |
| **v26** | **0.7P + 0.3 exp50** | **0.931** | reference |
| v28 | v26 + exp51 27-head additive | 0.929 | −0.002 |
| v29 | v26 + per-class OOF routing | 0.927 | −0.004 |
| v32 | v26 + R5 aliveness routing (4 cls) | 0.930 | −0.001 |
| **v33** | **v26 + L2c file-max coherence (α=0.10)** | **0.932** | **+0.001 vs v26, +0.003 vs v12** |
| v34 | v33 + mel-iVAE z-kNN (exp78, w_z=0.05) | 0.916 | −0.016 vs v33 |
| v36 | 5-way 0.5P + (exp50+exp59+exp73+exp84b) each 0.125 | 0.915 | −0.017 vs v33 |

**v36 post-mortem (2026-04-26)**: 5-way blend with 4 SEDs (HGNet exp50, ConvNeXt exp59, exp50+ext exp73, exp50+545ext exp84b). Local exp85 showed macro Δ +0.013, sp_row 0.999, Aves +0.024, all-taxa positive. Predicted by anti-correlation rule v3 to be class A (LB +0.002-0.004). **LB = 0.915, regressed −0.017.** Anti-correlation rule v3 has now failed TWO consecutive submissions in same magnitude (v34 −0.016, v36 −0.017).

**Deeper mechanism**: v36 reduced W_PERCH from 0.7 (v33) to 0.5, redistributing weight to 4 SEDs that are 0.97-0.99 Pearson-correlated. Effectively this is "amplify site-conflated SED signal at the cost of Perch". Perch is the only training-time-multi-region backbone; SEDs are all SS-train-fitted. The Aves +0.024 local was an artifact of same-site eval rewarding the site shortcut amplification.

**Anti-correlation rule v3 invalidated**: same-site eval with all-taxa-positive AND sp_row ≥ 0.99 does NOT predict positive LB transfer when the modification reduces W_PERCH below 0.7. The actual LB-protective invariant appears to be: **W_PERCH must stay ≥ 0.7 in any blend**.

**0.929 → 0.932 honest decomposition (post-v36 re-analysis, 2026-04-26)**: total +0.003 LB came from THREE independent +0.001 steps, each adding a new source of **site-invariance**:

| Step | Config change | Site-invariance mechanism added | LB |
|---|---|---|---|
| v12 (ref) | 0.8P + 0.2 SED29 | (Perch xeno-canto multi-region only) | 0.929 |
| v12 → v24 | SED29 → exp50 (weight 0.2) | **Training-time invariance via 2025 BG mixing** (Colombia ≠ Pantanal) | 0.930 |
| v24 → v26 | weight 0.2 → 0.3 | **Optimal mixing of two site-invariance sources** | 0.931 |
| v26 → v33 | + file-max coherence α=0.10 | **Universal physics decoupled from site** | 0.932 |

**Every regression matches one of three failure modes**:
1. **Duplicated existing site-conflated signal**: v17 (SED41f Pearson 0.69 with Perch), v36 (4 SEDs all 0.97-0.99 corr with exp50)
2. **Train-SS-fitted structure**: v18, v19, v22, v28, v29, v34 (centroid/cluster/lookup built on labeled SS)
3. **Reduced existing site-invariance**: v25 (W_PERCH 0.5), v36 (W_PERCH 0.5)

**exp89-97 deep diagnostic + new lever (2026-04-27)**: TP/FP/TN/FN signal hunt revealed:
- **Per-class teacher prediction (sed_on_c, cnxt_on_c) has TN_vs_FN AUC 0.98** (near-perfect FN detector)
- **Perch-SED disagreement |perch[i,c]−exp50[i,c]| has TP_vs_FP AUC 0.89** (strong FP detector)
- Perch latent FFT (ac_dc_ratio, low_high_ratio): TN_vs_FN AUC 0.68-0.70 (real signal)
- ConvNeXt-tiny latent: similar to Perch (0.67-0.69) — transformer hypothesis partially refuted
- SED50 latent: muted (0.35) — outlier, possibly due to BatchNorm + shallow architecture

**exp96 candidate lever (FN_RESCUE)**: per-(row,class) selective boost when v33 < threshold AND max(exp50, exp59) > delta. Preserves W_PERCH=0.7 globally. Best variants:
- Conservative: tl=0.4, α=0.5, δ=0.3 → macro_d +0.0068, sp_row 0.999, Aves +0.009
- Aggressive: tl=0.55, α=0.7, δ=0.2 → macro_d +0.010, sp_row 0.997, Aves +0.017

**exp97 robustness checks**: Train/eval ratio 0.11 (no train overfit), holdout-site robust (+0.004 to +0.013 across all single-site removals). BUT S08 (Insecta-only) shows −0.023 regression on lever — concern. Aves Δ +0.017 magnitude approaches v36's +0.024 which crashed LB; mechanism differs but transfer remains uncertain.

**Honest decision space (2026-04-27)**: Candidate recorded but not submitted. Bull case (LB +0.001~+0.004): different mechanism than v36 (per-class vs uniform), W_PERCH preserved. Bear case (LB −0.005~−0.015): teachers (exp50, exp59) inherit site shortcut from train SS, S08 regression is a warning sign.

**exp99-100 LR-correction lever (2026-04-27)**: built explicit FP/FN LR detectors using exp95 features.
- **Universal-only features Eval AUC**: FP 0.909, FN 0.987 (universal features alone retain almost all signal)
- **LOSO-site CV**: Mean LOSO AUC FP=0.973, FN=0.981 across 5 holdouts. STRONGEST cross-site validation we've ever shown — qualitative evidence past lever variants lacked.
- **Best correction (α_fn=0.5, β_fp=0.1)**: macro_d +0.005, sp_row 0.999, **Aves Δ +0.022, all-other-taxa 0**, S08 unchanged (exp97's −0.023 fixed).
- Mechanism: per-(row, class) LR-predicted FN-prob/FP-prob → soft boost/suppress. W_PERCH=0.7 preserved.
- **Strongest LB candidate since v33** but Aves Δ magnitude similar to v36 (+0.022 vs +0.024). LB transfer uncertain.

| Lever | macro_d | sp_row | Aves Δ | mechanism | LB |
|---|---|---|---|---|---|
| v33 (production) | +0.003 | 0.994 | +0.007 | universal physics | **+0.001** |
| v34 mel-iVAE | +0.026 | 0.990 | +0.008 | train-SS centroid | −0.016 |
| v36 5-way | +0.013 | 0.999 | +0.024 | uniform W_PERCH 0.5 | −0.017 |
| exp97 best | +0.010 | 0.997 | +0.017 | per-class threshold | ? |
| **exp100 LR** | **+0.005** | **0.999** | **+0.022** | **per-class LR-fitted, LOSO 0.97** | **?** |

**FINAL LB SWEEP RESULTS (2026-04-27 daily 5/5 slots, all regressed)**:

| Sub | Config | LB | Δ vs v33 |
|---|---|---|---|
| v38 | LR α=0.3 β=0.1 | 0.920 | **−0.012** |
| v39 | LR α=0.5 β=0.1 | 0.915 | **−0.017** |
| v40 | LR α=0.5 β=0.0 (FN-only) | 0.917 | **−0.015** |
| v41 | exp97 threshold rule (no LR) | 0.918 | **−0.014** |
| v42 | file-max α=0.20 (universal physics scaled) | 0.918 | **−0.014** |

**Critical findings**:
- **LOSO-site CV AUC 0.97 is INVALIDATED as LB-transfer predictor.** The strongest cross-site signal we ever produced still failed to transfer. Hidden-test site distribution is fundamentally different from labeled-site distribution.
- **LR vs simple threshold rule give nearly identical LB results (−0.014 vs −0.015).** LR-fitting adds no value beyond direct teacher boost. The mechanism (per-(row, class) teacher-boost on low v33) is structurally broken.
- **FP-suppress component costs −0.002 (v39 vs v40)**. β=0.1 made things slightly worse.
- **MOST IMPORTANT: file-max α=0.20 also −0.014**. Universal physics lever was thought to be the safest path. But scaling beyond α=0.10 (v33's value) **also crashes**. file-max has a sharp peak at α=0.10 — non-monotonic in α.
- **All post-hoc lever families exhausted**. v22, v28, v34, v36, v38-v42 = 11 distinct mechanisms tested. ALL regressed −0.012 to −0.017 from v33. v33 (0.932) is robust local minimum and likely true ceiling for our model zoo.

**Final lever taxonomy** (all 11 distinct LB-tested mechanisms):

| Mechanism | LB Δ |
|---|---|
| **Universal physics @ optimal α** (v33) | **+0.001** |
| Universal physics scaled (v42 α=0.20) | −0.014 |
| Per-class LR-fit | −0.012 to −0.017 |
| Per-class threshold rule | −0.014 |
| Train-SS centroid (v34 mel-iVAE) | −0.016 |
| Train-SS lookup (v22 site prior+cluster) | −0.016 |
| 27-head additive (v28) | −0.002 |
| Per-class OOF routing (v29) | −0.004 |
| R5 routing (v32) | −0.001 |
| Global rebalance W_PERCH=0.5 (v36) | −0.017 |
| SED swap (v17 SED41f) | −0.007 |

**Conclusion**: Local-LB anti-correlation rule v3 (Aves+, sp_row≥0.99, universal mechanism) is invalidated. **LOSO-site CV at 0.97 is also invalidated**. To break v33 0.932 ceiling, ONLY remaining options are:
1. Data augmentation (xeno-canto/iNat scale up, multi-region BG mixing) — adds new training-time site-invariance source
2. New transformer-based bioacoustic foundation model (Perch v3 release wait)
3. Accept v33 0.932 as production ceiling

**Implication for next lever**: must add a 4TH source of site-invariance independent of (Perch xeno-canto, exp50 2025-BG, file-max physics). Untested candidates:
- Multi-region BG mixing (extend 2025-only to 2023+2024+2025)
- Hour-of-day invariance via temporal augmentation
- Domain-adversarial training (site as adversary)
- Cross-window TTA at inference (we have ~30 min budget headroom per v37 dev wall)

These have falsifiable predictions per the framework: each should add ~+0.001 LB if mechanism is genuinely new and non-train-SS-fitted; each should regress if it merely duplicates an existing source.

**v34 post-mortem (2026-04-25)**: mel-iVAE (T_POOL=16 × N_MELS=128, 32-d z, 55/234 valid centroids from SS-train positives ≥3) z-kNN cosine-sim sigmoid(s·5) additive blend at 5%. Local exp77 (122 held-out eval): macro Δ +0.026, sp_row 0.990, Aves +0.008, Insecta +0.068, Reptilia +0.104. LB regressed −0.016. **4th independent local-positive → LB-negative case (after v18/v19/v22/v28).**

**Anti-correlation rule v3**: even Aves-positive Δ AND extremely high sp_row (0.990) AND a tiny blend weight (0.05) does NOT guarantee LB transfer when the lever's structure is derived from our 55-file labeled SS train split. Mel-iVAE inherited the same site-dependence as exp44c/g, exp51, v22 cluster maps. **Boredom-recipe + 2025-BG site-invariance training (exp50) remains the only train-SS-touching mechanism that has ever transferred to LB.** Anything fitted on our 55 SS files appears poisoned by site shortcut regardless of architecture cleverness or sp_row safety.

**Locked-in lesson (post-v34)**: stop probing levers derived from SS-train-only fits. Remaining unverified levers: (1) probe arch sweep on v26 base, (2) per-class OOF Platt on v33 output, (3) recipe-divergent SED loss (focal/soft-AUC/contrastive), (4) cross-model confidence masking via Perch+exp50 disagreement. None introduce new train-SS-fitted structure.

**exp79–80 post-v34 teardown (2026-04-26)**: 6-experiment mechanism audit confirms iVAE/disagreement noise lever family is structurally dead.
- **exp79 pilot**: same-distribution iVAE eval shows Insecta detection AUC 0.92 (Perch 0.50 random). Looks like big signal.
- **exp79b eval-only**: train-only iVAE+centroids, eval on 122 held-out → still 0.92, but eval Insecta sites ⊂ train Insecta sites (S08/S19/S23 in both).
- **exp79c unlabeled probe**: 64% of 3600 random unlabeled rows get Insecta top-1; 84% concentrated on son11+son20 (2 of 22 valid Insecta centroids). Class-imbalanced centroid space biases toward Insecta whenever audio is "non-Aves-like".
- **exp80a site-holdout**: train iVAE without S19 → S19 Insecta detection AUC = **0.073** (sub-random). Definitive: site fingerprint, not species acoustics.
- **exp80b big-pool iVAE (30k unlabeled)**: Aves +0.10, but Amphibia/Insecta DROP. Bigger pool dilutes already-site-fingerprint centroid.
- **exp80c GPU-MLP taxon classifier (Perch+iVAE concat)**: same-site Insecta AUC 0.97, **LOSO Insecta AUC 0.30** — 0.67-AUC site fingerprint gap. Even iVAE-augmented Perch+MLP cannot generalize Insecta detection across sites.

**Hard data limitation revealed**: All 168 labeled Insecta windows in our 55 train + 11 eval SS files come from just 4 sites (S08/S15/S19/S23). 25 Insecta sonotype species are mostly single-site (Gini=0). With this site-conflated label structure, **no architecture, no representation, no clever objective will produce site-invariant Insecta detection from our labels alone.** This is a data problem, not a model problem.

**Same applies to Mammalia (8 windows total) and Reptilia (1 species).** Only Aves has enough cross-site labeled diversity.

**Implication for next steps**: data-augmentation-first strategy is the only real path. Must bring in:
1. **xeno-canto Insecta clips** (geographically diverse) as positive signal injection during SED training.
2. **iNaturalist 25-sonotype recordings** for cross-site rare-class augmentation.
3. **2025 + 2024 SS clips** as background mix (exp50 already does 2025 BG only — extend).
4. **Synthetic mixup** of labeled Insecta source × multi-site BG (exp44g attempted this but with same-pool BG — confirmed regression v19; need geographically distinct BG).

The lever-pull-and-pray phase is over. To break v33 0.932, we need new positive examples not new prediction tricks.

**exp82-85 systematic Q1-Q4 audit (2026-04-26)**: Tested 4 hypotheses we had previously skipped or insufficiently measured. Critical findings:

| Hypothesis | Result | Mechanism |
|---|---|---|
| **Q1 architecture diversity (ConvNeXt-tiny exp59)** | ✓ +0.006 macro beyond pure weight effect | 0.98 Pearson with exp50 but residual 2% complementary |
| **Q2 recipe diversity (focal loss exp83)** | ✗ NEGATIVE | Pearson 0.98 with exp50, val_SS −0.011, hurts Reptilia in blend |
| **Q3a small external data (exp73, 21 clips)** | ≈ marginal +0.001 | Already largely captured by Q1 |
| **Q4 large external data (exp84b, 545 filtered clips)** | ✓✓ STRONGEST individual contributor | val_SS +0.025, Aves +0.020 alone, all-taxa-positive in blend |

**ALL SED variants Pearson 0.97-0.99 with each other** (exp50/exp59/exp73/exp83/exp84b). Recipe / architecture / external-data tweaks on the same Boredom-style HGNet base produce highly correlated predictions. The "diversity" benefit is at the 1-3% residual signal level. **Implication**: future ensemble strengthening requires architecturally distinct teacher families (non-CNN, Perch v3, transformer-based) — not loss/data tweaks.

**Best LB candidate (exp85)**: **5-way 0.5P + (exp50 + exp59 + exp73 + exp84b) each 0.125 → V9 gate → file-max α=0.10**.
- macro Δ +0.013 vs v33 baseline
- sp_row 0.999 (extremely safe rank preservation)
- Aves Δ +0.024, Insecta +0.003, Mammalia +0.008, Amphibia +0.014, Reptilia +0.028 — only config with ALL taxa positive
- Anti-correlation rule v3 prediction: LB transfer +0.002 to +0.004 → estimated 0.934-0.936

**What we've definitively ruled out (with local evidence)**: focal loss alone (Q2 negative). Recipe diversity hypothesis FAILED at the loss-function level on our pipeline.

**v26 (0.7P + 0.3 exp50)이 우리 파이프라인 최고**. 200+ 팀이 위에 있다는 건 우리가 못 찾은 lever 분명히 있다는 뜻 — "ceiling"이 아님. 미시도/불충분 시도 영역들:

- **Probe arch**: PCA32+LogReg는 v1 baseline 시절 최적화. v26 base 위에서 sweep 안 함.
- **TTA**: 과거 timeout. 90 min budget 내에서 조심스럽게 다시 가능.
- **State-space on logits/embeddings**: exp30 prediction-level Kalman 실패. logit/emb level 안 함.
- **Stacking**: LightGBM/XGBoost on (Perch + exp50 + sequential features) per-class. 표준 top-scorer 관행.
- **Recipe-divergent SED**: focal loss, soft AUC loss, contrastive — 우리 모든 SED는 BCE+BCE.
- **Cross-model confidence masking**: Perch가 confident-wrong일 때 exp50 출력으로 mask.

**v23 LB result (2026-04-24): 0.928** (−0.001 vs v12 0.929). Significance:

| Sub | Config | LB Δ |
|---|---|---|
| v12 (ref) | 0.8P + 0.2 SED29 | 0 |
| v13 | 0.8P + 0.1 S29 + 0.1 S41f | −0.003 |
| v14 | 0.8P + 0.2 SED38 | −0.003 |
| v15 | 0.8P + 0.1 S29 + 0.1 S39 | −0.007 |
| v17 | 0.8P + 0.2 S41f | −0.007 |
| **v23** | **0.8P + 0.1 S29 + 0.1 exp50** | **−0.001** |

**exp50 is the least-harmful teacher-blend addition by a factor of 3.** Residual-correlation-with-Perch hypothesis partially validated: exp50 (residual-corr 0.64 on 11-file val) shows measurably better LB transfer than sed41f (0.70) and sed29 (0.79 with sed41f) blends.

But v23 is still −0.001, not +. Teacher blending at this architecture does not push past v12 ceiling. To actually exceed 0.929, the architecture layer itself (not the final-layer blend) must change: different probe, per-class calibration from OOF, state space on different variables, or exp50 as PRIMARY teacher with Perch as secondary (reversed blend).

**Refined lever classification (2026-04-24 post-v23):**
- ✗ Post-hoc multiplicative lookup from train SS (v18, v19, v22): consistently large LB regressions
- ≈ Post-hoc multiplicative self-consistency (v20, v21 taxon gate): neutral at v12 ceiling
- ≈ Teacher blend additions with Perch-derived SEDs (v13, v14, v17): −0.003 to −0.007
- **≈↑ Teacher blend with Perch-independent SED (v23, exp50 trained w/ 2025 BG): −0.001, smallest margin**
- ? Completely unverified: exp50 as PRIMARY (reversed blend), different probe architecture, state-space on logits/embs, larger/per-class calibrated blend, ensemble OOF stacking

**v24 LB BREAKTHROUGH (2026-04-25): 0.930** — first submission to exceed v12 ceiling (0.929).

Config: `0.8 * Perch + 0.2 * exp50 + V9 gate + Gauss σ=0.5` (SED29 dropped entirely, exp50 takes its slot).

Comparison of same-structure (0.8P + 0.2 SED) submissions:

| Sub | SED used | Trained from | LB |
|---|---|---|---|
| v12 | SED29 (HGNet-B0) | train_audio only | 0.929 |
| v14 | SED38 (HGNet-B0) | train_audio + labeled SS 66f | 0.926 |
| v17 | SED41f (HGNet-B0) | Perch pseudo + labeled SS | 0.922 |
| **v24** | **exp50 (HGNet-B0, Boredom recipe + 2025 BG)** | **train_audio 80% + labeled SS 55f, no Perch pseudo** | **0.930** |

**Why v24 worked (detailed mechanism):**

1. **Ensemble diversity beats individual teacher strength.**
   Residual-corr(Perch, ·) on 11-file held-out: sed29 0.792, sed41f 0.695, exp47 0.640, **exp50 also ~0.64**. The three SEDs share architecture (HGNet-B0) and training data mostly (train_audio + labeled SS 55). What differs is the PSEUDO-LABEL EXPOSURE: SED41f trained on Perch pseudo (imports Perch errors), exp50 trained WITHOUT any teacher distillation (independent error structure). At Perch-0.8 blend, adding a teacher whose errors correlate with Perch provides negative value (v17 −0.007). Adding a teacher with truly orthogonal errors provides positive value.

2. **2025 soundscape BG mixing broke the site-shortcut.**
   Previous 2026-only SEDs (SED29, 38, 41f) had implicit site overfitting. exp50's `BG_MIX_P=0.4` with 2025 SS quiet clips (Colombian Upper Magdalena, geographically distinct from 2026 Pantanal) forced the model to produce predictions INVARIANT to site acoustic fingerprint. Since LB test uses hidden/different sites than our train SS (S03-S23), site-invariant predictions generalize while site-overfit ones regress.

3. **Perch-independence adds rare-class signal without touching common-class signal.**
   Perch near-saturates common Aves (>0.95 on most) — there is no room for a similar teacher to add common-class value, only room for a DIFFERENT teacher to add ORTHOGONAL information. exp50's independent rare-class detection (val_SS 0.838 vs SED29 0.715 held-out) supplies information Perch doesn't have without disturbing Perch-dominant common predictions. Previous teacher blends disturbed common-class predictions via correlated errors, netting negative LB.

4. **Boredom recipe 20s context helps sparse species detection.**
   20-second clips capture mammalian/amphibian/insect events that can be sparse within 5-sec windows (bafcur1 exp50=0.97 vs Perch 0.28). This rare-class rescue was the unique 516975/67107 gain observed in exp48f bottom-8 audit — it's exp50-specific.

5. **Local audit (exp55) now explainable via oracle:**
   Per-class oracle blend weights show 89% of classes prefer exp50-heavy (S50 ≥ 0.4). On 66 SS files mean best weight is (P 0.10, S29 0.05, S50 **0.85**). While these numbers are inflated by training-data memorization, the held-out eval (11 files) confirms the pattern qualitatively. We have been using the wrong primary teacher all along.

**LB slope estimate**: dLB / dw_exp50 ≈ +0.02 over the [0.10, 0.20] range (v23 −0.001 → v24 +0.001). Linear extrapolation suggests w_50 sweet spot possibly at 0.4-0.6, giving LB 0.936-0.943. Curve may bend, but direction confirmed.

**Next test (v25): 0.5P + 0.5*exp50 — aggressive bisection.** Tests whether slope continues or plateaus. If positive, exp50 should likely become primary (v26+ with w_50 > w_P). If flat/negative, peak is between 0.2 and 0.5.

**v25 LB = 0.928** (−0.001). Slope flipped — curve is CONCAVE, not linear. Peak is between w_50 = 0.2 and 0.5.

**v26 LB = 0.931** — new peak @ w_50 = 0.3. exp50 weight sweep so far:

| w_50 | 0.0 (v12) | 0.10 (v23) | 0.20 (v24) | **0.30 (v26)** | 0.50 (v25) |
|---|---|---|---|---|---|
| LB  | 0.929 | 0.928 | 0.930 | **0.931** | 0.928 |

Peak confirmed in the [0.2, 0.5] interval; best probed point is 0.3. True peak likely 0.3-0.4 range.

**exp56 mechanism analysis (v12 → v24 only-differ-by-SED comparison):**
- 37/234 classes show KS > 0.1 distribution shift (most are 25 Insecta sonotypes)
- Per-taxon local Δ on held-out 11 files: Aves **−0.012**, Amphibia **+0.075**, Insecta +0.042, Mammalia **+0.330**, Reptilia **+0.299**
- Only 1 class regressed by > 0.02 (47158son10, −0.053) — all others improved or stayed flat
- SED29 vs exp50 Pearson correlation = **0.194** — near-orthogonal teachers
- AUC-gain positively correlates with KS-stat (ρ=0.39, p<0.001) — classes that shifted most also gained most

**Concave curve explained**: v24 LB +0.001 arises from (Δ Aves × w_Aves_count + Δ non-Aves × w_non-Aves_count). At w_50 = 0.2, Aves taxon dips slightly (−0.012 local on 9 eval Aves) while non-Aves gain hugely (+0.08 to +0.33). At w_50 = 0.5, Aves dips much more (Perch dominance eroded) — the cost on common Aves exceeds additional rare gain, net negative. At w_50 = 0.3, sweet spot.

**Why Perch needs to stay primary**: LB evaluable classes are mostly Aves (~150-170 of ~200). Preserving Perch's common-Aves ranking is worth more LB macro than chasing rare-class AUC beyond what w_50 = 0.3 already unlocks.

**Class-conditional blending hypothesis (unverified)**: w_50 should be high for non-Aves species (where Perch is weak) and low for common Aves (where Perch is saturated). Currently we use a global weight. Per-class adaptive weights could push past 0.931 — test is low-risk since the local oracle showed 0.989 under per-class optimal (training-inflated but directionally informative).

**v28 LB = 0.929** (−0.002 vs v26 0.931). Config: v26 + exp51 27-head additive @ w=0.10 on 25 sonotype/Amphibia columns (Aves untouched, sp_row 0.998 vs v26).

Local was strongly positive (m11 +0.039, Aves Δ exactly 0). LB regressed −0.002. **27-class dedicated head still doesn't transfer despite 2025 BG site-invariance training.**

Pattern across all 27-class dedicated head submissions:
| Sub | Head | Base | LB |
|---|---|---|---|
| v18 | exp44c (no BG) | v17 (P+SED41f) | 0.916 |
| v19 | exp44g (synth-aug) | v17 | 0.907 |
| **v28** | **exp51 (2025 BG)** | **v26 (P+exp50)** | **0.929** |

Improvement order: v22(0.916) > v19(0.907), then v28(0.929) is best. But none POSITIVE vs base (v26 base would have been 0.931 alone). 2025 BG mixing gets us closest but still −0.002.

**Hypothesis confirmed**: 27 sonotypes are Gini=0 single-site in our train SS. ANY model trained on this data, no matter how clever the augmentation, will inherit some site-conditional behavior. exp58 Q3 showed son25's stdev across sites = 0.264 (range 100x) on held-out — this exact site-conditionality damages LB on hidden-site test rows.

**Practical implication**: 27-class species column predictions cannot be improved without genuinely multi-site labeled data. We should accept this and stop probing this lever family.

**Remaining levers (post-v28):**
- ✗ 27-class dedicated head additive (v18/v19/v28 all confirm failure)
- ✗ Post-hoc multiplicative lookups (v22)
- ≈ Higher exp50 weight beyond 0.3 (v25 hit 0.928)
- ? **Architecture-diverse 4-way blend** (exp59 ConvNeXt-tiny: at ep10 val_SS=0.866 already exceeds exp50's 0.826 — independence test pending)
- ? Per-class OOF Platt calibration on v26 output
- ? Class-conditional global blend (Aves vs non-Aves split, exp57 showed +0.024 local)

**NEW (exp47, 2026-04-24) — Boredom-style SED faithful reproduction (HGNet-B0 + 20s + raw-wave mixup + SpecAug + BCE(clip)+BCE(fmax), fp32, BN2d(n_mels)):**

Trained 21 ep (stopped; val_TA plateau 3+ ep, full run was 30). Two eval signals:
- **val_TA** (train_audio 20% held-out, stratified; 171 evaluable classes, common-Aves dominated, LB-proxy-like)
- **val_SS** (same 11 labeled-SS held-out as exp38/41f; 40 rare-heavy classes, our existing local eval)

| epoch | val_TA | val_SS |
|---|---|---|
| ep02 | 0.934 | **0.866 (peak)** |
| ep07 | 0.976 | 0.833 |
| ep14 | 0.985 | 0.814 |
| ep16 | **0.987 (peak)** | 0.786 |
| ep21 | 0.987 | 0.805 |

**Single-model val_TA 0.987** vs SED29 0.737 / SED41f 0.878 local (different eval set). Boredom's 2025 "0.947+ single model" claim most likely refers to a train_audio held-out metric like val_TA, not the hidden-site test LB. Our LB 0.929 ceiling suggests a **~0.06 gap between clean train_audio eval and test LB** — the same covariate-shift story as Perch.

**Key observation: val_TA↑ while val_SS↓ WITHIN one training run.**
- val_TA rises monotonically 0.895 → 0.987 across 20 ep
- val_SS peaks at ep02 0.866, then drops to 0.786 by ep16
- Trade-off is NOT about capacity ceiling — it is capacity *allocation* under class-imbalanced data:
  - 234-class BCE loss is dominated by common Aves gradient (majority of positives)
  - 27 double-blind species (Insecta sonotypes + Amphibia) get few/zero gradient signals
  - More epochs polish common classifiers AND memorize confident wrong predictions on rare
  - Net: "stronger model" actively hurts rare-class AUC even as common-class AUC saturates

**Therefore: SED strengthening has diminishing returns for Perch+SED blend.** Perch already near-saturates common-Aves (LB 0.929 without any SED). Better SED (SED29→41f→47) improves common-class alone-AUC but adds no new information that Perch doesn't already have on those classes. On rare classes (where the marginal LB would come from), all our SEDs remain weak — same imbalanced training data. **The SED axis is orthogonal only on rare species, and on rare species it is orthogonally weak.** This is why v17 (SED41f blend) regressed vs v12 (SED29 blend) despite SED41f being +0.14 stronger alone.

**Two ckpt strategy**: ep02 ckpt is the blend-friendly variant (higher val_SS = less rare-class over-suppression). ep16 ckpt is the common-Aves-alone variant. Save both.

Logs: `experiments/exp47_outputs/train.log`  Ckpt: `best_ckpt.pt` (ep16). ep02 ckpt must be re-trained or saved explicitly.

**NEW (exp48, 2026-04-24) — 5-phase deep dive on error patterns + post-hoc levers:**

Six-hour budget. All evaluated on 11-file held-out (same as exp38/41f/45).

### exp48a — Six novel error patterns
| # | Pattern | Quantification |
|---|---|---|
| 1 | Site variance | S08 macro 0.517 vs S13 0.833 — 0.316 spread, site is dominant error axis |
| 2 | Hour-of-day | 20-24h 0.794 vs 16-20h 0.592 — diurnal species overlap matters |
| 3 | Confident-wrong | All bottom-8 have neg_top10 > pos_median (e.g., 326272: 0.944 > 0.398) — NOT low-signal |
| 4 | **Structured confusion clusters** | 47158son21/22/23 all → {grhtan1, greant1, compot1}; 25073/67107/326272 all → {bcwfin2, rutjac1, tattin1} |
| 5 | Within-file SD = signal quality proxy | Bottom-8 SD 0.01–0.02 (flat); good classes 0.08+ |
| 6 | Full pred range compressed to [0.47, 0.51] | Multiplicative priors have enormous ranking leverage |

### exp48b-g — Post-hoc levers tested (leak-free from train 55-file)

| Lever | macro Δ (40 cls) | Aves Δ | Notes |
|---|---|---|---|
| Site prior (soft τ=0.75) | **+0.117** | +0.006 | P(species \| site) from train, `final *= τ·P + (1−τ)` |
| Cluster rewrite alone (α=2) | +0.034 | +0.003 | Leak-free; eval-derived was +0.090 (leak) |
| Site + cluster combined (τ=0.5, α=2) | +0.118 | +0.004 | Levers are non-redundant |
| **exp47 blend + site + cluster** (w47=0.25, τ=0.75, α=4) | **+0.141** | **+0.062** | Aves also rises |
| "Defensive" (w47=0, τ=0.75, α=2) | +0.120 | +0.006 | no Aves risk |

### exp48f — modality independence audit (KEY insight)
- Spearman on raw preds: perch↔exp47 **ρ=0.011** (excellent signal independence) vs perch↔sed29 0.003 vs sed41f↔exp47 0.675
- Residual correlation (pred − label): perch↔exp47 **0.640** (best) vs perch↔sed41f 0.695 vs sed29↔sed41f 0.901
- exp47 is genuinely error-independent from Perch; sed41f (Perch-pseudo-distilled) is not

### exp48f — exp47 uniquely rescues bottom-8
| Class | Perch | sed29 | sed41f | **exp47** |
|---|---|---|---|---|
| **516975 (Mammalia)** | 0.50 | **0.00** | **0.00** | **1.00** |
| **67107 (Amphibia)** | 0.36 | 0.41 | 0.49 | **0.87** |
| **bafcur1 (Aves)** | 0.28 | 0.59 | 0.95 | **0.97** |
| 74113 (Mammalia) | 0.38 | 0.50 | 0.82 | 0.62 |
| 25073 | 0.50 | 0.41 | **1.00** | 0.43 |
| 326272 | 0.41 | 0.44 | **0.76** | 0.44 |
| 116570 | 0.50 | 0.44 | 0.82 | 0.65 |
| 47158son11 | 0.50 | 0.48 | **1.00** | 0.99 |

**Mechanism for exp47's unique rescue (516975, 67107):**
1. **No pseudo-distillation contamination.** sed41f was Perch-pseudo-trained; for species Perch has no signal on (516975 is unmapped from Perch's 14,795), the pseudo pushes toward "Aves" confidently wrong, drowning the true labeled-SS positive windows. exp47 trains on clean labels only → 516975's S09 positive windows are learned cleanly.
2. **20s clip context.** Sparse mammalian/amphibian calls are better attention-pooled over 20s than 5s.
3. **Label-clean + BN2d + raw-wave mixup regularization.** Boredom recipe generalises from sparse positives.

**This refutes our earlier "stronger SED doesn't help" claim at the per-class level.** Common-Aves saturation held (Perch is already near-max there), but for rare classes in labeled SS, exp47 vs SED41f difference is pseudo-contamination, not capacity. → Perch-as-teacher is an active harm for rare classes, not a neutral addition.

### Anti-correlation rule refinement
| Sub | local Δ | Aves Δ | LB Δ |
|---|---|---|---|
| v18 | +0.131 | **−** | **−0.013** |
| v19 | +0.167 | **−** | **−0.022** |
| v20 | −0.032 | ≈0 | +0.007 |
| v21 | −0.033 | ≈0 | 0.000 |
| **exp48g aggressive** | **+0.141** | **+0.062** | **untested** |

Hypothesis: LB response depends on SIGN OF AVES Δ. v18/v19 had Aves drop (intervention traded common for rare on local; LB dominated by common → negative). v20/v21 had no Aves change (neutral). exp48g has Aves UP → outside the anti-correlation regime → LB gain plausible. Next LB slot should test this specific breakpoint.

### exp48 worst-case damage per top config
- 1491113 (Amphibia): 0.565 → **0.118** (−0.447, cluster definition wrong for this class)
- 47158son03: 0.905 → 0.733 (−0.171)
→ Consider per-class cluster definition whitelist, not blanket apply.

### Deprecation notes from exp48
- exp48c (DSP insect detector): single scalar DSP rule is AUC 0.526 on generic-insect detection. Works perfectly on subset (son17/21/22/23 AUC 1.0) but fails on others. Overlay −0.015 to −0.04. **Single physical rule insufficient** — each sonotype has distinct spectrum.
- iVAE on raw mel: NOT tested in exp48 budget. Given exp43e/k/n/o converging negatives on Perch feature, and exp43e's Spearman(raw Perch, iVAE) = 0.90 at scale, priority dropped. Re-evaluate only if new mechanism hypothesis emerges.

**Today's LB submissions (5/5 slots used):**
| v | Config | LB | vs v12 |
|---|---|---|---|
| v18 (×2) | SED41f base + exp44c overlay | 0.916/0.914 | −0.013/−0.015 |
| v19 | SED41f base + exp44g synth-aug | 0.907 | −0.022 |
| v20 | SED41f base + V9 taxon gate | **0.929** | 0 (gate recovered SED41f drag) |
| v21 | SED29 base (v12 config) + V9 taxon gate | **0.929** | 0 (gate neutral on clean base) |

**Gate profile confirmed on LB**: defensive post-processing that rescues weak bases (SED41f-inheriting-Perch-pseudo-errors) but does not push above v12 ceiling. v12 config (SED29-only, no pseudo) remains Pareto-optimal.

**Anti-correlation rule (4 independent verifications):**
- Local Δ macro positive → LB Δ negative (v18 +0.131 → −0.006, v19 +0.167 → −0.015)
- Local Δ negative → LB Δ zero/positive (v20 −0.032 → +0.007, v21 −0.033 → 0)

**Mechanism**: local 40-class fair eval is selection-biased toward rare/hard taxa underrepresented by Perch. LB 234-class macro is dominated by common classes Perch already handles. Interventions raising local macro (targeting the hard 40) cost predictions on the easy ~194, net LB loss.

**Fair eval reliability**: exp44c local macro +0.131 → LB −0.013. exp44g +0.036 over exp44c → LB −0.009 MORE. Same-site held-out eval does not generalize to hidden-site test LB.

**Pipeline failure audit (exp45, 2026-04-24):** See dedicated section below. Top findings:
1. **SED41f alone macro 0.878 >> v12 z-blend 0.714** on 11 held-out (40 eval classes). Blend weight or normalization leaves signal on the table (locally; LB transfer untested at higher W_SED41F).
2. **`n_train_audio` ∈ [1, 10] is the WORST band (mean AUC 0.491)** — rare-but-nonzero taxa (Mammalia, Reptilia, tail Amphibia) are the true structural drag, not 27-species double-blind.
3. **All bottom-8 non-Aves species get confidently predicted as Aves** — Perch's 14,795-class bird-centric training is a systemic inductive bias beyond just missing species.
4. **85% of windows have 3+ concurrent species** — BCE independent-class assumption mismatches ground truth.
5. bottom-8 offenders span all 5 taxa; 4 are Perch-mapped (problem is not just missing from Perch).

**Local ceiling (fair 11-file Val-A_v2):** 0.922 (exp41f Perch + ensemble-pseudo SED41f α=0.5). NOT reliable LB predictor — v17 local 0.922 → LB 0.922 (−0.007 vs v12 local 0.897 → LB 0.929).

**Active paper thesis:** **iVDFM** (Chang 2026, `ucrl-iclr2026-ivdfm`) — innovation-conditioned time-series iVAE with non-Gaussian (Laplace) prior + linear diagonal dynamics + regime embedding. Applied to Perch `spatial_embedding` (16 temporal patches × 4 freq × 1536) for species-identifiable factor trajectories per 5-sec window. Pooled-embedding iVAE (exp43e) was **negative** (Spearman with raw Perch kNN = 0.90) — pivot justified.

**Running jobs:** exp43a Perch ONNX-GPU extraction on all 10,658 SS (ETA ~20m).

**Top three orthogonal pseudo-confidence signals (exp43g evidence):**
| Signal | Flip-detect AUC | Correlation w/ teacher posterior |
|---|---|---|
| Teacher posterior (top1-top2 gap) | 0.687 | 1.000 |
| Perch/iVAE kNN disagreement | 0.606 | ~0.00 |
| Mahalanobis (exp41b) | **0.463 — useless** | +0.15 |

Best composite: Perch+Teacher rank-norm = 0.707. Mahalanobis added **hurts** (drops to 0.63). exp41 pipeline currently uses Mahal and should be migrated off.

**NEW (exp44c, 2026-04-24) — 27-species dedicated head:**

First positive result in exp44 suite. 27 double-blind species (25 Insecta `47158sonXX` + 2 Amphibia) trained from scratch on labeled SS 55 files (no Perch, no train_audio required). HGNet B0 SED, BN2d(N_MELS) input norm, **pure fp32 (AMP fp16 overflowed on RTX 5090 compute cap 12.0a — NaN batches). 20s clips, BCE+BCE, pos oversample 6x, LR 1e-4.

Best Val-A_v2 **0.848 @ ep04** (15/27 evaluable classes). Per-class bimodal: 6 species AUC>0.95 (site-specific Insecta sonotypes benefit from site shortcut), 2 species AUC ~0.53 (data too sparse). Theoretical macro-AUC boost when blended into 234-class pipeline: ~+0.04 (27/234 × 0.35 lift).

Key recipe lessons (apply to any future SED on labeled SS):
- AMP off (RTX 5090 compute 12.0a triggers fp16 overflow)
- BN2d(N_MELS) input norm mandatory
- Pos oversample 6x for rare class tail
- LR 1e-4, grad clip 1.0, NaN-skip in train loop

Ckpt: `experiments/exp44c_outputs/best_ckpt.pt`.  Val scores: `val_scores.npz`.

**NEW (exp44e, 2026-04-24) — exp44c + Perch blend verified on 11 held-out files:**

| Variant | macro_all (40 cls) | macro_mapped (25) | macro_unmapped (15) |
|---|---|---|---|
| raw Perch alone | 0.622 | 0.695 | **0.500** (random) |
| Perch + Gauss | 0.626 | 0.702 | 0.500 |
| **raw Perch + exp44c** | **0.752** | 0.695 | **0.848** |
| **Perch + Gauss + exp44c** | **0.757** | 0.702 | 0.848 |
| Perch + rank-blend exp44c 0.7 | 0.743 | 0.695 | 0.823 |

**Δ macro_all = +0.131** from exp44c blend. **Δ macro_unmapped = +0.348** (random → 0.848). Perch alone cannot detect 27 double-blind species at all (AUC 0.500 = random, exactly as predicted by exp43r's zero-score-on-unmapped-cols analysis). exp44c head closes this gap. **Dedicated Insecta/Amphibia head is the core lever this competition asks for** (2025 winner insight reproduced).

Next: push into perch-distill submission notebook — drop-in replacement for 27 columns of the 234-class output. Runtime impact: tiny (4M param inference on 600 × 12 = 7200 5-sec windows = maybe 30 sec on Kaggle CPU).

**NEW (exp44g, 2026-04-24) — synthetic augmentation for site-invariance:**

Problem: 16/27 double-blind species appear in SINGLE site only (S08=11, S19=4, S23=3, S15=1). exp44c risks learning site shortcut. Recipe: source = labeled-SS positive window, background = different-site unlabeled SS quiet clip, mix `α*source + (1-α)*bg`, α ~ U(0.4, 0.8), synth_p=0.75 during training. Same SED27 architecture (HGNet B0, fp32, LR 1e-4, BN2d mandatory).

Best Val-A_v2 **0.8842 @ ep04** (+0.036 over exp44c 0.8482). Bottom-10 per-class AUC improved from exp44c [0.53, 0.53, 0.71,...] → exp44g [0.62, 0.67, 0.75,...] — previously random-level species now have genuine signal. Site-shortcut partially broken.

Ckpt: `experiments/exp44g_outputs/best_ckpt.pt`. Candidate replacement for exp44c in the submission notebook if v18 LB confirms baseline.

**NEW (exp44f, 2026-04-24) — v18/v19 LB: BOTH NEGATIVE:**

Submitted `v18 = v17 base + exp44c overlay` → LB **0.916 / 0.914** (two runs, ±0.002 Kaggle variance). Then `v19 = v17 base + exp44g (synth-aug) overlay` → LB **0.907**. All against v17 base 0.922 and v12 baseline 0.929.

Decomposition:
- v12 → v17: −0.007 (SED29 → SED41f swap alone, already in current notebook)
- v17 → v18 (+exp44c): −0.006
- v17 → v19 (+exp44g): **−0.015** (worse despite higher local val)

**Critical finding: fair 11-file Val-A_v2 is ANTI-CORRELATED with LB for 27-species overlay.** exp44g's better local AUC (0.884 vs exp44c 0.848) translates to WORSE LB — synthetic augmentation learned confident wrong patterns. Same-site held-out eval does not generalize to hidden-site test LB.

**Lesson for paper**: "Dedicated sub-classifier" + overlay is fragile when training distribution is site-concentrated. Site shortcut cannot be broken by same-pool augmentation. Requires either (a) multi-site labeled data (not available), (b) domain-adaptation loss with unlabeled target-site data, or (c) abandon overlay approach entirely.

**Remaining LB slots today: 2.** Don't burn on v12 restore (we already know 0.929). Next lever should come from exp45 audit findings (see below).

**NEW (exp45, 2026-04-24) — Pipeline failure-mode audit on 11 held-out files:**

Eval set: 122 rows × 234 classes, 40 classes have both pos+neg → evaluable. 85% of rows have 3+ simultaneous species (multi-label is the norm).

**Macro AUC comparison (40 classes):**
| Model | Macro AUC |
|---|---|
| Perch probability alone | 0.622 |
| SED29 alone | 0.715 |
| v12 z-blend (0.80 P + 0.20 S29 + Gauss 0.5) | 0.714 |
| **SED41f alone** | **0.878** |

SED41f dominates locally. Current notebook W_SED41F = 0.20 may be suboptimal (or local-eval signal overstates it, since SED41f was trained on labeled SS proper).

**Stratification by `n_train_audio`:**
| Band | n_cls | Mean AUC |
|---|---|---|
| 0 | 16 | 0.706 |
| **1-10** | **6** | **0.491 (WORST)** |
| 11-50 | 9 | 0.765 |
| 51-200 | 6 | 0.771 |
| 200+ | 3 | 0.939 |

**Key insight**: 1-10-clip species are MORE broken than 0-clip species. At 0 clips Perch abstains (constant score → AUC 0.5 ≈ random). At 1-10 clips Perch learns a noisy signal that is CONFIDENTLY WRONG (AUC < 0.5). The data-scarcity cliff is not at "0 vs non-0" but at "enough to fit without enough to generalize".

**Stratification by n_sites (labeled SS site diversity):**
| Sites | n_cls | Mean AUC |
|---|---|---|
| 1 | 20 | 0.664 |
| 2 | 13 | 0.738 |
| 3 | 4 | 0.831 |
| 4+ | 3 | 0.790 |

**Bottom-8 species (AUC < 0.5) with predicted confusion targets:**
| Label | AUC | Taxon | n_TA | Confused with (all Aves) |
|---|---|---|---|---|
| 516975 (Mico melanurus) | **0.000** | Mammalia | 3 | litnig1, soulap1, whnjay1 |
| 67107 | 0.333 | Amphibia | 0 | 22961, 22930, bcwfin2 |
| 326272 | 0.358 | Amphibia | 0 | same as 67107 |
| bafcur1 | 0.398 | **Aves** | 125 | strcuc1, sobtyr1, undtin1 |
| 74113 | 0.400 | Mammalia | 2 | compau, osprey, bbwduc |
| 25073 | 0.407 | Amphibia | 0 | 22961, 22930, bcwfin2 |
| 116570 (Caiman yacare) | 0.441 | Reptilia | 1 | flawar1, chobla1, yebela1 |
| 47158son11 | 0.464 | Insecta | 0 | 22961, 22930, bcwfin2 |

Patterns:
- **All 4 non-Aves non-Insecta offenders (Mammalia 2 + Reptilia 1 + 1 Amphibia) get predicted as Aves species.** Perch's bird-centric training bias is structural — it does not "abstain" on non-birds, it actively mislabels them as the closest-sounding bird.
- **Aves species (`bafcur1`, 125 train_audio clips) at AUC 0.40** — this is a sister-species confusion problem (confused with other Aves), not taxonomic. Indicates distinct lever: acoustic species-pair discriminators for close relatives.
- **Three Amphibia + Insecta unmapped species all fall into the same confusion cluster** (22961, 22930, bcwfin2 — all Amphibia). Model defaults to this cluster for any non-bird non-insect sound.

**SED41f blend gain: biggest wins are on 27 double-blind Insecta sonotypes (+0.20–0.40 AUC).** These species are where SED41f adds the most to the blend. But the blend on LB hurts (v17 vs v12 = −0.007). Paradox: SED41f's signal IS species-informative locally but misaligns with test LB.

**Next lever candidates (ranked by evidence-based value):**
1. **Taxonomic hierarchical loss / two-stage classifier** (L1): First predict class_name (Aves/Amphibia/Insecta/Mammalia/Reptilia), then species within taxon. Directly addresses non-Aves → Aves confusion.
2. **Dedicated Mammalia/Reptilia head** (L2): 8-10 species with 1-10 train_audio clips. Needs site-diverse augmentation that doesn't preserve source-site fingerprint (exp44g's failure mode).
3. **Species-pair discriminators for close Aves relatives** (L3): bafcur1 vs strcuc1 type. Very targeted.
4. **SED41f weight sweep** (L4): local evidence W_SED41F=0.5+ better. Risky given local ≠ LB history.
5. **Multi-label-aware loss / CRF** (L5): 85% simultaneous multi-species, but per-class BCE. Possible gain but complex.

Priority: **L1 (hierarchical loss) + L2 (Mammalia head)** address 5 of the bottom-8 offenders directly. Combined potential macro gain: 5/234 × 0.4 = +0.009, IF LB transfer works (unclear).

**NEW (exp45a/b/c/d/e, 2026-04-24) — Hierarchical head experiments + SED sweep:**

### exp45a: multiplicative taxon gating
5-way taxon head trained on Perch embs (train_audio 35,549 + labeled SS 617). Multiplicative gate `base × taxon_prob` on Perch-only baseline: macro **0.622 → 0.745 (+0.123)**. Mammalia NEGATIVE (−0.117 from over-suppression of species where taxon head fails, e.g. 74113 0.375 → 0.042).

### exp45b: class-balanced taxon head
Added per-taxon `pos_weight = N / (5 * N_pos_taxon)` capped at 100. Macro 0.745 → 0.737 (slight regression). Doesn't fix 74113 (0.375 → 0.112 vs 0.042 in 45a) — fundamental issue is species-level, not taxon-imbalance.

### exp45c: soft gating variants
V9 (`base × (taxon_prob + 0.1)`) best: macro 0.622 → **0.767 (+0.145)**, ALL taxa positive, Mammalia +0.122.

### exp45d: SED weight sweep (critical finding)
3-way weight grid {W_P, W_S29, W_S41f} on local 40-class eval:

| Config | Local macro (40) | LB |
|---|---|---|
| v12 (0.8P + 0.2S29) | 0.714 | **0.929** |
| v17 (0.8P + 0.2S41f) | 0.814 | 0.922 |
| 3-way (0.7P + 0.15S29 + 0.15S41f) | 0.816 | 0.925 (v13) |
| **S41f alone (w41=1.0)** | **0.881** | **0.856 (v16)** |

**Perfect anti-correlation** between local macro and LB for W_SED41F. Higher local = worse LB. SED41f was trained on pseudo generated from Perch → inherits Perch's site-specific confidences, which manifest as local-eval gains (same site distribution) but LB losses (hidden sites).

Species-level: SED41f perfectly fixes bafcur1 (Aves, 0.278 → 0.946) and 27 sonotypes (0.500 → 1.000 for many). This fix does NOT transfer to LB.

### exp45e: Gate × base interaction (critical negative)
Applying V9 taxon gate to **already-good bases** HURTS:

| Base | Without gate | With gate | Δ |
|---|---|---|---|
| Perch alone | 0.622 | **0.767** | **+0.145** ✓ |
| v12 (P + S29) | 0.714 | 0.681 | **−0.033** ✗ |
| v17 (P + S41f) | 0.814 | 0.782 | **−0.032** ✗ |
| 3-way | 0.816 | 0.799 | −0.017 ✗ |

**Gate only helps when base lacks species-level signal (Perch alone).** Once SED provides real per-class ranking, multiplicative taxon-level gate adds noise. For bafcur1 (where SED41f had 0.95), applying gate drops to 0.14 — catastrophic.

**Takeaway**: The `+0.145` headline from exp45a/c was an artifact of evaluating against the wrong baseline (Perch-alone). Against the production baseline (Perch + SED blend), the gate is a loss. **v20 submission (v17 × V9 gate) is expected to regress ~−0.03 on LB based on this local analysis.**

**UPDATE (v20 LB result, 2026-04-24):** Actual v20 LB = **0.929** — tied with v12 ceiling and +0.007 over v17 (0.922). **Local analysis was wrong.** Gate *recovered* the SED41f-swap drag on LB.

Pattern across 3 recent submissions:
| Sub | Local Δ macro | LB Δ |
|---|---|---|
| v18 (exp44c overlay) | +0.131 | −0.006 |
| v19 (exp44g synth-aug) | +0.167 | −0.015 |
| **v20 (exp45c V9 gate)** | **−0.032** | **+0.007** |

**Local fair 11-file Val-A is reliably ANTI-CORRELATED with LB.** Tentative inversion rule: local negative → LB positive. This is consistent with covariate-shift reasoning: same-site 11-file eval rewards models that overfit to site-concentrated labeled SS, while hidden-site LB punishes that overfit. Any modification that LOWERS local macro must be removing some site-specific fit → generalizes better.

If the pattern holds for v12-base + gate (local predicted −0.033), LB could be 0.929 + ~0.007 = ~0.936. Remaining 1 slot today could test this.

**NEW (exp45f, 2026-04-24) — Per-class optimal blend analysis:**

On 40 local eval classes:
- Oracle per-class blend AUC = **0.903** (hypothetical upper bound)
- Bimodal α distribution: 22 classes want α < 0.25 (Perch-heavy), 15 want α > 0.75 (SED41f-heavy), only 3 middle
- Teacher dominance: SED41f 27/40, SED29 8/40, Perch 5/40

Per-taxon oracle:
| Taxon | Perch | SED29 | SED41f | Oracle |
|---|---|---|---|---|
| Aves | 0.690 | 0.849 | 0.962 | **0.965** |
| Amphibia | 0.701 | 0.728 | 0.888 | **0.895** |
| Insecta | 0.500 | 0.725 | 0.875 | **0.906** |
| Mammalia | 0.617 | 0.304 | 0.604 | **0.772** (data ceiling) |
| Reptilia | 0.500 | 0.441 | 0.818 | 0.818 |

**Re-interpretation of local/LB anti-correlation:**

Our local 40-class macro and LB ~234-class macro are **different distributions** — not apples-to-apples. Local 0.714 (v12) vs LB 0.929 (v12) means:
- Local 40 classes are a HARD SUBSET (many underperformers make it in as evaluable)
- LB has ~194 additional classes where Perch is already excellent
- These 194 are INVISIBLE to our local eval (no/too-few positives in 11 held-out)

Any pipeline modification that boosts local macro (40-class) likely comes at the cost of the 194 invisible classes. Gate + SED41f-heavy blends pull predictions in directions that help rare/hard species (which dominate our eval) but disturb Perch's already-near-perfect signal on the common 194 (which dominate LB).

This formalizes **why local is anti-correlated with LB**: our eval is a selection-biased subset that SYSTEMATICALLY overweights interventions favoring rare/difficult taxa, at the expense of common taxa that actually drive LB macro.

**Practical consequence**: intervention design should target local LOSS ~0.03 range (not gain) to match what historically produced LB gains. Higher local gains (+0.10+) are warning flags for rare-taxa overfitting.

### Next-lever implication
1. Multiplicative gates only for WEAK base classifiers
2. For blended bases, gate becomes noise on species SED already discriminates
3. Taxon gating as PREPROCESSING to a SED training (not post-processing) might still help — train SED conditional on taxon prior → SED learns species-within-taxon structure
4. For LB, the safest path is to keep v12 config as is. Any deviation requires v12-base local test first.

Paper framing beyond LB: "Data-scarcity cliff and taxonomic inductive bias in large-scale bioacoustic foundation models" — a clean negative result about when pre-trained bird-focused models fail on multi-taxa wildlife monitoring.



**NEW (exp43r, 2026-04-23) — Perch inductive bias on 31 unseen species:**

31/234 BirdCLEF 2026 species are NOT in Perch's 14,795-class training:

| Taxon | Total | In Perch | **Unmapped** |
|---|---|---|---|
| Aves | 162 | 162 | 0 |
| Amphibia | 35 | 32 | 3 |
| Mammalia | 8 | 6 | 2 |
| Insecta | 28 | 3 | **25** (all `47158sonXX` sonotypes) |
| Reptilia | 1 | 0 | 1 (Caiman) |

**260/739 labeled windows (35%) contain ≥1 unmapped species.**

**T1 (teacher score):** `scores` matrix assigns 0.000 to unmapped columns by construction; for 64 "only-unmapped" windows Perch confidently assigns mapped-species logits ~6.18 — structurally wrong but confident.

**T2 (confusion targets per unmapped species):**
- `Insect son07` (n=48) → Leptotila verreauxi / Conirostrum speciosum (Aves) with 200+ NN hits each
- `Insect son17-20, 25` → Ortalis canicollis (Aves) consistently
- `Insect son25` (n=84) → Ortalis canicollis (22) / Pithecopus azureus (3)
- `Adenomera guarani` (Amphibia, n=79) → Leptodactylus fuscus / Leptodactylus elenae (same taxon, less broken)
- `Sapajus cay` (Mammalia monkey, n=13) → Ortalis canicollis / Leptotila verreauxi (birds)

→ Perch consistently projects non-bird taxa onto nearby bird species.

**T3 (impostor detection by group):**
- Mapped classes (n_eval=14): mean AUC 0.94 (exp43o reproduced)
- Unmapped classes: **0 qualified** — teacher_score>0.3 rarely met → no teacher-c-positive group → impostor lever inapplicable.

**Implication**: iVDFM/iVAE on Perch features inherits this bias (only sees embeddings, not raw audio). exp43o centroid-distance works ONLY within mapped 203 classes. **Unmapped 31 species need Perch bypass**:
1. Labeled-SS-trained SED from scratch (SED29/38/39/41f already do this but under-trained for Insecta)
2. Raw mel-spec features for centroid distance in non-Perch space
3. Dedicated Insecta/Amphibia head with aggressive oversampling (2025 winner tactic)
4. **Insect sonotypes are Gini=0 single-site** (exp25) → site-conditioning may help since Perch's site-shortcut incidentally encodes sonotype presence in correlated features.

**NEW (exp43o, 2026-04-23) — impostor detection via centroid distance:**

Within a teacher-c-positive group (windows where Perch score(c)>0.3), how well can we detect windows that are actually NOT c (teacher mistakes) by latent-distance from true-positive centroid?

| Representation | Impostor-detection AUC (mean over 14 qualifying classes) |
|---|---|
| **raw_perch_pooled (1536-d)** | **0.940** |
| iVDFM_eta_win (16-d, posterior mean) | 0.908 |
| iVDFM_f_win (16-d, factor) | 0.906 |
| iVDFM_eta_flat (256-d) | 0.782 |
| iVDFM_f_flat (256-d) | 0.788 |

**Key implication for exp41 pipeline**: impostor-centroid-distance is a **MUCH stronger pseudo-refinement signal than Mahalanobis** (AUC 0.94 vs 0.46). Paired per-class: iVDFM ties or beats raw Perch in 5/14 classes but loses in 9/14 — complementarity exists but raw-Perch-only already captures most of this signal. **Highest-leverage lever: replace Mahal weight in exp41c with raw Perch centroid-distance impostor score**.

## Refactor status (2026-04-23)

Old work archived / deprecated; see `experiments/README.md` for the complete current chain.
- `experiments/_archive_2025/` — pre-Perch pivot SED pipeline (exp1-19), obsolete
- `experiments/_deprecated/` — replaced scripts (exp43 collapsed-iVAE, exp43a CPU-only extractor)
- `experiments/README.md` — single-source-of-truth index of active experiments
- Paper `paper/experiments.tex` is stale past exp39 — needs exp40+ append for working-note

## Project Overview

BirdCLEF+ 2026 Kaggle competition: identify wildlife species (birds, amphibians, mammals, reptiles, insects) from audio recordings in Brazil's Pantanal wetlands. 234 species classes, evaluation metric is macro-averaged ROC-AUC (skipping classes with no true positives). This is a **code competition** — CPU-only notebooks, 90 min runtime, no internet, output `submission.csv`.

Key competition details:
- Audio: 32 kHz ogg format, test soundscapes are 1 min long, predictions per 5-second window
- Species include non-bird taxa (Insecta, Amphibia, Reptilia, Mammalia) which are severely underrepresented
- Some species only appear in labeled train_soundscapes, not in train_audio
- Deadline: June 3, 2026. Working note deadline: June 17, 2026

## LB submission discipline (MUST READ before proposing any Kaggle push)

Kaggle is **hard rate-limited (5 submissions/day)** and each comp re-run costs ~1 hour of wall clock. Treat every submission as a precious final-validation slot, never a search tool.

**Rule (strict)**:
```
local Val-A verified positive     → eligible for LB test
local Val-A verified negative     → rejected. Do NOT re-try on LB.
local Val-A not measured yet      → measure locally FIRST. No LB.
```

**Only exception**: a specific, measurable hypothesis about WHY LB would disagree with local, combined with a single clear disagreement to resolve. Example: v5 priors-OFF was justified because OOF analysis said "priors memorize sites" but we suspected LB has the same sites — one test, one answer. This is NOT a license to re-try every locally-failed lever under "OOF ≠ LB".

**Anti-patterns to reject**:
- "Top scorer X uses technique Y, let's add Y" — check our local evidence first; Y may have already failed on our pipeline
- "OOF might be wrong, let's try on LB" — this is a rationalization, not a hypothesis. Only valid when you can state *exactly* what would make LB differ (e.g., OOF evalset missing certain classes) AND the test is designed to confirm/reject that specific mechanism
- Listing multiple "possible gains" without noting which are locally-verified vs guessed — always separate ✓ verified / ✗ verified negative / ⚠ unverified

**Already locally-verified negative (do NOT re-try on LB unless you have the narrow exception above)**:
- Kalman smoothing (exp30: worse than Gaussian across Q/R grid)
- Per-class rank normalization (exp28: Val-A drop)
- Platt/Isotonic calibration (exp34: Val-A drop — per-fold positives too sparse)
- MLP/Temporal-stack probes (exp33: no gain over LogReg C=0.25)
- ConvNeXtV2-tiny SED blend (exp34b: simplex weight 0)
- HGNet BG-mix SED (exp31: Val-A 0.604)
- Per-file centering / weak priors (exp26: Val-B collapse)
- SED 강한 규제 재학습 (exp36: SpecAugment 24/80 + WD=0.05 + LS=0.1 동시 → alone Val-A 0.60-0.65, SED29 0.737보다 하락. 3-way w_new=0)
- HGNetV2-B2 backbone swap (exp37b: exp29 동일 조건 ep5 0.58, B2가 오히려 해가 됨 — HGNet-B0이 이 도메인/데이터 크기의 sweet spot)
- ConvNeXt-small SED (exp40: 수렴 느림, ep4 0.627 ramp, SED39 EffNet보다 약해 블렌드 기여 미미)
- Linear site/file-mean centering of Perch embeddings (exp42 knn: cross-site kNN agreement 0.638 → 0.513/0.425 하락. Site shortcut 제거가 종 신호도 같이 제거함. 비선형 disentanglement (iVAE 등)은 미검증)
- r2 pseudo iteration from same teacher (exp41h: r1 ensemble student 0.922 → r2 0.914 후퇴. 라운드 추가는 다양한 teacher family 필요)
- Per-class rank / z-score / temperature normalization (exp42: AUC는 per-class monotonic transform에 불변 — 수학적으로 no-op. Cross-class variant만 미세한 차이 (-0.026))
- **Pooled Perch + standard iVAE (exp43e at n=128k, 2026-04-23)**: flip-AUC @k=10 0.562 (iVAE) vs 0.568 (raw Perch) — no improvement. Spearman(raw, iVAE)=**0.901** at n=128k (was 0.84 at n=708 → 스케일이 redundancy 악화). 결론: Perch pooled embedding은 이미 종-판별 압축 상태라 추가 disentangle할 site factor가 없음. **Paper thesis는 시간 구조 유지 입력이 필요** (iVDFM on spatial_embedding).
- **ONNX vs TF embedding drift**: exp43a ONNX-GPU 임베딩의 flip-AUC가 exp21 TF-CPU 캐시 대비 −0.037 낮음. 앞으로 ONNX 일관 사용. 캐시 혼용 금지.

**If a candidate is unverified**: run it locally under Val-A (+ Val-B sanity check) first. New models (SED variants, different backbones) are the main path to genuine LB gains beyond the current 0.91+ ceiling — but they require GPU training + local measurement, not direct LB shots.

## Local compute discipline

**GPU-first on this machine (RTX 5090 32GB, 16 CPU cores).** Kaggle's 90-min CPU limit applies ONLY to the final submission notebook. All local training, inference, and data extraction should use GPU unless a specific technical constraint prevents it.

**Perch v2 extraction — MUST use ONNX + CUDAExecutionProvider:**
- `perch_v2/` is a TF SavedModel **compiled with XLA for CPU only** (`The current platform CUDA is not among the platforms required by the module: [CPU]`). It literally refuses to run on GPU.
- ONNX version at `/tmp/perch_v2.onnx` (Kaggle dataset `rishikeshjani/perch-onnx-for-birdclef-2026`) runs on GPU. **ONNX GPU: 0.81s/batch vs TF CPU: 8s/batch = 10× speedup.**
- Use `onnxruntime-gpu` package. `CUDAExecutionProvider` works on RTX 5090.
- **Do NOT copy the `CUDA_VISIBLE_DEVICES=""` pattern from exp21.** That was correct for CPU-only SavedModel but WRONG now that ONNX is available.

**Embedding drift risk**: ONNX and TF SavedModel may produce marginally different embeddings due to float precision / kernel differences. Kaggle submission notebook uses ONNX too (perch-distill v10+), so local ONNX → Kaggle ONNX is consistent. TF-cached `exp21_outputs/perch_cache/` was built with CPU SavedModel and may differ; revalidate if re-using.

## Commands

```bash
# Dependency management
uv add <package>          # add dependency
uv run <script.py>        # run script with managed deps

# Kaggle notebook submission (for LB scoring only)
uv run kaggle kernels push notebooks/<notebook-dir>

# Kaggle notebook output check (for testing inference pipeline, NOT for scoring)
uv run kaggle kernels output <notebook-slug>

# Run experiment scripts
bash scripts/<experiment>.sh
```

## Repository Structure

- `src/` — shared library code from 2025 (did not perform well, needs rework for 2026)
  - `src/train/` — training loop, dataloader (mel-spectrogram based), EfficientNet models
  - `src/inference/` — ensemble inference with Kalman smoothing, OpenVINO conversion
  - `src/process/` — mel-spectrogram preprocessing, weight updates for rare species
  - `src/utils/` — taxonomy loading, label vectors, silence/VAD detection, metrics
- `experiments/` — trackable experiment scripts (e.g., `exp1_eda.py`). Each experiment is self-contained and importable from `src/`
- `scripts/` — shell scripts to launch experiments (e.g., `scripts/run_exp1.sh`)
- `notebooks/` — Kaggle submission notebooks. Push for inference pipeline testing only; use `kaggle kernels push` for LB scores
- `paper/` — LaTeX files for CLEF 2026 working note submission. Update with experiment results as they come
- `data/birdclef-2025/` — prior year competition data (Brazil Pantanal, reusable)
- `data/birdclef-2026/` — current year competition data
- `ir_models/` — legacy OpenVINO IR weight files (can be deleted)
- `OVERVIEW.md` — competition description and dataset documentation
- `WINNING_SOLUTION_2025.md` — top 3 solutions from BirdCLEF 2025

## Data Layout

```
data/birdclef-2026/
  train.csv                    # metadata: primary_label, secondary_labels, rating, filename, collection (XC/iNat)
  taxonomy.csv                 # 234 species with class_name (Aves, Amphibia, Mammalia, Insecta, Reptilia)
  train_audio/                 # short recordings per species from xeno-canto/iNaturalist
  train_soundscapes/           # 1-min field recordings, some with expert labels
  train_soundscapes_labels.csv # semicolon-separated multi-label annotations per 5-sec segment
  test_soundscapes/            # populated at submission time (~600 files)
  sample_submission.csv        # row_id = {filename}_{end_time}, 234 species probability columns
```

Both 2025 and 2026 data cover Brazilian Pantanal soundscapes and can potentially be combined.

## Architecture Decisions (from 2025 src/)

- **Preprocessing**: audio -> librosa mel-spectrogram -> resize to fixed (H,W) -> save as .npy. Config-driven via YAML
- **Training**: timm backbones (EfficientNet B0/B3, potential RegNet/NFNet), BCEWithLogitsLoss, soft labels with secondary label weighting, MixUp augmentation, WeightedRandomSampler for class imbalance, StratifiedKFold CV
- **Inference**: ensemble of fold models -> power transform on low-ranked columns -> Multivariate Bernoulli Kalman Filter smoothing across time chunks -> OpenVINO conversion for fast CPU inference

## Key Lessons from 2025 Winning Solutions

1. **Self-training is critical**: iterative pseudo-labeling on train_soundscapes with noise injection (mixup, dropout, stochastic depth)
2. **Chunk duration matters**: 20s chunks outperform 5s for training context; 5s windows for inference/submission
3. **Dedicated models for underrepresented taxa**: Insecta/Amphibia need separate training with more epochs, larger batch size
4. **Diverse ensembles survive shakeup**: mix backbone architectures and self-training iterations
5. **Power transform** on pseudo-labels reduces noise in multi-iteration self-training
6. **Background mixing** with prior year soundscapes and ESC-50 helps domain shift
7. **Soft AUC loss** showed promise (3rd place) as alternative to BCE

## Experiment Tracking

Experiments live in `experiments/` as numbered Python files. Each experiment should:
1. Have a corresponding shell script in `scripts/`
2. Log results that can be transferred to `paper/` LaTeX tables
3. Import shared utilities from `src/` rather than duplicating code

**Canonical experiment index**: `experiments/README.md` — always read it before proposing new experiments. Quick milestones below.

Canonical milestones (non-exhaustive):
- `exp21-27` Perch diagnostics & dual-validation — `R5` robust recipe (priors-OFF, PCA32 probes, Gauss σ=0.5); Val-A + Val-B both required.
- `exp29` SED29 HGNet-B0 on train_audio only — alone Val-A 0.737; **enduring diversity component in every blend**.
- `exp34` Perch + SED29 z-blend α=0.8 + Gauss σ=0.5 → **Val-A 0.9062**, LB 0.912 (v6), LB 0.927 (v6 reproducer), LB 0.929 (v12 = perch-distill + SED29 blend).
- `exp38` SED B0 + 20s + labeled SS training (55/11 split) → Val-A_v2 0.810, fair 3-way blend 0.902.
- `exp39` EffNet-NS-JFT SED — Val-A 0.789 alone, fair blend 0.909 (exp39).
- `exp41f` **ensemble-teacher pseudo student** (HGNet B0 + pseudo + SED29 + labeled SS) → Val-A_v2 0.902 alone, fair blend **0.922**. Submitted as v17 → LB 0.922 (regression).
- `exp42/43` identifiable latent pseudo-refinement (see README for exp43 chain).

**Eval regimes** (after exp27, both required):
- **Val-A** (seen-site, file-stratified): matches actual test regime where test soundscapes overlap training sites. Use as primary metric.
- **Val-B** (GroupKFold-by-site): hedge against unseen sites in private LB. Use as secondary metric.
- Decision rule: prefer recipes that win Val-A AND don't collapse on Val-B. R5 satisfies both.

Active notebooks (use `kaggle kernels push -p notebooks/<dir>` — never create new dirs per variant):
- `birdclef-2026-perch-distill` — **LB 0.929 (v12)** — TF-cache(train) + ONNX-Perch(test) + SED29 blend α=0.80 + Gauss σ=0.5. Current production notebook; v13-v17 variants all regressed
- `birdclef-2026-test-submission` — main inference sandbox (Perch v2)
- `birdclef-2026-exp20-submission` — **LB 0.910 (v1)**, frozen-config Perch v2 pipeline
- `birdclef-2026-exp20-to-exp30` — R1 recipe submission kernel
Retired paths: direct-SED-only (v16 LB 0.856 fail). Never re-submit same SED-swap variants under "maybe LB differs" — already 4 LB regressions confirm.

## Kaggle top-scorer insights (digested from `example_and_discussion/lb928.md`, 2026-04-20)

**Gap decomposition (hengck23, repeatedly cited):**
- `lb 0.930 = perch2 + proxy class + temporal + PCA + prior + state space + perclass normalisation`
- `lb 0.92+ = perch2 + proxy class + temporal + PCA`

**Verified single-model scores without KD/unlabeled data:**
- Boredom (1st): 0.947+ single EfficientNet, train_audio + labeled SS only, 13 min inference
- Ali Ozan (4th): 0.925 without unlabeled
- Salman (18th): 0.922 single SED B0 — **20s chunk, raw waveform mixup, CE(clipwise+framewise max), 20+ epoch**
- EliKal (732): 4x arch each ~0.906-0.909 with 10ep Perch-KD + 12ep finetune (KD stabilizes across backbones; NOT a score-boost mechanism)

**Kaggle competitor gotchas (take as priors):**
- Raw waveform mixup > spectrogram mixup (Salman, yukiZ)
- EfficientNet-NS (MBConv) underperforms vs HGNetV2-B0 / ConvNeXt-small (Salman, Murilo)
- Pseudo-labeling on unlabeled SS is risky: Don Mathis failed, top 5 all skipped. Don't invest here unless baseline is solid
- `train_soundscapes_labels.csv` has 739 exact-duplicate rows → always `.drop_duplicates()` or groupby-union
- Sonotypes `47158sonXX` are single-site (Gini=0) → harsh under GKF-by-site, but seen-site Val-A is the right regime for this comp

**What we ALREADY have in our notebook (`proxy_map`, `seq_features_1d`, `smooth_cols`):** proxy-class (genus-match for Amphibia) + temporal seq features (prev/next/mean/max of base_col per 12-window block) + texture-taxa Gaussian smoothing (alpha=0.35). So hengck23's 0.92 baseline isn't a trivial add for us — our gap to 0.92+ is **per-class normalisation + state space + probe hyperparams (starter freezes PCA 64, C=0.50; exp20 v1 was PCA 32, C=0.25)**.

**Single-model Perch-only ceiling is ~0.92**; beyond that needs a second-family model (SED CNN) and/or state space.

## Active research plan (2026-04-23)

**Current paper thesis (working-note direction):** **iVDFM on time-series Perch features** (`ucrl-iclr2026-ivdfm`, Chang 2026). Innovation-level conditioning with Laplace non-Gaussian prior + linear diagonal dynamics + regime embedding. Applied to `spatial_embedding (B, 16, 4, 1536)` from ONNX Perch to preserve per-5-sec temporal structure.

**Running now:** none.

**Completed (2026-04-23):**
- `exp43j` — spatial_embedding extracted (5.7 GB fp16)
- `exp43k` — iVDFM trained (ckpt saved). Regime collapsed to 1/4. Spearman(f, raw Perch) = 0.76.
- `exp43l` — η/f/π structural analysis. η magnitude probe AUC 0.53 (random). η_flat probe 0.81 < raw Perch 0.88.
- `exp43n` — conditional advantage tests (rare species, Zone C, taxonomy). All refuted. Raw Perch ≥ iVDFM in every regime.
- `exp43o` — impostor detection within teacher-c-positive group. **Strong signal across all reps (AUC 0.94 raw / 0.91 iVDFM)**. Paired per-class: iVDFM ties/wins only 5/14 classes. Raw-Perch centroid-distance alone is the high-leverage pseudo-refinement lever.

**6-test refutation of paper thesis as-framed:** iVDFM on Perch spatial_embedding does NOT provide species-identifiable structure beyond what raw Perch pooled already captures. Perch is supervised-pretrained on 14,795-class bioacoustics → little residual disentanglement left. Paper direction must pivot.

**Next steps (high-leverage → speculative):**
1. **`exp43p`** — replace Mahalanobis weight in `exp41c2_ensemble_prep.py` with **per-class centroid-distance impostor score** in raw Perch space. Rebuild pseudo-train_df with new weights. Retrain exp41f-style student. Gate: fair 11-file blend > 0.922 AND LB > 0.929. This is the only evidence-based lever with clear mechanism.
2. **`exp43q`** — ensemble impostor score (raw Perch + iVDFM rank-mean) per class. Check if iVDFM adds any unique signal on 5/14 winning classes in downstream pseudo cleanup. If marginal, drop iVDFM entirely.
3. **Paper pivot** — negative-result analysis: "Why identifiability-driven iVAE/iVDFM fails on supervised-pretrained feature extractors". 6 converging negatives + positive impostor-detection baseline = publishable insight. No more iVDFM engineering.

**Retired hypotheses (see negative-list above; do not re-attempt without new mechanism):**
- Linear site/file centering of Perch
- B2+ backbone swap
- Heavy regularization in SED retrain
- Mahalanobis-only pseudo confidence weighting
- Same-teacher r2 pseudo iteration
- Per-class rank/z-score normalization (AUC-invariant)
- **(NEW) iVAE/iVDFM on Perch features for pseudo refinement** — 6 independent tests (exp43e/k/l/n/o) refute distinct signal over raw Perch

### Timeout diagnostics (90 min CPU budget)

Every Perch v2 pass on ~600 test files × 12 windows ≈ 7200 × 5 s clips costs ~60 min on Kaggle CPU. That single pass alone consumes most of the budget, so:
- **TTA multiplies Perch cost linearly.** `TTA_SHIFTS = [0, 1, -1]` = 3× Perch = ~180 min → guaranteed timeout. Known cause of exp20 v2 timeout.
- **Cache miss doubles it.** If `jaejohn/perch-meta` cache isn't detected, the notebook runs Perch on ~107 fully-labeled train soundscapes first (~60 min) plus test (~60 min). Suspected cause of test-submission v9 timeout even with identical probe config to exp20 v1.
- **Verify cache mount in competition re-run.** Dataset path on Kaggle is `/kaggle/input/datasets/<owner>/<slug>`, not `/kaggle/input/<slug>`. Multiple candidate paths must be tried.
- **MLP probes are cheap** (few seconds per class), not a timeout driver. PCA 32 vs 64 is also negligible.
- Baseline frozen-config (PCA 32, C=0.25, alpha=0.40, no TTA, no MLP) ran under budget and scored **0.910**. Add features incrementally from here and measure wall time.

## Kaggle Submission Flow (Code Competition)

This is a **code competition**. Kaggle re-runs the notebook against hidden test data for scoring. The dev run output (from `kernels push`) is NOT what gets scored.

```bash
# 1. Upload model weights as Kaggle dataset (if needed)
uv run kaggle datasets version -p model-weights/ -m "description" -r zip

# 2. Push notebook — Kaggle runs it (dev run, no test data)
uv run kaggle kernels push -p notebooks/<notebook-dir>

# 3. Wait for dev run to complete (MUST be complete before submit)
uv run kaggle kernels status ultimatumgame/<notebook-slug>

# 4. Submit — Kaggle RE-RUNS the notebook with hidden test data for scoring
uv run kaggle competitions submit -c birdclef-2026 -f submission.csv \
    -k ultimatumgame/<notebook-slug> -v <VERSION> -m "description"

# 5. Check submission score
uv run kaggle competitions submissions -c birdclef-2026 --csv | head -5
```

**Key gotchas:**
- Step 4 triggers a SEPARATE re-run in competition environment (test data mounted, 90 min CPU limit)
- Competition re-run logs are only visible on Kaggle web UI, not via CLI
- If re-run fails (timeout, crash), publicScore is empty — check web UI for error details
- The `-f submission.csv` is the output filename the re-run should produce, not a local file upload
- All `dataset_sources`, `kernel_sources`, `model_sources` must be accessible in competition environment
- **Runtime budget**: with ~600 test files, Perch v2 inference alone can take 60-100 min — must optimize to fit 90 min

### 2026-04-20 SUBMISSION POST-MORTEM (all 4 failed, empty publicScore)

Wasted ~4 daily submission slots. Root causes:

| Kernel | Dev wall (20 files dry-run) | Comp estimate | Cause |
|---|---:|---:|---|
| `exp20-submission` v6 | 436s | ~220 min | TTA_SHIFTS=[0,1,-1] still enabled — labeled as "frozen LB 0.910 config" but v1's actual config had **no TTA** |
| `exp20-to-exp30` v3 | 287s | ~100 min | R1 recipe, TTA=[0], but dev wall already indicates comp will be tight — squeezed over 90 min |
| `test-submission` v10 | 571s | ~200 min | Log shows "Cache saved to /kaggle/working/cache" → jaejohn/perch-meta path not matched → Perch re-computed on 59 val files before test |
| `perch-distill` v3 | 504s | ~180 min | Perch + ProtoSSM CNN dual model |

**Rule of thumb** (derived from these failures):
- Dev run is 20 files. Comp run is ~600 files. **Scaling factor ≈ 30×** for Perch-bound work.
- If dev wall > 200s with TTA=[0], DO NOT SUBMIT — will timeout in comp.
- Target dev wall ≤ **180s** for a safe 90-min comp budget.

**Pre-submission checklist** (mandatory before any `kaggle competitions submit`):
1. `grep -n "TTA_SHIFTS" notebook.py` — must be `[0]` or removed entirely
2. `grep "Cache loaded from" dev_log` — MUST see this, not "Cache saved"
3. Dev run wall time ≤ 180s for 20-file dry-run
4. No secondary CNN/distillation model in the same kernel as Perch
5. `submission.csv` shape = (dry-run-files × 12, 235). Verify columns are 234 classes + row_id

**If cache miss suspected**: the path `/kaggle/input/datasets/jaejohn/perch-meta` must exist. The notebook tries multiple candidate paths; check which branch was taken in the dev log.

- Kaggle username: `ultimatumgame`
- Model weights dataset: `ultimatumgame/birdclef2026-model-weights`
- Model weights are stored locally in `model-weights/` (gitignored)
- **IMPORTANT**: Do NOT create new notebook directories for each submission variant. Edit the existing notebook in-place and push to create a new Kaggle version. Kaggle tracks versions automatically. The user will specify which notebook to use; do not create new notebook slugs without being asked.
