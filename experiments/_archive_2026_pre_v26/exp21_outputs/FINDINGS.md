# exp21 — OOF Ablation Findings

## Setup
- 59 fully-labeled train_soundscapes × 12 windows = 708 evaluation rows
- 71 active classes (Y_FULL.sum > 0) out of 234
- Frozen v1 hyperparameters: PCA=32, C=0.25, λ_event=0.4, λ_texture=1.0, smooth_α=0.35
- Two evaluation protocols:
  1. **GroupKFold by site** (5-fold) — true OOF generalization
  2. **In-sample** (no holdout) — prior tables fit on all labeled SS

## Results

| Condition | OOF (site holdout) | In-sample |
|---|---|---|
| A: raw Perch | 0.729 | 0.739 |
| B: + genus proxy | 0.739 | — |
| C: + prior fusion | **0.488** ⚠️ | **0.977** |
| D: + texture smoothing | 0.485 | — |
| E: + LogReg probes | 0.539 | — |
| F: + Gaussian (full v1) | 0.524 | — |

## Key Finding: LB ≈ in-sample, not OOF

LB 0.910 reproduces almost exactly the in-sample number (0.977 with smoothing/probes
likely puts it in the 0.91–0.93 range). The OOF protocol drops to 0.488 because:

- Prior tables fit on N−1 sites lose all (site, hour) cells specific to held-out sites.
- For held-out val sites, only `global_p` and `hour_p` updates fire.
- For prior-only Insecta sonotypes (`47158sonXX`), Perch logit is 0; prior is the
  *only* signal, but generalized prior is anti-correlated with true labels at unseen
  sites → AUC drops below 0.5.

**Implication**: the 0.910 LB is largely *site memorization* via priors. Test sites
must overlap heavily with training sites in this competition. If hidden test contains
truly new sites, performance would degrade to the OOF range.

## Per-class breakdown (OOF, fold-averaged)

| Class group | A_raw | C_prior | E_probe |
|---|---|---|---|
| Aves (mapped birds) | 0.890 | 0.803 | 0.824 |
| Mammalia (mostly mapped) | 0.945 | 0.875 | 0.906 |
| Amphibia (mix) | 0.804 | 0.524 | 0.599 |
| Insecta (sonotypes, no proxy) | 0.500 | 0.108 | 0.162 |
| Reptilia | 0.500 | 0.341 | 0.743 |

The probe step (E) is the only stage that *recovers* in OOF — it learns site-invariant
features from the embedding for active classes, and helps Reptilia substantially.

## Implication for exp22-25

Every subsequent experiment must report **both** numbers:
- **OOF AUC** (GroupKFold by site) — measures true generalization, robust value
- **In-sample AUC** (no holdout) — proxy for LB-style memorization upside

Acceptance criteria:
- A method that lifts **both** is unambiguously useful.
- A method that lifts only OOF is robust signal (good for paper / private LB).
- A method that lifts only in-sample is more memorization (likely lifts public LB
  but adds shake-up risk on private LB).
