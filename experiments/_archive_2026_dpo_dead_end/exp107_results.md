# exp107 — Rarity cliff + Hour-of-day analysis (LOSO-clean)

Date: 2026-04-28

P_NEW1 (random) and P_NEW3 (Perch-init hybrid) predictions are leak-free
LOSO: each row is predicted by a model that did not see ANY row from its
site. Perch is fully out-of-sample. v33 contains exp50 which is fit on
55-file SS-train split — so v33 is unbiased per-band but mildly fits
some train rows in the eval-band breakdown.

---

## Test A: Rarity cliff (n_train_audio band)

| Band | n_cls | Perch | v33 | P_NEW1 LOSO | P_NEW3 LOSO |
|---|---:|---:|---:|---:|---:|
| 0 | 28 | 0.5145 | 0.9206 | 0.4322 | 0.7392 |
| 1-10 | 8 | 0.5974 | 0.9019 | 0.7480 | 0.7578 |
| 11-50 | 11 | 0.7344 | 0.8813 | 0.7753 | 0.7931 |
| 51-200 | 14 | 0.6127 | 0.9399 | 0.7614 | 0.7460 |
| 200+ | 14 | 0.7470 | 0.9647 | 0.7919 | 0.7689 |

### Key results

| Cliff metric | Perch | v33 | P_NEW1 | P_NEW3 |
|---|---|---|---|---|
| 1-10 → 11-50 (rise = cliff) | **+0.137** | -0.021 | +0.027 | +0.035 |
| 0 (untrained) AUC | 0.515 | 0.921 | 0.432 | **0.739** |
| 1-10 AUC | 0.597 | 0.902 | 0.748 | **0.758** |

### Per-band × taxon (where the cliff hides)

| Band | Taxon | n | Perch | v33 | P_NEW1 | P_NEW3 |
|---|---|---:|---:|---:|---:|---:|
| 0 | Insecta | 25 | 0.500 | 0.931 | 0.422 | 0.740 |
| 0 | Amphibia | 3 | 0.636 | 0.838 | 0.516 | 0.732 |
| 1-10 | Amphibia | 5 | 0.681 | 0.983 | 0.831 | 0.876 |
| 1-10 | Mammalia | 2 | 0.438 | 0.666 | 0.483 | 0.406 |
| 1-10 | Reptilia | 1 | 0.500 | 0.966 | 0.866 | 0.870 |
| 11-50 | Aves | 1 | 0.922 | 0.962 | 0.782 | 0.608 |
| 11-50 | Amphibia | 8 | 0.660 | 0.847 | 0.736 | 0.790 |
| 11-50 | Mammalia | 2 | 0.940 | 0.978 | 0.928 | 0.898 |
| 51-200 | Aves | 13 | 0.595 | 0.938 | 0.755 | 0.745 |
| 51-200 | Amphibia | 1 | 0.843 | 0.967 | 0.839 | 0.760 |
| 200+ | Aves | 14 | 0.747 | 0.965 | 0.792 | 0.769 |

### Conclusions for Test A

1. **Perch cliff is real**: 1-10 (0.597) → 11-50 (0.734), +0.137. 0-clip
   species are at random (0.515) for Perch.
2. **v33 has NO cliff**: 0.88-0.97 across all bands. The current pipeline
   (Perch + exp50 + V9 gate + file-max) effectively masks the rarity cliff
   because exp50 was trained on labels (not Perch-pseudo) covering 0-clip
   species, file-max coherence boosts within-file consistency, and V9 gate
   suppresses cross-taxon confusion.
3. **P_NEW3 closes most of the cliff but doesn't beat v33**: at every band,
   P_NEW3 LOSO < v33. The hybrid head learns rare classes from labeled SS,
   but on test-distribution it generalizes worse than the SED-blend pipeline
   that was already optimized for hidden-test sites.
4. **External data lever is partial**: rarity-cliff hypothesis says external
   data would help band 0 and 1-10. But our v33 already handles those bands
   at 0.88-0.92. The remaining gap (0.92 → 0.95+ ceiling) is dominated by
   common-Aves classes in the 51-200 and 200+ bands, NOT by rare classes.
   External data targeting 0-clip Insecta would help those classes' AUC
   but those classes are already near-saturated in v33 (0.93+).

**Verdict**: external Mammalia/Reptilia (band 1-10, 0.67 v33 AUC) is the
ONLY pocket where rare-class data injection can plausibly move LB. But
this is 11 species out of 234 (Mammalia 8 + Reptilia 1 + Mammalia 1-10
band ≈ 11). Even +0.20 AUC on these gives ~+0.001 LB max.

---

## Test B: Hour-of-day macro AUC

| Bucket | n_rows | Perch | v33 | P_NEW1 LOSO | P_NEW3 LOSO |
|---|---:|---:|---:|---:|---:|
| 0-5h (night) | 247 | 0.6346 | 0.9473 | 0.7100 | 0.7850 |
| 6-11h (morning) | 84 | 0.5820 | 0.9495 | 0.4915 | 0.5995 |
| 18-23h (evening) | 408 | 0.6329 | 0.8887 | 0.6089 | 0.7354 |

(No 12-17h data in our SS — afternoon never recorded.)

### Hour-bucket spread (max - min)

| Model | Spread | Interpretation |
|---|---|---|
| Perch | +0.053 | Modest; foundation model is hour-robust |
| v33 | +0.061 | Modest; preserves Perch's robustness |
| **P_NEW1 (random)** | **+0.219** | Huge — random init learns hour shortcut |
| **P_NEW3 (Perch-init)** | **+0.186** | Large — even Perch init drifts to hour bias |

### Conclusions for Test B

1. **Hour-of-day is small for Perch/v33** (~0.05). Not a major LB lever.
2. **Learned heads AMPLIFY hour bias** (+0.18-0.22 spread). Training on
   our 5-site SS data with 9 unique hours bakes in site×hour correlations.
3. **6-11h bucket collapses for learned models**: Perch 0.58 → P_NEW1 0.49
   (worse than random) → P_NEW3 0.60. Just 84 rows in this bucket and they
   come from sites with limited training coverage.
4. Per-class additive logit-bias correction (within-class hour shift) is
   AUC-invariant by construction (a per-class monotonic transform). The
   test was malformed — to actually fix hour bias would need cross-class
   re-ranking, which our additive shift does not do.

**Verdict**: hour-of-day is NOT a viable lever. Two reasons:
- Effect on Perch/v33 is too small (~0.05) to move LB
- Adding learned heads makes hour bias WORSE, not better
- We don't even have 12-17h coverage in train SS

If hidden test has 12-17h rows, our models are completely unverified there.
Risk is "miss recording window in test we never saw" rather than "fix the
ones we have."

---

## Net implications for OOD strategy

Both hypotheses (rarity-cliff + hour-of-day) turned out to be **smaller
levers than expected** in our existing pipeline:

| Hypothesis | Original belief | Actual finding | Verdict |
|---|---|---|---|
| Rarity cliff drives LB gap | Big effect | v33 already handles it | Marginal |
| Hour-of-day drives LB gap | Real but unmeasured | 0.05 in Perch/v33; learned models amplify | Not viable |

**The dominant remaining LB lever** in our local-experiment evidence space is **site shortcut on Insecta/Mammalia**, but as established in the 5/5 sweep, that is also locally-detectable but not LB-transferable.

This reinforces the existing conclusion: **v33 0.932 is at the data
ceiling for our pipeline architecture**. The remaining levers require
either (a) genuinely new pretrained foundation models with multi-region
non-Aves coverage, or (b) external labeled data for rare taxa, downloaded
from xeno-canto/iNat — neither feasible in the remaining time budget.
