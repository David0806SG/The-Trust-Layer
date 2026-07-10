# Trust-Layer Report

*Generated 2026-07-06. One automated pass: your code and your data
are both audited for leakage, an honest leakage-safe cross-validation is re-run,
and a calibrated TabPFNv2 model reports what the data can really
support.*

---

## The one-line answer

A naive analysis of this dataset reports **AUC 0.98**. After closing every
leak we could find, the honest performance is **AUC 0.90** — the
**+0.08 AUC** difference was inflation, not biology.

---

## 1. Code audit (example_leaky.py)

**3 leakage pattern(s) detected automatically from the code** (no self-declaration needed):

| Line | Leak | Offending code |
|---|---|---|
| 14 | Preprocessing before split | `Xs = scaler.fit_transform(X)` |
| 18 | Feature selection before split | `Xsel = sel.fit_transform(Xs, y)` |
| 21 | Not group-aware | — |

## 2. Data audit

Dataset: **160 samples · 800 features** (p/n = 5.0). Batch–outcome confounding: Cramér's V = 0.67.

**4 critical**, **1 warning**:

| Check | Severity | Finding |
|---|---|---|
| dimensionality | **WARNING** | High-dimensional small-n regime: 800 features vs 160 samples (p/n = 5.0). Univariate feature selection on the full dataset will find spurious signal; nested CV is mandatory. |
| subject_leakage | **CRITICAL** | Repeated measures present: 40 unique subjects for 160 rows (up to 4 rows/subject). A plain train/test split or StratifiedKFold puts the same subject on both sides. Use GroupKFold on subject ID. |
| batch_confound | **CRITICAL** | Batch is confounded with the outcome (Cramer's V = 0.67, p = 1.4e-17). A model can score high by reading batch signatures in the features instead of biology. Correct for batch or split by batch. |
| leaky_feature_selection | **CRITICAL** | Feature selection was performed on the FULL dataset before the train/test split. With p >> n this alone can manufacture a 0.9+ AUC from pure noise. Move selection inside the CV folds. |
| leaky_preprocessing | **CRITICAL** | Scaling/imputation was fit on the full dataset (test rows informed the transform). Fit preprocessing on train folds only. |

## 3. How each leak inflated the score

| Cross-validation | AUC |
|---|---|
| Naive (preprocessing + selection on full data, random split) | 0.98 ± 0.01 |
| Subject-safe (GroupKFold on subject ID) | 0.94 ± 0.06 |
| Fully honest (+ in-fold batch centering) | 0.89 ± 0.06 |

## 4. Honest performance with calibrated uncertainty

**TabPFNv2** fit with leakage-safe, group-aware out-of-fold prediction:

- **Honest AUC = 0.90** (vs 0.98 naive)
- **Brier score = 0.13** — lower is better; 0.25 is uninformative. A well-calibrated model's stated probabilities can be trusted, not just its labels.

## 5. What to do

1. **Report the honest number (0.90), not the naive one** — and state that CV was group-aware with all preprocessing fit inside folds.
2. **Address the batch confound at the bench** — randomize case/control across batches, or model batch as a covariate; no analysis fully separates a confounded batch.
3. **Keep subjects intact** — never let a subject's replicates straddle a split.

---
*No code from the audited notebook was executed. Honest CV uses GroupKFold on subject ID with scaling, feature selection, and per-batch centering all fit inside each training fold. Probabilities are read from out-of-fold predictions.*
