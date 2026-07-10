# Trust-Layer Report — small-data biology classifier

*Generated 2026-07-05 · reviewed against a data-leakage taxonomy · honest performance re-estimated with leakage-safe cross-validation · uncertainty from TabPFN v2 (pretrained tabular foundation model).*

---

## The one-line answer

**Your dataset's reported accuracy is inflated by data leakage.** A naive
analysis reports **AUC 0.98**. After closing every leak we could
find, the model's *honest* performance is **AUC 0.90** — still
useful, but not the near-perfect number the naive pipeline suggested. The gap of
**0.08 AUC** was an artifact, not biology.

---

## 1. What we audited (leakage taxonomy)

We checked the dataset (160 samples · 800 features · 40 subjects
with 4 replicate measurements each) against the ways small-data
biology models most often fool themselves. **4 critical** issues
and **1 warning** were found:

| Issue | Severity | What it means for you |
|---|---|---|
| More features than samples | **WARNING** | High-dimensional small-n regime: 800 features vs 160 samples (p/n = 5.0). Univariate feature selection on the full dataset will find spurious signal; nested CV is mandatory. |
| Class balance | **OK** | Classes reasonably balanced (minority = 40%). |
| Duplicate rows | **OK** | No exact duplicate feature rows. |
| Repeated measures / subject leakage | **CRITICAL** | Repeated measures present: 40 unique subjects for 160 rows (up to 4 rows/subject). A plain train/test split or StratifiedKFold puts the same subject on both sides. Use GroupKFold on subject ID. |
| Batch confounded with outcome | **CRITICAL** | Batch is confounded with the outcome (Cramer's V = 0.67, p = 1.4e-17). A model can score high by reading batch signatures in the features instead of biology. Correct for batch or split by batch. |
| Feature selection before split | **CRITICAL** | Feature selection was performed on the FULL dataset before the train/test split. With p >> n this alone can manufacture a 0.9+ AUC from pure noise. Move selection inside the CV folds. |
| Scaling/imputation before split | **CRITICAL** | Scaling/imputation was fit on the full dataset (test rows informed the transform). Fit preprocessing on train folds only. |


## 2. How each leak inflated the score

We re-ran cross-validation, closing one leak at a time. Each fix lowers the
reported AUC toward the truth (Figure 1):

| Analysis | AUC | What changed |
|---|---|---|
| **Naive** (what inflates results) | **0.98 ± 0.01** | scaling + feature selection fit on the *whole* dataset, random train/test split |
| Subject-safe | 0.94 ± 0.06 | preprocessing/selection moved *inside* folds; split by patient so no subject is on both sides |
| **Fully honest** | **0.89 ± 0.06** | additionally removes the batch signature (learned on training rows only) |

The two biggest culprits here:

1. **Feature selection & scaling done before the split.** With 800
   features and only 160 samples, picking "the best features" on the
   full dataset lets the test set leak into training. This alone can manufacture
   a high AUC from pure noise.
2. **Repeated measurements from the same subjects.** 4
   rows per patient means a random split puts near-identical rows of the *same
   person* in both train and test — the model recognises the person, not the
   disease. Splitting by patient ID (GroupKFold) fixes this.
3. **A batch effect confounded with the outcome** (Cramér's V = 0.67). The
   plate/batch a sample was run on carries a strong signature, and here it lines
   up with the disease label — so a model can score high by reading the batch,
   not the biology.

## 3. What the model can *honestly* do — with calibrated uncertainty

We fit **TabPFN v2**, a pretrained tabular foundation model that needs no
training and is strong in the small-n / high-dimensional regime, using the
leakage-safe, patient-grouped scheme. Its out-of-fold predictions (Figure 2)
give an unbiased read on both accuracy and confidence:

- **Honest AUC = 0.90** (vs 0.98 naive).
- **Brier score = 0.13** (lower is better; 0.25 = uninformative).
  The calibration curve tracks the diagonal, so when the model says "70% likely
  disease," it's right about 70% of the time — you can trust its probabilities,
  not just its labels.

## 4. What to do next

1. **Report the honest number (0.90), not the naive one.** In any
   manuscript or model card, state that CV was patient-grouped and that all
   preprocessing/feature selection was fit inside folds.
2. **Deal with the batch confound at the bench, not just in software.** Because
   batch is confounded with the outcome, no analysis can fully separate them.
   Randomise disease/control across plates in future runs, or include batch as a
   modelled covariate.
3. **Keep subjects intact.** Never let replicate measurements of one subject
   straddle a train/test split.

---

*Method notes: honest CV uses GroupKFold on subject ID with StandardScaler,
SelectKBest, and per-batch mean-centering all fit inside each training fold.
TabPFN v2 (tabpfn 2.2.1, pretrained weights) supplies calibrated probabilities.
Uncertainty is read from out-of-fold predictions across 5 folds. The demonstration
dataset is synthetic, with a known modest true signal (25
informative features), deliberately embedded subject and batch leakage, so the
recovered honest AUC can be checked against ground truth.*
