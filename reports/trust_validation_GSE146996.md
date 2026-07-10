# Trust-Layer Validation on Real Biology (GEO GSE146996)

*Generated 2026-07-06. The trust layer was validated on a real
public expression dataset, not the synthetic demo.*

## Dataset

- **GSE146996** — Genome-wide mRNA profiling of human gastric normal vs gastric cancer tissue
- **65 samples** (50 gastric cancer, 15 normal) x **70,523 probes**
- p/n = 1085 — a genuine "far more features than samples" regime, exactly where leakage bites hardest
- Processed log-scale series matrix from GEO (no re-normalization by us)

## Result 1 — the tool does NOT invent leakage where there is none

Run through `trust_audit()` on the real labels:

| Cross-validation | AUC |
|---|---|
| Naive (global feature selection, random split) | 0.99 ± 0.03 |
| Honest (selection fit inside folds) | 0.97 ± 0.05 |

Inflation is only **+0.01 AUC**. Gastric
cancer vs normal tissue is a large, real transcriptional difference, so the high
naive AUC is *mostly earned* — and the trust layer correctly reports that. A tool
that cried "leakage!" on every high score would be useless; this one distinguishes
real signal from artifact.

## Result 2 — the null test: same data, shuffled labels

To isolate the leakage mechanism, the labels were randomly permuted so **no real
biology remains**. The two pipelines were run on the identical 70,523-feature matrix:

| Cross-validation | Real labels | Shuffled labels (null) |
|---|---|---|
| **Naive** (global feature selection) | 0.99 | **0.99** |
| **Honest** (selection inside folds) | 0.97 | **0.54** |

The naive pipeline reports **AUC 0.99 on pure noise** — it is
literally unable to tell gastric cancer from a coin flip, because selecting the 20
"best" of 70,523 features on the full dataset guarantees some will correlate with
any label by chance, and that correlation leaks into the test folds. Honest CV,
which selects features inside each training fold, sits at **0.54
(chance)** on the null and recovers the real signal (0.97) on
the true labels.

**This is the reproducibility problem in one figure:** a bench scientist running
the naive pipeline would report a near-perfect classifier that is +0.45
AUC of pure artifact, and it would never replicate.

## What this validates

- The leakage-safe CV engine behaves correctly on real high-dimensional data.
- The honest/naive gap tracks *actual* leakage — near zero when signal is real,
  ~0.45 AUC when it is manufactured.
- TabPFN v2 runs on real expression data (honest AUC 0.98,
  Brier 0.04) with calibrated probabilities.

*Method: probes with missing values dropped; for the TabPFN run an unsupervised
top-2000-variance prefilter was applied (variance uses no labels, so it is not
leakage). Naive vs honest pipelines differ only in whether StandardScaler +
SelectKBest are fit on the full data or inside each of 5 CV folds. Labels for the
null test permuted with a fixed seed.*
