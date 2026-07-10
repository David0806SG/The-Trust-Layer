# Trust-Layer Report — code + data audit

*Generated 2026-07-06. The trust layer now audits BOTH the dataset
and the analysis code. The notebook audit below is fully automatic — it reads the
scientist's code and detects leakage patterns without asking them to self-declare.
Line numbers refer to the companion file `example_leaky.py`.*

---

## Notebook audit (example_leaky.py)

**3 leakage pattern(s) found in the code itself.** These were detected by static analysis of the notebook — no manual declaration needed:

| Line | Leak type | Offending code | Why it inflates results |
|---|---|---|---|
| 14 | Preprocessing before split | `Xs = scaler.fit_transform(X)` | the transform is fit using test rows, so test information leaks into training. |
| 18 | Feature selection before split | `Xsel = sel.fit_transform(Xs, y)` | selecting features on the full dataset lets test-set signal pick the features. With p>>n this alone can manufacture a high AUC from noise. |
| 21 | Not group-aware | — | A subject/patient-like column is present but the split is not group-aware (no GroupKFold / groups= argument). Rows from the same subject can straddle train and test. Use GroupKFold on the subject ID. |


## What the code audit changes

Previously, the dataset audit had to *ask* the analyst "did you fit preprocessing
before the split? did you select features on the full data?" — questions a
scientist who already made the mistake wouldn't know to answer correctly. The
notebook analyzer answers them by reading the code directly, so the leak is caught
before any model is re-fit.

Each finding above maps to the same leakage taxonomy as the dataset audit, so the
two compose: the **code audit** says *what the analyst did wrong and where*; the
**data audit + honest CV** (see the companion report) says *how much it inflated
the score* (here, naive AUC 0.98 → honest 0.89).

---

*Method: `notebook_audit.audit_notebook()` parses a .ipynb/.py into an AST, tracks
transformer/​selector/​resampler constructor assignments, locates the train/test
split or CV `.split()` call, and flags any fit that precedes it. Group-aware CV is
verified by checking for GroupKFold or a `groups=` argument when a subject/patient
column is present. No code is executed.*
