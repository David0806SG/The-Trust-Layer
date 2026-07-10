# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Note: `/Users/davidgu/CLAUDE.md` (loaded as global context) describes an unrelated
> WrenAI/FabGPT project. It does **not** apply here. This repo is "The Trust Layer".

## What this is

An agentic **data-leakage audit** for small-data biology ML (n often < 200, features >> samples).
Point it at a dataset (and optionally the analysis notebook) and it: (1) statically audits the
analyst's code for leakage, (2) audits the dataset against a leakage taxonomy, (3) re-runs an
honest, leakage-safe cross-validation to quantify inflation, (4) fits a calibrated model with
reliable uncertainty, and (5) emits a plain-language markdown report.

**The central design invariant:** `naive_cv() − honest_cv() = the leakage`. Every "honest" code
path exists to be the leak-free counterpart of a deliberately-leaky one. When editing CV logic,
preserve this contrast — `naive_cv`/`naive_cv_task`/`naive_tuned_cv` fit preprocessing on the
full dataset **on purpose**; that is not a bug to fix.

## Running

```bash
cd code                    # all modules resolve siblings by filename, from cwd == code/
python demo.py             # full narrated demo on bundled sample data (~35s)
python demo.py --fast      # linear model only, skips HistGB (~24s)
python demo.py --no-finale # skip the XOR nonlinear-autoswitch finale
python demo.py --data DIR  # point at another sample_dataset/ dir
```

There is **no test suite, linter, or build step** and **no requirements.txt**. Dependencies are
the standard scientific stack only: `scikit-learn`, `scipy`, `pandas`, `numpy`. No network, no
model weights. `HistGradientBoostingClassifier` is used for the nonlinear model — a sklearn
built-in, so no extra dependency.

## Module load order (important, non-obvious)

The modules are **not a package** — there is no `__init__.py`, and they import each other by bare
name (`import trust_layer as tl`). They must be imported in dependency order with `code/` on the
path:

```python
import trust_layer, notebook_audit, trust_tasks   # siblings first
import trust_audit                                 # depends on all three
```

`demo.py::load_trust_layer()` does this explicitly via `importlib`, loading
`trust_layer → notebook_audit → trust_tasks → trust_audit`. Replicate that order in any new
entry point or you will get import errors.

## Architecture

Single entry point: **`trust_audit.trust_audit(X, y, groups=, batch=, notebook=)`** in
`trust_audit.py`. It chains the five stages and dispatches by task type
(`trust_tasks.detect_task` → `binary` | `multiclass` | `regression`). The binary path stays in
`trust_audit`; multiclass/regression route to `_trust_audit_general` and the `trust_tasks` engine.

Module responsibilities:

- **`trust_layer.py`** — binary-classification core. `audit_leakage` (taxonomy check),
  `naive_cv`/`honest_cv`/`honest_cv_batch`, `fit_trust_model` (calibrated LogReg + HistGB,
  auto-selected), `nested_model_selection`, the `BatchCenterer` transformer, `reliability_curve`.
- **`trust_tasks.py`** — generalizes the same naive-vs-honest story to multiclass/regression and
  adds **nested CV** (`naive_tuned_cv` vs `nested_cv`) to measure hyperparameter-selection optimism.
- **`notebook_audit.py`** — **static AST** analyzer (`audit_notebook`). Parses a `.py`/`.ipynb`,
  finds the train/test split line, and flags any scaler/imputer/selector/PCA/resampler fit
  *before* it. **Never executes audited code.**
- **`trust_audit.py`** — orchestrator + markdown report renderer (`_render_report`,
  `_render_general_report`). When a notebook is supplied, its **critical findings override** the
  self-declared leakage flags fed into the data audit (code beats self-report).
- **`kernel_sidecar.py`** — packages the *entire* API into one auto-loadable module
  (`build_trust_namespace()`) for the `small-data-leakage-audit` skill. It **duplicates** the
  source of the other modules inline; if you change core logic, this file drifts unless updated too.
- **`demo.py`** / **`demo_dataset.py`** — narrated live demo and synthetic dataset generator with
  known ground-truth leaks.

## Leakage-safety rules (the whole point — don't break these)

- In any **honest** path, all preprocessing (scaling, `SelectKBest`, per-batch centering) and
  probability calibration are fit **inside each training fold only** — see the `clone(pipe)`
  pattern in `honest_cv` and the manual per-fold fitting in `_fit_predict_fold`.
- Use **`GroupKFold` on subject IDs** whenever `groups` is present, so a subject's replicates never
  straddle a split. Falling back to `StratifiedKFold` when groups exist is a leak.
- `BatchCenterer` learns per-batch means on **train rows only**; unseen batches at predict time
  fall back to the global training mean. Set `.batch` on the instance before each `transform`
  (train batches for train, test batches for test) — this stateful pattern is deliberate.
- Model selection is itself mildly optimistic: `fit_trust_model` picks the best OOF model, but
  `nested_model_selection` gives the **selection-honest** number that the report actually leads with.

## Report/JSON contract

Each run writes `<outdir>/trust_report.md` and `<outdir>/trust_results.json` (default
`outdir="trust_out"`); `oof_proba` is stripped from the JSON. `fit_trust_model` returns the winning
model's metrics under **top-level keys** (`model`, `oof_proba`, `auc`, `brier`, `log_loss`) for
backward compatibility, plus `candidates`/`selected`. `tabpfn_model_path` is accepted and ignored
everywhere — a deprecated arg kept so old calls don't break; don't wire it to anything.

Generated `reports/`, `figures/`, `deck/`, and `sample_data/` are release artifacts, not inputs —
regenerate via the code rather than hand-editing.
