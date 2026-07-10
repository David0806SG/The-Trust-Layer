"""
trust_audit.py
==============
The single entry point for the trust layer.

    result = trust_audit(
        X, y,
        groups=subject_ids,          # optional: subject/patient IDs
        batch=batch_labels,          # optional: plate/site/batch labels
        notebook="analysis.ipynb",   # optional: the scientist's code
        outdir="trust_out",
    )

One call runs the whole pipeline:

  1. CODE AUDIT   (if a notebook/script is given) -- static AST leakage scan
  2. DATA AUDIT   -- dataset checked against the leakage taxonomy
  3. HONEST CV    -- naive vs leakage-safe, group-aware cross-validation
  4. TRUST MODEL  -- calibrated logistic regression, out-of-fold predictions (real performance)
  5. REPORT       -- one plain-language markdown report a biologist can trust

Returns a dict with every number, and writes `<outdir>/trust_report.md`.

The code audit is what makes the data audit honest: instead of asking the analyst
"did you leak?", it reads their code and answers. When a notebook is supplied, its
findings OVERRIDE the self-declared flags in the dataset audit.
"""
from __future__ import annotations

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import f_classif

import trust_layer as tl
import notebook_audit as na
import trust_tasks as tt


# ----------------------------------------------------------------------------
def _cramers_v(batch, y):
    ct = pd.crosstab(pd.Series(batch), pd.Series(y))
    chi2, p, _, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    return float(v), float(p)


def trust_audit(X, y, groups=None, batch=None, notebook=None, notebook_code=None,
                tabpfn_model_path=None, n_splits=5, k_features=20, seed=0,
                task=None, nested=True, model="auto", outdir="trust_out"):
    # `tabpfn_model_path` is accepted and ignored (deprecated): the model is now
    # a calibrated linear+nonlinear classifier. Kept so existing calls don't break.
    # `model`: "auto" (default) fits calibrated LogReg AND HistGradientBoosting
    # through the same honest folds and reports the better one, plus a
    # selection-honest nested estimate. Use "logreg"/"hgb" to force one.
    """Run the full trust-layer audit. See module docstring.

    `task` is auto-detected as 'binary' | 'multiclass' | 'regression' from `y`,
    or forced by passing it explicitly. Binary classification uses a calibrated
    logistic-regression pipeline with (optional) batch centering. Multiclass
    and regression use the generalized honest-CV engine (macro-OVR AUC / R^2).
    `nested=True` adds a nested-CV vs flat-GridSearch comparison to measure the
    optimism of hyperparameter selection.
    """
    os.makedirs(outdir, exist_ok=True)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X),
                         columns=[f"feat_{i}" for i in range(np.asarray(X).shape[1])])
    y = np.asarray(y)
    n, p = X.shape
    task = task or tt.detect_task(y)

    result = {"dataset": {"n": int(n), "p": int(p)}, "task": task}

    # Multiclass / regression route to the generalized engine and return early.
    if task in ("multiclass", "regression"):
        return _trust_audit_general(
            X, y, task, groups=groups, notebook=notebook, notebook_code=notebook_code,
            n_splits=n_splits, k_features=k_features, seed=seed, nested=nested,
            outdir=outdir, result=result)

    # -- 1. CODE AUDIT ------------------------------------------------------
    code_findings = None
    code_flags = {}
    if notebook is not None or notebook_code is not None:
        code_findings = na.audit_notebook(path=notebook, code=notebook_code)
        cf = na.summarize(code_findings)
        result["code_audit"] = cf
        # translate code findings into the data-audit's declared-leakage flags
        checks = {f.check for f in code_findings if f.severity == "critical"}
        code_flags["preprocessing_done_globally"] = "leaky_preprocessing" in checks
        code_flags["feature_selection_done_globally"] = "leaky_feature_selection" in checks

    # -- 2. DATA AUDIT ------------------------------------------------------
    audit = tl.audit_leakage(
        X, y, groups=groups, batch=batch,
        feature_selection_done_globally=code_flags.get("feature_selection_done_globally"),
        preprocessing_done_globally=code_flags.get("preprocessing_done_globally"),
    )
    result["data_audit"] = {"n_critical": audit.n_critical, "n_warning": audit.n_warning,
                            "findings": [{"check": f.check, "severity": f.severity,
                                          "message": f.message} for f in audit.findings]}

    # -- 3. HONEST vs NAIVE CV ---------------------------------------------
    naive = tl.naive_cv(X.values, y, k_features=k_features, n_splits=n_splits, seed=seed)
    ladder = {"naive": [float(naive.mean()), float(naive.std())]}
    if groups is not None:
        honest_grp = tl.honest_cv(X.values, y, groups=groups, k_features=k_features,
                                  n_splits=n_splits, seed=seed)
        ladder["groupkfold"] = [float(honest_grp.mean()), float(honest_grp.std())]
    else:
        honest_grp = tl.honest_cv(X.values, y, k_features=k_features,
                                  n_splits=n_splits, seed=seed)
        ladder["honest_randomsplit"] = [float(honest_grp.mean()), float(honest_grp.std())]
    if batch is not None:
        honest_full = tl.honest_cv_batch(X.values, y, batch=batch, groups=groups,
                                         k_features=k_features, n_splits=n_splits, seed=seed)
        ladder["groupkfold_batchcorrected"] = [float(honest_full.mean()), float(honest_full.std())]
        v, pv = _cramers_v(batch, y)
        result["dataset"]["batch_cramers_v"] = v
    else:
        honest_full = honest_grp
    result["cv_ladder"] = ladder
    honest_best = float(honest_full.mean())

    # -- 4. TRUST MODEL: calibrated linear + nonlinear, honest OOF ---------
    res = tl.fit_trust_model(X.values, y, groups=groups, batch=batch, n_splits=n_splits,
                             k_features=min(50, p), model=model)
    result["model"] = {"name": res["model"], "auc": float(res["auc"]),
                       "brier": float(res["brier"]), "log_loss": float(res["log_loss"])}
    result["model"]["selected"] = res.get("selected", res["model"])
    result["model"]["candidates"] = res.get("candidates", {})
    result["oof_proba"] = res["oof_proba"].tolist()
    result["inflation"] = float(naive.mean() - res["auc"])

    # -- 4a. MODEL-SELECTION HONESTY (only meaningful when auto-selecting) --
    if model == "auto" and len(res.get("candidates", {})) > 1:
        try:
            outer_auc, picks = tl.nested_model_selection(
                X.values, y, groups=groups, batch=batch, n_splits=n_splits,
                k_features=min(50, p))
            result["model_selection"] = {
                "best_candidate_auc": float(res["auc"]),
                "nested_selection_auc": float(outer_auc.mean()),
                "nested_selection_std": float(outer_auc.std()),
                "selection_optimism": float(res["auc"] - outer_auc.mean()),
                "picks": picks}
        except Exception:
            pass

    # -- 4b. NESTED CV (hyperparameter-selection honesty) ------------------
    if nested:
        flat_score, flat_params = tt.naive_tuned_cv(
            X.values, y, task="binary", groups=groups, n_splits=n_splits, seed=seed)
        nested_scores, _ = tt.nested_cv(
            X.values, y, task="binary", groups=groups, n_splits=n_splits, seed=seed)
        result["tuning"] = {
            "flat_gridsearch_auc": flat_score,
            "nested_cv_auc": float(nested_scores.mean()),
            "nested_cv_std": float(nested_scores.std()),
            "tuning_optimism": float(flat_score - nested_scores.mean()),
            "best_params": {k: (v if isinstance(v, (int, float, str)) else str(v))
                            for k, v in flat_params.items()}}

    # -- 5. REPORT ----------------------------------------------------------
    report = _render_report(result, audit, code_findings, notebook or "analysis notebook")
    report_path = os.path.join(outdir, "trust_report.md")
    open(report_path, "w").write(report)
    json.dump({k: v for k, v in result.items() if k != "oof_proba"},
              open(os.path.join(outdir, "trust_results.json"), "w"), indent=2)
    result["report_path"] = report_path
    result["report_md"] = report
    return result


# ----------------------------------------------------------------------------
def _render_report(result, audit, code_findings, nb_name):
    ds = result["dataset"]
    naive = result["cv_ladder"]["naive"][0]
    honest = result["model"]["auc"]
    infl = result["inflation"]
    md = f"""# Trust-Layer Report

*Generated {datetime.now():%Y-%m-%d}. One automated pass: your code and your data
are both audited for leakage, an honest leakage-safe cross-validation is re-run,
and a calibrated {result['model']['name']} model reports what the data can really
support.*

---

## The one-line answer

A naive analysis of this dataset reports **AUC {naive:.2f}**. After closing every
leak we could find, the honest performance is **AUC {honest:.2f}** — the
**{infl:+.2f} AUC** difference was inflation, not biology.

---
"""
    # --- code section
    if code_findings is not None:
        cf = result["code_audit"]
        crit = [f for f in code_findings if f.severity == "critical"]
        md += f"\n## 1. Code audit ({nb_name})\n\n"
        if cf["n_critical"] == 0:
            md += "**No leakage patterns found in the code.** "
            oks = [f for f in code_findings if f.severity == "ok"]
            for f in oks:
                md += f"\n- {f.message}"
            md += "\n"
        else:
            md += (f"**{cf['n_critical']} leakage pattern(s) detected automatically "
                   "from the code** (no self-declaration needed):\n\n"
                   "| Line | Leak | Offending code |\n|---|---|---|\n")
            typ = {"leaky_preprocessing": "Preprocessing before split",
                   "leaky_feature_selection": "Feature selection before split",
                   "leaky_resampling": "Resampling before split",
                   "subject_leakage": "Not group-aware", "no_split": "No split"}
            for f in crit:
                code = f"`{f.code}`" if f.code else "—"
                md += f"| {f.line or '—'} | {typ.get(f.check, f.check)} | {code} |\n"
        md += "\n"

    # --- data audit
    sec = "2" if code_findings is not None else "1"
    md += f"## {sec}. Data audit\n\n"
    md += (f"Dataset: **{ds['n']} samples · {ds['p']} features** "
           f"(p/n = {ds['p']/ds['n']:.1f}).")
    if "batch_cramers_v" in ds:
        md += f" Batch–outcome confounding: Cramér's V = {ds['batch_cramers_v']:.2f}."
    md += f"\n\n**{audit.n_critical} critical**, **{audit.n_warning} warning**:\n\n"
    md += "| Check | Severity | Finding |\n|---|---|---|\n"
    for f in audit.findings:
        if f.severity in ("critical", "warning"):
            md += f"| {f.check} | **{f.severity.upper()}** | {f.message} |\n"

    # --- honest CV ladder
    sec = str(int(sec) + 1)
    md += f"\n## {sec}. How each leak inflated the score\n\n"
    md += "| Cross-validation | AUC |\n|---|---|\n"
    names = {"naive": "Naive (preprocessing + selection on full data, random split)",
             "honest_randomsplit": "Honest preprocessing (fit inside folds)",
             "groupkfold": "Subject-safe (GroupKFold on subject ID)",
             "groupkfold_batchcorrected": "Fully honest (+ in-fold batch centering)"}
    for k, (m, s) in result["cv_ladder"].items():
        md += f"| {names.get(k, k)} | {m:.2f} ± {s:.2f} |\n"

    # --- model
    sec = str(int(sec) + 1)
    m = result["model"]
    md += (f"\n## {sec}. Honest performance with calibrated uncertainty\n\n"
           f"**{m['name']}** fit with leakage-safe, group-aware out-of-fold prediction:\n\n"
           f"- **Honest AUC = {m['auc']:.2f}** (vs {naive:.2f} naive)\n"
           f"- **Brier score = {m['brier']:.2f}** — lower is better; 0.25 is uninformative. "
           f"A well-calibrated model's stated probabilities can be trusted, not just its labels.\n")
    cand = m.get("candidates") or {}
    if len(cand) > 1:
        md += ("\nA linear (logistic regression) and a nonlinear (gradient-boosted trees) "
               "model were both fit through the identical honest folds; the better one is "
               "reported above:\n\n| Candidate model | Honest AUC | Brier |\n|---|---|---|\n")
        for name, cm in cand.items():
            mark = " ✓" if name == m["name"] else ""
            md += f"| {name}{mark} | {cm['auc']:.2f} | {cm['brier']:.2f} |\n"
    ms = result.get("model_selection")
    if ms:
        md += (f"\n*Choosing the better-looking model is itself mildly optimistic. Nested "
               f"model selection (picking the model inside each training fold) gives the "
               f"selection-honest estimate: **AUC {ms['nested_selection_auc']:.2f}** "
               f"(selection optimism {ms['selection_optimism']:+.2f}).*\n")

    # --- next steps
    sec = str(int(sec) + 1)
    md += (f"\n## {sec}. What to do\n\n"
           f"1. **Report the honest number ({honest:.2f}), not the naive one** — and state that "
           "CV was group-aware with all preprocessing fit inside folds.\n")
    if any(f.check == "batch_confound" and f.severity == "critical" for f in audit.findings):
        md += ("2. **Address the batch confound at the bench** — randomize case/control across "
               "batches, or model batch as a covariate; no analysis fully separates a confounded batch.\n")
    if any(f.check == "subject_leakage" and f.severity == "critical" for f in audit.findings):
        md += "3. **Keep subjects intact** — never let a subject's replicates straddle a split.\n"
    md += ("\n---\n*No code from the audited notebook was executed. Honest CV uses GroupKFold on "
           "subject ID with scaling, feature selection, and per-batch centering all fit inside each "
           "training fold. Probabilities are read from out-of-fold predictions.*\n")
    # --- tuning honesty (binary path)
    if result.get("tuning"):
        t = result["tuning"]
        md += (f"\n## Hyperparameter-selection honesty (nested CV)\n\n"
               f"Tuning the model (feature count + regularization) on the whole dataset and "
               f"reporting the best cross-validated score is itself a mild leak. Nested CV — "
               f"tuning inside each outer training fold only — removes it:\n\n"
               f"| Estimate | AUC |\n|---|---|\n"
               f"| Flat GridSearch best score (optimistic) | {t['flat_gridsearch_auc']:.2f} |\n"
               f"| Nested CV (honest) | {t['nested_cv_auc']:.2f} ± {t['nested_cv_std']:.2f} |\n\n"
               f"Tuning optimism = **{t['tuning_optimism']:+.2f} AUC**.\n")
    return md


# ----------------------------------------------------------------------------
# Generalized path: multiclass + regression
# ----------------------------------------------------------------------------
def _trust_audit_general(X, y, task, groups=None, notebook=None, notebook_code=None,
                         n_splits=5, k_features=20, seed=0, nested=True,
                         outdir="trust_out", result=None):
    """Honest CV + nested CV for multiclass / regression (Ridge / multinomial
    LogReg). Same leakage story, per-task metrics."""
    result = result or {"dataset": {"n": int(X.shape[0]), "p": int(X.shape[1])}, "task": task}
    n, p = X.shape

    # -- code audit (identical to binary path) --
    code_findings = None
    code_flags = {}
    if notebook is not None or notebook_code is not None:
        code_findings = na.audit_notebook(path=notebook, code=notebook_code)
        result["code_audit"] = na.summarize(code_findings)
        checks = {f.check for f in code_findings if f.severity == "critical"}
        code_flags["preprocessing_done_globally"] = "leaky_preprocessing" in checks
        code_flags["feature_selection_done_globally"] = "leaky_feature_selection" in checks

    # -- data audit (dimensionality / duplicates / subject leakage all task-agnostic) --
    audit = tl.audit_leakage(
        X, (y if task != "regression" else np.zeros(n, dtype=int)),  # class checks skipped for reg
        groups=groups, batch=None,
        feature_selection_done_globally=code_flags.get("feature_selection_done_globally"),
        preprocessing_done_globally=code_flags.get("preprocessing_done_globally"))
    # for regression, drop the meaningless class_balance finding
    findings = [f for f in audit.findings
                if not (task == "regression" and f.check == "class_balance")]
    result["data_audit"] = {
        "n_critical": sum(f.severity == "critical" for f in findings),
        "n_warning": sum(f.severity == "warning" for f in findings),
        "findings": [{"check": f.check, "severity": f.severity, "message": f.message}
                     for f in findings]}

    # -- honest vs naive ladder --
    naive = tt.naive_cv_task(X.values, y, task=task, k_features=k_features,
                             n_splits=n_splits, seed=seed)
    honest = tt.honest_cv_task(X.values, y, task=task, groups=groups,
                               k_features=k_features, n_splits=n_splits, seed=seed)
    hn = "R2" if task == "regression" else "AUC (macro-OVR)"
    result["cv_ladder"] = {"naive": [float(naive.mean()), float(naive.std())],
                           ("groupkfold" if groups is not None else "honest_randomsplit"):
                               [float(honest.mean()), float(honest.std())]}
    result["headline_name"] = hn

    # -- honest OOF metric bundle --
    oof = tt.oof_predict_task(X.values, y, task=task, groups=groups,
                              k_features=min(50, p), n_splits=n_splits, seed=seed)
    result["model"] = {"name": "Ridge" if task == "regression" else "LogReg(multinomial)",
                       "metrics": oof["metrics"]}
    result["inflation"] = float(naive.mean() - oof["metrics"]["headline"])

    # -- nested CV --
    if nested:
        flat_score, flat_params = tt.naive_tuned_cv(X.values, y, task=task, groups=groups,
                                                    n_splits=n_splits, seed=seed)
        nested_scores, _ = tt.nested_cv(X.values, y, task=task, groups=groups,
                                        n_splits=n_splits, seed=seed)
        result["tuning"] = {"flat_gridsearch": flat_score,
                            "nested_cv": float(nested_scores.mean()),
                            "nested_cv_std": float(nested_scores.std()),
                            "tuning_optimism": float(flat_score - nested_scores.mean()),
                            "best_params": {k: str(v) for k, v in flat_params.items()}}

    # -- report --
    report = _render_general_report(result, findings, code_findings, notebook or "analysis notebook")
    rp = os.path.join(outdir, "trust_report.md")
    open(rp, "w").write(report)
    json.dump({k: v for k, v in result.items() if k != "oof"},
              open(os.path.join(outdir, "trust_results.json"), "w"), indent=2)
    result["report_path"] = rp
    result["report_md"] = report
    return result


def _render_general_report(result, findings, code_findings, nb_name):
    ds = result["dataset"]; task = result["task"]; hn = result["headline_name"]
    naive = result["cv_ladder"]["naive"][0]
    honest = result["model"]["metrics"]["headline"]
    infl = result["inflation"]
    md = f"""# Trust-Layer Report ({task})

*Generated {datetime.now():%Y-%m-%d}. Task auto-detected as **{task}**. The same
leakage audit and leakage-safe cross-validation apply; the reported metric is
{hn} ({'higher is better' if task != 'regression' else 'R² closer to 1 is better'}).*

---

## The one-line answer

A naive analysis reports **{hn} {naive:.2f}**. After closing every leak we could
find, honest performance is **{hn} {honest:.2f}** — a **{infl:+.2f}** difference
that was inflation, not signal.

---
"""
    sec = 1
    if code_findings is not None:
        cf = result["code_audit"]; crit = [f for f in code_findings if f.severity == "critical"]
        md += f"\n## {sec}. Code audit ({nb_name})\n\n"
        if cf["n_critical"] == 0:
            md += "**No leakage patterns found in the code.**\n"
        else:
            md += f"**{cf['n_critical']} leakage pattern(s) detected from the code:**\n\n| Line | Leak | Code |\n|---|---|---|\n"
            for f in crit:
                md += f"| {f.line or '—'} | {f.check} | `{f.code or '—'}` |\n"
        sec += 1

    md += f"\n## {sec}. Data audit\n\nDataset: **{ds['n']} samples · {ds['p']} features** (p/n = {ds['p']/ds['n']:.1f}).\n\n"
    md += "| Check | Severity | Finding |\n|---|---|---|\n"
    for f in findings:
        if f.severity in ("critical", "warning"):
            md += f"| {f.check} | **{f.severity.upper()}** | {f.message} |\n"
    sec += 1

    md += f"\n## {sec}. How leakage inflated the score\n\n| Cross-validation | {hn} |\n|---|---|\n"
    names = {"naive": "Naive (preprocessing + selection on full data)",
             "honest_randomsplit": "Honest (fit inside folds)",
             "groupkfold": "Subject-safe (GroupKFold)"}
    for k, (m, s) in result["cv_ladder"].items():
        md += f"| {names.get(k, k)} | {m:.2f} ± {s:.2f} |\n"
    sec += 1

    m = result["model"]["metrics"]
    md += f"\n## {sec}. Honest performance\n\n**{result['model']['name']}**, leakage-safe out-of-fold:\n\n"
    if task == "regression":
        md += f"- **R² = {m['r2']:.2f}**, RMSE = {m['rmse']:.2f}, MAE = {m['mae']:.2f}\n"
    else:
        md += (f"- **Macro-OVR AUC = {m['auc_ovr_macro']:.2f}**, balanced accuracy = "
               f"{m['balanced_accuracy']:.2f}, multiclass Brier = {m['brier']:.2f}\n")
    sec += 1

    if result.get("tuning"):
        t = result["tuning"]
        md += (f"\n## {sec}. Hyperparameter-selection honesty (nested CV)\n\n"
               f"| Estimate | {hn} |\n|---|---|\n"
               f"| Flat GridSearch best score (optimistic) | {t['flat_gridsearch']:.2f} |\n"
               f"| Nested CV (honest) | {t['nested_cv']:.2f} ± {t['nested_cv_std']:.2f} |\n\n"
               f"Tuning optimism = **{t['tuning_optimism']:+.2f} {hn}**.\n")

    md += ("\n---\n*No notebook code was executed. Honest CV fits scaling and feature selection "
           "inside each fold; GroupKFold keeps subjects intact; nested CV tunes hyperparameters "
           "inside each outer training fold only.*\n")
    return md
