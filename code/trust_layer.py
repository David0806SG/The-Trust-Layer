"""
trust_layer.py
================
A "trust layer" for small-data biology classifiers.

Small biomedical datasets (n often < 200, features >> samples) are a minefield
for data leakage. A classifier that reports 0.95 AUC is frequently measuring an
artifact -- preprocessing fit on the full dataset, feature selection done before
the train/test split, replicate measurements from the same subject straddling
the split, or a batch variable confounded with the outcome.

This module provides:

  1. audit_leakage()   -- static + statistical audit against a leakage taxonomy
  2. honest_cv()       -- leakage-safe cross-validation (all preprocessing and
                          feature selection fit INSIDE each fold, group-aware
                          splitting when subject IDs are present)
  3. naive_cv()        -- deliberately leaky pipeline, to quantify the inflation
  4. fit_trust_model() -- calibrated linear + nonlinear classifiers with
                          calibrated uncertainty and honest model selection
  5. plain_language_report() -- a report a non-ML biologist can read

Design goal: the difference between naive_cv() and honest_cv() *is* the leakage.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


# ----------------------------------------------------------------------------
# 1. LEAKAGE AUDIT
# ----------------------------------------------------------------------------

@dataclass
class Finding:
    """One item in the leakage audit."""
    check: str
    severity: str          # "critical" | "warning" | "ok" | "info"
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class AuditReport:
    findings: list = field(default_factory=list)

    def add(self, check, severity, message, **detail):
        self.findings.append(Finding(check, severity, message, detail))

    @property
    def n_critical(self):
        return sum(f.severity == "critical" for f in self.findings)

    @property
    def n_warning(self):
        return sum(f.severity == "warning" for f in self.findings)

    def to_frame(self):
        return pd.DataFrame(
            [{"check": f.check, "severity": f.severity, "message": f.message}
             for f in self.findings]
        )


def _duplicate_analysis(X: np.ndarray, tol: float = 1e-9):
    """Count exact/near-duplicate rows -- these leak across any random split."""
    # exact duplicates via hashing rounded rows
    keys = [hash(row.tobytes()) for row in np.ascontiguousarray(np.round(X, 6))]
    _, counts = np.unique(keys, return_counts=True)
    n_exact_dupe_rows = int((counts[counts > 1]).sum() - (counts > 1).sum())
    return n_exact_dupe_rows


def audit_leakage(
    X: pd.DataFrame,
    y: Sequence,
    groups: Optional[Sequence] = None,
    batch: Optional[Sequence] = None,
    feature_selection_done_globally: Optional[bool] = None,
    preprocessing_done_globally: Optional[bool] = None,
) -> AuditReport:
    """
    Audit a dataset (and optionally what the analyst already did to it) against a
    leakage taxonomy.

    Parameters
    ----------
    X : DataFrame  (n_samples x n_features)
    y : array-like of labels
    groups : subject/patient IDs, one per row (for detecting subject leakage)
    batch  : batch/plate/site labels, one per row (for detecting batch confound)
    feature_selection_done_globally : if the analyst tells us they selected
        features on the full dataset before splitting -> critical.
    preprocessing_done_globally : if scaling/imputation was fit on the full data.
    """
    y = np.asarray(y)
    Xv = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    n, p = Xv.shape
    rep = AuditReport()

    # -- Shape / dimensionality regime -------------------------------------
    if p >= n:
        rep.add("dimensionality", "warning",
                f"High-dimensional small-n regime: {p} features vs {n} samples "
                f"(p/n = {p/n:.1f}). Univariate feature selection on the full "
                f"dataset will find spurious signal; nested CV is mandatory.",
                n=n, p=p, p_over_n=p / n)
    else:
        rep.add("dimensionality", "info",
                f"{n} samples, {p} features (p/n = {p/n:.2f}).", n=n, p=p)

    # -- Class balance ------------------------------------------------------
    classes, cnts = np.unique(y, return_counts=True)
    minf = cnts.min() / cnts.sum()
    if minf < 0.15:
        rep.add("class_balance", "warning",
                f"Imbalanced classes (minority = {minf:.0%}). AUC can look high; "
                f"report PR-AUC / balanced metrics too.", counts=dict(zip(map(str, classes), cnts.tolist())))
    else:
        rep.add("class_balance", "ok",
                f"Classes reasonably balanced (minority = {minf:.0%}).",
                counts=dict(zip(map(str, classes), cnts.tolist())))

    # -- Duplicate rows -----------------------------------------------------
    ndup = _duplicate_analysis(Xv)
    if ndup > 0:
        rep.add("duplicate_rows", "critical",
                f"{ndup} duplicated feature rows detected. Identical rows split "
                f"across train/test are memorised, not learned.", n_dupe=ndup)
    else:
        rep.add("duplicate_rows", "ok", "No exact duplicate feature rows.")

    # -- Subject / repeated-measures leakage --------------------------------
    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        max_rep = pd.Series(groups).value_counts().max()
        if max_rep > 1:
            rep.add("subject_leakage", "critical",
                    f"Repeated measures present: {n_groups} unique subjects for "
                    f"{n} rows (up to {max_rep} rows/subject). A plain "
                    f"train/test split or StratifiedKFold puts the same subject "
                    f"on both sides. Use GroupKFold on subject ID.",
                    n_subjects=n_groups, max_rows_per_subject=int(max_rep))
        else:
            rep.add("subject_leakage", "ok",
                    "One row per subject; no repeated-measures leakage.")
    else:
        rep.add("subject_leakage", "warning",
                "No subject/patient IDs supplied. If any subject contributed "
                "more than one row, results may be inflated. Provide `groups`.")

    # -- Batch confound -----------------------------------------------------
    if batch is not None:
        batch = np.asarray(batch)
        # chi-square: is batch associated with the label?
        ct = pd.crosstab(pd.Series(batch, name="batch"),
                         pd.Series(y, name="y"))
        chi2, pval, _, _ = stats.chi2_contingency(ct)
        cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        if pval < 0.05 and cramers_v > 0.3:
            rep.add("batch_confound", "critical",
                    f"Batch is confounded with the outcome "
                    f"(Cramer's V = {cramers_v:.2f}, p = {pval:.1e}). A model can "
                    f"score high by reading batch signatures in the features "
                    f"instead of biology. Correct for batch or split by batch.",
                    cramers_v=float(cramers_v), p=float(pval))
        else:
            rep.add("batch_confound", "ok",
                    f"Batch not strongly confounded with outcome "
                    f"(Cramer's V = {cramers_v:.2f}, p = {pval:.2f}).",
                    cramers_v=float(cramers_v), p=float(pval))
    else:
        rep.add("batch_confound", "info",
                "No batch/plate/site labels supplied -- batch leakage not checked.")

    # -- Analyst-declared procedural leakage --------------------------------
    if feature_selection_done_globally:
        rep.add("leaky_feature_selection", "critical",
                "Feature selection was performed on the FULL dataset before the "
                "train/test split. With p >> n this alone can manufacture a "
                "0.9+ AUC from pure noise. Move selection inside the CV folds.")
    elif feature_selection_done_globally is False:
        rep.add("leaky_feature_selection", "ok",
                "Feature selection reported as done inside CV folds.")

    if preprocessing_done_globally:
        rep.add("leaky_preprocessing", "critical",
                "Scaling/imputation was fit on the full dataset (test rows "
                "informed the transform). Fit preprocessing on train folds only.")
    elif preprocessing_done_globally is False:
        rep.add("leaky_preprocessing", "ok",
                "Preprocessing reported as fit inside CV folds.")

    return rep


from sklearn.base import BaseEstimator, TransformerMixin


class BatchCenterer(BaseEstimator, TransformerMixin):
    """
    Leakage-safe batch adjustment: subtract each feature's per-batch mean,
    learned on TRAINING rows only, then apply to any rows. This removes an
    additive batch/plate signature without peeking at test rows and without
    discarding features. Unseen batches at predict time fall back to the
    global training mean.

    Fit requires the batch label per row, passed via `fit(X, y, batch=...)`;
    for use inside a Pipeline, set batch on the instance before fitting.
    """
    def __init__(self, batch=None):
        self.batch = batch

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        b = np.asarray(self.batch)
        self.global_mean_ = X.mean(axis=0)
        self.batch_means_ = {}
        for bl in np.unique(b):
            self.batch_means_[bl] = X[b == bl].mean(axis=0)
        self.fit_batch_ = b
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        b = np.asarray(self.batch)
        out = X.copy()
        for i, bl in enumerate(b):
            m = self.batch_means_.get(bl, self.global_mean_)
            out[i] = X[i] - m
        return out


def honest_cv_batch(X, y, batch, groups=None, k_features=20, n_splits=5, seed=0):
    """
    Fully honest CV: per-batch centering + scaling + feature selection ALL fit
    inside each training fold, group-aware splitting on subject. Isolates the
    real biological signal from an additive batch effect without leakage.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y); batch = np.asarray(batch)
    if groups is not None:
        split_iter = GroupKFold(n_splits=n_splits).split(X, y, groups=np.asarray(groups))
    else:
        split_iter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed).split(X, y)
    aucs = []
    for tr, te in split_iter:
        bc = BatchCenterer(batch=batch[tr]).fit(X[tr])
        Xtr = bc.transform(X[tr])
        bc.batch = batch[te]; Xte = bc.transform(X[te])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        sel = SelectKBest(f_classif, k=min(k_features, Xtr.shape[1])).fit(Xtr, y[tr])
        Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
        p = LogisticRegression(max_iter=2000).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)


# ----------------------------------------------------------------------------
# 2/3. HONEST vs NAIVE CROSS-VALIDATION
# ----------------------------------------------------------------------------

def naive_cv(X, y, k_features=20, n_splits=5, seed=0):
    """
    Deliberately LEAKY pipeline -- reproduces the common mistake:
      * StandardScaler fit on the whole dataset
      * SelectKBest fit on the whole dataset
      * then cross-validate a classifier on the pre-processed, pre-selected data
    Returns the (inflated) mean AUC.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    # leakage: fit transforms on ALL data (train+test together)
    Xs = StandardScaler().fit_transform(X)
    sel = SelectKBest(f_classif, k=min(k_features, X.shape[1])).fit(Xs, y)
    Xsel = sel.transform(Xs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(Xsel, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xsel[tr], y[tr])
        p = clf.predict_proba(Xsel[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)


def honest_cv(X, y, groups=None, k_features=20, n_splits=5, seed=0):
    """
    Leakage-safe pipeline -- scaling AND feature selection are fit inside each
    training fold only. Uses GroupKFold when subject IDs are supplied so no
    subject appears on both sides of a split.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=min(k_features, X.shape[1]))),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=np.asarray(groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    aucs = []
    for tr, te in split_iter:
        p = clone(pipe).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return np.array(aucs)


# ----------------------------------------------------------------------------
# 4. TRUST MODEL: calibrated logistic regression w/ honest, group-aware CV
# ----------------------------------------------------------------------------

def _calibration_method(n_train):
    """Platt (sigmoid) calibration for small training folds, isotonic when
    there are enough rows to fit a monotone step function without overfitting."""
    return "isotonic" if n_train >= 200 else "sigmoid"


# Candidate trust models: a linear baseline and a nonlinear tree ensemble.
# Both are wrapped in CalibratedClassifierCV so their probabilities (not just
# their rankings) are trustworthy. HistGradientBoosting is a scikit-learn
# built-in -- no extra dependency.
MODEL_LABELS = {"logreg": "LogReg(calibrated)", "hgb": "HistGB(calibrated)"}


def _base_estimator(kind):
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=3,
            early_stopping=False, random_state=0)
    return LogisticRegression(max_iter=2000)


def _make_splits(X, y, groups, n_splits, seed):
    """Materialize the CV folds ONCE so every candidate model is scored on the
    identical splits (a fair comparison) and folds can be reused for nesting."""
    if groups is not None:
        it = GroupKFold(n_splits=n_splits).split(X, y, groups=np.asarray(groups))
    else:
        it = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=seed).split(X, y)
    return [(tr, te) for tr, te in it]


def _fit_predict_fold(Xtr, Xte, ytr, batch_tr, batch_te, kind, k_features):
    """One leakage-safe fold: per-batch centering, scaling, and feature
    selection fit on TRAIN rows only, then a calibrated estimator. Returns
    P(positive class) on the held-out rows."""
    from sklearn.calibration import CalibratedClassifierCV
    Xtr = np.asarray(Xtr, dtype=float); Xte = np.asarray(Xte, dtype=float)
    if batch_tr is not None:
        bc = BatchCenterer(batch=batch_tr).fit(Xtr)
        Xtr = bc.transform(Xtr)
        bc.batch = batch_te; Xte = bc.transform(Xte)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    if k_features is not None and k_features < Xtr.shape[1]:
        sel = SelectKBest(f_classif, k=k_features).fit(Xtr, ytr)
        Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
    _, class_counts = np.unique(ytr, return_counts=True)
    inner_cv = int(max(2, min(3, class_counts.min())))
    clf = CalibratedClassifierCV(_base_estimator(kind),
                                 method=_calibration_method(len(ytr)), cv=inner_cv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]


def _oof_for_kind(X, y, kind, splits, batch, k_features):
    """Out-of-fold P(positive) for one model kind over pre-materialized folds."""
    oof = np.full(len(y), np.nan)
    for tr, te in splits:
        bt = None if batch is None else batch[tr]
        be = None if batch is None else batch[te]
        oof[te] = _fit_predict_fold(X[tr], X[te], y[tr], bt, be, kind, k_features)
    return oof


def _metrics(y, oof, kind):
    return {"model": MODEL_LABELS[kind], "oof_proba": oof,
            "auc": float(roc_auc_score(y, oof)),
            "brier": float(brier_score_loss(y, oof)),
            "log_loss": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)))}


def fit_trust_model(X, y, groups=None, batch=None, n_splits=5, seed=0, k_features=None,
                    model="auto", tabpfn_model_path=None):
    """
    Honest, group-aware out-of-fold prediction with a calibrated classifier.

    `model` selects the estimator:
      * "logreg" -- calibrated logistic regression (linear baseline)
      * "hgb"    -- calibrated HistGradientBoosting (nonlinear tree ensemble)
      * "auto"   -- fit BOTH through the identical honest folds and report the
                    one with the higher out-of-fold AUC (default).

    All preprocessing (per-batch centering, scaling, feature selection) and the
    probability calibration are fit INSIDE each training fold only, so the
    reported numbers are leakage-safe.

    Returns the winning model's out-of-fold metrics under the top-level keys
    (`model`, `oof_proba`, `auc`, `brier`, `log_loss`) for backward
    compatibility, plus `candidates` (every model's honest OOF metrics) and
    `selected`. NOTE: picking the best-looking model on the same OOF used to
    report is itself mildly optimistic -- `nested_model_selection()` gives the
    selection-honest number, and `trust_audit` reports it.

    `tabpfn_model_path` is accepted and ignored for backward compatibility.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    batch = None if batch is None else np.asarray(batch)
    splits = _make_splits(X, y, groups, n_splits, seed)

    kinds = ["logreg", "hgb"] if model == "auto" else [model]
    candidates = {}
    for kind in kinds:
        oof = _oof_for_kind(X, y, kind, splits, batch, k_features)
        candidates[kind] = _metrics(y, oof, kind)

    best = max(candidates, key=lambda k: candidates[k]["auc"])
    out = dict(candidates[best])
    out["y"] = y
    out["selected"] = MODEL_LABELS[best]
    out["candidates"] = {c["model"]: {"auc": c["auc"], "brier": c["brier"],
                                      "log_loss": c["log_loss"]}
                         for c in candidates.values()}
    return out


def nested_model_selection(X, y, groups=None, batch=None,
                           kinds=("logreg", "hgb"), n_splits=5, inner_splits=3,
                           seed=0, k_features=None):
    """
    Selection-honest performance when the MODEL itself is chosen from the data.

    On each outer training fold an inner CV picks the model with the best inner
    OOF AUC; that model is then fit on the whole outer-training fold and scored
    on the untouched outer-test fold. The mean outer score is an unbiased
    estimate of "pick the best-looking model" -- and the gap between the naive
    best-candidate OOF AUC and this number is the optimism of model selection.

    Returns (per_outer_fold_auc, chosen_model_per_fold).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    batch = None if batch is None else np.asarray(batch)
    groups = None if groups is None else np.asarray(groups)
    outer = _make_splits(X, y, groups, n_splits, seed)

    outer_scores, chosen = [], []
    for tr, te in outer:
        gtr = None if groups is None else groups[tr]
        inner = _make_splits(X[tr], y[tr], gtr, inner_splits, seed)
        btr_full = None if batch is None else batch[tr]
        inner_auc = {}
        for kind in kinds:
            oof_in = _oof_for_kind(X[tr], y[tr], kind, inner, btr_full, k_features)
            inner_auc[kind] = roc_auc_score(y[tr], oof_in)
        pick = max(inner_auc, key=inner_auc.get)
        chosen.append(MODEL_LABELS[pick])
        bt = None if batch is None else batch[tr]
        be = None if batch is None else batch[te]
        p = _fit_predict_fold(X[tr], X[te], y[tr], bt, be, pick, k_features)
        outer_scores.append(roc_auc_score(y[te], p))
    return np.array(outer_scores), chosen


def reliability_curve(y, p, n_bins=10):
    """Binned observed-vs-predicted for a calibration plot."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ws.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ws)
