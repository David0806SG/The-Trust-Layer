"""
trust_tasks.py
==============
Extends the trust layer beyond binary classification to:

  * MULTICLASS classification  (macro one-vs-rest AUC, multiclass log-loss/Brier)
  * REGRESSION                 (R^2, RMSE, MAE)
  * NESTED CV                  (hyperparameter-selection honesty)

The leakage story is identical across task types: preprocessing / feature
selection fit on the full dataset before the split inflates the score, and
subject/batch structure straddling the split inflates it further. This module
generalizes `naive_cv` / `honest_cv` to any task, and adds the one leak the
binary engine did not cover -- **tuning hyperparameters on the whole dataset**
and reporting that same cross-validated score.

Design goal (unchanged): naive - honest = leakage; and for tuning,
`best_score_ from a flat GridSearch` - `nested-CV score` = the optimism of
hyperparameter selection.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (StratifiedKFold, GroupKFold, KFold,
                                     GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, log_loss, balanced_accuracy_score,
                             r2_score, mean_squared_error, mean_absolute_error)


# ----------------------------------------------------------------------------
# Task detection & per-task building blocks
# ----------------------------------------------------------------------------

def detect_task(y, max_classes=20):
    """Infer 'binary' | 'multiclass' | 'regression' from the outcome vector.

    A float outcome with many distinct values is regression; otherwise the task
    is classification, binary when there are exactly two classes.
    """
    y = np.asarray(y)
    uniq = np.unique(y[~pd.isna(y)]) if y.dtype.kind == "f" else np.unique(y)
    n_uniq = len(uniq)
    if y.dtype.kind == "f" and n_uniq > max_classes:
        return "regression"
    if n_uniq == 2:
        return "binary"
    if n_uniq <= max_classes:
        return "multiclass"
    return "regression"


def _selector(task, k):
    score_func = f_regression if task == "regression" else f_classif
    return SelectKBest(score_func, k=k)


def _estimator(task):
    if task == "regression":
        return Ridge()
    return LogisticRegression(max_iter=2000)


def _cv_splitter(task, groups, n_splits, seed):
    """Return (splitter, uses_groups). GroupKFold when subjects are supplied;
    StratifiedKFold for classification; KFold for regression."""
    if groups is not None:
        return GroupKFold(n_splits=n_splits), True
    if task == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed), False
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), False


def _split_iter(task, X, y, groups, n_splits, seed):
    splitter, uses_groups = _cv_splitter(task, groups, n_splits, seed)
    if uses_groups:
        return splitter.split(X, y, groups=np.asarray(groups))
    return splitter.split(X, y)


def _aligned_proba(est, Xte, classes):
    """predict_proba aligned to the GLOBAL class order (a training fold may miss
    a rare class)."""
    proba = est.predict_proba(Xte)
    full = np.zeros((Xte.shape[0], len(classes)))
    cls_index = {c: j for j, c in enumerate(classes)}
    for j, c in enumerate(est.classes_):
        full[:, cls_index[c]] = proba[:, j]
    return full


def task_metrics(y_true, pred, task, classes=None):
    """Compute the per-task metric bundle.

    `pred` is: regression -> point predictions; binary -> P(positive class);
    multiclass -> (n, n_classes) probability matrix aligned to `classes`.
    Returns a dict; `headline` is the single number used for the naive/honest
    ladder (R^2, AUC, or macro-OVR AUC).
    """
    y_true = np.asarray(y_true)
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
        m = {"r2": float(r2_score(y_true, pred)), "rmse": rmse,
             "mae": float(mean_absolute_error(y_true, pred))}
        m["headline"] = m["r2"]; m["headline_name"] = "R2"
        return m
    if task == "binary":
        pred = np.asarray(pred)
        m = {"auc": float(roc_auc_score(y_true, pred)),
             "log_loss": float(log_loss(y_true, np.clip(pred, 1e-6, 1 - 1e-6)))}
        # Brier for the positive class
        pos = classes[1] if classes is not None else np.unique(y_true)[1]
        yb = (y_true == pos).astype(float)
        m["brier"] = float(np.mean((pred - yb) ** 2))
        m["headline"] = m["auc"]; m["headline_name"] = "AUC"
        return m
    # multiclass
    classes = np.unique(y_true) if classes is None else np.asarray(classes)
    P = np.asarray(pred)
    auc = float(roc_auc_score(y_true, P, multi_class="ovr", average="macro",
                              labels=list(classes)))
    ll = float(log_loss(y_true, P, labels=list(classes)))
    yhat = classes[np.argmax(P, axis=1)]
    Y = np.zeros_like(P)
    for j, c in enumerate(classes):
        Y[y_true == c, j] = 1.0
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    m = {"auc_ovr_macro": auc, "log_loss": ll,
         "balanced_accuracy": float(balanced_accuracy_score(y_true, yhat)),
         "brier": brier}
    m["headline"] = auc; m["headline_name"] = "AUC (macro-OVR)"
    return m


def _fold_pred(est, task, Xte, classes):
    if task == "regression":
        return est.predict(Xte)
    if task == "binary":
        proba = est.predict_proba(Xte)
        pos_col = list(est.classes_).index(classes[1])
        return proba[:, pos_col]
    return _aligned_proba(est, Xte, classes)


# ----------------------------------------------------------------------------
# Generalized naive vs honest CV (any task)
# ----------------------------------------------------------------------------

def naive_cv_task(X, y, task=None, k_features=20, n_splits=5, seed=0):
    """LEAKY pipeline for any task: scaling + feature selection fit on the FULL
    dataset, then cross-validate. Returns per-fold headline scores."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    task = task or detect_task(y)
    classes = None if task == "regression" else np.unique(y)
    Xs = StandardScaler().fit_transform(X)
    sel = _selector(task, min(k_features, X.shape[1])).fit(Xs, y)   # leak
    Xsel = sel.transform(Xs)
    scores = []
    for tr, te in _split_iter(task, Xsel, y, None, n_splits, seed):
        est = clone(_estimator(task)).fit(Xsel[tr], y[tr])
        pred = _fold_pred(est, task, Xsel[te], classes)
        scores.append(task_metrics(y[te], pred, task, classes)["headline"])
    return np.array(scores)


def honest_cv_task(X, y, task=None, groups=None, k_features=20, n_splits=5, seed=0):
    """Leakage-safe pipeline for any task: scaling + feature selection fit
    INSIDE each training fold; GroupKFold when subjects are supplied."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    task = task or detect_task(y)
    classes = None if task == "regression" else np.unique(y)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("select", _selector(task, min(k_features, X.shape[1]))),
                     ("est", _estimator(task))])
    scores = []
    for tr, te in _split_iter(task, X, y, groups, n_splits, seed):
        est = clone(pipe).fit(X[tr], y[tr])
        pred = _fold_pred(est, task, X[te], classes)
        scores.append(task_metrics(y[te], pred, task, classes)["headline"])
    return np.array(scores)


def oof_predict_task(X, y, task=None, groups=None, k_features=20, n_splits=5, seed=0):
    """Honest out-of-fold predictions + full metric bundle for any task."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    task = task or detect_task(y)
    classes = None if task == "regression" else np.unique(y)
    n = len(y)
    if task == "multiclass":
        oof = np.zeros((n, len(classes)))
    else:
        oof = np.full(n, np.nan)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("select", _selector(task, min(k_features, X.shape[1]))),
                     ("est", _estimator(task))])
    for tr, te in _split_iter(task, X, y, groups, n_splits, seed):
        est = clone(pipe).fit(X[tr], y[tr])
        pred = _fold_pred(est, task, X[te], classes)
        if task == "multiclass":
            oof[te, :] = pred
        else:
            oof[te] = pred
    metrics = task_metrics(y, oof, task, classes)
    return {"task": task, "oof": oof, "metrics": metrics,
            "classes": None if classes is None else classes.tolist()}


# ----------------------------------------------------------------------------
# Nested CV -- hyperparameter-selection honesty
# ----------------------------------------------------------------------------

def _scoring(task):
    if task == "regression":
        return "r2"
    if task == "binary":
        return "roc_auc"
    return "roc_auc_ovr"


def _tuning_pipeline(task, p):
    return Pipeline([("scale", StandardScaler()),
                     ("select", _selector(task, min(20, p))),
                     ("est", _estimator(task))])


def default_param_grid(task, p):
    """A small, sensible grid over feature count + model regularization."""
    ks = sorted({k for k in [5, 10, 20, 50] if k <= p}) or [min(5, p)]
    if task == "regression":
        return {"select__k": ks, "est__alpha": [0.1, 1.0, 10.0, 100.0]}
    return {"select__k": ks, "est__C": [0.01, 0.1, 1.0, 10.0]}


def naive_tuned_cv(X, y, task=None, param_grid=None, groups=None,
                   n_splits=5, seed=0):
    """The hyperparameter leak: run ONE GridSearchCV over the whole dataset and
    report its best cross-validated score. That number is optimistic -- the grid
    was chosen to maximize this very score. Returns (best_score, best_params)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    task = task or detect_task(y)
    grid = param_grid or default_param_grid(task, X.shape[1])
    splitter, uses_groups = _cv_splitter(task, groups, n_splits, seed)
    gs = GridSearchCV(_tuning_pipeline(task, X.shape[1]), grid,
                      scoring=_scoring(task), cv=splitter, n_jobs=1, refit=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if uses_groups:
            gs.fit(X, y, groups=np.asarray(groups))
        else:
            gs.fit(X, y)
    return float(gs.best_score_), gs.best_params_


def nested_cv(X, y, task=None, param_grid=None, groups=None,
              n_splits=5, inner_splits=3, seed=0):
    """Honest hyperparameter estimate: an inner GridSearchCV tunes on each outer
    training fold only; the outer fold is scored with the tuned model. Returns
    per-outer-fold headline scores + the params chosen in each fold."""
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    task = task or detect_task(y)
    grid = param_grid or default_param_grid(task, X.shape[1])
    classes = None if task == "regression" else np.unique(y)
    outer_scores, chosen = [], []
    groups_arr = None if groups is None else np.asarray(groups)
    for tr, te in _split_iter(task, X, y, groups, n_splits, seed):
        inner_groups = None if groups_arr is None else groups_arr[tr]
        inner_splitter, inner_uses_groups = _cv_splitter(
            task, inner_groups, inner_splits, seed)
        gs = GridSearchCV(_tuning_pipeline(task, X.shape[1]), grid,
                          scoring=_scoring(task), cv=inner_splitter,
                          n_jobs=1, refit=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if inner_uses_groups:
                gs.fit(X[tr], y[tr], groups=inner_groups)
            else:
                gs.fit(X[tr], y[tr])
        pred = _fold_pred(gs.best_estimator_, task, X[te], classes)
        outer_scores.append(task_metrics(y[te], pred, task, classes)["headline"])
        chosen.append(gs.best_params_)
    return np.array(outer_scores), chosen
