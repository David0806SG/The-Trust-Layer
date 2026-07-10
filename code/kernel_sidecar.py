"""Trust-layer sidecar: leakage audit + honest CV (binary/multiclass/regression) + nested CV + calibrated linear/nonlinear models + notebook analysis."""

def build_trust_namespace():
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


    """
    notebook_audit.py
    =================
    Static leakage analyzer for a scientist's analysis notebook or script.

    The trust-layer's dataset audit (trust_layer.audit_leakage) can only *ask* the
    analyst "did you fit preprocessing before the split?". This module answers that
    question automatically by reading the code: it parses a .ipynb / .py into an AST,
    reconstructs the linear order of operations, locates where the train/test split
    (or CV loop) happens, and flags any preprocessing, feature selection, imputation,
    or resampling that was *fit on the full data before* that split.

    It is a static analyzer -- no code is executed. It reports line-anchored findings
    mapped to the same leakage taxonomy as trust_layer, so the two audits compose.

    Detected patterns
    -----------------
      * leaky_preprocessing     -- Scaler/Imputer/Normalizer .fit()/.fit_transform()
                                   on the full data before the split
      * leaky_feature_selection -- SelectKBest/RFE/etc. or a y-referencing selection
                                   fit before the split
      * leaky_resampling        -- SMOTE/over/under-sampling applied before the split
      * subject_leakage         -- a subject/patient/id column is present but the CV
                                   splitter is KFold/StratifiedKFold (not GroupKFold)
      * no_split_found          -- no train/test split or CV detected at all
      * good_practice           -- Pipeline / inside-CV usage that AVOIDS a leak
    """

    import ast
    import json
    import re
    from dataclasses import dataclass, field
    from typing import Optional


    # --- vocabulary -------------------------------------------------------------
    SPLIT_FUNCS = {"train_test_split"}
    CV_SPLITTERS = {"KFold", "StratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit",
                    "RepeatedKFold", "RepeatedStratifiedKFold", "LeaveOneOut"}
    GROUP_SPLITTERS = {"GroupKFold", "StratifiedGroupKFold", "LeaveOneGroupOut",
                       "GroupShuffleSplit"}
    CV_HELPERS = {"cross_val_score", "cross_validate", "cross_val_predict"}

    SCALERS = {"StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
               "Normalizer", "PowerTransformer", "QuantileTransformer"}
    IMPUTERS = {"SimpleImputer", "KNNImputer", "IterativeImputer"}
    SELECTORS = {"SelectKBest", "SelectPercentile", "SelectFdr", "SelectFpr",
                 "SelectFwe", "RFE", "RFECV", "SelectFromModel", "VarianceThreshold",
                 "SequentialFeatureSelector", "GenericUnivariateSelect"}
    DECOMP = {"PCA", "TruncatedSVD", "KernelPCA", "FastICA", "NMF"}
    RESAMPLERS = {"SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler",
                  "BorderlineSMOTE", "SMOTEENN", "SMOTETomek", "NearMiss"}
    GROUP_COL_HINTS = re.compile(r"(subject|patient|donor|animal|mouse|participant|"
                                 r"individual|sample_id|subj|pid|group)", re.I)


    @dataclass
    class NBFinding:
        check: str
        severity: str            # "critical" | "warning" | "ok" | "info"
        line: Optional[int]
        message: str
        code: str = ""

        def as_dict(self):
            return {"check": self.check, "severity": self.severity,
                    "line": self.line, "message": self.message, "code": self.code}


    # ----------------------------------------------------------------------------
    # notebook / script -> flat source with cell-aware line numbers
    # ----------------------------------------------------------------------------

    def _load_source(path: str) -> str:
        """Return the concatenated code of a .ipynb (code cells only) or a .py file."""
        if path.endswith(".ipynb"):
            nb = json.load(open(path))
            cells = []
            for c in nb.get("cells", []):
                if c.get("cell_type") == "code":
                    src = c.get("source", [])
                    cells.append("".join(src) if isinstance(src, list) else src)
            return "\n\n".join(cells)
        return open(path).read()


    def source_from_string(code: str) -> str:
        return code


    # ----------------------------------------------------------------------------
    # AST walk
    # ----------------------------------------------------------------------------

    class _LeakageVisitor(ast.NodeVisitor):
        def __init__(self, source_lines):
            self.lines = source_lines
            # name -> constructor class (e.g. sc -> "StandardScaler")
            self.var_class = {}
            # ordered list of (lineno, kind, class_or_func, node)
            self.events = []
            self.split_line = None          # first train/test split OR cv .split()
            self.cv_splitter_used = None    # class name of splitter feeding a loop
            self.uses_pipeline = False
            self.group_col_seen = False
            self.groups_kw_seen = False

        # -- track constructor assignments: sc = StandardScaler() ---------------
        def visit_Assign(self, node):
            cls = self._constructor_name(node.value)
            if cls:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        self.var_class[tgt.id] = cls
            # detect a groups/subject column being pulled out: groups = df['subject']
            self._scan_group_column(node)
            self.generic_visit(node)

        def _scan_group_column(self, node):
            txt = self._seg(node)
            if txt and GROUP_COL_HINTS.search(txt):
                # only count if it looks like a column/array assignment, not a comment
                if "=" in txt and ("[" in txt or ".loc" in txt or "groups" in txt.lower()):
                    self.group_col_seen = True

        @staticmethod
        def _constructor_name(value):
            if isinstance(value, ast.Call):
                f = value.func
                if isinstance(f, ast.Name):
                    return f.id
                if isinstance(f, ast.Attribute):
                    return f.attr
            return None

        def _seg(self, node):
            try:
                return ast.get_source_segment("\n".join(self.lines), node)
            except Exception:
                return None

        # -- every call: classify it -------------------------------------------
        def visit_Call(self, node):
            fname = None
            recv = None
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
                if isinstance(node.func.value, ast.Name):
                    recv = node.func.value.id

            ln = getattr(node, "lineno", None)

            # pipeline / make_pipeline anywhere -> good practice signal
            if fname in ("Pipeline", "make_pipeline", "make_column_transformer",
                         "ColumnTransformer"):
                self.uses_pipeline = True

            # split detection
            if fname in SPLIT_FUNCS:
                if self.split_line is None or ln < self.split_line:
                    self.split_line = ln
                self.events.append((ln, "split", fname, node))
            if fname in CV_SPLITTERS:
                self.cv_splitter_used = fname
                self.events.append((ln, "cv_splitter", fname, node))
            if fname in GROUP_SPLITTERS:
                self.cv_splitter_used = fname
                self.events.append((ln, "group_splitter", fname, node))
            if fname in CV_HELPERS:
                self.events.append((ln, "cv_helper", fname, node))
                # groups= passed to cross_validate?
                if any(kw.arg == "groups" for kw in node.keywords):
                    self.groups_kw_seen = True
            if fname == "split" and recv is not None:
                # splitter.split(...) -- the CV loop start
                if self.split_line is None:
                    self.split_line = ln
                self.events.append((ln, "cv_split_call", recv, node))
                if any(kw.arg == "groups" for kw in node.keywords) or \
                   (len(node.args) >= 3):   # split(X, y, groups)
                    self.groups_kw_seen = True

            # fit / fit_transform on a tracked transformer
            if fname in ("fit", "fit_transform") and recv is not None:
                cls = self.var_class.get(recv)
                if cls:
                    kind = self._transformer_kind(cls)
                    if kind:
                        self.events.append((ln, kind, cls, node))
            # inline constructor .fit_transform(): StandardScaler().fit_transform(X)
            if fname in ("fit", "fit_transform") and isinstance(node.func, ast.Attribute):
                inner = node.func.value
                cls = self._constructor_name(inner)
                if cls:
                    kind = self._transformer_kind(cls)
                    if kind:
                        self.events.append((ln, kind, cls, node))

            # resamplers: fit_resample
            if fname == "fit_resample" and recv is not None:
                cls = self.var_class.get(recv, recv)
                self.events.append((ln, "resample", cls, node))
            if fname in RESAMPLERS:
                # SMOTE() constructed; look for fit_resample separately, but flag construct too
                pass

            # groups kwarg on train_test_split-like or GridSearchCV.fit(groups=)
            if any(kw.arg == "groups" for kw in node.keywords):
                self.groups_kw_seen = True

            self.generic_visit(node)

        @staticmethod
        def _transformer_kind(cls):
            if cls in SCALERS:
                return "scale"
            if cls in IMPUTERS:
                return "impute"
            if cls in SELECTORS:
                return "select"
            if cls in DECOMP:
                return "decompose"
            if cls in RESAMPLERS:
                return "resample"
            return None


    # ----------------------------------------------------------------------------
    # main entry
    # ----------------------------------------------------------------------------

    def audit_notebook(path: str = None, code: str = None):
        """
        Statically audit a notebook/script for data-leakage patterns.
        Provide either `path` (.ipynb or .py) or raw `code`.
        Returns list[NBFinding].
        """
        src = source_from_string(code) if code is not None else _load_source(path)
        lines = src.split("\n")
        findings = []

        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            return [NBFinding("parse_error", "info", getattr(e, "lineno", None),
                              f"Could not parse source: {e}")]

        v = _LeakageVisitor(lines)
        v.visit(tree)

        kind_labels = {
            "scale":      ("leaky_preprocessing", "Scaling"),
            "impute":     ("leaky_preprocessing", "Imputation"),
            "decompose":  ("leaky_preprocessing", "Dimensionality reduction (PCA/SVD)"),
            "select":     ("leaky_feature_selection", "Feature selection"),
            "resample":   ("leaky_resampling", "Class resampling (SMOTE/over/under)"),
        }

        split_ln = v.split_line
        # -- no split at all ----------------------------------------------------
        if split_ln is None and not any(e[1] in ("cv_helper",) for e in v.events):
            findings.append(NBFinding(
                "no_split", "critical", None,
                "No train/test split or cross-validation was detected. Performance "
                "measured on training data alone is not an estimate of generalization."))

        # -- preprocessing/selection/resampling fit BEFORE the split -----------
        # If a Pipeline is used AND fits happen after split, that's the safe pattern.
        for ln, kind, cls, node in v.events:
            if kind not in kind_labels:
                continue
            taxonomy, human = kind_labels[kind]
            code_seg = lines[ln - 1].strip() if ln and ln <= len(lines) else ""
            before_split = (split_ln is not None and ln < split_ln)
            no_split_ref = (split_ln is None)
            if before_split or no_split_ref:
                sev = "critical"
                if kind == "select":
                    why = ("selecting features on the full dataset lets test-set signal "
                           "pick the features. With p>>n this alone can manufacture a "
                           "high AUC from noise.")
                elif kind == "resample":
                    why = ("resampling before the split copies/synthesizes rows that can "
                           "land in both train and test.")
                elif kind == "decompose":
                    why = ("fitting PCA/SVD on all rows lets test rows shape the "
                           "components used to train.")
                else:
                    why = ("the transform is fit using test rows, so test information "
                           "leaks into training.")
                loc = f"before the train/test split (line {split_ln})" if before_split \
                      else "with no train/test split in scope"
                findings.append(NBFinding(
                    taxonomy, sev, ln,
                    f"{human} is fit on the full data {loc}: {why}",
                    code=code_seg))
            else:
                findings.append(NBFinding(
                    kind_labels[kind][0], "ok", ln,
                    f"{human} appears after the split (line {ln} > split line {split_ln}).",
                    code=code_seg))

        # -- subject/group leakage in the splitter ------------------------------
        used = v.cv_splitter_used
        if v.group_col_seen and not v.groups_kw_seen:
            if used in CV_SPLITTERS or (used is None and split_ln is not None):
                findings.append(NBFinding(
                    "subject_leakage", "critical", split_ln,
                    "A subject/patient-like column is present but the split is not "
                    "group-aware (no GroupKFold / groups= argument). Rows from the same "
                    "subject can straddle train and test. Use GroupKFold on the subject ID."))
            elif used in GROUP_SPLITTERS or v.groups_kw_seen:
                pass
        elif used in GROUP_SPLITTERS or v.groups_kw_seen:
            findings.append(NBFinding(
                "subject_leakage", "ok", split_ln,
                f"Group-aware splitting detected ({used or 'groups= argument'})."))

        # -- pipeline good-practice note ---------------------------------------
        if v.uses_pipeline:
            findings.append(NBFinding(
                "good_practice", "info", None,
                "A scikit-learn Pipeline / ColumnTransformer is used -- when the whole "
                "pipeline is fit inside CV, preprocessing stays leak-free."))

        # -- de-dup identical (check,line,message) -----------------------------
        seen, out = set(), []
        for f in findings:
            key = (f.check, f.line, f.message)
            if key not in seen:
                seen.add(key); out.append(f)
        return out


    def summarize(findings):
        crit = sum(f.severity == "critical" for f in findings)
        warn = sum(f.severity == "warning" for f in findings)
        return {"n_critical": crit, "n_warning": warn,
                "findings": [f.as_dict() for f in findings]}


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

    import os
    import json
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.feature_selection import f_classif



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
        task = task or detect_task(y)

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
            code_findings = audit_notebook(path=notebook, code=notebook_code)
            cf = summarize(code_findings)
            result["code_audit"] = cf
            # translate code findings into the data-audit's declared-leakage flags
            checks = {f.check for f in code_findings if f.severity == "critical"}
            code_flags["preprocessing_done_globally"] = "leaky_preprocessing" in checks
            code_flags["feature_selection_done_globally"] = "leaky_feature_selection" in checks

        # -- 2. DATA AUDIT ------------------------------------------------------
        audit = audit_leakage(
            X, y, groups=groups, batch=batch,
            feature_selection_done_globally=code_flags.get("feature_selection_done_globally"),
            preprocessing_done_globally=code_flags.get("preprocessing_done_globally"),
        )
        result["data_audit"] = {"n_critical": audit.n_critical, "n_warning": audit.n_warning,
                                "findings": [{"check": f.check, "severity": f.severity,
                                              "message": f.message} for f in audit.findings]}

        # -- 3. HONEST vs NAIVE CV ---------------------------------------------
        naive = naive_cv(X.values, y, k_features=k_features, n_splits=n_splits, seed=seed)
        ladder = {"naive": [float(naive.mean()), float(naive.std())]}
        if groups is not None:
            honest_grp = honest_cv(X.values, y, groups=groups, k_features=k_features,
                                      n_splits=n_splits, seed=seed)
            ladder["groupkfold"] = [float(honest_grp.mean()), float(honest_grp.std())]
        else:
            honest_grp = honest_cv(X.values, y, k_features=k_features,
                                      n_splits=n_splits, seed=seed)
            ladder["honest_randomsplit"] = [float(honest_grp.mean()), float(honest_grp.std())]
        if batch is not None:
            honest_full = honest_cv_batch(X.values, y, batch=batch, groups=groups,
                                             k_features=k_features, n_splits=n_splits, seed=seed)
            ladder["groupkfold_batchcorrected"] = [float(honest_full.mean()), float(honest_full.std())]
            v, pv = _cramers_v(batch, y)
            result["dataset"]["batch_cramers_v"] = v
        else:
            honest_full = honest_grp
        result["cv_ladder"] = ladder
        honest_best = float(honest_full.mean())

        # -- 4. TRUST MODEL: calibrated linear + nonlinear, honest OOF ---------
        res = fit_trust_model(X.values, y, groups=groups, batch=batch, n_splits=n_splits,
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
                outer_auc, picks = nested_model_selection(
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
            flat_score, flat_params = naive_tuned_cv(
                X.values, y, task="binary", groups=groups, n_splits=n_splits, seed=seed)
            nested_scores, _ = nested_cv(
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
            code_findings = audit_notebook(path=notebook, code=notebook_code)
            result["code_audit"] = summarize(code_findings)
            checks = {f.check for f in code_findings if f.severity == "critical"}
            code_flags["preprocessing_done_globally"] = "leaky_preprocessing" in checks
            code_flags["feature_selection_done_globally"] = "leaky_feature_selection" in checks

        # -- data audit (dimensionality / duplicates / subject leakage all task-agnostic) --
        audit = audit_leakage(
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
        naive = naive_cv_task(X.values, y, task=task, k_features=k_features,
                                 n_splits=n_splits, seed=seed)
        honest = honest_cv_task(X.values, y, task=task, groups=groups,
                                   k_features=k_features, n_splits=n_splits, seed=seed)
        hn = "R2" if task == "regression" else "AUC (macro-OVR)"
        result["cv_ladder"] = {"naive": [float(naive.mean()), float(naive.std())],
                               ("groupkfold" if groups is not None else "honest_randomsplit"):
                                   [float(honest.mean()), float(honest.std())]}
        result["headline_name"] = hn

        # -- honest OOF metric bundle --
        oof = oof_predict_task(X.values, y, task=task, groups=groups,
                                  k_features=min(50, p), n_splits=n_splits, seed=seed)
        result["model"] = {"name": "Ridge" if task == "regression" else "LogReg(multinomial)",
                           "metrics": oof["metrics"]}
        result["inflation"] = float(naive.mean() - oof["metrics"]["headline"])

        # -- nested CV --
        if nested:
            flat_score, flat_params = naive_tuned_cv(X.values, y, task=task, groups=groups,
                                                        n_splits=n_splits, seed=seed)
            nested_scores, _ = nested_cv(X.values, y, task=task, groups=groups,
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

    return {n: v for n, v in dict(locals()).items()}

TRUST_CACHE = {}

def trust_ns():
    if 'ns' not in TRUST_CACHE:
        TRUST_CACHE['ns'] = build_trust_namespace()
    return TRUST_CACHE['ns']

def audit_leakage(*a, **k):
    return trust_ns()['audit_leakage'](*a, **k)

def naive_cv(*a, **k):
    return trust_ns()['naive_cv'](*a, **k)

def honest_cv(*a, **k):
    return trust_ns()['honest_cv'](*a, **k)

def honest_cv_batch(*a, **k):
    return trust_ns()['honest_cv_batch'](*a, **k)

def BatchCenterer(*a, **k):
    return trust_ns()['BatchCenterer'](*a, **k)

def fit_trust_model(*a, **k):
    return trust_ns()['fit_trust_model'](*a, **k)

def nested_model_selection(*a, **k):
    return trust_ns()['nested_model_selection'](*a, **k)

def reliability_curve(*a, **k):
    return trust_ns()['reliability_curve'](*a, **k)

def AuditReport(*a, **k):
    return trust_ns()['AuditReport'](*a, **k)

def Finding(*a, **k):
    return trust_ns()['Finding'](*a, **k)

def NBFinding(*a, **k):
    return trust_ns()['NBFinding'](*a, **k)

def audit_notebook(*a, **k):
    return trust_ns()['audit_notebook'](*a, **k)

def summarize(*a, **k):
    return trust_ns()['summarize'](*a, **k)

def trust_audit(*a, **k):
    return trust_ns()['trust_audit'](*a, **k)

def detect_task(*a, **k):
    return trust_ns()['detect_task'](*a, **k)

def naive_cv_task(*a, **k):
    return trust_ns()['naive_cv_task'](*a, **k)

def honest_cv_task(*a, **k):
    return trust_ns()['honest_cv_task'](*a, **k)

def oof_predict_task(*a, **k):
    return trust_ns()['oof_predict_task'](*a, **k)

def task_metrics(*a, **k):
    return trust_ns()['task_metrics'](*a, **k)

def nested_cv(*a, **k):
    return trust_ns()['nested_cv'](*a, **k)

def naive_tuned_cv(*a, **k):
    return trust_ns()['naive_tuned_cv'](*a, **k)

def default_param_grid(*a, **k):
    return trust_ns()['default_param_grid'](*a, **k)


# ---------------------------------------------------------------------------
# Deprecated: the model no longer uses TabPFN; no external checkpoint needed.
# Kept as a no-op so existing calls don't break.
# ---------------------------------------------------------------------------
def ensure_tabpfn_checkpoint(cache_dir="./tabpfn_cache"):
    """Deprecated no-op. Returns None; fit_trust_model uses calibrated
    scikit-learn models directly."""
    return None
