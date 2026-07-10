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
from __future__ import annotations

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
