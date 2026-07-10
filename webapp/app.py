"""
The Trust Layer — web UI backend.

A thin FastAPI wrapper around the existing `trust_audit.trust_audit()` engine.
No audit code is changed here; this only handles uploads, column mapping, and
JSON serialization of the result for the frontend.
"""
from __future__ import annotations

import io
import os
import sys
import time
import uuid
import tempfile
import threading
import importlib.util

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------------
# Load the Trust Layer modules in their required sibling order (see CLAUDE.md).
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _load_trust_layer():
    mods = {}
    for name in ("trust_layer", "notebook_audit", "trust_tasks", "trust_audit"):
        path = os.path.join(CODE, f"{name}.py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


MODS = _load_trust_layer()
tl, na, tt, trust_audit = (MODS["trust_layer"], MODS["notebook_audit"],
                           MODS["trust_tasks"], MODS["trust_audit"])

app = FastAPI(title="The Trust Layer")


@app.middleware("http")
async def no_cache(request, call_next):
    """Never let the browser cache the UI assets — during development stale
    HTML/CSS/JS is a constant source of confusion."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _to_native(obj):
    """Recursively convert numpy scalars/arrays to JSON-native Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    return obj


def _read_csv(upload: UploadFile) -> pd.DataFrame:
    raw = upload.file.read()
    return pd.read_csv(io.BytesIO(raw))


def _derive_y(series: pd.Series):
    """Two-value categorical -> 0/1 (positive = a disease-like label when present).
    Non-numeric with >2 classes -> integer codes (multiclass). Numeric -> passed
    through for the task auto-detector (binary / multiclass / regression)."""
    s = series
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    uniq_vals = pd.unique(s.dropna())
    if numeric and len(uniq_vals) > 2:
        return s.values, None                      # multiclass ints / regression
    if numeric and len(uniq_vals) == 2:
        # already numeric two-class: normalize to 0/1
        lo, hi = sorted(uniq_vals)
        return (s == hi).astype(int).values, str(hi)
    # non-numeric (string/category/bool)
    uniq = sorted(map(str, uniq_vals))
    if len(uniq) == 2:
        pos = next((u for u in uniq if u.lower() in
                    ("disease", "case", "positive", "pos", "1", "true", "yes",
                     "tumor", "cancer", "abnormal")), uniq[-1])
        return (s.astype(str) == pos).astype(int).values, pos
    return s.astype(str).astype("category").cat.codes.values, None


def _feature_matrix(df: pd.DataFrame, drop_cols) -> pd.DataFrame:
    """Drop id / mapping columns, keep numeric feature columns only."""
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # drop obvious id columns and any non-numeric leftovers
    for c in list(X.columns):
        if c.lower() in ("sample_id", "id", "index", "unnamed: 0"):
            X = X.drop(columns=[c])
    X = X.select_dtypes(include=[np.number])
    return X


# --------------------------------------------------------------------------
# Job registry + real, non-blocking progress.
#
# Each audit runs in its own thread so the event loop stays responsive. A
# single AUDIT_LOCK serializes audits — they are CPU-bound (running several at
# once would only thrash), and serializing also makes it safe to temporarily
# wrap the engine's functions to report genuine stage progress.
# --------------------------------------------------------------------------
JOBS = {}
JOBS_LOCK = threading.Lock()
AUDIT_LOCK = threading.Lock()

# (module, function name, caption shown to the user, target %) — reached when
# that stage of the real audit actually begins. Names not present are skipped.
STAGES = [
    (na, "audit_notebook",         "Auditing your code…", 10),
    (tl, "audit_leakage",          "Checking the data against the leakage taxonomy…", 22),
    (tl, "naive_cv",               "Running naive vs honest cross-validation…", 38),
    (tt, "naive_cv_task",          "Running naive vs honest cross-validation…", 38),
    (tl, "fit_trust_model",        "Fitting a calibrated model, inside each fold…", 62),
    (tt, "oof_predict_task",       "Fitting the model, inside each fold…", 62),
    (tl, "nested_model_selection", "Measuring model-selection optimism…", 80),
    (tt, "nested_cv",              "Measuring hyperparameter-tuning optimism…", 90),
]


def _prune():
    """Drop old finished jobs so the registry doesn't grow unbounded."""
    if len(JOBS) <= 60:
        return
    stale = sorted((j for j in JOBS.values() if j["status"] in ("done", "error")),
                   key=lambda j: j.get("finished", 0))
    for j in stale[:len(JOBS) - 60]:
        JOBS.pop(j["id"], None)


def _new_job():
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {"id": jid, "status": "queued", "stage": "Queued…", "pct": 3}
        _prune()
    return jid


def _set(jid, **kw):
    with JOBS_LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)


def _run_job(jid, X, y, groups, batch, notebook_code, model, nested, input_meta):
    queued_since = time.time()
    # If another audit is in flight, reflect that instead of a frozen spinner.
    if AUDIT_LOCK.locked():
        _set(jid, stage="Waiting for another audit to finish…", pct=4)
    with AUDIT_LOCK:
        _set(jid, status="running", stage="Auditing your code…", pct=6)
        originals = []
        try:
            for mod, name, stage, pct in STAGES:
                if not hasattr(mod, name):
                    continue
                orig = getattr(mod, name)

                def make(orig, stage, pct):
                    def wrapper(*a, **k):
                        _set(jid, stage=stage, pct=pct)
                        return orig(*a, **k)
                    return wrapper

                setattr(mod, name, make(orig, stage, pct))
                originals.append((mod, name, orig))

            with tempfile.TemporaryDirectory() as outdir:
                res = trust_audit.trust_audit(
                    X, y, groups=groups, batch=batch, notebook_code=notebook_code,
                    model=model, nested=nested, outdir=outdir)
            res.pop("oof_proba", None)          # large, not needed by the UI
            res = _to_native(res)
            res["input"] = input_meta
            _set(jid, status="done", stage="Done", pct=100, result=res,
                 finished=time.time())
        except Exception as e:
            _set(jid, status="error", pct=100, finished=time.time(),
                 error=f"Audit failed: {type(e).__name__}: {e}")
        finally:
            for mod, name, orig in originals:
                setattr(mod, name, orig)


def _start_job(X, y, groups, batch, notebook_code, model, nested, input_meta):
    jid = _new_job()
    threading.Thread(
        target=_run_job,
        args=(jid, X, y, groups, batch, notebook_code, model, nested, input_meta),
        daemon=True,
    ).start()
    return jid


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.post("/api/analyze")
async def analyze(
    expression: UploadFile = File(...),
    metadata: UploadFile = File(None),
    script: UploadFile = File(None),
    label_col: str = Form(None),
    group_col: str = Form(None),
    batch_col: str = Form(None),
    model: str = Form("auto"),
    nested: bool = Form(True),
):
    try:
        expr_df = _read_csv(expression)
    except Exception as e:
        return JSONResponse({"error": f"Could not read expression CSV: {e}"}, 400)

    groups = batch = None
    pos_label = None

    if metadata is not None:
        try:
            meta = _read_csv(metadata)
        except Exception as e:
            return JSONResponse({"error": f"Could not read metadata CSV: {e}"}, 400)
        if len(meta) != len(expr_df):
            return JSONResponse(
                {"error": f"Row mismatch: expression has {len(expr_df)} rows, "
                          f"metadata has {len(meta)}. They must align by row."}, 400)
        if not label_col or label_col not in meta.columns:
            return JSONResponse(
                {"error": f"Label column '{label_col}' not found in metadata. "
                          f"Available: {list(meta.columns)}"}, 400)
        y, pos_label = _derive_y(meta[label_col])
        if group_col and group_col in meta.columns:
            groups = meta[group_col].values
        if batch_col and batch_col in meta.columns:
            batch = meta[batch_col].values
        X = _feature_matrix(expr_df, drop_cols=[label_col, group_col, batch_col])
    else:
        # no metadata: label must be a column of the expression file
        if not label_col or label_col not in expr_df.columns:
            return JSONResponse(
                {"error": "No metadata uploaded, so the label must be a column in "
                          f"the data file. Column '{label_col}' not found. "
                          f"Available: {list(expr_df.columns)}"}, 400)
        y, pos_label = _derive_y(expr_df[label_col])
        X = _feature_matrix(expr_df, drop_cols=[label_col])

    if X.shape[1] == 0:
        return JSONResponse({"error": "No numeric feature columns found."}, 400)

    notebook_code = None
    if script is not None:
        raw = script.file.read()
        try:
            notebook_code = raw.decode("utf-8")
        except Exception:
            notebook_code = raw.decode("latin-1", errors="ignore")
        if (script.filename or "").endswith(".ipynb"):
            # let notebook_audit parse the .ipynb via a temp path instead
            notebook_code = _ipynb_to_code(notebook_code)

    input_meta = {
        "n": int(X.shape[0]), "p": int(X.shape[1]),
        "label_col": label_col, "positive_label": pos_label,
        "group_col": group_col if groups is not None else None,
        "batch_col": batch_col if batch is not None else None,
        "has_script": script is not None,
    }
    jid = _start_job(X, y, groups, batch, notebook_code, model, nested, input_meta)
    return JSONResponse({"job_id": jid})


def _ipynb_to_code(text: str) -> str:
    import json
    try:
        nb = json.loads(text)
    except Exception:
        return text
    cells = []
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            src = c.get("source", [])
            cells.append("".join(src) if isinstance(src, list) else src)
    return "\n\n".join(cells)


@app.post("/api/sample")
async def sample(clean: int = 0):
    """Run the audit on the bundled sample dataset (leaky by default)."""
    d = os.path.join(CODE, "sample_dataset_clean" if clean else "sample_dataset")
    if not os.path.isdir(d):
        return JSONResponse({"error": f"Sample dataset not found at {d}."}, 404)
    expr = pd.read_csv(os.path.join(d, "expression_matrix.csv"))
    meta = pd.read_csv(os.path.join(d, "sample_metadata.csv"))
    y, pos = _derive_y(meta["diagnosis"])
    X = _feature_matrix(expr, drop_cols=["diagnosis", "patient_id", "plate"])
    nb_name = "analysis_clean.py" if clean else "analysis_leaky.py"
    nb_path = os.path.join(d, nb_name)
    code = open(nb_path).read() if os.path.exists(nb_path) else None
    input_meta = {
        "n": int(X.shape[0]), "p": int(X.shape[1]),
        "label_col": "diagnosis", "positive_label": pos,
        "group_col": "patient_id", "batch_col": "plate",
        "has_script": code is not None,
        "sample": "clean" if clean else "leaky",
    }
    jid = _start_job(X, y, meta["patient_id"].values, meta["plate"].values,
                     code, "auto", True, input_meta)
    return JSONResponse({"job_id": jid})


@app.get("/api/progress/{jid}")
def progress(jid: str):
    with JOBS_LOCK:
        job = JOBS.get(jid)
        job = dict(job) if job else None
    if job is None:
        return JSONResponse({"error": "Unknown job id."}, 404)
    out = {"status": job["status"], "stage": job.get("stage"), "pct": job.get("pct", 0)}
    if job["status"] == "done":
        out["result"] = job["result"]
    elif job["status"] == "error":
        out["error"] = job.get("error", "Audit failed.")
    return JSONResponse(out)


app.mount("/", StaticFiles(directory=STATIC), name="static")
