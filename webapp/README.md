# The Trust Layer — Web UI

A modern web interface over the Trust Layer leakage-audit engine
(`code/trust_audit.py`). Upload your data (and optionally your analysis script),
click **Analyze**, and get back the naive-vs-honest verdict, the leakage ladder,
code + data audits, calibrated model performance, and the full report — no code.

## Run

One command — creates `.venv`, installs pinned deps, and serves:

```bash
./webapp/run.sh              # http://localhost:8000
PORT=9000 ./webapp/run.sh    # custom port
```

Or manually, from the **repo root** (so `webapp.app` resolves):

```bash
uv pip install --python .venv/bin/python -r webapp/requirements.txt   # first time
.venv/bin/uvicorn webapp.app:app --port 8000
```

Then open http://localhost:8000.

> Always run from the **repo root**, not inside `webapp/`. The backend adds
> `../code` to the path and imports the Trust Layer modules in their required
> order (`trust_layer → notebook_audit → trust_tasks → trust_audit`).

Dependencies are pinned in `requirements.txt` (tested on Python 3.12).

## Using it

- **Try sample data** — runs the bundled leaky serum-proteomics benchmark
  (expect ~0.98 → ~0.87, three code leaks, four critical data findings).
- **Try a clean dataset** — the clean counterpart (expect ~0.98 → ~0.96, no leaks).
- **Your own data:**
  - *Expression matrix* (required): CSV, rows = samples, columns = features. An
    `sample_id`/id column is dropped automatically; non-numeric columns are ignored.
  - *Metadata* (optional): CSV aligned by row order, holding the outcome, subject,
    and batch columns. After you drop it in, pick the columns from the dropdowns.
    Providing subject and batch unlocks the repeated-measures and batch-confound
    checks — the tool's headline value.
  - *Analysis script* (optional): a `.py` or `.ipynb`. It is **statically** audited
    (AST only) — no uploaded code is ever executed.

## API

Audits run as background **jobs** so the server stays responsive and the UI can
show real progress:

- `POST /api/analyze` — multipart: `expression` (CSV, required), `metadata` (CSV),
  `script` (.py/.ipynb); form fields `label_col`, `group_col`, `batch_col`,
  `model` (`auto`/`logreg`/`hgb`), `nested` (bool). Validates the input, then
  returns `{"job_id": "..."}` (or a `400` with a clear message).
- `POST /api/sample?clean=0|1` — runs the bundled leaky/clean sample dataset;
  also returns `{"job_id": ...}`.
- `GET /api/progress/{job_id}` — `{status, stage, pct}` while running; adds
  `result` when `status == "done"` or `error` when `status == "error"`.

Audits are serialized by a lock (they're CPU-bound) and each runs in its own
thread; the reported stages are wrapped around the real engine calls, so the
progress bar reflects actual work. The backend is a thin wrapper — it does not
modify any audit code in `../code`.
