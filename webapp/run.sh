#!/usr/bin/env bash
# Launch the Trust Layer web UI.
#
#   ./webapp/run.sh            # http://localhost:8000
#   PORT=9000 ./webapp/run.sh  # custom port
#
# Creates ./.venv on first run (via uv if available, else python3 -m venv),
# installs pinned deps, then serves the app. Run from anywhere in the repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8000}"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "Creating virtualenv at .venv …"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

echo "Installing dependencies …"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" -r "$ROOT/webapp/requirements.txt" -q
else
  "$PY" -m pip install -q -r "$ROOT/webapp/requirements.txt"
fi

echo "Serving on http://localhost:$PORT  (Ctrl+C to stop)"
exec "$VENV/bin/uvicorn" webapp.app:app --host 0.0.0.0 --port "$PORT"
