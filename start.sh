#!/usr/bin/env bash
# Standard way to launch the speakIelts web UI.
# Serves the React app AND the voice WebSocket on a single port (:8080).
set -e

cd "$(dirname "$0")"

# Use the venv's Python/fastapi if present.
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Build the UI only if it hasn't been built yet.
if [ ! -f web/dist/index.html ]; then
  echo "web/dist not found — building the UI (first time, this takes a minute)…"
  (cd web && npm install && npm run build)
fi

echo "Starting server at http://localhost:8080  (Ctrl+C to stop)"
# Pass server.py explicitly so discovery can never mistake main.py (the
# terminal script, which is NOT a FastAPI app) for the web entrypoint.
exec python -m fastapi dev server.py --port 8080
