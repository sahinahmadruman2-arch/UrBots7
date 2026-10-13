#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export KARLLO_ROOT="$ROOT"
export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
echo "Starting Karllo API on :8000 and UI on :5173"
python3 -m uvicorn karllo.api.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT
cd "$ROOT/frontend"
npm run dev
