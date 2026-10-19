#!/usr/bin/env bash
set -euo pipefail

SIDECAR_PID=""

if [[ "${USE_INDIC_PARLER:-1}" != "0" ]]; then
  /opt/parler-venv/bin/python3 /app/indic_parler_sidecar.py &
  SIDECAR_PID=$!
fi

cleanup() {
  if [[ -n "${SIDECAR_PID}" ]]; then
    kill "${SIDECAR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

exec uvicorn salad_app:app --host "::" --port 8000
