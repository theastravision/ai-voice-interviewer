#!/usr/bin/env bash
# Run on EC2 (or anywhere) to verify Salad gateway exposes Free Chat + auth.
set -euo pipefail

GATEWAY="${SALAD_GATEWAY_URL:-}"
SECRET="${SALAD_API_SECRET:-}"

if [[ -z "$GATEWAY" || -z "$SECRET" ]]; then
  echo "Set SALAD_GATEWAY_URL and SALAD_API_SECRET (e.g. source ~/ai-voice-interview/.env)"
  exit 1
fi

GATEWAY="${GATEWAY%/}"
echo "=== GET $GATEWAY/health ==="
HEALTH_JSON="$(curl -sS "$GATEWAY/health")"
echo "$HEALTH_JSON" | python3 -m json.tool

API_VER="$(echo "$HEALTH_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('api_version',''))" 2>/dev/null || true)"
HAS_CHAT="$(echo "$HEALTH_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('has_free_chat', False))" 2>/dev/null || echo "False")"

if [[ "$HAS_CHAT" != "True" && "$HAS_CHAT" != "true" ]]; then
  echo ""
  echo "ERROR: Salad /health reports has_free_chat=$HAS_CHAT (api_version=$API_VER)."
  echo "Redeploy Salad with DOCKERHUB_USER/interview-salad-gpu:latest from latest main."
  exit 1
fi

echo ""
echo "=== POST $GATEWAY/chat-reply (text-only probe) ==="
curl -sS -w "\nHTTP %{http_code}\n" -X POST "$GATEWAY/chat-reply" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $SECRET" \
  -d '{"chat_text":"say hi in one word","skip_tts":true,"speech_lang":"en-in"}'
