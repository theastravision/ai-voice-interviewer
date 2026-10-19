#!/usr/bin/env bash
# Run on EC2 (or anywhere) to verify Salad gateway exposes Free Chat + auth.
set -euo pipefail

GATEWAY="${SALAD_GATEWAY_URL:-}"
APP_KEY="${SALAD_API_SECRET:-}"
GATEWAY_KEY="${SALAD_GATEWAY_API_KEY:-$APP_KEY}"

if [[ -z "$GATEWAY" || -z "$APP_KEY" ]]; then
  echo "Set SALAD_GATEWAY_URL and SALAD_API_SECRET (e.g. source ~/ai-voice-interview/.env)"
  exit 1
fi

GATEWAY="${GATEWAY%/}"
AUTH_HEADERS=(-H "x-api-key: $APP_KEY")
if [[ -n "$GATEWAY_KEY" ]]; then
  AUTH_HEADERS+=(-H "Salad-Api-Key: $GATEWAY_KEY")
fi

echo "=== GET $GATEWAY/health ==="
HEALTH_JSON="$(curl -sS "${AUTH_HEADERS[@]}" "$GATEWAY/health")"
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
  "${AUTH_HEADERS[@]}" \
  -d '{"chat_text":"say hi in one word","skip_tts":true,"speech_lang":"en-in"}'
