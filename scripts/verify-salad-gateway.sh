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
curl -sS "$GATEWAY/health" | python3 -m json.tool

echo ""
echo "=== POST $GATEWAY/chat-reply (text-only probe) ==="
curl -sS -w "\nHTTP %{http_code}\n" -X POST "$GATEWAY/chat-reply" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $SECRET" \
  -d '{"chat_text":"say hi in one word","skip_tts":true,"speech_lang":"en-in"}'
