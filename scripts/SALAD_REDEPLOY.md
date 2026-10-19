# Salad GPU redeploy (required for Free Chat)

After pushing to `main`, CI publishes `DOCKERHUB_USER/interview-salad-gpu:latest`.

1. Salad Cloud → your GPU container group → set image to `interview-salad-gpu:latest` → **Redeploy**.
2. On EC2:
   ```bash
   export $(grep -E '^SALAD_' ~/ai-voice-interview/.env | xargs)
   bash scripts/verify-salad-gateway.sh
   curl -s http://localhost/api/diagnostics | python3 -m json.tool
   ```
3. Expect `salad_has_free_chat: true` and `salad_api_version` ≥ `2025-10-20-indic`.
