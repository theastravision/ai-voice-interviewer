# Salad GPU redeploy (required for Free Chat)

## Indic Parler (gated Hugging Face model)

1. Log in at [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) and **accept the model license** (once per HF account).
2. Create a Hugging Face **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Where to put `HF_TOKEN`

| Where | Required? | Purpose |
|--------|-----------|---------|
| **Salad Cloud** → GPU container → **Environment variables** → `HF_TOKEN` = `hf_...` | **Yes** (unless weights were fully baked into the image at build) | Parler sidecar downloads/loads the model at runtime |
| **GitHub** → repo **Settings → Secrets and variables → Actions** → secret name **`HF_TOKEN`** | Recommended | CI `docker build` preloads model into the image (faster Salad cold start) |

Either name works on Salad: `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.

**AWS EC2 / `.env` on EC2:** do **not** add `HF_TOKEN` — inference runs on Salad only.

If Parler cannot authenticate, TTS falls back to Kokoro. Set `USE_INDIC_PARLER=0` on Salad to skip Parler entirely.

---

After pushing to `main`, CI publishes `DOCKERHUB_USER/interview-salad-gpu:latest`.

1. Salad Cloud → GPU container → image `interview-salad-gpu:latest` → **Redeploy**.
2. On EC2:
   ```bash
   export $(grep -E '^SALAD_' ~/ai-voice-interview/.env | xargs)
   bash scripts/verify-salad-gateway.sh
   curl -s http://localhost/api/diagnostics | python3 -m json.tool
   ```
3. Expect `salad_has_free_chat: true` and `salad_api_version` ≥ `2025-10-20-indic`.

**First boot:** CPU Parler sidecar starts before the main API; startup can take several minutes while the model loads.
