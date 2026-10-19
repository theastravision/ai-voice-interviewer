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

## Salad Container Gateway auth (403 on `/health`)

If the gateway shows **Protected** / a lock icon, every request needs **two** headers (EC2 backend sends both):

| Secret | Where to get it | EC2 `.env` |
|--------|------------------|------------|
| **Salad gateway key** | Salad portal → container group → **API Endpoints** (or Authentication) → copy API key | `SALAD_GATEWAY_API_KEY` |
| **App key** | Same value as `SALAD_API_SECRET` on the **Salad container env** | `SALAD_API_SECRET` |

If you use one shared password for both, set only `SALAD_API_SECRET` on EC2 (backend falls back for gateway auth).

Salad requires **IPv6** (`::` in `start.sh`) and port **8000** matching the gateway.

---

After pushing to `main`, CI publishes `DOCKERHUB_USER/interview-salad-gpu:latest`.

1. Salad Cloud → GPU container → image `interview-salad-gpu:latest` → **Redeploy**.
2. On EC2:
   ```bash
   export $(grep -E '^SALAD_' ~/ai-voice-interview/.env | xargs)
   bash scripts/verify-salad-gateway.sh
   curl -s http://localhost/api/diagnostics | python3 -m json.tool
   ```
3. Expect `salad_has_free_chat: true`, `salad_chat_probe_ok: true`, and `salad_api_version` **`2026-03-23-free-chat`** (or newer).

**First boot:** CPU Parler sidecar starts before the main API; startup can take several minutes while the model loads.
