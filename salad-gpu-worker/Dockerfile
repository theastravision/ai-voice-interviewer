FROM vllm/vllm-openai:latest

WORKDIR /app
RUN apt-get update && apt-get install -y wget libsndfile1 espeak-ng ffmpeg && rm -rf /var/lib/apt/lists/*

# App stack (must not downgrade vLLM's transformers / protobuf / huggingface_hub)
RUN pip install --no-cache-dir \
    faster-whisper kokoro-onnx soundfile fastapi "uvicorn[standard]" httpx

RUN wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx && \
    wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Isolated venv for Parler-TTS (transformers==4.46.1 — incompatible with vLLM in main env)
ENV HF_HOME=/app/hf_cache
RUN python3 -m venv /opt/parler-venv && \
    /opt/parler-venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/parler-venv/bin/pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu && \
    /opt/parler-venv/bin/pip install --no-cache-dir \
      "transformers==4.46.1" parler-tts sentencepiece soundfile accelerate huggingface_hub \
      fastapi uvicorn numpy

# Gated model — optional bake-in when HF_TOKEN is passed as a BuildKit secret (GitHub Actions).
RUN --mount=type=secret,id=HF_TOKEN \
    if [ -f /run/secrets/HF_TOKEN ] && [ -s /run/secrets/HF_TOKEN ]; then \
      export HF_TOKEN="$(tr -d '\n' < /run/secrets/HF_TOKEN)"; \
      export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; \
      /opt/parler-venv/bin/python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ai4bharat/indic-parler-tts'); \
print('Indic Parler weights cached at build time')"; \
    else \
      echo "No HF_TOKEN at build — weights download at container start (set HF_TOKEN on Salad)."; \
    fi

COPY indic_parler_sidecar.py indic_tts.py salad_app.py hf_auth.py /app/
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENV USE_INDIC_PARLER=1
ENV INDIC_PARLER_URL=http://127.0.0.1:8001
ENV VLLM_GPU_MEMORY_UTIL=0.48
ENV WHISPER_MODEL=small

ENTRYPOINT ["/app/start.sh"]
