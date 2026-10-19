FROM vllm/vllm-openai:latest

WORKDIR /app
RUN apt-get update && apt-get install -y wget libsndfile1 espeak-ng ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    faster-whisper kokoro-onnx soundfile fastapi "uvicorn[standard]" \
    parler-tts transformers accelerate nltk

RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

RUN wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx && \
    wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

ENV HF_HOME=/app/hf_cache
RUN pip install --no-cache-dir huggingface_hub && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('ai4bharat/indic-parler-tts'); \
print('Indic Parler weights cached')"

COPY indic_tts.py salad_app.py /app/

ENV USE_INDIC_PARLER=1
ENV VLLM_GPU_MEMORY_UTIL=0.48
ENV WHISPER_MODEL=small

ENTRYPOINT ["uvicorn", "salad_app:app", "--host", "::", "--port", "8000"]
