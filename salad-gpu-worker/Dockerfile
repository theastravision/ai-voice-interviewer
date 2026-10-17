FROM vllm/vllm-openai:latest

WORKDIR /app
RUN apt-get update && apt-get install -y wget libsndfile1 espeak-ng ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir faster-whisper kokoro-onnx soundfile fastapi "uvicorn[standard]"

RUN wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx && \
    wget -q https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

COPY salad_app.py /app/salad_app.py

ENTRYPOINT ["uvicorn", "salad_app:app", "--host", "::", "--port", "8000"]
