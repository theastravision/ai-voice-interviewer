FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential git curl ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True)"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .
RUN mkdir -p /app/chroma_data

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
