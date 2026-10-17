FROM python:3.11-slim

WORKDIR /app

# Step 1: Install system build essentials and audio compilation tools first
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    cmake \
    pkg-config \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Step 2: Install PyTorch CPU version first cleanly
RUN pip install --no-cache-dir torch==2.4.1+cpu torchaudio==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Pre-cache Silero VAD and MiniLM-L6-v2 inside the image
RUN python -c "import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=True)"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .
RUN mkdir -p /app/chroma_data

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]