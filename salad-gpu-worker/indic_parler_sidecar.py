"""Isolated Parler-TTS service (separate venv — avoids vLLM transformers conflict)."""
import base64
import io
import os
import re

import hf_auth

hf_auth.apply_hf_auth()

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

INDIC_PARLER_MODEL = os.getenv("INDIC_PARLER_MODEL", "ai4bharat/indic-parler-tts")
MAX_CHARS = int(os.getenv("INDIC_TTS_MAX_CHARS", "280"))
PORT = int(os.getenv("INDIC_PARLER_PORT", "8001"))

VOICE_DESCRIPTIONS: dict[str, str] = {
    "en-in": (
        "A male speaker with a clear Indian English accent delivers conversational speech "
        "at a moderate pace and pitch. The recording is high quality with a warm, natural tone."
    ),
    "hi": (
        "Rohit speaks with a clear Hindi voice at a moderate pace in a conversational Indian tone. "
        "The recording is high quality with no background noise."
    ),
    "mr": (
        "Sunita speaks slowly in a calm, moderate-pitched Marathi voice, delivering speech "
        "with a neutral conversational tone. The recording is very high quality."
    ),
    "gu": (
        "Aditi speaks at a moderate pace and pitch, with a clear Gujarati conversational tone. "
        "The recording is very high quality with no background noise."
    ),
}

app = FastAPI()
_model = None
_tokenizer = None
_description_tokenizer = None
_sample_rate = 44100
_device = "cpu"


class SynthRequest(BaseModel):
    text: str
    speech_lang: str = "en-in"


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= MAX_CHARS:
        return [text]
    parts = re.split(r"(?<=[.!?।])\s+", text)
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= MAX_CHARS:
            buf = f"{buf} {p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p[:MAX_CHARS]
    if buf:
        chunks.append(buf)
    return chunks or [text[:MAX_CHARS]]


@app.on_event("startup")
def load_model():
    global _model, _tokenizer, _description_tokenizer, _sample_rate
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    _model = ParlerTTSForConditionalGeneration.from_pretrained(
        INDIC_PARLER_MODEL,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(_device)
    _tokenizer = AutoTokenizer.from_pretrained(INDIC_PARLER_MODEL)
    _description_tokenizer = AutoTokenizer.from_pretrained(_model.config.text_encoder._name_or_path)
    _sample_rate = int(_model.config.sampling_rate)
    _model.eval()


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="loading")
    return {"status": "ready", "device": _device}


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Parler model not loaded")
    lang = req.speech_lang if req.speech_lang in VOICE_DESCRIPTIONS else "en-in"
    description = VOICE_DESCRIPTIONS[lang]
    parts: list[np.ndarray] = []
    for chunk in _split_text(req.text):
        desc_ids = _description_tokenizer(description, return_tensors="pt").to(_device)
        prompt_ids = _tokenizer(chunk, return_tensors="pt").to(_device)
        with torch.inference_mode():
            generation = _model.generate(
                input_ids=desc_ids.input_ids,
                attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
            )
        parts.append(generation.cpu().numpy().squeeze().astype(np.float32))
    audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
    buf = io.BytesIO()
    sf.write(buf, audio, _sample_rate, format="WAV", subtype="PCM_16")
    return {"tts_wav_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
