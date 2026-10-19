"""AI4Bharat Indic Parler-TTS — Indian English + Indic languages."""
from __future__ import annotations

import io
import logging
import os
import re

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("uvicorn.error")

INDIC_PARLER_MODEL = os.getenv("INDIC_PARLER_MODEL", "ai4bharat/indic-parler-tts")
MAX_CHARS_PER_CHUNK = int(os.getenv("INDIC_TTS_MAX_CHARS", "280"))

# Speaker style descriptions (accent / prosody) per UI language
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

_model = None
_tokenizer = None
_description_tokenizer = None
_sample_rate: int | None = None


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_model():
    global _model, _tokenizer, _description_tokenizer, _sample_rate
    if _model is not None:
        return
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = _device()
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    logger.info("Loading Indic Parler-TTS %s on %s", INDIC_PARLER_MODEL, device)
    _model = ParlerTTSForConditionalGeneration.from_pretrained(
        INDIC_PARLER_MODEL,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    _tokenizer = AutoTokenizer.from_pretrained(INDIC_PARLER_MODEL)
    _description_tokenizer = AutoTokenizer.from_pretrained(_model.config.text_encoder._name_or_path)
    _sample_rate = int(_model.config.sampling_rate)
    _model.eval()
    logger.info("Indic Parler-TTS ready (sr=%s)", _sample_rate)


def preload_indic_tts() -> None:
    """Call at startup to avoid first-request latency."""
    try:
        _load_model()
    except Exception as exc:
        logger.warning("Indic Parler preload skipped: %s", exc)


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= MAX_CHARS_PER_CHUNK:
        return [text]
    parts = re.split(r"(?<=[.!?।])\s+", text)
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= MAX_CHARS_PER_CHUNK:
            buf = f"{buf} {p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p if len(p) <= MAX_CHARS_PER_CHUNK else p[:MAX_CHARS_PER_CHUNK]
    if buf:
        chunks.append(buf)
    return chunks or [text[:MAX_CHARS_PER_CHUNK]]


def _synthesize_chunk(prompt: str, description: str) -> np.ndarray:
    _load_model()
    assert _model is not None and _tokenizer is not None and _description_tokenizer is not None

    device = _device()
    desc_ids = _description_tokenizer(description, return_tensors="pt").to(device)
    prompt_ids = _tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        generation = _model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
    audio = generation.cpu().numpy().squeeze().astype(np.float32)
    return audio


def synthesize_to_wav_b64(text: str, speech_lang: str) -> str:
    """Return base64-encoded WAV PCM16."""
    import base64

    description = VOICE_DESCRIPTIONS.get(speech_lang, VOICE_DESCRIPTIONS["en-in"])
    chunks = _split_text(text)
    wav_parts: list[np.ndarray] = []
    for chunk in chunks:
        wav_parts.append(_synthesize_chunk(chunk, description))

    if not wav_parts:
        raise ValueError("empty TTS input")

    audio = np.concatenate(wav_parts) if len(wav_parts) > 1 else wav_parts[0]
    sr = _sample_rate or 44100
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
