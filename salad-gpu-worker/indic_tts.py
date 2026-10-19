"""Client for Indic Parler sidecar (runs in isolated Python venv)."""
from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger("uvicorn.error")

SIDECAR_URL = os.getenv("INDIC_PARLER_URL", "http://127.0.0.1:8001").rstrip("/")
_ready = False


def wait_for_sidecar(timeout_sec: float = 300.0) -> bool:
    global _ready
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = httpx.get(f"{SIDECAR_URL}/health", timeout=5.0)
            if r.status_code == 200:
                _ready = True
                logger.info("Indic Parler sidecar ready at %s", SIDECAR_URL)
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    logger.error("Indic Parler sidecar not ready after %ss", timeout_sec)
    return False


def preload_indic_tts() -> None:
    wait_for_sidecar()


def synthesize_to_wav_b64(text: str, speech_lang: str) -> str:
    if not _ready and not wait_for_sidecar(timeout_sec=60.0):
        raise RuntimeError("Indic Parler sidecar unavailable")
    r = httpx.post(
        f"{SIDECAR_URL}/synthesize",
        json={"text": text, "speech_lang": speech_lang},
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    b64 = data.get("tts_wav_b64", "")
    if not b64:
        raise RuntimeError("Empty TTS from sidecar")
    return b64
