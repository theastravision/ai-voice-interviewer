import os
import io
import json
import re
import uuid
import base64
import asyncio
import logging
import numpy as np
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

import indic_tts

API_SECRET = os.getenv("SALAD_API_SECRET", "my-super-secret-salad-key")
API_VERSION = "2026-03-23-free-chat"
CHAT_HTTP_PATHS = ("/chat-reply", "/free-chat", "/chat")
logger = logging.getLogger("uvicorn.error")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
VLLM_GPU_UTIL = float(os.getenv("VLLM_GPU_MEMORY_UTIL", "0.48"))

stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="int8")
tts_model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-3B-Instruct-AWQ",
    quantization="awq",
    gpu_memory_utilization=VLLM_GPU_UTIL,
    max_model_len=6144,
)
llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

SPEECH_LANG_CONFIG: dict[str, dict[str, str]] = {
    "en-in": {
        "whisper": "en",
        "whisper_prompt": "Indian English conversation, clear pronunciation.",
        "kokoro_lang": "hi",
        "kokoro_voice": "hm_omega",
        "llm_hint": "Indian English (natural spoken Indian phrasing, short sentences)",
    },
    "hi": {
        "whisper": "hi",
        "whisper_prompt": "हिंदी बातचीत, स्पष्ट उच्चारण।",
        "kokoro_lang": "hi",
        "kokoro_voice": "hm_omega",
        "llm_hint": "Hindi in Devanagari script, conversational",
    },
    "mr": {
        "whisper": "mr",
        "whisper_prompt": "मराठी संभाषण, स्पष्ट उच्चार.",
        "kokoro_lang": "hi",
        "kokoro_voice": "hm_omega",
        "llm_hint": "Marathi in Devanagari script, conversational",
    },
    "gu": {
        "whisper": "gu",
        "whisper_prompt": "ગુજરાતી વાતચીત, સ્પષ્ટ ઉચ્ચાર.",
        "kokoro_lang": "hi",
        "kokoro_voice": "hf_alpha",
        "llm_hint": "Gujarati in Gujarati script, conversational",
    },
}
DEFAULT_SPEECH_LANG = "en-in"
IM_END = "<|" + "im_end" + "|>"
USE_INDIC_PARLER = os.getenv("USE_INDIC_PARLER", "1") != "0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if USE_INDIC_PARLER:
        asyncio.create_task(asyncio.to_thread(indic_tts.preload_indic_tts))
    paths = sorted({getattr(r, "path", "") for r in app.routes if getattr(r, "path", "")})
    logger.info(
        "Salad GPU ready api_version=%s chat_paths=%s all_routes=%s",
        API_VERSION,
        list(CHAT_HTTP_PATHS),
        paths,
    )
    yield


app = FastAPI(title="Salad Cloud GPU Inference Node", lifespan=lifespan)


def normalize_speech_lang(lang: str | None) -> str:
    if lang and lang in SPEECH_LANG_CONFIG:
        return lang
    return DEFAULT_SPEECH_LANG


def check_auth(key: str):
    if key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API Key")


def transcribe_audio(f32: np.ndarray, speech_lang: str) -> str:
    cfg = SPEECH_LANG_CONFIG[normalize_speech_lang(speech_lang)]
    segs, _ = stt_model.transcribe(
        f32,
        beam_size=1,
        without_timestamps=True,
        language=cfg["whisper"],
        initial_prompt=cfg["whisper_prompt"],
    )
    return " ".join([s.text for s in segs]).strip()


def _kokoro_tts_b64(text: str, speech_lang: str) -> str:
    cfg = SPEECH_LANG_CONFIG[normalize_speech_lang(speech_lang)]
    samples, sr = tts_model.create(
        text, voice=cfg["kokoro_voice"], speed=1.05, lang=cfg["kokoro_lang"]
    )
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_tts(text: str, speech_lang: str | None = None) -> str:
    lang = normalize_speech_lang(speech_lang)
    if USE_INDIC_PARLER:
        try:
            return indic_tts.synthesize_to_wav_b64(text, lang)
        except Exception as exc:
            logger.warning("Indic Parler TTS failed, Kokoro fallback: %s", exc)
    return _kokoro_tts_b64(text, lang)


def llm_language_instruction(speech_lang: str | None) -> str:
    cfg = SPEECH_LANG_CONFIG[normalize_speech_lang(speech_lang)]
    return (
        f"Always respond in {cfg['llm_hint']}. "
        "Use natural spoken language suitable for voice conversation (short sentences)."
    )


@app.get("/health")
async def health():
    routes = sorted({getattr(r, "path", "") for r in app.routes if getattr(r, "path", "")})
    return {
        "status": "ready",
        "api_version": API_VERSION,
        "tts_engine": "indic-parler" if USE_INDIC_PARLER else "kokoro",
        "whisper_model": WHISPER_MODEL_SIZE,
        "has_free_chat": all(p in routes for p in CHAT_HTTP_PATHS[:2]),
        "chat_paths": list(CHAT_HTTP_PATHS),
        "routes": routes,
    }


async def llm_generate(prompt: str, *, max_tokens: int, temperature: float) -> str:
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    out = None
    async for res in llm_engine.generate(prompt, params, uuid.uuid4().hex):
        out = res
    if out is None or not out.outputs:
        raise HTTPException(status_code=503, detail="LLM returned no output")
    return out.outputs[0].text.strip()


def safe_tts(text: str, speech_lang: str | None, skip: bool = False) -> str:
    if skip or not text.strip():
        return ""
    try:
        return generate_tts(text, speech_lang)
    except Exception as exc:
        logger.exception("TTS failed: %s", exc)
        return ""


class QARequest(BaseModel):
    context: str


def repair_json(raw: str) -> dict:
    text = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    last_complete = text.rfind("}")
    if last_complete > 0:
        truncated = text[: last_complete + 1]
        truncated += "]" * max(0, open_brackets) + "}" * max(0, open_braces - 1)
        truncated = re.sub(r",\s*([}\]])", r"\1", truncated)
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not parse LLM JSON output")


@app.post("/generate-15-qa")
async def generate_15_qa(req: QARequest, x_api_key: str = Header(...)):
    check_auth(x_api_key)
    if not req.context.strip():
        raise HTTPException(status_code=400, detail="Empty resume/JD context")
    prompt = f"""<|im_start|>system
You are a Senior Technical Interviewer. Based on the Resume and Job Description context below, output ONLY valid JSON containing 15 technical questions and concise reference answers:
{{"questions": [{{"id": 1, "question": "...", "reference_answer": "..."}}]}}
Context:
{req.context}
{IM_END}
<|im_start|>assistant
{{"""

    max_retries = 2
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            body = await llm_generate(prompt, max_tokens=3500, temperature=0.2 + (attempt * 0.1))
            raw_text = "{" + body
            start, end = raw_text.find("{"), raw_text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Model output contained no JSON object")
            parsed = repair_json(raw_text[start : end + 1])
            if not parsed.get("questions"):
                raise ValueError("JSON had no questions array")
            return parsed
        except HTTPException:
            raise
        except (json.JSONDecodeError, ValueError, IndexError, AttributeError) as e:
            last_error = str(e)
            logger.warning("generate-15-qa attempt %s failed: %s", attempt + 1, last_error)
            continue

    raise HTTPException(status_code=500, detail=f"JSON parse failed after {max_retries+1} attempts: {last_error}")


class TTSRequest(BaseModel):
    text: str
    speech_lang: str | None = DEFAULT_SPEECH_LANG


@app.post("/tts-only")
async def tts_only(req: TTSRequest, x_api_key: str = Header(...)):
    check_auth(x_api_key)
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty TTS text")
    tts_b64 = safe_tts(req.text, req.speech_lang)
    if not tts_b64:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")
    return {"tts_wav_b64": tts_b64}


STRICT_INTERVIEWER_RULES = """STRICT RULES — you MUST follow these:
1. You are ONLY an interviewer. Your ONLY job is to ask questions and give brief feedback.
2. NEVER answer any technical question from the candidate.
3. NEVER explain any concept, topic, or technology.
4. If candidate asks you a question, say: "I'm here to conduct your interview, not answer questions. Let's continue."
5. If candidate asks for hints or explanations, say: "I cannot provide explanations during the interview. Please answer based on your knowledge."
6. If candidate discusses anything unrelated to the interview, say: "Let's stay focused on the interview."
7. Compare their answer to the reference answer. Give ONE brief sentence of feedback (note if strong or partial).
8. ALWAYS move to the next question after their answer — whether correct, partial, or wrong. Never block or re-ask the same question.
9. Then ask the next question verbatim.
10. Under 50 words total."""


class TurnRequest(BaseModel):
    pcm16_base64: str | None = None
    chat_text: str | None = None
    current_question: str
    reference_answer: str
    next_question: str | None = None
    speech_lang: str | None = DEFAULT_SPEECH_LANG


@app.post("/process-turn")
async def process_turn(req: TurnRequest, x_api_key: str = Header(...)):
    check_auth(x_api_key)
    speech_lang = normalize_speech_lang(req.speech_lang)
    user_text = req.chat_text
    if req.pcm16_base64:
        pcm = base64.b64decode(req.pcm16_base64)
        f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        user_text = transcribe_audio(f32, speech_lang)

    if not user_text:
        return {"user_transcript": "", "ai_text": "", "tts_wav_b64": ""}

    lang_rule = llm_language_instruction(speech_lang)
    if req.next_question:
        prompt = (
            f"<|im_start|>system\n{STRICT_INTERVIEWER_RULES}\n{lang_rule}\n"
            f"Question asked: {req.current_question}\n"
            f"Reference Answer: {req.reference_answer}\n"
            f"Candidate said: {user_text}\n"
            f"Give 1 brief sentence of feedback, then ask verbatim: \"{req.next_question}\".\n"
            f"{IM_END}\n<|im_start|>assistant\n"
        )
    else:
        prompt = (
            f"<|im_start|>system\n{STRICT_INTERVIEWER_RULES}\n{lang_rule}\n"
            f"Candidate answered the final question: \"{user_text}\". "
            f"Thank them warmly in 2 sentences. Do not answer any questions.\n"
            f"{IM_END}\n<|im_start|>assistant\n"
        )

    ai_reply = await llm_generate(prompt, max_tokens=120, temperature=0.3)
    return {
        "user_transcript": user_text,
        "ai_text": ai_reply,
        "tts_wav_b64": safe_tts(ai_reply, speech_lang),
    }


class ChatRequest(BaseModel):
    pcm16_base64: str | None = None
    chat_text: str | None = None
    history: list[dict] | None = None
    skip_tts: bool = False
    speech_lang: str | None = DEFAULT_SPEECH_LANG
    speak_replies: bool = False


async def _chat_reply_handler(req: ChatRequest) -> dict:
    speech_lang = normalize_speech_lang(req.speech_lang)
    user_text = req.chat_text
    if req.pcm16_base64:
        pcm = base64.b64decode(req.pcm16_base64)
        f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        user_text = transcribe_audio(f32, speech_lang)

    if not user_text:
        return {"user_transcript": "", "ai_text": "", "tts_wav_b64": ""}

    history_str = ""
    if req.history:
        for msg in req.history[-6:]:
            role = "user" if msg.get("role") == "user" else "assistant"
            history_str += f"<|im_start|>{role}\n{msg.get('text', '')}\n{IM_END}\n"

    lang_rule = llm_language_instruction(speech_lang)
    prompt = (
        "<|im_start|>system\n"
        "You are a friendly, knowledgeable AI assistant with an Indian personality. "
        f"{lang_rule} Be conversational, warm, and helpful. "
        "You can discuss any topic freely — technology, culture, sports, philosophy, anything. "
        "Keep responses concise (under 60 words) and natural.\n"
        f"{IM_END}\n"
        f"{history_str}"
        f"<|im_start|>user\n{user_text}\n{IM_END}\n"
        "<|im_start|>assistant\n"
    )

    ai_reply = await llm_generate(prompt, max_tokens=150, temperature=0.7)
    skip = req.skip_tts and not req.pcm16_base64 and not req.speak_replies

    return {
        "user_transcript": user_text,
        "ai_text": ai_reply,
        "tts_wav_b64": safe_tts(ai_reply, speech_lang, skip=skip),
    }


@app.post("/chat-reply")
@app.post("/free-chat")
@app.post("/chat")
async def chat_reply(req: ChatRequest, x_api_key: str = Header(...)):
    check_auth(x_api_key)
    return await _chat_reply_handler(req)
