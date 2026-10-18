import os
import io
import json
import re
import uuid
import base64
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

app = FastAPI(title="Salad Cloud GPU Inference Node")
API_SECRET = os.getenv("SALAD_API_SECRET", "my-super-secret-salad-key")

stt_model = WhisperModel("small", device="cuda", compute_type="int8")
tts_model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-3B-Instruct-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.55,
    max_model_len=6144
)
llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

# speech_lang from frontend → Whisper + Kokoro settings
SPEECH_LANG_CONFIG: dict[str, dict[str, str]] = {
    "en-in": {"whisper": "en", "tts_lang": "en-gb", "tts_voice": "hm_psi", "llm_hint": "Indian English"},
    "hi": {"whisper": "hi", "tts_lang": "hi", "tts_voice": "hm_psi", "llm_hint": "Hindi (Devanagari)"},
    "mr": {"whisper": "mr", "tts_lang": "hi", "tts_voice": "hm_psi", "llm_hint": "Marathi (Devanagari)"},
    "gu": {"whisper": "gu", "tts_lang": "hi", "tts_voice": "hm_psi", "llm_hint": "Gujarati (Gujarati script)"},
}
DEFAULT_SPEECH_LANG = "en-in"

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
    )
    return " ".join([s.text for s in segs]).strip()

def generate_tts(text: str, speech_lang: str | None = None) -> str:
    """Generate TTS audio and return base64 WAV."""
    cfg = SPEECH_LANG_CONFIG[normalize_speech_lang(speech_lang)]
    samples, sr = tts_model.create(
        text, voice=cfg["tts_voice"], speed=1.05, lang=cfg["tts_lang"]
    )
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def llm_language_instruction(speech_lang: str | None) -> str:
    cfg = SPEECH_LANG_CONFIG[normalize_speech_lang(speech_lang)]
    return f"Always respond in {cfg['llm_hint']}."

@app.get("/health")
async def health():
    return {"status": "ready"}

# ─── Common JSON repair ─────────────────────────────────────────────
class QARequest(BaseModel):
    context: str

def repair_json(raw: str) -> dict:
    """Try to fix common LLM JSON issues and parse."""
    text = re.sub(r',\s*([}\]])', r'\1', raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    last_complete = text.rfind('}')
    if last_complete > 0:
        truncated = text[:last_complete + 1]
        truncated += ']' * max(0, open_brackets) + '}' * max(0, open_braces - 1)
        truncated = re.sub(r',\s*([}\]])', r'\1', truncated)
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not parse LLM JSON output")

# ─── Generate 15 Q&A ────────────────────────────────────────────────
@app.post("/generate-15-qa")
async def generate_15_qa(req: QARequest, x_api_key: str = Header(...)):
    check_auth(x_api_key)
    prompt = f"""<|im_start|>system
You are a Senior Technical Interviewer. Based on the Resume and Job Description context below, output ONLY valid JSON containing 15 technical questions and concise reference answers:
{{"questions": [{{"id": 1, "question": "...", "reference_answer": "..."}}]}}
Context:
{req.context}<|im_end|>
<|im_start|>assistant
{{"""

    max_retries = 2
    last_error = None
    for attempt in range(max_retries + 1):
        params = SamplingParams(temperature=0.2 + (attempt * 0.1), max_tokens=3500)
        out = None
        async for res in llm_engine.generate(prompt, params, uuid.uuid4().hex):
            out = res
        raw_text = "{" + out.outputs[0].text.strip()
        try:
            start, end = raw_text.find("{"), raw_text.rfind("}")
            parsed = repair_json(raw_text[start:end+1])
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            continue

    raise HTTPException(status_code=500, detail=f"JSON parse failed after {max_retries+1} attempts: {last_error}")

# ─── TTS Only ────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    speech_lang: str | None = DEFAULT_SPEECH_LANG

@app.post("/tts-only")
async def tts_only(req: TTSRequest, x_api_key: str = Header(...)):
    check_auth(x_api_key)
    return {"tts_wav_b64": generate_tts(req.text, req.speech_lang)}

# ─── Interview Turn (STRICT mode) ───────────────────────────────────
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
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
    else:
        prompt = (
            f"<|im_start|>system\n{STRICT_INTERVIEWER_RULES}\n{lang_rule}\n"
            f"Candidate answered the final question: \"{user_text}\". "
            f"Thank them warmly in 2 sentences. Do not answer any questions.\n"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )

    params = SamplingParams(temperature=0.3, max_tokens=120)
    out = None
    async for res in llm_engine.generate(prompt, params, uuid.uuid4().hex):
        out = res
    ai_reply = out.outputs[0].text.strip()

    return {
        "user_transcript": user_text,
        "ai_text": ai_reply,
        "tts_wav_b64": generate_tts(ai_reply, speech_lang),
    }

# ─── Free Chat (multilingual conversational) ────────────────────────
class ChatRequest(BaseModel):
    pcm16_base64: str | None = None
    chat_text: str | None = None
    history: list[dict] | None = None
    skip_tts: bool = False
    speech_lang: str | None = DEFAULT_SPEECH_LANG

@app.post("/chat-reply")
async def chat_reply(req: ChatRequest, x_api_key: str = Header(...)):
    check_auth(x_api_key)
    speech_lang = normalize_speech_lang(req.speech_lang)
    user_text = req.chat_text
    if req.pcm16_base64:
        pcm = base64.b64decode(req.pcm16_base64)
        f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        user_text = transcribe_audio(f32, speech_lang)

    if not user_text:
        return {"user_transcript": "", "ai_text": "", "tts_wav_b64": ""}

    # Build conversation history
    history_str = ""
    if req.history:
        for msg in req.history[-6:]:  # last 6 messages for context
            role = "user" if msg.get("role") == "user" else "assistant"
            history_str += f"<|im_start|>{role}\n{msg.get('text', '')}<|im_end|>\n"

    lang_rule = llm_language_instruction(speech_lang)
    prompt = (
        "<|im_start|>system\n"
        "You are a friendly, knowledgeable AI assistant with an Indian personality. "
        f"{lang_rule} Be conversational, warm, and helpful. "
        "You can discuss any topic freely — technology, culture, sports, philosophy, anything. "
        "Keep responses concise (under 60 words) and natural.\n"
        "<|im_end|>\n"
        f"{history_str}"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    params = SamplingParams(temperature=0.7, max_tokens=150)
    out = None
    async for res in llm_engine.generate(prompt, params, uuid.uuid4().hex):
        out = res
    ai_reply = out.outputs[0].text.strip()

    return {
        "user_transcript": user_text,
        "ai_text": ai_reply,
        "tts_wav_b64": "" if req.skip_tts else generate_tts(ai_reply, speech_lang),
    }
