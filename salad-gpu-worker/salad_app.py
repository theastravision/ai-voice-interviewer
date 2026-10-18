import os
import io
import json
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

stt_model = WhisperModel("small.en", device="cuda", compute_type="int8")
tts_model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

engine_args = AsyncEngineArgs(
    model="Qwen/Qwen2.5-3B-Instruct-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.55,
    max_model_len=6144
)
llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

def check_auth(key: str):
    if key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/health")
async def health():
    return {"status": "ready"}

import re

class QARequest(BaseModel):
    context: str

def repair_json(raw: str) -> dict:
    """Try to fix common LLM JSON issues and parse."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', raw)
    # Try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # If truncated, try closing open structures
    # Count braces/brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    # Remove last incomplete object (after last complete })
    last_complete = text.rfind('}')
    if last_complete > 0:
        truncated = text[:last_complete + 1]
        truncated += ']' * max(0, open_brackets) + '}' * max(0, open_braces - 1)
        truncated = re.sub(r',\s*([}\]])', r'\1', truncated)
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse LLM JSON output")

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

class TTSRequest(BaseModel):
    text: str

@app.post("/tts-only")
async def tts_only(req: TTSRequest, x_api_key: str = Header(...)):
    check_auth(x_api_key)
    samples, sr = tts_model.create(req.text, voice="af_heart", speed=1.05, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return {"tts_wav_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}

class TurnRequest(BaseModel):
    pcm16_base64: str | None = None
    chat_text: str | None = None
    current_question: str
    reference_answer: str
    next_question: str | None = None

@app.post("/process-turn")
async def process_turn(req: TurnRequest, x_api_key: str = Header(...)):
    check_auth(x_api_key)
    user_text = req.chat_text
    if req.pcm16_base64:
        pcm = base64.b64decode(req.pcm16_base64)
        f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segs, _ = stt_model.transcribe(f32, beam_size=1, language="en", without_timestamps=True)
        user_text = " ".join([s.text for s in segs]).strip()

    if not user_text:
        return {"user_transcript": "", "ai_text": "", "tts_wav_b64": ""}

    if req.next_question:
        prompt = f"<|im_start|>system\nQuestion asked: {req.current_question}\nReference Answer: {req.reference_answer}\nCandidate answered: {user_text}\nGive 1 brief sentence of feedback, then ask verbatim: \"{req.next_question}\". Under 45 words.<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>system\nCandidate answered final question: \"{user_text}\". Thank them warmly in 2 sentences.<|im_end|>\n<|im_start|>assistant\n"

    params = SamplingParams(temperature=0.3, max_tokens=90)
    out = None
    async for res in llm_engine.generate(prompt, params, uuid.uuid4().hex):
        out = res
    ai_reply = out.outputs[0].text.strip()

    samples, sr = tts_model.create(ai_reply, voice="af_heart", speed=1.05, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return {
        "user_transcript": user_text,
        "ai_text": ai_reply,
        "tts_wav_b64": base64.b64encode(buf.getvalue()).decode("utf-8")
    }
