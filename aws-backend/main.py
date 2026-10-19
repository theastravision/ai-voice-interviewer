import os
import json
import time
import base64
import asyncio
import logging
import numpy as np
import torch
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Depends, HTTPException
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import Base, writer_engine, WriteSessionLocal, ReadSessionLocal, InterviewSessionDB, get_write_db, get_read_db
from audio_dsp import AudioDSPPipeline
from rag_engine import ChromaRAGEngine

app = FastAPI(title="AI Interviewer AWS Orchestrator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SALAD_GATEWAY_URL = os.getenv("SALAD_GATEWAY_URL", "http://localhost:8001").rstrip("/")
SALAD_API_SECRET = os.getenv("SALAD_API_SECRET", "my-super-secret-salad-key")
# Salad Container Gateway auth (portal → API Endpoints). Falls back to SALAD_API_SECRET if unset.
SALAD_GATEWAY_API_KEY = (os.getenv("SALAD_GATEWAY_API_KEY") or SALAD_API_SECRET).strip()


def salad_gateway_headers() -> dict[str, str]:
    """Headers for requests through Salad's HTTPS gateway (auth + in-container app key)."""
    headers = {"x-api-key": SALAD_API_SECRET}
    if SALAD_GATEWAY_API_KEY:
        headers["Salad-Api-Key"] = SALAD_GATEWAY_API_KEY
    return headers
INTERVIEW_DURATION_SEC = 1800
VALID_SPEECH_LANGS = frozenset({"en-in", "hi", "mr", "gu"})

logger = logging.getLogger("uvicorn.error")


def normalize_speech_lang(lang: str | None) -> str:
    if lang and lang in VALID_SPEECH_LANGS:
        return lang
    return "en-in"


def ws_connected(websocket: WebSocket) -> bool:
    return websocket.client_state == WebSocketState.CONNECTED


async def ws_send_json(websocket: WebSocket, payload: dict) -> bool:
    if not ws_connected(websocket):
        return False
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


async def ws_send_bytes(websocket: WebSocket, payload: bytes) -> bool:
    if not ws_connected(websocket):
        return False
    try:
        await websocket.send_bytes(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
rag_engine = ChromaRAGEngine()
http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0))
vad_model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=True)


async def ws_receive(websocket: WebSocket) -> dict | None:
    """Return the next WS message, or None if the client disconnected."""
    try:
        msg = await websocket.receive()
    except (WebSocketDisconnect, RuntimeError):
        return None
    if msg.get("type") == "websocket.disconnect":
        return None
    return msg


SALAD_CHAT_PATHS = ("/chat-reply", "/free-chat", "/chat")


async def post_salad_chat(payload: dict) -> tuple[bool, dict]:
    """POST Free Chat to Salad; try every known chat path (old/new images)."""
    return await post_salad(SALAD_CHAT_PATHS[0], payload, alt_paths=SALAD_CHAT_PATHS[1:])


async def post_salad(path: str, payload: dict, *, alt_paths: tuple[str, ...] = ()) -> tuple[bool, dict]:
    """POST to Salad GPU. Returns (ok, json_body_or_error_detail)."""
    paths = (path,) + alt_paths
    last_status = 0
    last_detail = ""

    for p in paths:
        url = f"{SALAD_GATEWAY_URL}{p}"
        try:
            resp = await http_client.post(
                url,
                json=payload,
                headers=salad_gateway_headers(),
            )
        except httpx.HTTPError as exc:
            logger.error("Salad unreachable %s: %s", url, exc)
            return False, {"detail": f"Cannot reach AI GPU at {SALAD_GATEWAY_URL}. Check SALAD_GATEWAY_URL on EC2."}

        if resp.status_code == 200:
            try:
                return True, resp.json()
            except json.JSONDecodeError:
                logger.error("Salad %s returned non-JSON: %s", p, resp.text[:200])
                return False, {"detail": "GPU returned an invalid response."}

        last_status = resp.status_code
        last_detail = resp.text[:200]
        if resp.status_code == 404 and p != paths[-1]:
            logger.warning("Salad %s HTTP 404, trying alternate path…", p)
            continue
        logger.error("Salad %s HTTP %s: %s", p, resp.status_code, resp.text[:800])
        break

    detail = last_detail
    if last_status == 403:
        detail = (
            "Salad gateway rejected the request (403). Set SALAD_GATEWAY_API_KEY on EC2 to your "
            "Salad portal API key (Container Gateway → Authentication), or disable gateway auth."
        )
    elif last_status == 503 or (last_detail and "503 Service Unavailable" in last_detail):
        detail = (
            "Salad GPU is not ready (503). Container may still be starting — check Salad Log Explorer. "
            "Ensure the image is interview-salad-gpu:latest and readiness probe passes."
        )
    elif last_status == 401:
        detail = "Invalid SALAD_API_SECRET (401 from GPU app)."
    elif last_status == 404:
        detail = (
            "GPU container is missing /chat-reply (404). "
            "Redeploy Salad Cloud with the latest interview-salad-gpu Docker image."
        )
    return False, {"detail": detail, "status_code": last_status}

def salad_user_error(failure: dict) -> str:
    detail = failure.get("detail", "")
    if "SALAD_GATEWAY_URL" in detail or failure.get("status_code") == 0:
        return "Cannot reach the AI GPU. On EC2, verify SALAD_GATEWAY_URL in ~/ai-voice-interview/.env points to your Salad gateway."
    if failure.get("status_code") == 403 or "SALAD_GATEWAY_API_KEY" in detail:
        return (
            "Salad gateway auth failed (403). Add SALAD_GATEWAY_API_KEY to EC2 ~/ai-voice-interview/.env "
            "(Salad portal → your container group → API Endpoints / Authentication key)."
        )
    if failure.get("status_code") == 503:
        return "AI GPU is starting or unavailable (503). Wait 2–5 minutes and try again; check Salad instance logs."
    if failure.get("status_code") == 401 or "SALAD_API_SECRET" in detail:
        return "AI GPU rejected the app API key. Ensure SALAD_API_SECRET matches on EC2 and Salad container env."
    if failure.get("status_code") == 404 or "missing /chat-reply" in detail:
        return (
            "Salad GPU is running an old image without Free Chat. "
            "In Salad Cloud, redeploy container image "
            "DOCKERHUB_USER/interview-salad-gpu:latest (push from CI or docker pull + restart)."
        )
    return f"Sorry — the AI GPU had an error. {detail[:120] if detail else 'Check aws-backend logs.'}"


@app.on_event("startup")
async def startup_event():
    logger.info("Salad gateway URL: %s", SALAD_GATEWAY_URL)
    try:
        resp = await http_client.get(
            f"{SALAD_GATEWAY_URL}/health",
            headers=salad_gateway_headers(),
            timeout=15.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            routes = data.get("routes") or []
            if not any(p in routes for p in SALAD_CHAT_PATHS):
                logger.error(
                    "Salad GPU health OK but missing Free Chat routes (%s). "
                    "Redeploy interview-salad-gpu:latest on Salad Cloud "
                    "(expect api_version >= 2026-03-23-free-chat).",
                    routes,
                )
            else:
                logger.info("Salad GPU api_version=%s routes=%s", data.get("api_version"), routes)
        elif resp.status_code == 403:
            logger.error(
                "Salad gateway /health HTTP 403 — set SALAD_GATEWAY_API_KEY on EC2 "
                "(Salad portal API key for Container Gateway auth)."
            )
        else:
            logger.warning("Salad health check HTTP %s", resp.status_code)
    except httpx.HTTPError as exc:
        logger.warning("Salad health check failed at startup: %s", exc)
    async with writer_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "aws-fastapi-orchestrator"}


@app.get("/api/diagnostics")
async def diagnostics():
    """Quick check for Salad connectivity (open /api/diagnostics in the browser)."""
    salad_ok = False
    salad_detail = ""
    salad_routes: list[str] = []
    salad_api_version = ""
    salad_tts_engine = ""
    try:
        resp = await http_client.get(
            f"{SALAD_GATEWAY_URL}/health",
            headers=salad_gateway_headers(),
            timeout=10.0,
        )
        salad_ok = resp.status_code == 200
        salad_detail = resp.text[:300]
        if salad_ok:
            try:
                health = resp.json()
                salad_routes = health.get("routes") or []
                salad_api_version = health.get("api_version") or ""
                salad_tts_engine = health.get("tts_engine") or ""
            except json.JSONDecodeError:
                pass
    except httpx.HTTPError as exc:
        salad_detail = str(exc)

    chat_route_ok = any(p in salad_routes for p in SALAD_CHAT_PATHS)

    chat_probe_ok = False
    chat_probe_detail = ""
    if chat_route_ok:
        try:
            probe_chat = await http_client.post(
                f"{SALAD_GATEWAY_URL}/chat-reply",
                json={"chat_text": "ping", "skip_tts": True, "speech_lang": "en-in"},
                headers=salad_gateway_headers(),
                timeout=120.0,
            )
            chat_probe_ok = probe_chat.status_code == 200
            chat_probe_detail = f"HTTP {probe_chat.status_code}"
            if not chat_probe_ok:
                chat_probe_detail += f": {probe_chat.text[:120]}"
        except httpx.HTTPError as exc:
            chat_probe_detail = str(exc)
    else:
        chat_probe_detail = "chat routes missing on Salad /health"

    auth_probe_ok = False
    auth_detail = ""
    try:
        probe = await http_client.post(
            f"{SALAD_GATEWAY_URL}/tts-only",
            json={"text": "ok"},
            headers=salad_gateway_headers(),
            timeout=60.0,
        )
        auth_probe_ok = probe.status_code == 200
        auth_detail = f"HTTP {probe.status_code}"
        if not auth_probe_ok:
            auth_detail += f": {probe.text[:120]}"
    except httpx.HTTPError as exc:
        auth_detail = str(exc)

    return {
        "salad_gateway_url": SALAD_GATEWAY_URL,
        "salad_health_ok": salad_ok,
        "salad_health_detail": salad_detail,
        "salad_api_version": salad_api_version,
        "salad_tts_engine": salad_tts_engine,
        "salad_routes": salad_routes,
        "salad_has_free_chat": chat_route_ok,
        "salad_chat_probe_ok": chat_probe_ok,
        "salad_chat_probe_detail": chat_probe_detail,
        "salad_auth_ok": auth_probe_ok,
        "salad_auth_detail": auth_detail,
        "salad_gateway_key_configured": bool(SALAD_GATEWAY_API_KEY),
        "rds_writer_configured": bool(os.getenv("RDS_WRITER_URL")),
        "rds_reader_configured": bool(os.getenv("RDS_READER_URL")),
    }

# ─── Initialize Interview ────────────────────────────────────────────
@app.post("/api/initialize-interview")
async def initialize_interview(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    write_db: AsyncSession = Depends(get_write_db)
):
    pdf_bytes = await resume.read()
    session_id, context = await asyncio.to_thread(
        rag_engine.build_context_from_pdf_and_jd, pdf_bytes, job_description
    )

    resp = await http_client.post(
        f"{SALAD_GATEWAY_URL}/generate-15-qa",
        json={"context": context},
        headers=salad_gateway_headers(),
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Salad GPU Error: {resp.text}")

    questions = resp.json().get("questions", [])[:15]

    new_session = InterviewSessionDB(
        session_id=session_id,
        questions_json=json.dumps(questions),
        transcript_json="[]",
        current_index=0
    )
    write_db.add(new_session)
    await write_db.commit()

    return {"session_id": session_id, "total_questions": len(questions), "questions": questions}

@app.get("/api/interview/{session_id}")
async def get_interview_from_read_replica(
    session_id: str,
    read_db: AsyncSession = Depends(get_read_db)
):
    result = await read_db.execute(
        select(InterviewSessionDB).where(InterviewSessionDB.session_id == session_id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found in Read Replica")
    return {
        "session_id": row.session_id,
        "current_index": row.current_index,
        "questions": json.loads(row.questions_json),
        "transcript": json.loads(row.transcript_json),
        "database_source": "AWS_RDS_READ_REPLICA"
    }

# ─── Interview WebSocket (with interruption + strict rules) ──────────
@app.websocket("/ws/interview/{session_id}")
async def interview_ws(websocket: WebSocket, session_id: str, sr: int = 16000, lang: str = "en-in"):
    await websocket.accept()
    speech_lang = normalize_speech_lang(lang)

    async with ReadSessionLocal() as read_db:
        res = await read_db.execute(
            select(InterviewSessionDB).where(InterviewSessionDB.session_id == session_id)
        )
        row = res.scalar_one_or_none()
        if not row:
            await websocket.send_json({"type": "error", "message": "Session not found"})
            await websocket.close()
            return
        questions = json.loads(row.questions_json)
        current_idx = row.current_index
        transcript = json.loads(row.transcript_json)

    dsp = AudioDSPPipeline()
    start_ts = time.time()
    ai_speaking = False  # Track if TTS is being sent (for interruption)

    q1_text = f"Welcome! You have 30 minutes for 15 questions. Question 1: {questions[0]['question']}"
    await websocket.send_json({"type": "ai_text", "text": q1_text, "q_index": 1, "remaining_seconds": INTERVIEW_DURATION_SEC})

    ok_tts, tts_data = await post_salad("/tts-only", {"text": q1_text, "speech_lang": speech_lang})
    if ok_tts and tts_data.get("tts_wav_b64"):
        wav_bytes = base64.b64decode(tts_data["tts_wav_b64"])
        pcm_out = await asyncio.to_thread(dsp.process_salad_tts_wav, wav_bytes, sr)
        ai_speaking = True
        await ws_send_json(websocket, {"type": "status", "status": "ai_speaking"})
        await ws_send_bytes(websocket, pcm_out)
        await ws_send_json(websocket, {"type": "status", "status": "ai_done_speaking"})
        ai_speaking = False

    audio_buffer = bytearray()
    is_speaking = False
    silence_frames = 0
    SILENCE_LIMIT = 22

    try:
        while True:
            elapsed = time.time() - start_ts
            if elapsed >= INTERVIEW_DURATION_SEC:
                await websocket.send_json({"type": "ai_text", "text": "30 minutes completed. Thank you!", "completed": True})
                break

            msg = await ws_receive(websocket)
            if msg is None:
                break
            candidate_chat = None
            clean_pcm16_b64 = None

            if "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "chat_answer":
                    candidate_chat = data.get("text", "").strip()
                elif data.get("type") == "set_language":
                    speech_lang = normalize_speech_lang(data.get("lang"))
                    continue
                elif data.get("type") == "interrupt":
                    # User interrupted AI speech — acknowledge and reset
                    ai_speaking = False
                    audio_buffer.clear()
                    await websocket.send_json({"type": "status", "status": "interrupted"})
                    continue

            elif "bytes" in msg and msg["bytes"]:
                chunk = msg["bytes"]
                audio_buffer.extend(chunk)
                pcm_float = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                pcm_16k = np.repeat(pcm_float, 2) if sr == 8000 else pcm_float

                if len(pcm_16k) >= 512:
                    prob = vad_model(torch.from_numpy(pcm_16k[:512]), 16000).item()
                    if prob > 0.55:
                        is_speaking = True
                        silence_frames = 0
                        # If AI is speaking and user starts talking → interrupt
                        if ai_speaking:
                            ai_speaking = False
                            await websocket.send_json({"type": "status", "status": "interrupted"})
                    elif is_speaking:
                        silence_frames += 1

                if is_speaking and silence_frames < SILENCE_LIMIT:
                    await websocket.send_json({"type": "status", "status": "listening"})

                if is_speaking and silence_frames >= SILENCE_LIMIT:
                    raw_audio = bytes(audio_buffer)
                    audio_buffer.clear()
                    is_speaking = False
                    silence_frames = 0

                    if len(raw_audio) > (sr * 0.4):
                        await websocket.send_json({"type": "status", "status": "processing"})
                        clean_bytes = await asyncio.to_thread(dsp.clean_mic_to_16k_pcm_bytes, raw_audio, sr)
                        clean_pcm16_b64 = base64.b64encode(clean_bytes).decode("utf-8")

            if candidate_chat or clean_pcm16_b64:
                await websocket.send_json({"type": "status", "status": "processing"})
                cur_q = questions[current_idx]
                next_q = questions[current_idx + 1] if (current_idx + 1) < len(questions) else None

                salad_payload = {
                    "pcm16_base64": clean_pcm16_b64,
                    "chat_text": candidate_chat,
                    "current_question": cur_q["question"],
                    "reference_answer": cur_q["reference_answer"],
                    "next_question": next_q["question"] if next_q else None,
                    "speech_lang": speech_lang,
                }
                ok, turn_data = await post_salad("/process-turn", salad_payload)
                if not ws_connected(websocket):
                    break
                if not ok:
                    await websocket.send_json({
                        "type": "ai_text",
                        "text": salad_user_error(turn_data),
                    })
                    await websocket.send_json({"type": "status", "status": "idle"})
                    continue
                user_text = turn_data.get("user_transcript", "")
                ai_reply = turn_data.get("ai_text", "")
                tts_b64 = turn_data.get("tts_wav_b64", "")

                if not user_text:
                    continue

                # Send finalized transcript to frontend
                if clean_pcm16_b64:
                    await websocket.send_json({"type": "user_transcript", "text": user_text})

                transcript.append({"q": cur_q["question"], "a": user_text, "ai_feedback": ai_reply})
                current_idx += 1
                is_done = next_q is None

                async with WriteSessionLocal() as write_db:
                    await write_db.execute(
                        update(InterviewSessionDB)
                        .where(InterviewSessionDB.session_id == session_id)
                        .values(current_index=current_idx, transcript_json=json.dumps(transcript))
                    )
                    await write_db.commit()

                await websocket.send_json({
                    "type": "ai_text",
                    "text": ai_reply,
                    "q_index": min(current_idx + 1, 15),
                    "completed": is_done,
                    "remaining_seconds": int(INTERVIEW_DURATION_SEC - (time.time() - start_ts))
                })

                if tts_b64 and ws_connected(websocket):
                    wav_bytes = base64.b64decode(tts_b64)
                    pcm_out = await asyncio.to_thread(dsp.process_salad_tts_wav, wav_bytes, sr)
                    ai_speaking = True
                    await ws_send_json(websocket, {"type": "status", "status": "ai_speaking"})
                    await ws_send_bytes(websocket, pcm_out)
                    await ws_send_json(websocket, {"type": "status", "status": "ai_done_speaking"})
                    ai_speaking = False

                await ws_send_json(websocket, {"type": "status", "status": "idle"})

                if is_done:
                    break
    except (WebSocketDisconnect, RuntimeError):
        pass

# ─── Free Chat WebSocket (multilingual, any topic) ───────────────────
@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket, sr: int = 16000, lang: str = "en-in"):
    await websocket.accept()
    speech_lang = normalize_speech_lang(lang)
    dsp = AudioDSPPipeline()
    chat_history: list[dict] = []
    voice_session = False
    speak_replies = False

    audio_buffer = bytearray()
    is_speaking = False
    silence_frames = 0
    SILENCE_LIMIT = 22

    # Welcome message
    await websocket.send_json({
        "type": "ai_text",
        "text": "Hey! I'm your AI buddy. You can talk to me in English, Hindi, Marathi, Gujarati — any language! What's on your mind? 😊"
    })
    welcome_tts = (
        "Hey! I'm your AI buddy. You can talk to me in Indian English, Hindi, Marathi, or Gujarati. What's on your mind?"
        if speech_lang == "en-in"
        else "नमस्ते! मैं आपका AI साथी हूँ। बताइए, क्या सोच रहे हैं?"
    )
    ok_tts, tts_data = await post_salad("/tts-only", {"text": welcome_tts, "speech_lang": speech_lang})
    if ok_tts and tts_data.get("tts_wav_b64"):
        wav_bytes = base64.b64decode(tts_data["tts_wav_b64"])
        pcm_out = await asyncio.to_thread(dsp.process_salad_tts_wav, wav_bytes, sr)
        await ws_send_json(websocket, {"type": "status", "status": "ai_speaking"})
        await ws_send_bytes(websocket, pcm_out)
        await ws_send_json(websocket, {"type": "status", "status": "ai_done_speaking"})

    try:
        while True:
            msg = await ws_receive(websocket)
            if msg is None:
                break
            user_text_input = None
            clean_pcm16_b64 = None

            if "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "chat_message":
                    user_text_input = data.get("text", "").strip()
                    if data.get("speak_replies") is not None:
                        speak_replies = bool(data.get("speak_replies"))
                elif data.get("type") == "voice_session":
                    voice_session = True
                    speak_replies = bool(data.get("speak_replies", True))
                    continue
                elif data.get("type") == "set_language":
                    speech_lang = normalize_speech_lang(data.get("lang"))
                    continue

            elif "bytes" in msg and msg["bytes"]:
                chunk = msg["bytes"]
                audio_buffer.extend(chunk)
                pcm_float = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                pcm_16k = np.repeat(pcm_float, 2) if sr == 8000 else pcm_float

                if len(pcm_16k) >= 512:
                    prob = vad_model(torch.from_numpy(pcm_16k[:512]), 16000).item()
                    if prob > 0.55:
                        is_speaking = True
                        silence_frames = 0
                    elif is_speaking:
                        silence_frames += 1

                if is_speaking and silence_frames < SILENCE_LIMIT:
                    await websocket.send_json({"type": "status", "status": "listening"})

                if is_speaking and silence_frames >= SILENCE_LIMIT:
                    raw_audio = bytes(audio_buffer)
                    audio_buffer.clear()
                    is_speaking = False
                    silence_frames = 0

                    if len(raw_audio) > (sr * 0.4):
                        await websocket.send_json({"type": "status", "status": "processing"})
                        clean_bytes = await asyncio.to_thread(dsp.clean_mic_to_16k_pcm_bytes, raw_audio, sr)
                        clean_pcm16_b64 = base64.b64encode(clean_bytes).decode("utf-8")

            if user_text_input or clean_pcm16_b64:
                await websocket.send_json({"type": "status", "status": "processing"})
                text_only = bool(user_text_input and not clean_pcm16_b64)
                skip_tts = text_only and not (voice_session and speak_replies)
                ok, resp_data = await post_salad_chat(
                    {
                        "pcm16_base64": clean_pcm16_b64,
                        "chat_text": user_text_input,
                        "history": chat_history[-6:],
                        "skip_tts": skip_tts,
                        "speak_replies": speak_replies,
                        "speech_lang": speech_lang,
                    },
                )
                if not ws_connected(websocket):
                    break
                if not ok:
                    await websocket.send_json({
                        "type": "ai_text",
                        "text": salad_user_error(resp_data),
                    })
                    await websocket.send_json({"type": "status", "status": "idle"})
                    continue
                user_text = resp_data.get("user_transcript", "")
                ai_reply = resp_data.get("ai_text", "")
                tts_b64 = resp_data.get("tts_wav_b64", "")

                if not (user_text or user_text_input):
                    continue

                final_user = user_text or user_text_input

                if clean_pcm16_b64 and user_text:
                    await websocket.send_json({"type": "user_transcript", "text": user_text})

                chat_history.append({"role": "user", "text": final_user})
                chat_history.append({"role": "assistant", "text": ai_reply})

                if ai_reply.strip():
                    await websocket.send_json({"type": "ai_text", "text": ai_reply})
                else:
                    await websocket.send_json({
                        "type": "ai_text",
                        "text": "I didn't catch a full reply — could you say that again?",
                    })

                await websocket.send_json({"type": "status", "status": "idle"})

                if tts_b64 and ws_connected(websocket):
                    wav_bytes = base64.b64decode(tts_b64)
                    pcm_out = await asyncio.to_thread(dsp.process_salad_tts_wav, wav_bytes, sr)
                    await ws_send_json(websocket, {"type": "status", "status": "ai_speaking"})
                    await ws_send_bytes(websocket, pcm_out)
                    await ws_send_json(websocket, {"type": "status", "status": "ai_done_speaking"})

    except (WebSocketDisconnect, RuntimeError):
        pass
