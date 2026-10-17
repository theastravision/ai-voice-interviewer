import os
import json
import time
import base64
import asyncio
import numpy as np
import torch
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Depends, HTTPException
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
INTERVIEW_DURATION_SEC = 1800

rag_engine = ChromaRAGEngine()
http_client = httpx.AsyncClient(timeout=45.0)
vad_model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=True)

@app.on_event("startup")
async def startup_event():
    async with writer_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "aws-fastapi-orchestrator"}

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
        headers={"x-api-key": SALAD_API_SECRET}
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

@app.websocket("/ws/interview/{session_id}")
async def interview_ws(websocket: WebSocket, session_id: str, sr: int = 16000):
    await websocket.accept()

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

    q1_text = f"Welcome! You have 30 minutes for 15 questions. Question 1: {questions[0]['question']}"
    await websocket.send_json({"type": "ai_text", "text": q1_text, "q_index": 1, "remaining_seconds": INTERVIEW_DURATION_SEC})

    tts_resp = await http_client.post(
        f"{SALAD_GATEWAY_URL}/tts-only",
        json={"text": q1_text},
        headers={"x-api-key": SALAD_API_SECRET}
    )
    if tts_resp.status_code == 200:
        wav_bytes = base64.b64decode(tts_resp.json()["tts_wav_b64"])
        pcm_out = await asyncio.to_thread(dsp.process_salad_tts_wav, wav_bytes, sr)
        await websocket.send_bytes(pcm_out)

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

            msg = await websocket.receive()
            candidate_chat = None
            clean_pcm16_b64 = None

            if "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "chat_answer":
                    candidate_chat = data.get("text", "").strip()

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

                if is_speaking and silence_frames >= SILENCE_LIMIT:
                    raw_audio = bytes(audio_buffer)
                    audio_buffer.clear()
                    is_speaking = False
                    silence_frames = 0

                    if len(raw_audio) > (sr * 0.4):
                        clean_bytes = await asyncio.to_thread(dsp.clean_mic_to_16k_pcm_bytes, raw_audio, sr)
                        clean_pcm16_b64 = base64.b64encode(clean_bytes).decode("utf-8")

            if candidate_chat or clean_pcm16_b64:
                cur_q = questions[current_idx]
                next_q = questions[current_idx + 1] if (current_idx + 1) < len(questions) else None

                salad_payload = {
                    "pcm16_base64": clean_pcm16_b64,
                    "chat_text": candidate_chat,
                    "current_question": cur_q["question"],
                    "reference_answer": cur_q["reference_answer"],
                    "next_question": next_q["question"] if next_q else None
                }
                turn_resp = await http_client.post(
                    f"{SALAD_GATEWAY_URL}/process-turn",
                    json=salad_payload,
                    headers={"x-api-key": SALAD_API_SECRET}
                )
                turn_data = turn_resp.json()
                user_text = turn_data.get("user_transcript", "")
                ai_reply = turn_data.get("ai_text", "")
                tts_b64 = turn_data.get("tts_wav_b64", "")

                if not user_text:
                    continue

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

                if tts_b64:
                    wav_bytes = base64.b64decode(tts_b64)
                    pcm_out = await asyncio.to_thread(dsp.process_salad_tts_wav, wav_bytes, sr)
                    await websocket.send_bytes(pcm_out)

                if is_done:
                    break
    except WebSocketDisconnect:
        pass
