import React, { useState, useRef, useEffect } from "react";

export default function App() {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jdText, setJdText] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<{ role: string; text: string }[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [qIndex, setQIndex] = useState(1);
  const [remainingSec, setRemainingSec] = useState(1800);
  const [connected, setConnected] = useState(false);
  const [replicaStatus, setReplicaStatus] = useState<string>("");

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!connected) return;
    const t = setInterval(() => setRemainingSec((s) => (s > 0 ? s - 1 : 0)), 1000);
    return () => clearInterval(t);
  }, [connected]);

  const handleInitInterview = async () => {
    if (!resumeFile || !jdText.trim()) return alert("Upload Resume PDF and enter JD!");
    setLoading(true);
    try {
      const form = new FormData();
      form.append("resume", resumeFile);
      form.append("job_description", jdText);
      const res = await fetch("/api/initialize-interview", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Initialization failed");
      setSessionId(data.session_id);
    } catch (e: any) {
      alert(e.message);
    } finally {
      setLoading(false);
    }
  };

  const checkReadReplicaSync = async () => {
    if (!sessionId) return;
    const res = await fetch(`/api/interview/${sessionId}`);
    const data = await res.json();
    setReplicaStatus(`Synced! Read from ${data.database_source} (Question Index: ${data.current_index})`);
  };

  const startVoiceSession = async (sr: 8000 | 16000) => {
    if (!sessionId) return;
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/interview/${sessionId}?sr=${sr}`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    const audioCtx = new AudioContext({ sampleRate: sr });

    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        if (msg.type === "ai_text") {
          setMessages((p) => [...p, { role: "AI Interviewer", text: msg.text }]);
          if (msg.q_index) setQIndex(msg.q_index);
          if (msg.remaining_seconds) setRemainingSec(msg.remaining_seconds);
        } else if (msg.type === "user_transcript") {
          setMessages((p) => [...p, { role: "Candidate (Voice)", text: msg.text }]);
        }
      } else if (ev.data instanceof ArrayBuffer) {
        const int16 = new Int16Array(ev.data);
        const f32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768.0;
        const buf = audioCtx.createBuffer(1, f32.length, sr);
        buf.getChannelData(0).set(f32);
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(audioCtx.destination);
        src.start();
      }
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: sr, channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
      const src = audioCtx.createMediaStreamSource(stream);
      const proc = audioCtx.createScriptProcessor(512, 1, 1);
      proc.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const ch = e.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(ch.length);
        for (let i = 0; i < ch.length; i++) pcm[i] = Math.max(-1, Math.min(1, ch[i])) * 0x7fff;
        ws.send(pcm.buffer);
      };
      src.connect(proc);
      proc.connect(audioCtx.destination);
    } catch {
      console.warn("Mic unavailable or HTTP origin; falling back to Chat-only mode with TTS output.");
    }
    setConnected(true);
  };

  const sendChat = () => {
    if (!chatInput.trim() || !wsRef.current) return;
    wsRef.current.send(JSON.stringify({ type: "chat_answer", text: chatInput }));
    setMessages((p) => [...p, { role: "Candidate (Chat)", text: chatInput }]);
    setChatInput("");
  };

  return (
    <div style={{ maxWidth: 860, margin: "30px auto", padding: 24, fontFamily: "system-ui" }}>
      <h1>AI Voice + Chat Technical Interviewer</h1>
      {!sessionId ? (
        <div style={{ background: "#1e293b", padding: 24, borderRadius: 12 }}>
          <h3>Step 1: Upload Candidate Resume (PDF) & Job Description</h3>
          <input type="file" accept=".pdf" onChange={(e) => setResumeFile(e.target.files?.[0] || null)} />
          <textarea
            rows={6}
            style={{ width: "100%", marginTop: 12, padding: 10, borderRadius: 6 }}
            placeholder="Paste Job Description here..."
            value={jdText}
            onChange={(e) => setJdText(e.target.value)}
          />
          <button
            onClick={handleInitInterview}
            disabled={loading}
            style={{ marginTop: 12, padding: "10px 20px", background: "#3b82f6", color: "#fff", border: "none", borderRadius: 6 }}
          >
            {loading ? "Generating 15 RAG Q&As via Salad GPU..." : "Generate 15 Questions & Initialize"}
          </button>
        </div>
      ) : (
        <div style={{ background: "#1e293b", padding: 24, borderRadius: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
            <span><strong>Question:</strong> {qIndex} / 15</span>
            <span><strong>Timer:</strong> {Math.floor(remainingSec / 60)}:{String(remainingSec % 60).padStart(2, "0")}</span>
            <button onClick={checkReadReplicaSync} style={{ padding: "4px 10px" }}>Test RDS Read Replica Sync</button>
          </div>
          {replicaStatus && <p style={{ color: "#4ade80", fontSize: 13 }}>{replicaStatus}</p>}

          {!connected && (
            <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
              <button onClick={() => startVoiceSession(16000)} style={{ padding: 10, background: "#10b981", color: "#fff", border: "none", borderRadius: 6 }}>
                Connect 16kHz Wideband Voice + Chat
              </button>
              <button onClick={() => startVoiceSession(8000)} style={{ padding: 10, background: "#f59e0b", color: "#fff", border: "none", borderRadius: 6 }}>
                Connect 8kHz Low-Bandwidth Voice + Chat
              </button>
            </div>
          )}

          <div style={{ height: 380, overflowY: "auto", background: "#0f172a", padding: 16, borderRadius: 8 }}>
            {messages.map((m, i) => (
              <div key={i} style={{ marginBottom: 12 }}>
                <strong style={{ color: m.role.includes("AI") ? "#38bdf8" : "#a78bfa" }}>{m.role}: </strong>
                <span>{m.text}</span>
              </div>
            ))}
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            <input
              style={{ flex: 1, padding: 10, borderRadius: 6 }}
              placeholder="Speak into mic OR type your answer here..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendChat()}
            />
            <button onClick={sendChat} style={{ padding: "10px 18px", background: "#3b82f6", color: "#fff", border: "none", borderRadius: 6 }}>
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
