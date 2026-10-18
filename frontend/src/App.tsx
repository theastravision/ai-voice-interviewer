import React, { useState, useRef, useEffect, useCallback } from "react";

/* ─── Types ──────────────────────────────────────────────────────── */
type Message = { role: "ai" | "candidate-voice" | "candidate-chat"; text: string };
type SttStatus = "idle" | "listening" | "processing";
type Tab = "interview" | "chat";

/* ─── Theme ──────────────────────────────────────────────────────── */
const T = {
  bg: "#06080f", card: "rgba(15,23,42,0.85)", glass: "rgba(30,41,59,0.55)",
  border: "rgba(100,116,139,0.25)", accent: "#3b82f6", accentGlow: "rgba(59,130,246,0.35)",
  green: "#22c55e", red: "#ef4444", purple: "#a78bfa", cyan: "#22d3ee",
  orange: "#f59e0b", txt: "#f1f5f9", txt2: "#94a3b8", txt3: "#64748b",
};

const css = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:${T.bg};color:${T.txt}}
.fade{animation:fadeIn .4s ease-out forwards;opacity:0;transform:translateY(10px)}
@keyframes fadeIn{to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.08)}}
@keyframes ring{0%{transform:scale(.85);opacity:.6}70%,100%{transform:scale(1.4);opacity:0}}
@keyframes bars{0%,100%{height:4px}50%{height:var(--bh,20px)}}
.mic-btn{position:relative;cursor:pointer;border:none;outline:none}
.mic-btn .r{position:absolute;inset:-8px;border-radius:50%;border:3px solid ${T.accent};opacity:0}
.mic-btn.on .r{animation:ring 1.5s cubic-bezier(.22,1,.36,1) infinite}
.mic-btn.on .d{animation:pulse 1.5s ease-in-out infinite}
.bars{display:flex;align-items:flex-end;gap:3px;height:28px}
.bars span{width:4px;border-radius:2px;background:${T.accent};animation:bars .8s ease-in-out infinite}
.scroll::-webkit-scrollbar{width:5px}
.scroll::-webkit-scrollbar-track{background:transparent}
.scroll::-webkit-scrollbar-thumb{background:${T.border};border-radius:3px}
.bubble{padding:12px 16px;border-radius:14px;max-width:85%;backdrop-filter:blur(12px);border:1px solid ${T.border};line-height:1.55;font-size:14px}
.inp:focus{outline:none;border-color:${T.accent};box-shadow:0 0 0 3px ${T.accentGlow}}
.btn{border:none;border-radius:12px;padding:11px 22px;font-weight:600;cursor:pointer;font-size:13.5px;transition:all .2s}
.btn:hover{transform:translateY(-1px)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btn-p{background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;box-shadow:0 4px 14px rgba(59,130,246,.3)}
.btn-d{background:linear-gradient(135deg,#ef4444,#dc2626);color:#fff;box-shadow:0 4px 12px rgba(239,68,68,.2)}
.btn-g{background:linear-gradient(135deg,#10b981,#059669);color:#fff}
.btn-o{background:linear-gradient(135deg,#f59e0b,#d97706);color:#fff}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot.on{background:${T.green};box-shadow:0 0 8px ${T.green}}
.dot.off{background:${T.red};box-shadow:0 0 8px ${T.red}}
.badge{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:18px;font-size:11.5px;font-weight:500;backdrop-filter:blur(8px);border:1px solid ${T.border}}
.tab{padding:10px 24px;border:none;border-radius:10px 10px 0 0;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s}
.tab.active{background:${T.card};color:${T.cyan};border-bottom:2px solid ${T.cyan}}
.tab.inactive{background:transparent;color:${T.txt3}}
.tab.inactive:hover{color:${T.txt2}}
`;

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("interview");

  return (
    <>
      <style>{css}</style>
      <div style={{ maxWidth: 940, margin: "0 auto", padding: "28px 20px", minHeight: "100vh" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          <h1 style={{ fontSize: 26, fontWeight: 700, background: "linear-gradient(135deg,#3b82f6,#22d3ee)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            🎙️ AI Voice Technical Interviewer
          </h1>
          <p style={{ color: T.txt3, fontSize: 13, marginTop: 4 }}>Salad Cloud GPU • Real-time STT + Indian English TTS</p>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 0 }}>
          <button className={`tab ${activeTab === "interview" ? "active" : "inactive"}`} onClick={() => setActiveTab("interview")}>🎤 Interview</button>
          <button className={`tab ${activeTab === "chat" ? "active" : "inactive"}`} onClick={() => setActiveTab("chat")}>💬 Free Chat</button>
        </div>

        {activeTab === "interview" ? <InterviewPanel /> : <ChatPanel />}
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   INTERVIEW PANEL
   ═══════════════════════════════════════════════════════════════════ */
function InterviewPanel() {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jdText, setJdText] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [qIndex, setQIndex] = useState(1);
  const [remainingSec, setRemainingSec] = useState(1800);
  const [connected, setConnected] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [sttStatus, setSttStatus] = useState<SttStatus>("idle");
  const [audioLevel, setAudioLevel] = useState(0);
  const [aiSpeaking, setAiSpeaking] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const micActiveRef = useRef(false);
  const ttsSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animRef = useRef(0);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { micActiveRef.current = micActive; }, [micActive]);
  useEffect(() => {
    if (!connected) return;
    const t = setInterval(() => setRemainingSec((s) => (s > 0 ? s - 1 : 0)), 1000);
    return () => clearInterval(t);
  }, [connected]);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const startMeter = useCallback((a: AnalyserNode) => {
    const d = new Uint8Array(a.frequencyBinCount);
    const tick = () => { a.getByteFrequencyData(d); setAudioLevel(d.reduce((x, y) => x + y, 0) / d.length / 255); animRef.current = requestAnimationFrame(tick); };
    tick();
  }, []);

  const stopTtsPlayback = () => {
    try { ttsSourceRef.current?.stop(); } catch {}
    ttsSourceRef.current = null;
    setAiSpeaking(false);
  };

  const playTts = (audioCtx: AudioContext, pcmBytes: ArrayBuffer, sr: number) => {
    const int16 = new Int16Array(pcmBytes);
    const f32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768.0;
    const buf = audioCtx.createBuffer(1, f32.length, sr);
    buf.getChannelData(0).set(f32);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    ttsSourceRef.current = src;
    setAiSpeaking(true);
    src.onended = () => { ttsSourceRef.current = null; setAiSpeaking(false); };
    src.start();
  };

  const handleInit = async () => {
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
    } catch (e: any) { alert(e.message); } finally { setLoading(false); }
  };

  const startVoice = async (sr: 8000 | 16000) => {
    if (!sessionId) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/interview/${sessionId}?sr=${sr}`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    const audioCtx = new AudioContext({ sampleRate: sr });
    audioCtxRef.current = audioCtx;

    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const m = JSON.parse(ev.data);
        if (m.type === "ai_text") {
          setSttStatus("idle");
          setMessages((p) => [...p, { role: "ai", text: m.text }]);
          if (m.q_index) setQIndex(m.q_index);
          if (m.remaining_seconds) setRemainingSec(m.remaining_seconds);
        } else if (m.type === "user_transcript") {
          setSttStatus("idle");
          setMessages((p) => [...p, { role: "candidate-voice", text: m.text }]);
        } else if (m.type === "status") {
          if (m.status === "listening") setSttStatus("listening");
          else if (m.status === "processing") setSttStatus("processing");
          else if (m.status === "ai_done_speaking") setAiSpeaking(false);
          else if (m.status === "interrupted") { stopTtsPlayback(); setSttStatus("listening"); }
        }
      } else if (ev.data instanceof ArrayBuffer) {
        playTts(audioCtx, ev.data, sr);
      }
    };
    ws.onclose = () => { setConnected(false); setMicActive(false); setSttStatus("idle"); };

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: sr, channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      streamRef.current = stream;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser(); analyser.fftSize = 256;
      analyserRef.current = analyser; source.connect(analyser);

      const proc = audioCtx.createScriptProcessor(512, 1, 1);
      proc.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN || !micActiveRef.current) return;
        const ch = e.inputBuffer.getChannelData(0);
        // Check if user is speaking while AI is speaking → interrupt
        const rms = Math.sqrt(ch.reduce((s, v) => s + v * v, 0) / ch.length);
        if (rms > 0.02 && ttsSourceRef.current) {
          stopTtsPlayback();
          ws.send(JSON.stringify({ type: "interrupt" }));
        }
        const pcm = new Int16Array(ch.length);
        for (let i = 0; i < ch.length; i++) pcm[i] = Math.max(-1, Math.min(1, ch[i])) * 0x7fff;
        ws.send(pcm.buffer);
      };
      source.connect(proc); proc.connect(audioCtx.destination);
      setMicActive(true); micActiveRef.current = true;
      startMeter(analyser);
    } catch { console.warn("Mic unavailable; Chat-only mode."); }
    setConnected(true);
  };

  const toggleMic = () => {
    if (!streamRef.current) return;
    const next = !micActive;
    streamRef.current.getAudioTracks().forEach((t) => (t.enabled = next));
    setMicActive(next); micActiveRef.current = next;
    if (!next) setSttStatus("idle");
  };

  const endInterview = () => {
    wsRef.current?.close();
    cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    stopTtsPlayback();
    setConnected(false); setMicActive(false); setSttStatus("idle");
  };

  const sendChat = () => {
    if (!chatInput.trim() || !wsRef.current) return;
    wsRef.current.send(JSON.stringify({ type: "chat_answer", text: chatInput }));
    setMessages((p) => [...p, { role: "candidate-chat", text: chatInput }]);
    setChatInput("");
  };

  const fmtTime = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

  if (!sessionId) {
    return (
      <div className="fade" style={{ background: T.card, borderRadius: "0 20px 20px 20px", padding: 28, border: `1px solid ${T.border}`, boxShadow: "0 8px 32px rgba(0,0,0,.3)" }}>
        <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 6 }}>📄 Setup Your Interview</h2>
        <p style={{ color: T.txt2, fontSize: 13, marginBottom: 20 }}>Upload resume + paste JD to generate 15 AI interview questions.</p>
        <div style={{ border: `2px dashed ${T.border}`, borderRadius: 14, padding: 20, textAlign: "center", marginBottom: 16, cursor: "pointer", background: resumeFile ? "rgba(34,197,94,.08)" : "transparent" }}
          onClick={() => document.getElementById("r-inp")?.click()}>
          <input id="r-inp" type="file" accept=".pdf" style={{ display: "none" }} onChange={(e) => setResumeFile(e.target.files?.[0] || null)} />
          <div style={{ fontSize: 32, marginBottom: 6 }}>{resumeFile ? "✅" : "📎"}</div>
          <div style={{ fontWeight: 500, fontSize: 13.5 }}>{resumeFile ? resumeFile.name : "Click to upload Resume PDF"}</div>
        </div>
        <textarea rows={5} className="inp" placeholder="Paste Job Description here..."
          style={{ width: "100%", padding: 14, borderRadius: 12, fontSize: 13.5, background: T.glass, border: `1px solid ${T.border}`, color: T.txt, resize: "vertical", lineHeight: 1.55 }}
          value={jdText} onChange={(e) => setJdText(e.target.value)} />
        <button className="btn btn-p" onClick={handleInit} disabled={loading} style={{ marginTop: 16, width: "100%", padding: "13px 0", fontSize: 14.5 }}>
          {loading ? "⏳ Generating 15 RAG Questions via Salad GPU..." : "🚀 Generate 15 Questions & Start"}
        </button>
      </div>
    );
  }

  return (
    <div className="fade" style={{ background: T.card, borderRadius: "0 20px 20px 20px", padding: 24, border: `1px solid ${T.border}`, boxShadow: "0 8px 32px rgba(0,0,0,.3)" }}>
      {/* Top Bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 10 }}>
        <div style={{ display: "flex", gap: 12 }}>
          <Stat label="Question" value={`${qIndex}/15`} color={T.cyan} />
          <Stat label="Timer" value={fmtTime(remainingSec)} color={remainingSec < 300 ? T.red : T.green} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className={`dot ${connected ? "on" : "off"}`} />
          <span style={{ fontSize: 11, color: T.txt2 }}>{connected ? "Live" : "Off"}</span>
          {aiSpeaking && <span className="badge" style={{ background: "rgba(59,130,246,.15)", color: T.accent }}>🔊 AI Speaking</span>}
          {connected && <button className="btn btn-d" style={{ padding: "7px 14px" }} onClick={endInterview}>End</button>}
        </div>
      </div>

      {/* Connect Buttons */}
      {!connected && (
        <div className="fade" style={{ display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
          <button className="btn btn-g" style={{ flex: 1 }} onClick={() => startVoice(16000)}>🎤 16kHz Wideband Voice</button>
          <button className="btn btn-o" style={{ flex: 1 }} onClick={() => startVoice(8000)}>📡 8kHz Low-BW</button>
        </div>
      )}

      {/* Mic Bar */}
      {connected && <MicBar micActive={micActive} toggleMic={toggleMic} audioLevel={audioLevel} sttStatus={sttStatus} />}

      {/* Messages */}
      <ChatBox messages={messages} chatEndRef={chatEndRef} />

      {/* Input */}
      <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
        <input className="inp" placeholder="Type your answer..." value={chatInput}
          style={{ flex: 1, padding: "12px 16px", borderRadius: 12, fontSize: 13.5, background: T.glass, border: `1px solid ${T.border}`, color: T.txt }}
          onChange={(e) => setChatInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendChat()} />
        <button className="btn btn-p" onClick={sendChat}>Send ↗</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   FREE CHAT PANEL
   ═══════════════════════════════════════════════════════════════════ */
function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [micActive, setMicActive] = useState(false);
  const [sttStatus, setSttStatus] = useState<SttStatus>("idle");
  const [audioLevel, setAudioLevel] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const micActiveRef = useRef(false);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animRef = useRef(0);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { micActiveRef.current = micActive; }, [micActive]);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const startMeter = useCallback((a: AnalyserNode) => {
    const d = new Uint8Array(a.frequencyBinCount);
    const tick = () => { a.getByteFrequencyData(d); setAudioLevel(d.reduce((x, y) => x + y, 0) / d.length / 255); animRef.current = requestAnimationFrame(tick); };
    tick();
  }, []);

  const connectChat = async (sr: 16000 | 8000 = 16000) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/chat?sr=${sr}`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    const audioCtx = new AudioContext({ sampleRate: sr });
    audioCtxRef.current = audioCtx;

    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const m = JSON.parse(ev.data);
        if (m.type === "ai_text") {
          setSttStatus("idle");
          setMessages((p) => [...p, { role: "ai", text: m.text }]);
        } else if (m.type === "user_transcript") {
          setSttStatus("idle");
          setMessages((p) => [...p, { role: "candidate-voice", text: m.text }]);
        } else if (m.type === "status") {
          if (m.status === "listening") setSttStatus("listening");
          else if (m.status === "processing") setSttStatus("processing");
        }
      } else if (ev.data instanceof ArrayBuffer) {
        const int16 = new Int16Array(ev.data);
        const f32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768.0;
        const buf = audioCtx.createBuffer(1, f32.length, sr);
        buf.getChannelData(0).set(f32);
        const src = audioCtx.createBufferSource();
        src.buffer = buf; src.connect(audioCtx.destination); src.start();
      }
    };
    ws.onclose = () => { setConnected(false); setMicActive(false); };

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: sr, channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      streamRef.current = stream;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser(); analyser.fftSize = 256;
      analyserRef.current = analyser; source.connect(analyser);
      const proc = audioCtx.createScriptProcessor(512, 1, 1);
      proc.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN || !micActiveRef.current) return;
        const ch = e.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(ch.length);
        for (let i = 0; i < ch.length; i++) pcm[i] = Math.max(-1, Math.min(1, ch[i])) * 0x7fff;
        ws.send(pcm.buffer);
      };
      source.connect(proc); proc.connect(audioCtx.destination);
      setMicActive(true); micActiveRef.current = true;
      startMeter(analyser);
    } catch { console.warn("Mic unavailable; text-only mode."); }
    setConnected(true);
  };

  const toggleMic = () => {
    if (!streamRef.current) return;
    const next = !micActive;
    streamRef.current.getAudioTracks().forEach((t) => (t.enabled = next));
    setMicActive(next); micActiveRef.current = next;
    if (!next) setSttStatus("idle");
  };

  const sendChat = () => {
    if (!chatInput.trim() || !wsRef.current) return;
    wsRef.current.send(JSON.stringify({ type: "chat_message", text: chatInput }));
    setMessages((p) => [...p, { role: "candidate-chat", text: chatInput }]);
    setChatInput("");
  };

  const disconnect = () => {
    wsRef.current?.close(); cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setConnected(false); setMicActive(false);
  };

  return (
    <div className="fade" style={{ background: T.card, borderRadius: "0 20px 20px 20px", padding: 24, border: `1px solid ${T.border}`, boxShadow: "0 8px 32px rgba(0,0,0,.3)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div>
          <h2 style={{ fontSize: 17, fontWeight: 600 }}>💬 Free Conversation</h2>
          <p style={{ color: T.txt3, fontSize: 12, marginTop: 2 }}>Talk in English, Hindi, Marathi, Gujarati — any language!</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className={`dot ${connected ? "on" : "off"}`} />
          {connected && <button className="btn btn-d" style={{ padding: "7px 14px" }} onClick={disconnect}>Disconnect</button>}
        </div>
      </div>

      {!connected && (
        <div className="fade" style={{ display: "flex", gap: 10, marginBottom: 16 }}>
          <button className="btn btn-g" style={{ flex: 1 }} onClick={() => connectChat(16000)}>🎤 Connect Voice + Chat</button>
          <button className="btn btn-o" style={{ flex: 1 }} onClick={() => connectChat(8000)}>📡 Low-Bandwidth</button>
        </div>
      )}

      {connected && <MicBar micActive={micActive} toggleMic={toggleMic} audioLevel={audioLevel} sttStatus={sttStatus} />}

      <ChatBox messages={messages} chatEndRef={chatEndRef} />

      <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
        <input className="inp" placeholder="Type in any language..." value={chatInput}
          style={{ flex: 1, padding: "12px 16px", borderRadius: 12, fontSize: 13.5, background: T.glass, border: `1px solid ${T.border}`, color: T.txt }}
          onChange={(e) => setChatInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendChat()} />
        <button className="btn btn-p" onClick={sendChat}>Send ↗</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   SHARED COMPONENTS
   ═══════════════════════════════════════════════════════════════════ */
function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ background: T.glass, padding: "6px 14px", borderRadius: 10, border: `1px solid ${T.border}` }}>
      <span style={{ color: T.txt2, fontSize: 11 }}>{label}</span>
      <div style={{ fontWeight: 700, fontSize: 16, color }}>{value}</div>
    </div>
  );
}

function MicBar({ micActive, toggleMic, audioLevel, sttStatus }: { micActive: boolean; toggleMic: () => void; audioLevel: number; sttStatus: SttStatus }) {
  return (
    <div className="fade" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 20, marginBottom: 16, padding: "14px 20px", borderRadius: 14, background: T.glass, border: `1px solid ${T.border}` }}>
      {/* Audio Bars */}
      <div className="bars" style={{ opacity: micActive ? 1 : 0.2 }}>
        {[0, 1, 2, 3, 4].map((i) => (
          <span key={i} style={{ "--bh": `${Math.max(4, audioLevel * 28 * (0.5 + Math.random() * 0.5))}px`, animationDelay: `${i * 0.12}s` } as React.CSSProperties} />
        ))}
      </div>

      {/* Mic Button */}
      <button className={`mic-btn ${micActive ? "on" : ""}`} onClick={toggleMic}
        style={{
          width: 56, height: 56, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
          background: micActive ? "linear-gradient(135deg,#ef4444,#dc2626)" : "linear-gradient(135deg,#22c55e,#16a34a)",
          boxShadow: micActive ? "0 0 20px rgba(239,68,68,.35)" : "0 0 20px rgba(34,197,94,.25)",
          transition: "all .3s",
        }}>
        <span className="r" /><span className="d" style={{ fontSize: 24 }}>{micActive ? "🎤" : "🔇"}</span>
      </button>

      {/* Status */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {sttStatus !== "idle" && (
          <div className="badge fade" style={{
            background: sttStatus === "listening" ? "rgba(34,197,94,.12)" : "rgba(59,130,246,.12)",
            color: sttStatus === "listening" ? T.green : T.accent,
          }}>
            {sttStatus === "listening" ? "🎤 Listening..." : "⏳ Processing..."}
          </div>
        )}
        <span style={{ color: T.txt3, fontSize: 10.5 }}>{micActive ? "Tap to mute" : "Tap to unmute"}</span>
      </div>
    </div>
  );
}

function ChatBox({ messages, chatEndRef }: { messages: Message[]; chatEndRef: React.RefObject<HTMLDivElement> }) {
  return (
    <div className="scroll" style={{ height: 360, overflowY: "auto", borderRadius: 14, padding: 14, background: "rgba(2,6,18,.55)", border: `1px solid ${T.border}` }}>
      {messages.length === 0 && (
        <div style={{ textAlign: "center", color: T.txt3, marginTop: 120, fontSize: 13 }}>Messages will appear here...</div>
      )}
      {messages.map((m, i) => {
        const isAI = m.role === "ai";
        const isVoice = m.role === "candidate-voice";
        const bg = isAI ? "rgba(59,130,246,.1)" : isVoice ? "rgba(167,139,250,.1)" : "rgba(34,197,94,.1)";
        const bc = isAI ? "rgba(59,130,246,.18)" : isVoice ? "rgba(167,139,250,.18)" : "rgba(34,197,94,.18)";
        const lc = isAI ? T.cyan : isVoice ? T.purple : T.green;
        const icon = isAI ? "🤖" : isVoice ? "🎤" : "⌨️";
        const label = isAI ? "AI Interviewer" : isVoice ? "You (Voice)" : "You (Chat)";
        return (
          <div key={i} className="fade" style={{ display: "flex", justifyContent: isAI ? "flex-start" : "flex-end", marginBottom: 12, animationDelay: `${i * 0.04}s` }}>
            <div className="bubble" style={{ background: bg, borderColor: bc }}>
              <div style={{ fontSize: 10.5, fontWeight: 600, marginBottom: 5, color: lc }}>{icon} {label}</div>
              <div>{m.text}</div>
            </div>
          </div>
        );
      })}
      <div ref={chatEndRef} />
    </div>
  );
}