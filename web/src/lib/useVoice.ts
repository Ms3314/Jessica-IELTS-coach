import { useCallback, useEffect, useRef, useState } from "react";
import { AudioEngine } from "./audio";

export type EngineState =
  | "idle"
  | "listening"
  | "transcribing"
  | "thinking"
  | "speaking"
  | "error";

const WS_BASE = `${location.protocol === "https:" ? "wss" : "ws"}://${
  location.host
}/ws`;
const ACTIVE_SESSION_KEY = "speakielts.activeSession";

/** The browser's active voice session (a LangGraph thread in the server's SQLite). */
function loadActiveSessionId(): string {
  const existing = localStorage.getItem(ACTIVE_SESSION_KEY);
  if (existing) return existing;
  const fresh = crypto.randomUUID();
  localStorage.setItem(ACTIVE_SESSION_KEY, fresh);
  return fresh;
}

export function useVoice() {
  const engineRef = useRef<AudioEngine | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // Sent as ?session=... so reconnects resume the same conversation memory.
  const sessionRef = useRef(loadActiveSessionId());
  const [sessionId, setSessionId] = useState(sessionRef.current);
  // True while the server wants mic audio: listening, and also transcribing /
  // thinking — if it finalized too early, the user keeps talking and the server
  // must still hear them to stitch the continuation.
  const captureRef = useRef(false);
  const engineReadyRef = useRef(false);

  const [connected, setConnected] = useState(false);
  const [started, setStarted] = useState(false);
  const [state, setState] = useState<EngineState>("idle");
  const [userText, setUserText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const pauseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Live "is the user actually talking?" flag (drives the status text), a
  // "Jessica is audible" flag (the server finishes synthesizing before the
  // browser finishes playing, so the client owns this), and barge-in
  // bookkeeping: levels are read on a timer, so the UI re-renders only when a
  // boolean flips.
  const [voiceActive, setVoiceActive] = useState(false);
  const [ttsActive, setTtsActive] = useState(false);
  const [micMuted, setMicMuted] = useState(false);
  const stateRef = useRef(state);
  stateRef.current = state;
  const ignoreAudioRef = useRef(false); // drop stale reply audio after barge-in
  const ignoreTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const bargeTicksRef = useRef(0);
  const micMutedRef = useRef(false);
  micMutedRef.current = micMuted;

  const clearPause = useCallback(() => {
    if (pauseTimerRef.current) {
      clearTimeout(pauseTimerRef.current);
      pauseTimerRef.current = null;
    }
    setPaused(false);
  }, []);

  const showPause = useCallback(() => {
    if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
    setPaused(true);
    // transient cue: fades if the user stays silent a moment longer
    pauseTimerRef.current = setTimeout(() => setPaused(false), 2200);
  }, []);

  const syncCapturing = useCallback(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.setCapturing(
      captureRef.current && engineReadyRef.current && !micMutedRef.current
    );
  }, []);

  const bargeIn = useCallback(() => {
    bargeTicksRef.current = 0;
    setTtsActive(false); // teal ends immediately
    // Drop any reply audio still in flight; cleared shortly after in case the
    // server never sends another "listening" event (e.g. it was already done).
    ignoreAudioRef.current = true;
    if (ignoreTimerRef.current) clearTimeout(ignoreTimerRef.current);
    ignoreTimerRef.current = setTimeout(() => {
      ignoreAudioRef.current = false;
      ignoreTimerRef.current = null;
    }, 1500);
    engineRef.current?.stopPlayback(); // silence Jessica immediately
    // Keep capturing; the server will record the continuation from here.
    captureRef.current = true;
    syncCapturing();
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "barge_in" }));
    }
    setState("listening");
  }, [syncCapturing]);

  useEffect(() => {
    const engine = new AudioEngine();
    engineRef.current = engine;
    engine.onPCM = (pcm) => {
      // Mic mute: keep capturing locally for levels, but don't send audio to
      // the server while the user is muted.
      if (micMutedRef.current) return;
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(pcm);
      }
    };

    let stopped = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (stopped) return;
      const url = `${WS_BASE}?session=${encodeURIComponent(sessionRef.current)}`;
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        // A fresh socket means any "ignore stale audio" window is over.
        ignoreAudioRef.current = false;
        setConnected(true);
      };
      ws.onclose = () => {
        setConnected(false);
        // The server is gone, so we're no longer capturing until a new
        // session's state arrives after reconnect.
        captureRef.current = false;
        syncCapturing();
        // Reconnect (the dev server reloads on file changes, which drops the
        // socket; the browser must reconnect to stay live).
        if (!stopped) reconnectTimer = setTimeout(connect, 1000);
      };
      ws.onerror = () => ws.close();

      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          const msg = JSON.parse(ev.data);
          if (msg.type === "state") {
            const s = msg.state as EngineState;
            // Mic stays live through transcribing/thinking so a continuation
            // after an early finalize still reaches the server.
            captureRef.current =
              s === "listening" || s === "transcribing" || s === "thinking";
            if (s !== "listening") clearPause();
            setState(s);
            syncCapturing();
          } else if (msg.type === "pause") {
            showPause();
          } else if (msg.type === "user") {
            // Voice-only UI: keep just the latest thing the user said, shown
            // as a grey caption (no chat bubbles).
            setUserText(msg.text as string);
          } else if (msg.type === "error") {
            setError(msg.message);
          }
        } else if (!ignoreAudioRef.current) {
          // binary TTS PCM (Float32, 24 kHz) — agent voice always plays
          const f32 = new Float32Array(ev.data as ArrayBuffer);
          engine.playTTS(f32);
        }
      };
    };

    connect();

    return () => {
      stopped = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (pauseTimerRef.current) clearTimeout(pauseTimerRef.current);
      if (ignoreTimerRef.current) clearTimeout(ignoreTimerRef.current);
      engine.stop();
      wsRef.current?.close();
    };
  }, [syncCapturing, clearPause, showPause]);

  // Voice-activity + "Jessica is audible" watchdog. The server emits "idle"
  // and "listening" as soon as synthesis finishes — but the browser is still
  // playing the tail, so we keep the speaking look until playback truly ends.
  // Also triggers barge-in: talking over her for ~3 ticks cuts the audio.
  useEffect(() => {
    let lastAudible = 0;
    const id = setInterval(() => {
      const engine = engineRef.current;
      if (!engine || !engineReadyRef.current) return;
      const now = performance.now();

      const audible = engine.ttsPlaying || engine.mouth > 0.04;
      if (audible) lastAudible = now;
      const speakingNow = audible || now - lastAudible < 350; // bridge gaps
      setTtsActive((prev) => (prev === speakingNow ? prev : speakingNow));

      const mic = engine.mic;
      const active = mic > 0.09;
      setVoiceActive((prev) => (prev === active ? prev : active));

      const st = stateRef.current;
      const bargeable =
        (st === "transcribing" || st === "thinking" || st === "speaking" || speakingNow) &&
        !ignoreAudioRef.current;
      if (bargeable) {
        // While she's audible, require the user to clearly exceed the
        // (echo-cancelled) playback bleed before counting it as an interruption.
        const threshold = speakingNow ? Math.max(0.1, engine.mouth * 0.5) : 0.1;
        bargeTicksRef.current = mic > threshold ? bargeTicksRef.current + 1 : 0;
        if (bargeTicksRef.current >= 3) bargeIn();
      } else {
        bargeTicksRef.current = 0;
      }
    }, 120);
    return () => clearInterval(id);
  }, [bargeIn]);

  const start = useCallback(async () => {
    setError(null);
    try {
      await engineRef.current?.init();
      engineReadyRef.current = true;
      setStarted(true);
      // Tell the server to spin up the engine and play the opening question.
      wsRef.current?.send(JSON.stringify({ type: "start" }));
      syncCapturing();
    } catch {
      setError("Microphone permission denied");
    }
  }, [syncCapturing]);

  const quit = useCallback(() => {
    // End this call: stop sending, release the mic, and close the socket. The
    // server tears its engine down; the reconnect resumes the same session
    // memory (the id lives in localStorage) so Start picks up where you left.
    captureRef.current = false;
    engineReadyRef.current = false;
    ignoreAudioRef.current = false;
    if (ignoreTimerRef.current) {
      clearTimeout(ignoreTimerRef.current);
      ignoreTimerRef.current = null;
    }
    clearPause();
    setVoiceActive(false);
    setTtsActive(false);
    setStarted(false);
    setState("idle");
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "quit" }));
    }
    ws?.close();
    engineRef.current?.stop();
  }, [clearPause]);

  const switchSession = useCallback(
    (id: string) => {
      if (id === sessionRef.current) return;
      sessionRef.current = id;
      localStorage.setItem(ACTIVE_SESSION_KEY, id);
      setSessionId(id);
      if (ignoreTimerRef.current) clearTimeout(ignoreTimerRef.current);
      ignoreAudioRef.current = true; // cleared when the new socket opens
      engineRef.current?.stopPlayback();
      clearPause();
      setVoiceActive(false);
      setTtsActive(false);
      setUserText(null);
      setError(null);
      setState("idle");
      setStarted(false);
      wsRef.current?.close();
    },
    [clearPause]
  );

  const newSession = useCallback(() => {
    // Fresh LangGraph thread on the server: new id, then reconnect (the
    // socket's onclose schedules it) so the request carries the new session.
    switchSession(crypto.randomUUID());
  }, [switchSession]);

  const toggleMicMute = useCallback(() => {
    setMicMuted((prev) => {
      const next = !prev;
      micMutedRef.current = next; // apply immediately for syncCapturing
      syncCapturing();
      return next;
    });
  }, [syncCapturing]);

  return {
    connected,
    started,
    state,
    userText,
    paused,
    voiceActive,
    ttsActive,
    micMuted,
    error,
    engine: engineRef,
    sessionId,
    start,
    quit,
    newSession,
    switchSession,
    toggleMicMute,
  };
}
