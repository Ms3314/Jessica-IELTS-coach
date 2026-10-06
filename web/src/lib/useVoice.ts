import { useCallback, useEffect, useRef, useState } from "react";
import { AudioEngine } from "./audio";

export type EngineState =
  | "idle"
  | "listening"
  | "transcribing"
  | "thinking"
  | "speaking"
  | "error";

const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${
  location.host
}/ws`;

export function useVoice() {
  const engineRef = useRef<AudioEngine | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const serverListeningRef = useRef(false);
  const engineReadyRef = useRef(false);

  const [connected, setConnected] = useState(false);
  const [started, setStarted] = useState(false);
  const [state, setState] = useState<EngineState>("idle");
  const [userText, setUserText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const pauseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Live "is the user actually talking?" flag (drives the status text) plus
  // barge-in bookkeeping: engine.mic is read on a timer, so the UI re-renders
  // only when the boolean flips.
  const [voiceActive, setVoiceActive] = useState(false);
  const stateRef = useRef(state);
  stateRef.current = state;
  const ignoreAudioRef = useRef(false); // drop stale reply audio after barge-in
  const bargeTicksRef = useRef(0);

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
    engine.setCapturing(serverListeningRef.current && engineReadyRef.current);
  }, []);

  const bargeIn = useCallback(() => {
    bargeTicksRef.current = 0;
    ignoreAudioRef.current = true;
    engineRef.current?.stopPlayback(); // silence Jessica immediately
    // Start capturing right away; the server's "listening" event confirms.
    serverListeningRef.current = true;
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
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(pcm);
      }
    };

    let stopped = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (stopped) return;
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        // The server is gone, so we're no longer capturing until a new
        // session's "listening" state arrives after reconnect.
        serverListeningRef.current = false;
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
            serverListeningRef.current = s === "listening";
            if (s === "listening") ignoreAudioRef.current = false;
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
          // binary TTS PCM (Float32, 24 kHz)
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
      engine.stop();
      wsRef.current?.close();
    };
  }, [syncCapturing, clearPause, showPause]);

  // Voice-activity + barge-in watchdog: if the user starts talking over a
  // reply (3 consecutive ~120ms ticks above the threshold), cut the audio and
  // tell the server to cancel the rest of the reply.
  useEffect(() => {
    const id = setInterval(() => {
      const engine = engineRef.current;
      if (!engine || !engineReadyRef.current) return;

      const active = engine.mic > 0.09;
      setVoiceActive((prev) => (prev === active ? prev : active));

      if (stateRef.current === "speaking" && !ignoreAudioRef.current) {
        bargeTicksRef.current = active ? bargeTicksRef.current + 1 : 0;
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
      syncCapturing();
    } catch {
      setError("Microphone permission denied");
    }
  }, [syncCapturing]);

  const quit = useCallback(() => {
    // Drop the session now: stop sending, release the mic, and close the
    // socket. The server tears its engine down; the reconnect starts a fresh
    // session so Start works again.
    serverListeningRef.current = false;
    engineReadyRef.current = false;
    ignoreAudioRef.current = false;
    clearPause();
    setVoiceActive(false);
    setStarted(false);
    setState("idle");
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "quit" }));
    }
    ws?.close();
    engineRef.current?.stop();
  }, [clearPause]);

  return {
    connected,
    started,
    state,
    userText,
    paused,
    voiceActive,
    error,
    engine: engineRef,
    start,
    quit,
  };
}
