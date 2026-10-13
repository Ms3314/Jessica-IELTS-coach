import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { AudioEngine } from "./audio-engine";
import { INITIAL_EXAM, examReducer, type ExamEvent } from "@/lib/exam";
import type { ExamLength } from "@/lib/settings";
import { addAgentSentence, addUserText, markInterrupted, type Line } from "@/lib/transcript";

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
export const ACTIVE_SESSION_KEY = "speakielts.activeSession";
// Loud-enough checks (one per 120 ms) in a row before talking over Jessica cuts
// her off. ~0.6 s: a door slam or a pen drop dies out sooner, real speech doesn't.
const BARGE_TICKS = 5;

/** The browser's active voice session (a LangGraph thread in the server's SQLite). */
function loadActiveSessionId(): string {
  const existing = localStorage.getItem(ACTIVE_SESSION_KEY);
  if (existing) return existing;
  const fresh = crypto.randomUUID();
  localStorage.setItem(ACTIVE_SESSION_KEY, fresh);
  return fresh;
}

export interface VoiceOptions {
  /** Allow cutting Jessica off by talking over her (the Stop button always works). */
  voiceInterrupt: boolean;
  /** Seconds of quiet before the server decides the user has finished. */
  pauseS: number;
  /** "chat" = free conversation, "ielts" = the mock test. */
  mode: "chat" | "ielts";
  /** Which mock test to take (only used when mode is "ielts"). */
  examLength: ExamLength;
}

export function useVoice({ voiceInterrupt, pauseS, mode, examLength }: VoiceOptions) {
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
  // True from pressing start until the first sound from Jessica reaches the
  // browser: the server needs a moment to produce it, and until then the screen
  // would look live while nothing is audible.
  const [preparing, setPreparing] = useState(false);
  const preparingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const clearPreparing = useCallback(() => {
    if (preparingTimerRef.current) {
      clearTimeout(preparingTimerRef.current);
      preparingTimerRef.current = null;
    }
    setPreparing(false);
  }, []);
  const [state, setState] = useState<EngineState>("idle");
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
  const voiceInterruptRef = useRef(voiceInterrupt);
  voiceInterruptRef.current = voiceInterrupt;
  const pauseSRef = useRef(pauseS);
  pauseSRef.current = pauseS;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const examLengthRef = useRef(examLength);
  examLengthRef.current = examLength;

  // In the mock test the candidate records by hand: the microphone is only open
  // between pressing Record and pressing Stop, and nothing else decides when an
  // answer ends.
  const [recording, setRecording] = useState(false);
  const recordingRef = useRef(false);

  // IELTS mock test progress, driven by the server's "exam" events.
  const [exam, dispatchExam] = useReducer(examReducer, INITIAL_EXAM);
  const examRef = useRef(exam);
  examRef.current = exam;
  // The conversation so far (shown live during the call).
  const [transcript, setTranscript] = useState<Line[]>([]);
  const freshLineRef = useRef(true);

  // Changing the pause in Settings mid-call takes effect immediately.
  useEffect(() => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "settings", pause_s: pauseS }));
    }
  }, [pauseS]);

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
    const wanted = modeRef.current === "ielts" ? recordingRef.current : captureRef.current;
    engine.setCapturing(wanted && engineReadyRef.current && !micMutedRef.current);
  }, []);

  const setRecordingNow = useCallback(
    (on: boolean) => {
      recordingRef.current = on;
      setRecording(on);
      syncCapturing();
    },
    [syncCapturing]
  );

  // by: "voice" = the user talked over her, "button" = pressed Stop. The server
  // stitches a voice interruption onto the previous message but treats a Stop
  // as the start of a fresh one.
  const bargeIn = useCallback((by: "voice" | "button") => {
    // What the user actually heard, read BEFORE playback is torn down.
    const heardIdx = engineRef.current?.lastStartedIdx ?? -1;
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
    setTranscript((prev) => markInterrupted(prev));
    // Keep capturing; the server will record the continuation from here.
    captureRef.current = true;
    syncCapturing();
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "barge_in", by, heard_idx: heardIdx }));
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
        // The server numbers sentences per socket, so forget the old count.
        if (engineRef.current) engineRef.current.lastStartedIdx = -1;
        setConnected(true);
      };
      ws.onclose = () => {
        clearPreparing();
        recordingRef.current = false;
        setRecording(false);
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
            if (s !== "speaking") freshLineRef.current = true; // next sentence starts a new line
            // Mic stays live through transcribing/thinking so a continuation
            // after an early finalize still reaches the server.
            captureRef.current =
              s === "listening" || s === "transcribing" || s === "thinking";
            if (s !== "listening") clearPause();
            setState(s);
            syncCapturing();
          } else if (msg.type === "exam") {
            dispatchExam({ type: "event", event: msg as ExamEvent, now: Date.now() });
            if (msg.event === "answer_end") setRecordingNow(false);
          } else if (msg.type === "agent") {
            const fresh = freshLineRef.current;
            setTranscript((prev) => addAgentSentence(prev, msg.text as string, fresh));
            freshLineRef.current = false;
            // Sentence boundary in the audio stream: lets us tell the server
            // which sentences were really played if the user interrupts.
            if (typeof msg.idx === "number" && !ignoreAudioRef.current) {
              engine.markSentence(msg.idx);
            }
          } else if (msg.type === "pause") {
            showPause();
          } else if (msg.type === "user") {
            setTranscript((prev) => addUserText(prev, msg.text as string));
          } else if (msg.type === "error") {
            clearPreparing();
            setError(msg.message);
          }
        } else if (!ignoreAudioRef.current) {
          // binary TTS PCM (Float32, 24 kHz) — agent voice always plays
          const f32 = new Float32Array(ev.data as ArrayBuffer);
          engine.playTTS(f32);
          clearPreparing(); // the first voice is here: show the live screen
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
  }, [syncCapturing, clearPause, showPause, clearPreparing, setRecordingNow]);

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
        voiceInterruptRef.current &&
        modeRef.current !== "ielts" &&
        (st === "transcribing" || st === "thinking" || st === "speaking" || speakingNow) &&
        !ignoreAudioRef.current;
      if (bargeable) {
        // While she's audible, require the user to clearly exceed the
        // (echo-cancelled) playback bleed before counting it as an interruption.
        const threshold = speakingNow ? Math.max(0.1, engine.mouth * 0.5) : 0.1;
        bargeTicksRef.current = mic > threshold ? bargeTicksRef.current + 1 : 0;
        if (bargeTicksRef.current >= BARGE_TICKS) bargeIn("voice");
      } else {
        bargeTicksRef.current = 0;
      }
    }, 120);
    return () => clearInterval(id);
  }, [bargeIn]);

  /** Stop button: cut Jessica off on purpose (her answer was wrong/irrelevant). */
  const stopReply = useCallback(() => bargeIn("button"), [bargeIn]);

  const start = useCallback(async () => {
    setError(null);
    try {
      await engineRef.current?.init();
      engineReadyRef.current = true;
      setStarted(true);
      setPreparing(true);
      // never leave the loader up forever if the voice doesn't come
      if (preparingTimerRef.current) clearTimeout(preparingTimerRef.current);
      preparingTimerRef.current = setTimeout(() => setPreparing(false), 25000);
      if (modeRef.current === "ielts") dispatchExam({ type: "reset" });
      setTranscript([]);
      // Tell the server to spin up the engine and play the opening question.
      wsRef.current?.send(
        JSON.stringify({
          type: "start",
          mode: modeRef.current,
          exam_length: examLengthRef.current,
          settings: { pause_s: pauseSRef.current },
        })
      );
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
    clearPreparing();
    setRecordingNow(false);
    dispatchExam({ type: "ended" }); // hanging up mid-test ends it (unfinished)
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
  }, [clearPause, clearPreparing, setRecordingNow]);

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
      setTranscript([]);
      setError(null);
      setState("idle");
      setStarted(false);
      clearPreparing();
      setRecordingNow(false);
      dispatchExam({ type: "reset" });
      wsRef.current?.close();
    },
    [clearPause, clearPreparing, setRecordingNow]
  );

  const newSession = useCallback(() => {
    // Fresh LangGraph thread on the server: new id, then reconnect (the
    // socket's onclose schedules it) so the request carries the new session.
    switchSession(crypto.randomUUID());
  }, [switchSession]);

  // Once a test is over the report is made on the server and may arrive after
  // the socket is gone, so fetch it until it appears.
  const { complete, examId, report: examReport, reportError } = exam;
  useEffect(() => {
    if (!complete || !examId || examReport || reportError) return;
    let stopped = false;
    const deadline = Date.now() + 10 * 60 * 1000;
    const poll = async () => {
      try {
        const res = await fetch(`/exams/${encodeURIComponent(sessionRef.current)}/${encodeURIComponent(examId)}`);
        if (res.ok) {
          const data = await res.json();
          if (data.report && !stopped) {
            dispatchExam(
              data.report.error
                ? { type: "reportError", message: data.report.error }
                : { type: "report", report: data.report }
            );
            return;
          }
        }
      } catch {
        /* server restarting: keep trying */
      }
      if (!stopped && Date.now() < deadline) timer = setTimeout(poll, 3000);
      else if (!stopped) dispatchExam({ type: "reportError", message: "The report is taking too long. Check back later." });
    };
    let timer = setTimeout(poll, 1500);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [complete, examId, examReport, reportError]);

  const resetExam = useCallback(() => dispatchExam({ type: "reset" }), []);

  // When the test is over, hang up by itself: the screen goes straight to the
  // analysis instead of waiting for the candidate to press End.
  useEffect(() => {
    if (complete && started && modeRef.current === "ielts") quit();
  }, [complete, started, quit]);

  /** Begin recording an answer (only while the examiner is waiting for one). */
  const startRecording = useCallback(() => {
    if (!examRef.current.answerWindow || recordingRef.current) return;
    setRecordingNow(true);
  }, [setRecordingNow]);

  /** Finish the answer. A moment's grace first, so the last words aren't clipped. */
  const stopRecording = useCallback(() => {
    if (!recordingRef.current) return;
    setTimeout(() => {
      if (!recordingRef.current) return;
      setRecordingNow(false);
      const ws = wsRef.current;
      if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "record_stop" }));
    }, 250);
  }, [setRecordingNow]);

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
    preparing,
    state,
    paused,
    voiceActive,
    ttsActive,
    micMuted,
    error,
    engine: engineRef,
    sessionId,
    exam,
    transcript,
    recording,
    startRecording,
    stopRecording,
    resetExam,
    start,
    quit,
    stopReply,
    newSession,
    switchSession,
    toggleMicMute,
  };
}
