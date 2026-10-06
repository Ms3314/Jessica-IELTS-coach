import type { EngineState } from "../lib/useVoice";

interface Props {
  connected: boolean;
  started: boolean;
  state: EngineState;
  paused: boolean;
  voiceActive: boolean;
  onStart: () => void;
  onQuit: () => void;
}

const LABEL: Record<EngineState, string> = {
  idle: "Ready",
  listening: "Listening…",
  transcribing: "Transcribing…",
  thinking: "Thinking…",
  speaking: "Speaking…",
  error: "Error",
};

export default function Controls({
  connected,
  started,
  state,
  paused,
  voiceActive,
  onStart,
  onQuit,
}: Props) {
  const idleListening = started && state === "listening" && !voiceActive;
  // Before Start (and while the user is silent) the state is not "live".
  const displayState: EngineState =
    !connected || !started
      ? "idle"
      : paused
        ? "listening"
        : state === "listening" && !voiceActive
          ? "idle"
          : state;
  const status = !connected
    ? "Connecting…"
    : !started
      ? "Ready — press Start"
      : paused
        ? "Take your time…"
        : idleListening
          ? "Your turn…"
          : LABEL[state];

  return (
    <div className="controls">
      <span className={`status ${displayState}${paused ? " paused" : ""}`}>
        <span className="dot" />
        {status}
      </span>
      <button
        className="btn primary"
        onClick={onStart}
        disabled={!connected || started}
      >
        Start
      </button>
      <button className="btn" onClick={onQuit} disabled={!connected || !started}>
        Quit
      </button>
    </div>
  );
}
