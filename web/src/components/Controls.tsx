import { useEffect } from "react";
import { Mic, MicOff, PhoneOff, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { EngineState } from "@/lib/useVoice";

interface Props {
  connected: boolean;
  started: boolean;
  state: EngineState;
  paused: boolean;
  voiceActive: boolean;
  micMuted: boolean;
  color: string;
  onStart: () => void;
  onQuit: () => void;
  onStop: () => void;
  onToggleMicMute: () => void;
}

const LABEL: Record<EngineState, string> = {
  idle: "Ready",
  listening: "Listening…",
  transcribing: "Understanding…",
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
  micMuted,
  color,
  onStart,
  onQuit,
  onStop,
  onToggleMicMute,
}: Props) {
  // Stop is offered while Jessica is working on / delivering a reply.
  const canStop =
    started &&
    (state === "transcribing" || state === "thinking" || state === "speaking");

  useEffect(() => {
    if (!canStop) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onStop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canStop, onStop]);

  const quietTurn = started && state === "listening" && !voiceActive;
  const displayState: EngineState =
    !connected || !started ? "idle" : paused ? "listening" : quietTurn ? "idle" : state;
  const status = !connected
    ? "Connecting…"
    : !started
      ? "Tap the mic to start"
      : micMuted
        ? "Microphone muted"
        : paused
          ? "Take your time…"
          : quietTurn
            ? "Your turn…"
            : LABEL[state];

  return (
    <div className="controls">
      <span
        className={`status ${displayState}${paused ? " paused" : ""}`}
        style={{ ["--session-color" as string]: color }}
      >
        <span className="dot" />
        {status}
      </span>

      <div className="control-cluster">
        {canStop && (
          <Button
            variant="outline"
            onClick={onStop}
            aria-label="Stop Jessica (Esc)"
            title="Stop Jessica (Esc)"
          >
            <Square className="size-4" />
            Stop
          </Button>
        )}

        {started && (
          <Button
            variant={micMuted ? "destructive" : "outline"}
            size="icon"
            onClick={onToggleMicMute}
            aria-label={micMuted ? "Unmute microphone" : "Mute microphone"}
          >
            {micMuted ? <MicOff className="size-4" /> : <Mic className="size-4" />}
          </Button>
        )}

        {!started ? (
          <Button
            size="icon-lg"
            className="mic-button"
            style={{ ["--session-color" as string]: color }}
            onClick={onStart}
            disabled={!connected}
            aria-label="Start conversation"
          >
            <Mic className="size-6" />
          </Button>
        ) : (
          <Button
            size="icon-lg"
            variant="destructive"
            className="mic-button"
            onClick={onQuit}
            aria-label="End conversation"
          >
            <PhoneOff className="size-6" />
          </Button>
        )}
      </div>
    </div>
  );
}
