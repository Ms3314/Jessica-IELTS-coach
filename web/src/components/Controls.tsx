import { Mic, MicOff, PhoneOff } from "lucide-react";
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
  onToggleMicMute,
}: Props) {
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
