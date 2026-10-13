import { useEffect, useState } from "react";
import { Circle, Mic, MicOff, PhoneOff, Square } from "lucide-react";
import { BarVisualizer } from "@/components/ui/bar-visualizer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "cn";
import type { Mode } from "@/lib/api";
import type { FrameEmitter } from "@/lib/audio/frame-source";
import type { VisualFrame } from "@/lib/audio/types";
import type { EngineState } from "@/lib/useVoice";

interface Props {
  mode: Mode;
  state: EngineState;
  preparing: boolean;
  paused: boolean;
  voiceActive: boolean;
  micMuted: boolean;
  source: FrameEmitter<VisualFrame> | null;
  /** The examiner is waiting for an answer (test only). */
  canRecord: boolean;
  /** The candidate is recording an answer (test only). */
  recording: boolean;
  onRecord: () => void;
  onStopRecording: () => void;
  onStop: () => void;
  onEnd: () => void;
  onToggleMic: () => void;
}

const LABEL: Record<EngineState, string> = {
  idle: "Ready",
  listening: "Listening",
  transcribing: "Understanding",
  thinking: "Thinking",
  speaking: "Speaking",
  error: "Error",
};

const DOT: Record<string, string> = {
  idle: "bg-muted-foreground/50",
  listening: "bg-emerald-600",
  transcribing: "bg-amber-500 animate-[blink_0.9s_ease-in-out_infinite]",
  thinking: "bg-amber-500 animate-[blink_0.9s_ease-in-out_infinite]",
  speaking: "bg-blue-600 animate-[blink_1.2s_ease-in-out_infinite]",
  error: "bg-red-600",
  recording: "bg-red-600 animate-[blink_1s_ease-in-out_infinite]",
};

/** The strip under a live call: what Jessica is doing, a level meter, and the controls. */
export default function CallBar({
  mode, state, preparing, paused, voiceActive, micMuted, source, canRecord, recording,
  onRecord, onStopRecording, onStop, onEnd, onToggleMic,
}: Props) {
  const exam = mode === "ielts";
  // Stop is for chat only: a real examiner can't be interrupted.
  const canStop = !exam && !preparing && (state === "transcribing" || state === "thinking" || state === "speaking");

  useEffect(() => {
    if (!canStop) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onStop();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canStop, onStop]);

  // Space records and stops an answer in the test (unless you're typing somewhere)
  useEffect(() => {
    if (!exam) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (recording) {
        e.preventDefault();
        onStopRecording();
      } else if (canRecord) {
        e.preventDefault();
        onRecord();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [exam, recording, canRecord, onRecord, onStopRecording]);

  // how long the current answer has been recording
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!recording) return setSeconds(0);
    const started = Date.now();
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 250);
    return () => clearInterval(id);
  }, [recording]);

  const quietTurn = state === "listening" && !voiceActive && !paused;
  const status = preparing
    ? "Getting ready…"
    : exam && recording
      ? "Recording your answer"
      : exam && canRecord
        ? "Your turn: press Record"
        : micMuted
      ? "Microphone muted"
      : paused
        ? "Take your time…"
        : quietTurn
          ? "Your turn"
          : LABEL[state];
  const tone = preparing ? "idle" : exam && recording ? "recording" : exam && canRecord ? "listening" : paused || quietTurn ? "listening" : state;

  return (
    <div className="-mx-7 grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-5 border-t bg-muted/40 px-7 py-3 max-md:grid-cols-1 max-md:justify-items-start">
      <Badge variant="outline" className="h-7 min-w-36 justify-start gap-2 bg-background px-3 text-[13px] font-medium">
        <span className={cn("size-2 rounded-full", DOT[tone])} aria-hidden="true" />
        {status}
      </Badge>

      <BarVisualizer
        aria-label="Voice activity"
        className="h-7 w-full max-w-80 justify-self-center"
        style={{ color: "#9ca3af" }}
        source={source}
        barCount={28}
        mirrored
        idle="pulse"
      />

      <div className="flex items-center gap-2">
        {exam && (
          <Button
            size="sm"
            variant={recording ? "destructive" : "default"}
            onClick={recording ? onStopRecording : onRecord}
            disabled={!recording && !canRecord}
            className="min-w-36"
          >
            {recording ? (
              <>
                <Square className="size-3.5 fill-current" /> Stop · {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")}
              </>
            ) : (
              <>
                <Circle className="size-3.5 fill-red-500 text-red-500" /> Record answer
              </>
            )}
            <kbd className="ml-1 text-[11px] opacity-70">Space</kbd>
          </Button>
        )}
        {canStop && (
          <Button variant="outline" size="sm" onClick={onStop} className="bg-background">
            <Square className="size-3.5" /> Stop <kbd className="ml-1 text-[11px] text-muted-foreground">Esc</kbd>
          </Button>
        )}
        {!exam && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="outline"
              size="icon-sm"
              onClick={onToggleMic}
              aria-label={micMuted ? "Unmute microphone" : "Mute microphone"}
              aria-pressed={micMuted}
              className="bg-background"
            >
              {micMuted ? <MicOff className="size-4" /> : <Mic className="size-4" />}
            </Button>
          </TooltipTrigger>
          <TooltipContent>{micMuted ? "Unmute microphone" : "Mute microphone"}</TooltipContent>
        </Tooltip>
        )}
        <Button variant="destructive" size="sm" onClick={onEnd}>
          <PhoneOff className="size-4" /> {exam ? "End test" : "End call"}
        </Button>
      </div>
    </div>
  );
}
