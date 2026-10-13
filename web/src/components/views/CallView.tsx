import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import CallBar from "@/components/call/CallBar";
import ExamPanel from "@/components/exam/ExamPanel";
import Loader from "@/components/brand/Loader";
import Transcript from "@/components/call/Transcript";
import type { Mode } from "@/lib/api";
import type { ExamState } from "@/lib/exam";
import type { FrameEmitter } from "@/lib/audio/frame-source";
import type { VisualFrame } from "@/lib/audio/types";
import type { Line } from "@/lib/transcript";
import type { EngineState } from "@/lib/voice/useVoice";

interface Props {
  mode: Mode;
  state: EngineState;
  preparing: boolean;
  paused: boolean;
  voiceActive: boolean;
  micMuted: boolean;
  error: string | null;
  transcript: Line[];
  exam: ExamState;
  source: FrameEmitter<VisualFrame> | null;
  recording: boolean;
  onRecord: () => void;
  onStopRecording: () => void;
  onStop: () => void;
  onEnd: () => void;
  onToggleMic: () => void;
}

/** A call in progress: the examiner's progress (for a test), the live transcript, and controls. */
export default function CallView(p: Props) {
  const exam = p.mode === "ielts";
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 px-7 pt-4">
      {exam && !p.preparing && <ExamPanel exam={p.exam} />}
      {p.error && (
        <Alert variant="destructive">
          <AlertCircle />
          <AlertDescription>{p.error}</AlertDescription>
        </Alert>
      )}
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        {p.preparing ? (
          <div className="grid flex-1 place-items-center">
            <Loader
              label={exam ? "Getting your examiner ready…" : "Getting Jessica ready…"}
              hint="She'll start speaking in a moment."
            />
          </div>
        ) : p.transcript.length === 0 ? (
          <div className="grid flex-1 place-items-center text-sm text-muted-foreground">
            Listening… start speaking whenever you're ready.
          </div>
        ) : (
          <div className="mx-auto w-full max-w-3xl pb-3">
            <Transcript lines={p.transcript} follow assistantName={exam ? "Examiner" : "Jessica"} />
          </div>
        )}
      </div>
      <CallBar
        mode={p.mode}
        state={p.state}
        preparing={p.preparing}
        paused={p.paused}
        voiceActive={p.voiceActive}
        micMuted={p.micMuted}
        source={p.source}
        canRecord={exam && p.exam.answerWindow !== null}
        recording={p.recording}
        onRecord={p.onRecord}
        onStopRecording={p.onStopRecording}
        onStop={p.onStop}
        onEnd={p.onEnd}
        onToggleMic={p.onToggleMic}
      />
    </div>
  );
}
