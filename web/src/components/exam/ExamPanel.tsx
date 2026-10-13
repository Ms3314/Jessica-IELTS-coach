import { Check } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useNow } from "@/hooks/use-now";
import { cn } from "cn";
import { PART_TITLES, PHASE_LABEL, formatClock, remainingS, stepStatus, type ExamState } from "@/lib/exam";

interface Props {
  exam: ExamState;
}

const TIMER_LABEL = {
  part: "Time left in this part",
  prep: "Preparation time",
  talk: "Speaking time left",
} as const;

/** Shown during the test: part stepper, the running countdown, and the Part 2 cue card. */
export default function ExamPanel({ exam }: Props) {
  const now = useNow(250, exam.active);
  const timer = exam.timer;
  const left = timer ? remainingS(timer, now) : null;
  const frac = timer && timer.totalS > 0 ? Math.min(1, Math.max(0, (left ?? 0) / timer.totalS)) : 0;
  const urgent = left !== null && left <= 10;
  const elapsed = exam.startedAt ? (now - exam.startedAt) / 1000 : 0;
  const showCard = exam.card && exam.part === 2;

  return (
    <Card className="gap-3 py-4" aria-label="Test progress">
      <div className="flex flex-wrap items-center justify-between gap-5 px-5">
        <ol className="flex flex-wrap gap-6">
          {([1, 2, 3] as const).map((p) => {
            const status = stepStatus(p, exam);
            return (
              <li
                key={p}
                aria-current={status === "active" ? "step" : undefined}
                className={cn("flex items-center gap-2.5", status === "upcoming" && "opacity-45")}
              >
                <span
                  className={cn(
                    "grid size-6 place-items-center rounded-full border text-xs font-semibold",
                    status === "active" && "border-primary bg-primary text-primary-foreground",
                    status === "done" && "border-transparent bg-muted text-muted-foreground"
                  )}
                >
                  {status === "done" ? <Check className="size-3" /> : p}
                </span>
                <span className="grid leading-tight">
                  <strong className="text-[13px] font-medium">{PART_TITLES[p].name}</strong>
                  <small className="text-xs text-muted-foreground">{PART_TITLES[p].blurb}</small>
                </span>
              </li>
            );
          })}
        </ol>

        <div className="grid min-w-36 justify-items-end leading-none">
          {timer && left !== null ? (
            <>
              <span className="text-[10.5px] font-semibold tracking-wider text-muted-foreground uppercase">
                {TIMER_LABEL[timer.kind]}
              </span>
              <span
                className={cn(
                  "mt-1 text-4xl font-semibold tabular-nums",
                  urgent && "animate-[blink_1s_ease-in-out_infinite] text-red-700"
                )}
              >
                {formatClock(left)}
              </span>
            </>
          ) : (
            <>
              <span className="text-[10.5px] font-semibold tracking-wider text-muted-foreground uppercase">Now</span>
              <span className="mt-1 text-lg font-medium">{PHASE_LABEL[exam.phase]}</span>
            </>
          )}
        </div>
      </div>

      <div className="grid gap-1.5 px-5">
        <Progress value={frac * 100} indicatorClassName={cn(urgent && "bg-red-700")} aria-hidden="true" />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{timer ? PHASE_LABEL[exam.phase] : "No timer in this step"}</span>
          <span>Test time {formatClock(elapsed)}</span>
        </div>
      </div>

      {showCard && exam.card && (
        <div className="mx-5 mt-1 animate-[pop_0.2s_ease-out] rounded-lg border border-dashed bg-muted/40 px-5 py-4">
          <p className="text-base font-semibold">{exam.card.title}</p>
          <p className="mt-2 text-xs text-muted-foreground">You should say:</p>
          <ul className="mt-1 grid list-disc gap-0.5 pl-5 text-sm">
            {exam.card.points.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
          <p className="mt-2 text-sm">{exam.card.explain}</p>
        </div>
      )}
    </Card>
  );
}
