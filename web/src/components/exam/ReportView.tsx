import { useEffect, useState } from "react";
import { AlertTriangle, ArrowRight, BarChart3, Check, Loader2, Quote } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { BAND_CRITERIA, type Criterion, type Report } from "@/lib/exam";
import { cn } from "cn";

interface Props {
  sessionId: string;
  examId: string | null;
  /** null while the report is still being made */
  report: Report | null;
  error: string | null;
  onAgain: () => void;
  onChat: () => void;
  /** open the progress screen (shown when the scores were saved to it) */
  onOpenProfile?: () => void;
  /** start the analysis again (offered when it failed or never ran) */
  onRetry?: () => void;
  /** false when no analysis is running and no report exists (a past test that was never analysed) */
  analysing?: boolean;
}

interface SavedTurn {
  n: number;
  part: number;
  question: string;
  answer: string;
  audio: string[];
  no_answer: boolean;
}

const RELIABILITY: Record<string, { label: string; className: string }> = {
  low: { label: "Low reliability: very little speech to judge", className: "border-amber-200 bg-amber-50 text-amber-800" },
  medium: { label: "Medium reliability", className: "" },
  good: { label: "Good amount of speech to judge", className: "border-emerald-200 bg-emerald-50 text-emerald-800" },
};

function CriterionCard({ title, c }: { title: string; c: Criterion }) {
  return (
    <Card className="gap-4 py-5">
      <CardHeader className="gap-3 px-5">
        <div className="flex items-baseline justify-between gap-3">
          <CardTitle className="text-[15px]">{title}</CardTitle>
          <span className="text-3xl font-semibold tabular-nums">{c.band.toFixed(1)}</span>
        </div>
        <Progress value={(c.band / 9) * 100} aria-label={`Band ${c.band} of 9`} />
      </CardHeader>
      <CardContent className="grid gap-3 px-5 text-sm">
        <p className="leading-relaxed text-muted-foreground">{c.summary}</p>
        {c.basis && <p className="text-xs text-muted-foreground italic">{c.basis}</p>}
        {c.strengths.length > 0 && (
          <ul className="grid gap-1.5">
            {c.strengths.map((s) => (
              <li key={s} className="flex items-start gap-2">
                <Check className="mt-1 size-3.5 shrink-0 text-emerald-700" /> {s}
              </li>
            ))}
          </ul>
        )}
        {c.improvements.length > 0 && (
          <ul className="grid gap-1.5">
            {c.improvements.map((s) => (
              <li key={s} className="flex items-start gap-2">
                <ArrowRight className="mt-1 size-3.5 shrink-0" /> {s}
              </li>
            ))}
          </ul>
        )}
        {c.evidence.length > 0 && (
          <div className="grid gap-1.5 pt-1">
            {c.evidence.map((q) => (
              <blockquote key={q} className="border-l-2 pl-3 text-xs text-muted-foreground italic">
                <Quote className="mr-1 inline size-3 align-[-1px]" />
                {q}
              </blockquote>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/** What you see after the test: the estimated bands, feedback, and your own answers to replay. */
export default function ReportView({
  sessionId,
  examId,
  report,
  error: reportError,
  onAgain,
  onChat,
  onOpenProfile,
  onRetry,
  analysing = true,
}: Props) {
  const [turns, setTurns] = useState<SavedTurn[]>([]);

  // the saved answers (for replay) come from the server once the report exists
  useEffect(() => {
    if (!examId || !report) return;
    let cancelled = false;
    fetch(`/exams/${encodeURIComponent(sessionId)}/${encodeURIComponent(examId)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => !cancelled && data && setTurns(data.turns ?? []))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [sessionId, examId, report]);

  const actions = (
    <div className="flex gap-2.5 pb-2">
      <Button onClick={onAgain}>Take another test</Button>
      <Button variant="outline" onClick={onChat}>
        New conversation
      </Button>
      {report?.profile_saved && onOpenProfile && (
        <Button variant="ghost" onClick={onOpenProfile}>
          <BarChart3 className="size-4" /> View your progress
        </Button>
      )}
    </div>
  );

  if (reportError) {
    return (
      <div className="mx-auto grid w-full max-w-3xl gap-4">
        <Alert variant="destructive">
          <AlertTriangle />
          <AlertTitle>The report couldn't be made</AlertTitle>
          <AlertDescription>
            <p>{reportError}</p>
            <p>Your answers are saved. You can try analysing them again.</p>
            {onRetry && (
              <Button size="sm" variant="outline" className="mt-1 w-fit" onClick={onRetry}>
                Analyse again
              </Button>
            )}
          </AlertDescription>
        </Alert>
        {actions}
      </div>
    );
  }

  if (!report && !analysing) {
    return (
      <div className="mx-auto grid w-full max-w-3xl gap-4">
        <Alert variant="warning">
          <AlertTriangle />
          <AlertTitle>No report for this test yet</AlertTitle>
          <AlertDescription>
            <p>Your answers are saved, but this test was never analysed.</p>
            {onRetry && (
              <Button size="sm" className="mt-1 w-fit" onClick={onRetry}>
                Analyse now
              </Button>
            )}
          </AlertDescription>
        </Alert>
        {actions}
      </div>
    );
  }

  if (!report) {
    return (
      <div className="mx-auto grid w-full max-w-3xl gap-4">
        <Alert role="status">
          <Loader2 className="animate-spin" />
          <AlertTitle>Your test is being analysed in the background</AlertTitle>
          <AlertDescription>
            <p>
              This takes about a minute. You don't need to wait here: start something else, and find the report later
              in your history. It appears on this page as soon as it's ready, and your score is added to Your
              progress.
            </p>
          </AlertDescription>
        </Alert>
        {actions}
      </div>
    );
  }

  if (report.insufficient) {
    return (
      <div className="mx-auto grid w-full max-w-3xl gap-4">
        <Alert variant="warning">
          <AlertTriangle />
          <AlertTitle>Not enough to assess</AlertTitle>
          <AlertDescription>
            <p>{report.message ?? "The test ended before enough questions were answered to give feedback."}</p>
          </AlertDescription>
        </Alert>
        {actions}
      </div>
    );
  }

  const rel = report.reliability ? RELIABILITY[report.reliability] : null;

  return (
    <div className="mx-auto grid w-full max-w-4xl gap-4" aria-label="Your IELTS speaking report">
      <Card className="py-6">
        <CardContent className="flex flex-wrap items-center gap-8 px-7">
          <div className="grid justify-items-center gap-1">
            <span className="text-[10.5px] font-semibold tracking-wider text-muted-foreground uppercase">
              Estimated overall band
            </span>
            <span className="text-6xl leading-none font-semibold tabular-nums">{report.overall_band?.toFixed(1)}</span>
          </div>
          <div className="grid min-w-64 flex-1 gap-2.5">
            {rel && (
              <Badge variant="outline" className={cn("w-fit font-normal", rel.className)}>
                {rel.label}
              </Badge>
            )}
            {report.examiner_note && <p className="leading-relaxed">{report.examiner_note}</p>}
            {report.profile_saved && (
              <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
                <Check className="size-4 text-emerald-700" /> Saved to your progress
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        {BAND_CRITERIA.map(({ key, title }) => {
          const c = report[key] as Criterion | undefined;
          return c ? <CriterionCard key={key} title={title} c={c} /> : null;
        })}
      </div>

      {report.top_priorities && report.top_priorities.length > 0 && (
        <Card className="gap-3 py-5">
          <CardHeader className="px-6">
            <CardTitle className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">
              Practise these first
            </CardTitle>
          </CardHeader>
          <CardContent className="px-6">
            <ol className="grid list-decimal gap-2 pl-5 text-sm leading-relaxed">
              {report.top_priorities.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ol>
          </CardContent>
        </Card>
      )}

      {turns.some((t) => t.audio.length > 0) && (
        <Card className="gap-3 py-5">
          <CardHeader className="px-6">
            <CardTitle className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">
              Listen back to your answers
            </CardTitle>
          </CardHeader>
          <CardContent className="grid px-6">
            {turns
              .filter((t) => t.audio.length > 0)
              .map((t, i) => (
                <div key={t.n}>
                  {i > 0 && <Separator className="my-4" />}
                  <p className="flex items-center gap-2 text-sm font-medium">
                    <Badge variant="secondary" className="font-normal">
                      Part {t.part}
                    </Badge>
                    {t.question}
                  </p>
                  <p className="mt-1.5 mb-2 text-sm leading-relaxed text-muted-foreground italic">{t.answer}</p>
                  {t.audio.map((name) => (
                    <audio
                      key={name}
                      controls
                      preload="none"
                      className="h-9 w-full"
                      src={`/exams/${encodeURIComponent(sessionId)}/${encodeURIComponent(examId ?? "")}/audio/${name}`}
                    />
                  ))}
                </div>
              ))}
          </CardContent>
        </Card>
      )}

      {report.caveats && report.caveats.length > 0 && (
        <Alert>
          <AlertTitle>Good to know</AlertTitle>
          <AlertDescription>
            <ul className="grid list-disc gap-1 pl-4">
              {report.caveats.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {actions}
    </div>
  );
}
