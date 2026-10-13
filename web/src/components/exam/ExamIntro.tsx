import { useEffect, useState } from "react";
import { Clock, Mic, ShieldCheck, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { PART_TITLES } from "@/lib/exam";

interface ScheduleEntry {
  part: number;
  limit_s?: number;
  prep_s?: number;
  talk_s?: number;
}

interface Props {
  connected: boolean;
  onBegin: () => void;
}

const FALLBACK: ScheduleEntry[] = [
  { part: 1, limit_s: 300 },
  { part: 2, prep_s: 60, talk_s: 120 },
  { part: 3, limit_s: 300 },
];

const minutes = (s: number) => {
  const m = s / 60;
  return Number.isInteger(m) ? `${m} min` : `${m.toFixed(1)} min`;
};

function duration(e: ScheduleEntry): string {
  if (e.part === 2) return `${minutes(e.prep_s ?? 60)} to prepare, then up to ${minutes(e.talk_s ?? 120)} talking`;
  return `about ${minutes(e.limit_s ?? 300)}`;
}

const DESCRIPTION: Record<number, string> = {
  1: "Jessica asks you about yourself and everyday topics, one question at a time.",
  2: "You get a topic card and prepare in silence, then talk on your own. She stops you when time is up.",
  3: "A discussion of more abstract questions linked to your Part 2 topic.",
};

function Points({ icon: Icon, title, items }: { icon: typeof Mic; title: string; items: string[] }) {
  return (
    <div className="grid content-start gap-2.5">
      <h3 className="flex items-center gap-2 text-xs font-semibold tracking-wider text-muted-foreground uppercase">
        <Icon className="size-4" /> {title}
      </h3>
      <ul className="grid list-disc gap-1.5 pl-5 text-sm leading-relaxed text-muted-foreground">
        {items.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
    </div>
  );
}

/** The "Before you begin" screen: what the test is, how it works, what happens to your voice. */
export default function ExamIntro({ connected, onBegin }: Props) {
  const [schedule, setSchedule] = useState<ScheduleEntry[]>(FALLBACK);

  // The server knows the real timings (they can be shortened for development).
  useEffect(() => {
    fetch("/ielts/schedule")
      .then((r) => (r.ok ? r.json() : null))
      .then((s) => Array.isArray(s) && s.length === 3 && setSchedule(s))
      .catch(() => undefined);
  }, []);

  const total =
    schedule.reduce((sum, e) => sum + (e.limit_s ?? 0) + (e.prep_s ?? 0) + (e.talk_s ?? 0), 0) + 60; // + intro and closing

  return (
    <Card className="mx-auto w-full max-w-3xl" aria-labelledby="exam-intro-title">
      <CardHeader>
        <CardTitle id="exam-intro-title" className="text-xl">
          IELTS Speaking mock test
        </CardTitle>
        <CardDescription className="flex items-center gap-1.5">
          <Clock className="size-4" /> About {Math.round(total / 60)} minutes · Jessica is your examiner
        </CardDescription>
      </CardHeader>

      <CardContent className="grid gap-6">
        <ol className="grid gap-4">
          {schedule.map((e) => {
            const t = PART_TITLES[e.part as 1 | 2 | 3];
            return (
              <li key={e.part} className="grid grid-cols-[2rem_1fr] items-start gap-3.5">
                <span className="grid size-7 place-items-center rounded-full bg-primary text-[13px] font-semibold text-primary-foreground">
                  {e.part}
                </span>
                <div>
                  <p className="font-medium">
                    {t.name} · {t.blurb}
                  </p>
                  <p className="text-xs text-muted-foreground">{duration(e)}</p>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{DESCRIPTION[e.part]}</p>
                </div>
              </li>
            );
          })}
        </ol>

        <Separator />

        <div className="grid gap-6 md:grid-cols-2">
          <Points
            icon={Sparkles}
            title="How it works"
            items={[
              "Jessica behaves like a real examiner. She won't comment on or correct your answers during the test.",
              "Each part has a countdown on screen, and a part ends when its time is up.",
              "You can't pause or interrupt her. If you missed a question, say “Sorry, could you repeat that?”",
              "At the end you get estimated band scores for the four IELTS criteria, with specific tips.",
            ]}
          />
          <div className="grid content-start gap-6">
            <Points
              icon={Mic}
              title="Before you start"
              items={[
                "Find a quiet room and allow microphone access when asked.",
                "Speak naturally and give full answers, a few sentences in Part 1.",
                "Take your time to think. A short pause won't end your answer.",
              ]}
            />
            <Points
              icon={ShieldCheck}
              title="About your voice"
              items={[
                "Your answers are recorded and kept on this computer with this session, and deleted if you delete the session.",
                "Speech is transcribed locally. The transcript and measurements are sent to OpenAI, which writes the questions and your report.",
                "Scores are practice estimates, not official results. Pronunciation is the least certain.",
              ]}
            />
          </div>
        </div>
      </CardContent>

      <CardFooter className="sticky bottom-0 gap-4 rounded-b-xl border-t bg-card/95 pt-6 backdrop-blur">
        <Button size="lg" onClick={onBegin} disabled={!connected}>
          {connected ? "Begin test" : "Connecting…"}
        </Button>
        <span className="text-sm text-muted-foreground">The clock starts as soon as Jessica begins speaking.</span>
      </CardFooter>
    </Card>
  );
}
