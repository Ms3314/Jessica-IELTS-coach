import { useEffect, useState } from "react";
import { AlertTriangle, ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { fetchProfile, type ProfileData, type ScoreRow } from "@/lib/api";
import { changeLabel, trendChart } from "@/lib/profile";

interface Props {
  /** bump to reload (e.g. after a new report has been saved) */
  refreshKey: number;
  onOpenTest: (row: ScoreRow) => void;
  onNewTest: () => void;
}

const CRITERIA: { key: "fluency" | "lexical" | "grammar" | "pronunciation"; label: string }[] = [
  { key: "fluency", label: "Fluency & Coherence" },
  { key: "lexical", label: "Lexical Resource" },
  { key: "grammar", label: "Grammatical Range & Accuracy" },
  { key: "pronunciation", label: "Pronunciation" },
];

const band = (v: number | null | undefined) => (v === null || v === undefined ? "–" : v.toFixed(1));

function Stat({ label, value, note }: { label: string; value: string; note?: React.ReactNode }) {
  return (
    <Card className="gap-1 py-4">
      <CardHeader className="gap-1 px-5">
        <CardDescription className="text-xs font-medium">{label}</CardDescription>
        <div className="flex items-baseline gap-2">
          <CardTitle className="text-3xl font-semibold tabular-nums">{value}</CardTitle>
          {note}
        </div>
      </CardHeader>
    </Card>
  );
}

function TrendCard({ values }: { values: number[] }) {
  const W = 520;
  const H = 220;
  const chart = trendChart(values, W, H, 34);
  return (
    <Card className="gap-3 py-5">
      <CardHeader className="gap-1 px-6">
        <CardTitle className="text-[15px]">Overall band over time</CardTitle>
        <CardDescription>Each point is one test, oldest on the left.</CardDescription>
      </CardHeader>
      <CardContent className="px-4">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Line chart of your overall band for each test" className="h-auto w-full">
          {chart.ticks.map((t) => (
            <g key={t.value}>
              <line x1={34} x2={W - 34} y1={t.y} y2={t.y} className="stroke-border" strokeDasharray="3 4" />
              <text x={26} y={t.y + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">
                {t.value}
              </text>
            </g>
          ))}
          {chart.path && <path d={chart.path} fill="none" className="stroke-foreground" strokeWidth={2} strokeLinejoin="round" />}
          {chart.points.map((p, i) => (
            <g key={i}>
              <circle cx={p.x} cy={p.y} r={i === chart.points.length - 1 ? 5 : 3.5} className="fill-background stroke-foreground" strokeWidth={2} />
              <text x={p.x} y={p.y - 11} textAnchor="middle" className="fill-foreground text-[11px] font-medium">
                {p.value.toFixed(1)}
              </text>
            </g>
          ))}
        </svg>
      </CardContent>
    </Card>
  );
}

/** Your progress: every mock test you have taken, scored and tracked over time. */
export default function ProfileView({ refreshKey, onOpenTest, onNewTest }: Props) {
  const [data, setData] = useState<ProfileData | null | undefined>(undefined);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setData(undefined);
    fetchProfile().then((d) => !cancelled && setData(d));
    return () => {
      cancelled = true;
    };
  }, [refreshKey, attempt]);

  if (data === undefined) {
    return (
      <div className="mx-auto grid w-full max-w-4xl gap-4 p-7">
        <div className="grid grid-cols-4 gap-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-64" />
      </div>
    );
  }

  // null means the server couldn't be asked (not "no tests"): say so, don't pretend it's empty
  if (data === null) {
    return (
      <div className="mx-auto grid w-full max-w-3xl gap-4 p-7">
        <Alert variant="warning">
          <AlertTriangle />
          <AlertTitle>Couldn't load your progress</AlertTitle>
          <AlertDescription>
            <p>
              The server didn't answer. If you updated the app recently, restart it so it picks up the latest version,
              then try again. Your saved tests are not lost.
            </p>
          </AlertDescription>
        </Alert>
        <div>
          <Button variant="outline" onClick={() => setAttempt((n) => n + 1)}>
            Try again
          </Button>
        </div>
      </div>
    );
  }

  const summary = data.summary;
  if (!summary || summary.tests === 0) {
    return (
      <div className="mx-auto w-full max-w-3xl p-7">
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">No mock tests yet</CardTitle>
            <CardDescription className="text-[15px] leading-relaxed">
              When you finish an IELTS mock test, its scores are saved here automatically so you can watch your
              progress over time. They stay on this computer.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={onNewTest}>Take a mock test</Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const change = changeLabel(summary.change);
  const ChangeIcon = summary.change === null || summary.change === 0 ? Minus : summary.change > 0 ? ArrowUpRight : ArrowDownRight;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-7">
      <div className="mx-auto grid w-full max-w-4xl gap-4">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Tests taken" value={String(summary.tests)} />
          <Stat
            label="Latest overall band"
            value={band(summary.latest)}
            note={
              change && (
                <Badge variant="outline" className="gap-1 font-normal">
                  <ChangeIcon /> {change}
                </Badge>
              )
            }
          />
          <Stat label="Best overall band" value={band(summary.best)} />
          <Stat label="Average overall band" value={band(summary.average)} />
        </div>

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
          <TrendCard values={summary.trend} />
          <Card className="gap-4 py-5">
            <CardHeader className="gap-1 px-6">
              <CardTitle className="text-[15px]">Average by criterion</CardTitle>
              <CardDescription>Where you are strongest, across all tests.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 px-6">
              {CRITERIA.map(({ key, label }) => {
                const v = summary.criteria_average[key];
                return (
                  <div key={key} className="grid gap-1.5">
                    <div className="flex items-baseline justify-between text-sm">
                      <span>{label}</span>
                      <span className="font-semibold tabular-nums">{band(v)}</span>
                    </div>
                    <Progress value={v === null ? 0 : (v / 9) * 100} aria-label={`${label}: ${band(v)} of 9`} />
                  </div>
                );
              })}
            </CardContent>
          </Card>
        </div>

        <Card className="gap-3 py-5">
          <CardHeader className="gap-1 px-6">
            <CardTitle className="text-[15px]">All tests</CardTitle>
            <CardDescription>Open a test to read its full report and listen back to your answers.</CardDescription>
          </CardHeader>
          <CardContent className="px-3">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Date</TableHead>
                  <TableHead>Test</TableHead>
                  <TableHead className="text-right">Overall</TableHead>
                  <TableHead className="text-right">Fluency</TableHead>
                  <TableHead className="text-right">Lexical</TableHead>
                  <TableHead className="text-right">Grammar</TableHead>
                  <TableHead className="text-right">Pronun.</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.history.map((r) => (
                  <TableRow
                    key={`${r.session_id}/${r.exam_id}`}
                    tabIndex={0}
                    onClick={() => onOpenTest(r)}
                    onKeyDown={(e) => e.key === "Enter" && onOpenTest(r)}
                    className="cursor-pointer"
                  >
                    <TableCell>
                      {new Date(r.taken_at * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" })}
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="font-normal">
                        {r.length === "short" ? "Short" : "Full"}
                      </Badge>
                      {!r.finished && (
                        <Badge variant="outline" className="ml-1.5 font-normal text-muted-foreground">
                          ended early
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right font-semibold tabular-nums">{band(r.overall)}</TableCell>
                    <TableCell className="text-right tabular-nums">{band(r.fluency)}</TableCell>
                    <TableCell className="text-right tabular-nums">{band(r.lexical)}</TableCell>
                    <TableCell className="text-right tabular-nums">{band(r.grammar)}</TableCell>
                    <TableCell className="text-right tabular-nums">{band(r.pronunciation)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
