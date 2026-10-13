import { useEffect, useState } from "react";
import { MessageSquare } from "lucide-react";
import ReportView from "@/components/exam/ReportView";
import Transcript from "@/components/Transcript";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchExam, fetchExams, type ChatMessage, type SessionMeta } from "@/lib/api";
import type { Report } from "@/lib/exam";

/** A past conversation: its summary and full transcript, with a way to carry on. */
export function ChatReview({
  session,
  messages,
  onContinue,
}: {
  session: SessionMeta | undefined;
  messages: ChatMessage[];
  onContinue: () => void;
}) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-7">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
        <header className="flex items-start justify-between gap-4">
          <div className="grid gap-2">
            <h1 className="text-xl font-semibold tracking-tight">{session?.name ?? "Conversation"}</h1>
            <Badge variant="secondary" className="w-fit gap-1.5 font-normal">
              <MessageSquare /> {messages.length} message{messages.length === 1 ? "" : "s"}
            </Badge>
          </div>
          <Button onClick={onContinue}>Continue conversation</Button>
        </header>

        {session?.summary && (
          <Card className="gap-2 py-4">
            <CardHeader className="gap-0">
              <CardTitle className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">
                Summary
              </CardTitle>
            </CardHeader>
            <CardContent className="text-[15px] leading-relaxed">{session.summary}</CardContent>
          </Card>
        )}

        <Transcript lines={messages} />
      </div>
    </div>
  );
}

/** A past mock test: loads its latest report and answers from the server. */
export function TestReview({
  sessionId,
  onAgain,
  onChat,
}: {
  sessionId: string;
  onAgain: () => void;
  onChat: () => void;
}) {
  const [examId, setExamId] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      const exams = await fetchExams(sessionId);
      const latest = exams[0];
      if (!latest) return !cancelled && setLoading(false);
      const data = await fetchExam(sessionId, latest.exam_id);
      if (cancelled) return;
      setExamId(latest.exam_id);
      setReport(data?.report ?? null);
      setLoading(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  if (loading) {
    return (
      <div className="mx-auto grid w-full max-w-3xl gap-4 p-7">
        <Skeleton className="h-28" />
        <div className="grid grid-cols-2 gap-4">
          <Skeleton className="h-52" />
          <Skeleton className="h-52" />
        </div>
      </div>
    );
  }
  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-7">
      <ReportView sessionId={sessionId} examId={examId} report={report} error={null} onAgain={onAgain} onChat={onChat} />
    </div>
  );
}
