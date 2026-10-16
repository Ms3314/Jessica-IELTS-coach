import { useCallback, useEffect, useRef, useState } from "react";
import CallView from "@/components/views/CallView";
import HistorySidebar from "@/components/layout/HistorySidebar";
import ReportView from "@/components/exam/ReportView";
import SettingsDialog from "@/components/layout/SettingsDialog";
import ProfileView from "@/components/views/ProfileView";
import StartView from "@/components/views/StartView";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ChatReview, TestReview } from "@/components/views/ReviewView";
import {
  deleteSession,
  fetchExams,
  analyseExam,
  fetchMessages,
  fetchServerSettings,
  fetchSessions,
  updateSession,
  type ChatMessage,
  type ExamListItem,
  type Mode,
  type ScoreRow,
  type ServerSettings,
  type SessionMeta,
} from "@/lib/api";
import { sessionMode } from "@/lib/history";
import { useSettings } from "@/lib/settings";
import { useVoice } from "@/lib/voice/useVoice";

interface SavedHistory {
  loading: boolean;
  messages: ChatMessage[];
  exams: ExamListItem[];
}

/** What has already happened in a session (its transcript, and any mock tests). */
function useSessionHistory(sessionId: string, reloadKey: number): SavedHistory {
  const [state, setState] = useState<SavedHistory>({ loading: true, messages: [], exams: [] });
  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, loading: true }));
    Promise.all([fetchMessages(sessionId), fetchExams(sessionId)]).then(([messages, exams]) => {
      if (!cancelled) setState({ loading: false, messages, exams });
    });
    return () => {
      cancelled = true;
    };
  }, [sessionId, reloadKey]);
  return state;
}

export default function App() {
  const { settings, update: updateSettings } = useSettings();
  const [mode, setMode] = useState<Mode>("chat");
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [server, setServer] = useState<ServerSettings | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [historyKey, setHistoryKey] = useState(0);
  // the main area shows either the current conversation/test, or your progress
  const [view, setView] = useState<"work" | "profile">("work");

  const voice = useVoice({
    voiceInterrupt: settings.voiceInterrupt,
    pauseS: settings.pauseS,
    mode,
    examLength: settings.examLength,
  });
  const { connected, started, sessionId, exam, resetExam, retryReport, newSession, switchSession, start, quit } = voice;

  const refreshSessions = useCallback(async () => setSessions(await fetchSessions()), []);
  const refreshServer = useCallback(async () => setServer(await fetchServerSettings()), []);

  useEffect(() => {
    refreshServer();
  }, [refreshServer]);
  useEffect(() => {
    refreshSessions();
  }, [refreshSessions, sessionId]);

  // When a call ends: reload its saved transcript now, and again once the server has
  // written its summary a moment later.
  const wasStarted = useRef(false);
  useEffect(() => {
    if (wasStarted.current && !started) {
      setHistoryKey((k) => k + 1);
      refreshSessions();
      const t = setTimeout(() => {
        refreshSessions();
        setHistoryKey((k) => k + 1);
      }, 2500);
      return () => clearTimeout(t);
    }
    wasStarted.current = started;
  }, [started, refreshSessions]);

  const history = useSessionHistory(sessionId, historyKey);
  const activeSession = sessions.find((s) => s.id === sessionId);
  const needsKey = server !== null && !server.keys.openai_api_key.set;

  // --- sidebar actions ---
  const openNew = useCallback(
    (next: Mode) => {
      setView("work");
      setMode(next);
      resetExam();
      newSession();
    },
    [newSession, resetExam]
  );
  const openSession = useCallback(
    (s: SessionMeta) => {
      setView("work");
      setMode(sessionMode(s));
      resetExam();
      switchSession(s.id);
    },
    [resetExam, switchSession]
  );
  const rename = useCallback(
    async (id: string, name: string) => {
      await updateSession(id, { name });
      refreshSessions();
    },
    [refreshSessions]
  );
  const remove = useCallback(
    async (s: SessionMeta) => {
      if (!confirm(`Delete "${s.name}"? This also deletes its recordings and can't be undone.`)) return;
      await deleteSession(s.id);
      if (s.id === sessionId) openNew(mode);
      refreshSessions();
    },
    [mode, openNew, refreshSessions, sessionId]
  );

  // --- what the workspace shows ---
  const showProfile = view === "profile" && !started;
  const openTestFromProfile = useCallback(
    (row: ScoreRow) => {
      const s = sessions.find((x) => x.id === row.session_id);
      if (s) openSession(s);
    },
    [sessions, openSession]
  );
  const title = showProfile
    ? "Your progress"
    : started
      ? mode === "ielts"
        ? "IELTS mock test"
        : "Conversation"
      : activeSession?.name ?? (mode === "ielts" ? "New IELTS test" : "New conversation");

  let content;
  if (showProfile) {
    content = <ProfileView refreshKey={historyKey} onOpenTest={openTestFromProfile} onNewTest={() => openNew("ielts")} />;
  } else if (started) {
    content = (
      <CallView
        mode={mode}
        state={voice.state}
        preparing={voice.preparing}
        paused={voice.paused}
        voiceActive={voice.voiceActive}
        micMuted={voice.micMuted}
        error={voice.error}
        transcript={voice.transcript}
        exam={exam}
        source={voice.engine.current?.visualSource ?? null}
        recording={voice.recording}
        onRecord={voice.startRecording}
        onStopRecording={voice.stopRecording}
        onStop={voice.stopReply}
        onEnd={quit}
        onToggleMic={voice.toggleMicMute}
      />
    );
  } else if (mode === "ielts" && exam.complete) {
    // the test that just ended: its report (arrives once analysis finishes)
    content = (
      <div className="min-h-0 flex-1 overflow-y-auto p-7">
        <ReportView
          sessionId={sessionId}
          examId={exam.examId}
          report={exam.report}
          error={exam.reportError}
          onAgain={() => openNew("ielts")}
          onChat={() => openNew("chat")}
          onOpenProfile={() => setView("profile")}
          onRetry={() => exam.examId && analyseExam(sessionId, exam.examId).then(retryReport)}
        />
      </div>
    );
  } else if (history.loading) {
    content = (
      <div className="mx-auto grid w-full max-w-3xl gap-4 p-7">
        <Skeleton className="h-10 w-1/2" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
    );
  } else if (mode === "chat" && history.messages.length > 0) {
    content = <ChatReview session={activeSession} messages={history.messages} onContinue={start} />;
  } else if (mode === "ielts" && history.exams.length > 0) {
    content = (
      <TestReview
        sessionId={sessionId}
        onAgain={() => openNew("ielts")}
        onChat={() => openNew("chat")}
        onOpenProfile={() => setView("profile")}
      />
    );
  } else {
    content = (
      <StartView
        mode={mode}
        connected={connected}
        needsKey={needsKey}
        examLength={settings.examLength}
        onExamLength={(examLength) => updateSettings({ examLength })}
        onStart={start}
        onOpenSettings={() => setSettingsOpen(true)}
      />
    );
  }

  return (
    <TooltipProvider>
      <div className="grid h-full grid-cols-[288px_minmax(0,1fr)] max-md:grid-cols-1 max-md:grid-rows-[auto_minmax(0,1fr)]">
        <HistorySidebar
          sessions={sessions}
          activeId={sessionId}
          locked={started}
          onNewChat={() => openNew("chat")}
          onNewTest={() => openNew("ielts")}
          onOpen={openSession}
          onRename={rename}
          onDelete={remove}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenProfile={() => setView("profile")}
          profileActive={showProfile}
        />

        <main className="flex h-full min-h-0 min-w-0 flex-col">
          <header className="flex min-h-14 items-center gap-3 border-b px-7">
            <h1 className="truncate text-[15px] font-semibold">{title}</h1>
            {started && (
              <Badge variant="outline" className="gap-1.5 border-red-200 bg-red-50 text-red-700">
                <span className="size-1.5 animate-[blink_1.4s_ease-in-out_infinite] rounded-full bg-red-600" /> Live
              </Badge>
            )}
          </header>
          <div className="flex min-h-0 flex-1 flex-col">{content}</div>
        </main>

        <SettingsDialog
          open={settingsOpen}
          onOpenChange={setSettingsOpen}
          settings={settings}
          onChange={updateSettings}
          server={server}
          onKeysSaved={refreshServer}
        />
      </div>
    </TooltipProvider>
  );
}
