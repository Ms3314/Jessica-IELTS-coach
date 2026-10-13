import { useCallback, useEffect, useState } from "react";
import { PanelLeft } from "lucide-react";
import SettingsDialog from "@/components/SettingsDialog";
import { BarVisualizer } from "@/components/ui/bar-visualizer";
import Controls from "@/components/Controls";
import SessionSidebar, { type Session } from "@/components/SessionSidebar";
import { Button } from "@/components/ui/button";
import { useSettings } from "@/lib/settings";
import { useVoice } from "@/lib/useVoice";

export default function App() {
  const { settings, update: updateSettings } = useSettings();
  const {
    connected,
    started,
    state,
    userText,
    paused,
    voiceActive,
    ttsActive,
    micMuted,
    error,
    engine,
    sessionId,
    start,
    quit,
    stopReply,
    newSession,
    switchSession,
    toggleMicMute,
  } = useVoice({ voiceInterrupt: settings.voiceInterrupt });

  const [sessions, setSessions] = useState<Session[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const refreshSessions = useCallback(async () => {
    try {
      const res = await fetch("/sessions");
      const data = await res.json();
      setSessions(data);
    } catch {
      // ignore — the socket will reconnect
    }
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions, sessionId]);

  // The server auto-summarises a session when its socket closes. After a
  // reconnect, give it a moment and pull the fresh summary in.
  useEffect(() => {
    if (!connected) return;
    const t = setTimeout(refreshSessions, 2500);
    return () => clearTimeout(t);
  }, [connected, refreshSessions]);

  const activeSession = sessions.find((s) => s.id === sessionId);
  const sessionColor = activeSession?.color || "#3b82f6";

  const handleUpdate = useCallback(
    async (id: string, patch: Partial<Session>) => {
      await fetch(`/sessions/${id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      refreshSessions();
    },
    [refreshSessions]
  );

  const handleDelete = useCallback(
    async (id: string) => {
      if (!confirm("Delete this session? This cannot be undone.")) return;
      await fetch(`/sessions/${id}`, { method: "DELETE" });
      if (id === sessionId) newSession();
      else refreshSessions();
    },
    [sessionId, newSession, refreshSessions]
  );

  const handleSwitch = useCallback(
    (id: string) => {
      switchSession(id);
      refreshSessions();
    },
    [switchSession, refreshSessions]
  );

  const handleNew = useCallback(() => {
    newSession();
    refreshSessions();
  }, [newSession, refreshSessions]);



  // While reply audio is still playing, keep the "speaking" look even if the
  // server already moved on to "idle"/"listening" (synthesis ends before
  // playback does).
  const visualState = ttsActive ? ("speaking" as const) : state;

  return (
    <div className="app-shell">
      <SessionSidebar
        sessions={sessions}
        activeId={sessionId}
        open={sidebarOpen}
        onSelect={handleSwitch}
        onNew={handleNew}
        onUpdate={handleUpdate}
        onDelete={handleDelete}
      />

      <div className="main">
        <header className="topbar">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setSidebarOpen((o) => !o)}
            aria-label="Toggle sessions"
          >
            <PanelLeft className="size-5" />
          </Button>
          <div className="topbar-title">
            <h1>Jessica</h1>
          </div>
          <div className="session-actions">
            <SettingsDialog settings={settings} onChange={updateSettings} />
          </div>
        </header>

        <section className="summary-panel">
          <div className="summary-panel-head">
            <h2>Summary</h2>
          </div>
          <p className="summary-panel-text">
            {activeSession?.summary ||
              "Your conversation summary will appear here after the call ends."}
          </p>
        </section>

        <main className="stage">
          <div className="visualizer-wrap">
            <BarVisualizer
              aria-label="Voice activity"
              className="h-40 w-full max-w-md"
              style={{ color: sessionColor }}
              source={engine.current?.visualSource ?? null}
              mirrored
              idle="pulse"
            />
          </div>
          {error && <div className="error-banner">{error}</div>}

          <div className="caption-wrap">
            {userText && (
              <p key={userText} className="caption" aria-live="polite">
                {userText}
              </p>
            )}
          </div>
        </main>

        <Controls
          connected={connected}
          started={started}
          state={visualState}
          paused={paused}
          voiceActive={voiceActive}
          micMuted={micMuted}
          color={sessionColor}
          onStart={start}
          onQuit={quit}
          onStop={stopReply}
          onToggleMicMute={toggleMicMute}
        />
      </div>
    </div>
  );
}
