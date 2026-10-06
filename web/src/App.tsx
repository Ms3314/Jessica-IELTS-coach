import Aura from "./components/Aura";
import Controls from "./components/Controls";
import { useVoice } from "./lib/useVoice";

export default function App() {
  const {
    connected,
    started,
    state,
    userText,
    paused,
    voiceActive,
    error,
    engine,
    start,
    quit,
  } = useVoice();

  return (
    <div className="app">
      <header className="topbar">
        <h1>Jessica</h1>
        <p className="subtitle">IELTS Speaking Coach · local voice agent</p>
      </header>

      <main className="stage">
        <Aura engineRef={engine} state={state} />
        {error && <div className="error-banner">{error}</div>}
      </main>

      {/* Voice-only UI: a grey caption of the latest thing you said. */}
      <p
        key={userText ?? "empty"}
        className={`caption${userText ? "" : " empty"}`}
        aria-live="polite"
      >
        {userText}
      </p>

      <Controls
        connected={connected}
        started={started}
        state={state}
        paused={paused}
        voiceActive={voiceActive}
        onStart={start}
        onQuit={quit}
      />
    </div>
  );
}
