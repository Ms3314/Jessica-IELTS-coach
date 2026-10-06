# TalkingAngela

A local-first, low-latency voice assistant for macOS (Apple Silicon).

Talk into your mic; Angela listens, thinks, and answers out loud — with search
when she needs facts.

## Pipeline

```
mic ──> Whisper large-v3-turbo (local, MLX)     # speech -> text
     ─> agent (OpenAI gpt-5.6-luna + Tavily)    # reasoning + web search, streamed
     ─> Kokoro TTS (local)                       # sentence-by-sentence streaming playback
```

Latency tricks:
- Records only while you talk (energy-based silence detection), trims dead air
- Agent reply is **streamed**: each finished sentence is spoken while the rest
  is still being generated
- One persistent mic + one persistent speaker stream (repeated PortAudio
  open/close was wedging macOS CoreAudio)
- Whisper is warm-loaded at startup, audio goes straight to it as a numpy
  array (no temp WAV, no ffmpeg)
- **Silero VAD** for voice activity detection — neural network that understands
  what speech sounds like, far fewer false triggers than RMS energy

## Project structure

```
main.py                    # terminal entry point — the conversation loop
server.py                  # FastAPI + WebSocket server; serves the web UI
app/
  ├── config.py            # loads .env, exposes shared settings
  ├── engine.py            # Session: shared conversation flow (both frontends)
  ├── io.py                # pluggable AudioSource / Sink / EventHandler
  ├── recorder.py          # streaming VAD + endpointing for web mic audio
  ├── vad.py               # Silero VAD — neural voice activity detection
  ├── asr.py               # mic capture + local Whisper transcription
  ├── agent.py             # LangChain agent, streams text deltas
  └── tts.py               # Kokoro sentence-streaming playback (sink-aware)
web/                       # React + Vite UI (avatar, transcript, mic capture)
  ├── src/lib/audio.ts     # mic capture, resample, TTS playback, lipsync
  └── src/lib/useVoice.ts  # WebSocket client + React state
tests/
  ├── test_asr.py          # recording logic (fake mic stream, no device needed)
  ├── test_tts.py          # sentence splitter edge cases
  └── test_vad.py          # VAD model (silence/noise rejection, dtype handling)
.env                       # API keys (not committed)
requirements.txt           # pinned dependencies for this machine
TODO.md                    # design log: every change and WHY it was made
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg        # only needed if you feed audio files to Whisper
```

`.env` at the project root:

```
OPENAI_API_KEY=...        # agent brain
TAVILY_API_KEY=...        # web search for the agent
# optional overrides:
# OPENAI_MODEL=gpt-5.6-luna
```

## Run

### Terminal (original)

```bash
python main.py
```

Say **"goodbye"** (the whole utterance) to quit. First launch downloads the
Kokoro + Whisper models and takes a few seconds; after that, the Whisper
weights warm-load at startup.

### Web UI (browser audio + talking-head avatar)

The same engine also runs behind a FastAPI + WebSocket server. The browser
captures the mic and plays the reply; Whisper, the agent, and Kokoro all stay
on the Mac.

```bash
# 1. build the React UI once (only needed the first time / after frontend edits)
cd web
npm install
npm run build      # outputs web/dist
cd ..

# 2. start the server (serves the UI + WebSocket on :8080)
python -m fastapi dev server.py --port 8080
```

`python -m fastapi dev` is the standard way to start — it auto-reloads on
backend edits and serves **both** the built UI and the WebSocket API on the
single port 8080, so no separate frontend server is needed to use the app.
(Use `python -m fastapi …` rather than bare `fastapi` so the venv's CLI is used.)

### Why `server.py` must be named explicitly

This repo has two top-level entry files: `main.py` (the terminal script) and
`server.py` (the FastAPI app). The FastAPI CLI auto-discovers a file to run
when you omit it, and its default search order finds `main.py` **first** —
which is not a FastAPI app. That produces:

```
Could not find FastAPI app in module, try using --app
```

To prevent this, `pyproject.toml` pins the entrypoint to `server:app`, so the
bare command also works:

```bash
python -m fastapi dev          # reads [tool.fastapi] entrypoint from pyproject.toml
python -m fastapi dev server.py --port 8080   # explicit, always works
```

Do **not** pass `--app server:app` together with the file path — the FastAPI
CLI's app discovery misbehaves in that combination; auto-detection is enough.

One-command convenience: `./start.sh` (builds the UI if missing, then starts).

Open <http://localhost:8080>, click **Start** (grants mic access), and talk.
The avatar's mouth lipsyncs to the streamed TTS; the transcript updates live.

For backend-only no-reload: `python -m uvicorn server:app --port 8080`.
For frontend hot-reload development: `cd web && npm run dev` (Vite on :5173,
which proxies `/ws` to the Python server on :8080).

## Testing

```bash
python tests/test_tts.py   # sentence splitter edge cases (fast, no devices)
python tests/test_asr.py   # recording state machine with a fake mic (no mic needed)
python tests/test_vad.py   # VAD model: silence/noise rejection, dtype handling
```

Headless tests use a fake audio stream — no microphone, no speakers, no API keys.
