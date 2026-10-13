# speakIelts

A local-first, low-latency voice assistant for macOS (Apple Silicon).

Talk into your mic; Jessica listens, thinks, and answers out loud — with search
when she needs facts.

## Pipeline

```
mic ──> Whisper large-v3-turbo (local, MLX)     # speech -> text
     ─> agent (OpenAI gpt-5.6-luna + Tavily)    # reasoning + web search, streamed
     ─> Kokoro TTS (local)                       # sentence-by-sentence streaming playback
```

Latency tricks:
- Records only while you talk (Silero VAD endpointing), trims dead air
- Agent reply is **streamed**: each finished sentence is spoken while the rest
  is still being generated
- One persistent mic + one persistent speaker stream (repeated PortAudio
  open/close was wedging macOS CoreAudio)
- Whisper is warm-loaded at startup, audio goes straight to it as a numpy
  array (no temp WAV, no ffmpeg)

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
  ├── agent.py             # LangChain agent + SQLite session memory (streams deltas)
  ├── tts.py               # Kokoro sentence-streaming playback (sink-aware)
  ├── sessions.py          # session metadata, colours, summaries
  └── greeting.py          # opening question (new session = random, returning = summary-based)
web/                       # React + Vite UI (blob visual, caption, mic capture)
  ├── src/lib/audio.ts     # mic capture, resample, TTS playback, levels
  └── src/lib/useVoice.ts  # WebSocket client + React state
tests/
  ├── test_asr.py          # recording logic (fake mic stream, no device needed)
  ├── test_vad.py          # VAD model (silence/noise rejection, dtype handling)
  ├── test_recorder.py     # streaming two-tier endpointing + backfill
  ├── test_tts.py          # sentence splitter edge cases
  ├── test_io.py           # event serialization plumbing
  └── test_engine_stitch.py# barge-in continuation stitching
.env                       # API keys (not committed)
requirements.txt           # pinned dependencies for this machine
TODO.md                    # design log: every change and WHY it was made
```

## IELTS mock test (backend)

Click **New IELTS test** in the sidebar and choose a **Short (about 6 min)** or **Full
(about 14 min)** test. Jessica runs the real
structure — intro, Part 1, Part 2 (cue card, 1 min prep, up to 2 min
talk), Part 3 — without commenting on your answers, with a countdown for each
part. Every answer is saved as audio under `data/exams/<session>/<exam>/`; when
the test ends (or you quit after 3+ answers) the recordings are re-transcribed
with word timestamps and scored into an estimated report. You record each answer yourself (Record, then Stop, or the Space key); a
part's countdown is a hard limit, so a recording still running when it reaches
zero is cut off and the test moves on. When the test ends it goes straight to
the analysis, and the scores are saved to **Your progress** (`data/profile.db`),
which charts your bands over time. Past tests and conversations are listed in the
History sidebar.

```
app/ielts.py          # the exam's fixed structure + the saved record
app/examiner.py       # fixed exam lines, prompts, cue card, question generation
app/exam_engine.py    # ExamSession: the examiner's loop
app/exam_analysis.py  # fluency / vocabulary / grammar / pitch / word-confidence numbers
app/exam_report.py    # second-pass transcription + band estimates -> report.json
```

REST: `GET /exams/{session}` · `GET /exams/{session}/{exam}` (turns, analysis,
report) · `GET /exams/{session}/{exam}/audio/{turnNN.wav}` ·
`POST /exams/{session}/{exam}/analyse` (re-run). Deleting a session deletes its
recordings. Bands are practice **estimates**; pronunciation is inferred from
indicators (word confidence, pitch variation), not assessed by ear.

## Settings and API keys

Add your OpenAI key (and optionally a Tavily key for web search) in the app's
**Settings**. Keys are saved in `data/settings.json` (readable only by you),
used immediately, and never sent back to the browser. A `.env` file still
works as a fallback. The server listens on `127.0.0.1` only; set `HOST=0.0.0.0`
to reach it from another device.

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

### Web UI (browser audio + voice visual)

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

Open <http://localhost:8080>, then tap the **mic button** to grant mic access
and start the call. Jessica opens with a question: a random starter for a new
session, or a follow-up based on the previous conversation summary for a
returning one.

The blob glows in your chosen session colour and follows the live audio; your
last sentence appears as a simple italic caption below it. Long answers don't
squash the visual.

The **left sidebar** lists your sessions. You can rename a session, pick its
colour, delete it, or ask for a one-paragraph summary. A **mic-mute** button
lets you silence your own microphone without ending the call — Jessica's voice
keeps playing normally.

Conversation memory lives in **SQLite** (`checkpoints.db`), one thread per
voice session. Quit/Start and page reloads resume the same memory; **New
session** starts a fresh thread.

For backend-only no-reload: `python -m uvicorn server:app --port 8080`.
For frontend hot-reload development: `cd web && npm run dev` (Vite on :5173,
which proxies `/ws` to the Python server on :8080).

## Testing

```bash
python tests/test_asr.py            # recording state machine with a fake mic
python tests/test_vad.py            # VAD: silence/noise rejection, dtype handling
python tests/test_recorder.py       # streaming endpointing (1.5s cue, 2.5s finalize)
python tests/test_tts.py            # sentence splitter edge cases
python tests/test_io.py             # event serialization
python tests/test_engine_stitch.py  # barge-in continuation stitching
```

Headless tests use a fake audio stream — no microphone, no speakers, no API keys.
