# TalkingAngela

A local-first, low-latency voice assistant for macOS (Apple Silicon).

Talk into your mic; Angela listens, thinks, and answers out loud — with search
when she needs facts.

## Pipeline

```
mic ──> Whisper large-v3-turbo (local, MLX)     # speech -> text
     ─> cleanup LLM (Groq, gpt-oss-20b free)     # fix STT noise: fillers, homophones
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

## Project structure

```
main.py                    # entry point — the conversation loop
app/
  ├── config.py            # loads .env, exposes shared settings
  ├── asr.py               # mic capture + local Whisper transcription
  ├── cleanup.py           # STT transcript bettering (Groq)
  ├── agent.py             # LangChain agent, streams text deltas
  └── tts.py               # Kokoro sentence-streaming playback
tests/
  ├── test_asr.py          # recording logic (fake mic stream, no device needed)
  └── test_tts.py          # sentence splitter edge cases
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
GROQ_API_KEY=...          # transcript cleanup (free tier)
TAVILY_API_KEY=...        # web search for the agent
# optional overrides:
# OPENAI_MODEL=gpt-5.6-luna
# GROQ_MODEL=openai/gpt-oss-20b
```

## Run

```bash
python main.py
```

Say **"goodbye"** (the whole utterance) to quit. First launch downloads the
Kokoro + Whisper models and takes a few seconds; after that, the Whisper
weights warm-load at startup.

## Testing

```bash
python tests/test_tts.py   # sentence splitter edge cases (fast, no devices)
python tests/test_asr.py   # recording state machine with a fake mic (no mic needed)
```

Headless tests use a fake audio stream — no microphone, no speakers, no API keys.
