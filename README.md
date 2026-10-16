# Jessica: a local voice assistant and IELTS speaking coach

Talk out loud, get an answer out loud. Jessica listens with a local speech model, thinks with
an LLM (OpenAI, or a model running on your own computer through Ollama), and speaks back
with a local voice. She can also run a **mock IELTS Speaking test** and give you an estimated
band report.

<!--
  IMAGE 1 (hero): a screenshot of the call screen, or a short GIF of a conversation.
  Save it as docs/images/hero.png (or hero.gif) and keep the line below.
-->
![Jessica: the call screen](docs/images/hero.png)

> **Status:** works well on Apple Silicon Macs. A one-command install and Windows support are
> planned (see [Roadmap](#roadmap)). Today you set it up from source, as described below.

## What it does

- **Free conversation.** Speak naturally. You can interrupt her, and she remembers what you
  actually heard.
- **IELTS mock test.** Short (about 6 minutes) or full (about 14 minutes). The examiner's lines
  and the cue card are written by the model each time, with a countdown for every part.
- **Report and progress.** After a test you get estimated bands for fluency, vocabulary,
  grammar and pronunciation, with quotes from your answers and what to practise first. Scores
  are saved to **Your progress**, which charts them over time.
- **History.** Every conversation and test is saved. Reopen it, rename it, delete it, or listen
  back to your own answers.
- **Private by default.** Speech recognition, the voice and your recordings stay on your
  computer. The LLM can run there too.

<!--
  IMAGE 2: the IELTS screen with the countdown timer and the cue card.
  Save as docs/images/ielts-test.png
-->
![The IELTS mock test with timer and cue card](docs/images/ielts-test.png)

<!--
  IMAGE 3: a finished report (overall band, the four criteria).
  Save as docs/images/report.png
-->
![An estimated band report](docs/images/report.png)

<!--
  IMAGE 4: the "Your progress" page with the chart.
  Save as docs/images/progress.png
-->
![Your progress over time](docs/images/progress.png)

<!--
  DEMO VIDEO: upload the demo to YouTube (or drag the file into a GitHub issue to get a
  link) and replace VIDEO_URL below. The thumbnail can be docs/images/hero.png.
-->
**Demo:** [Watch a 3-minute mock test](VIDEO_URL)

## How it works

```
mic ──> Silero VAD (is someone talking?)        # decides when you start and stop
     ─> Whisper large-v3-turbo (local, MLX)     # speech -> text
     ─> agent: OpenAI or Ollama (+ Tavily)      # reasoning + optional web search, streamed
     ─> Kokoro TTS (local)                      # speaks each sentence as soon as it is ready
```

Latency tricks:
- Records only while you talk (voice-activity detection) and trims dead air.
- The reply is **streamed**: each finished sentence is spoken while the rest is still being written.
- One persistent mic stream and one speaker stream. Opening and closing them repeatedly was
  wedging macOS CoreAudio.
- Whisper and the voice are loaded at startup, and audio goes straight to Whisper as an array
  (no temp files, no ffmpeg).

Noise handling: speech must be clearly louder than the room's background level to count, and
very short blips (a door closing, a pen drop) are ignored.

<!--
  IMAGE 5 (optional): an architecture or latency diagram. Save as docs/images/architecture.png
-->
![Architecture](docs/images/architecture.png)

## Requirements

| | Needed | Notes |
|---|---|---|
| Computer | Apple Silicon Mac (M1 or newer) | tested on an M4 only |
| Memory | 16 GB recommended | a local 4B LLM uses about 5 GB on top of the speech models; with OpenAI as the LLM, less may work (not tested) |
| Disk | a few GB for the speech models, plus about 4 GB for a local LLM | exact sizes not measured yet |
| Software | Python 3.11, Node 22+, a microphone | plus Ollama if you want the local LLM |

Only tested on an M4 MacBook with 16 GB so far. Windows and Linux are not supported yet,
because the speech-to-text backend (MLX) is Apple-only.

## Setup

```bash
git clone https://github.com/Ms3314/Jessica-IELTS-coach.git
cd Jessica-IELTS-coach
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then start it:

```bash
./start.sh
```

This builds the web UI the first time, then serves everything on <http://localhost:8080>. Open
it, allow microphone access, and press the mic button. The first launch downloads the Whisper
and Kokoro models, which takes a few minutes.

### Choose the LLM

**Option A: OpenAI (the default).** Open **Settings** in the app and paste your OpenAI key. A
Tavily key is optional and only adds web search to normal chat. Keys are saved in
`data/settings.json` (readable only by you) and are never sent back to the browser. A `.env`
file also works:

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...        # optional
# OPENAI_MODEL=gpt-5.6-luna
```

**Option B: a local model with Ollama (no key, no cloud).**

```bash
ollama pull qwen3.5:4b
```

then add this to `.env` and restart:

```
LLM_PROVIDER=ollama
# OLLAMA_MODEL=qwen3.5:4b
```

Small local models are good for conversation (replies start in well under a second once
loaded) but weaker at the structured parts. In my tests a 4B model wrote poor cue cards, and
its band scoring is untested. If you care about the report quality, use OpenAI for now.

## The IELTS mock test

Click **New IELTS test** and choose **Short** or **Full**. The real structure is followed: an
introduction, Part 1 (familiar topics), Part 2 (a cue card, preparation time, then a long
talk), and Part 3 (a discussion). The examiner never comments on your answers.

- You record each answer yourself: **Record**, then **Stop** (or press Space).
- Each part has a countdown that is a hard limit: a recording still running at zero is cut off
  and the test moves on.
- When the test ends it goes straight to the analysis, which runs in the background. You can
  start something else and open the report when it is ready.
- Every answer is saved as audio under `data/exams/`, re-transcribed with word timings, and
  analysed for pace, pauses, vocabulary, grammar and pitch before the report is written.

The bands are practice **estimates** from an AI, not an official score. Pronunciation is
inferred from indicators (word confidence, pitch variation) rather than judged by ear.

## Project structure

**[ARCHITECTURE.md](ARCHITECTURE.md)** is the full code map, with diagrams and a "where do I
change X" table.

```
main.py                  # terminal entry point
server.py                # web entry point (FastAPI)
app/
  ├── config.py          # settings from .env and the Settings screen
  ├── io.py              # pluggable audio source / sink / event handler
  ├── api/               # REST routes and the /ws call socket
  ├── chat/              # free conversation: loop, greeting, sessions, history
  ├── exam/              # IELTS: structure, examiner, loop, analysis, report, profile
  ├── speech/            # VAD, recorder, Whisper (asr), Kokoro (tts)
  └── llm/               # the agent, its memory, interruption handling
web/                     # React + Vite UI (shadcn components)
tests/                   # one script per backend module (no mic, no API keys)
web/tests/               # frontend helper tests
```

## Development

```bash
python -m fastapi dev server.py --port 8080   # auto-reloads on backend edits
cd web && npm run dev                          # UI hot-reload on :5173 (proxies /ws to :8080)
```

Pass `server.py` explicitly. The repository also has `main.py` (the terminal script), and the
FastAPI CLI would find that first and fail with "Could not find FastAPI app".

The server listens on `127.0.0.1` only. Set `HOST=0.0.0.0` to reach it from another device.

Tests need no microphone, speakers or API keys:

```bash
for t in tests/test_*.py; do python "$t" || break; done   # backend
node --test web/tests/                                     # frontend (Node 22+)
```

Note: the dev server restarts when a file changes, which also stops any analysis that is
running at that moment. Re-run it from the test's page with **Analyse again**.

## Roadmap

- [ ] One-command install (`pip` or a shell installer that also downloads the models and Ollama)
- [ ] Windows support (needs a different speech-to-text backend)
- [ ] Re-run unfinished analyses automatically when the server restarts
- [ ] A pronunciation check that listens to the sounds, not just confidence scores
- [ ] Test the local LLM against the report and cue-card tasks, and pick a recommended model

## Licence

To be decided. Some dependencies are GPL, which affects what this project can be released under.

## Credits

[Whisper](https://github.com/openai/whisper) and [MLX](https://github.com/ml-explore/mlx),
[Kokoro](https://github.com/hexgrad/kokoro), [Silero VAD](https://github.com/snakers4/silero-vad),
[LangChain](https://github.com/langchain-ai/langchain), [Ollama](https://ollama.com),
[FastAPI](https://fastapi.tiangolo.com) and [shadcn/ui](https://ui.shadcn.com).
