# TODO — Bug Fixes & Improvements

_Tracking file: what we're fixing, **why**, and status. Update status as each is done._

---

## 1. Filter agent stream to AIMessageChunk only (agent.py)

**Why:** CONFIRMED LIVE — when the agent calls the Tavily search tool, the tool's
result is a `ToolMessage` whose `.content` is a *string* (raw JSON). The current
filter accepts any chunk with string content → the Tavily JSON is yielded to the
TTS and Jessica literally reads out `{"query": "latest AI news today...` out loud.

**Fix:** Only accept `isinstance(chunk, AIMessageChunk)` — ToolMessage / HumanMessage
also have `.content` and must not leak into speech.

**Status:** [x] done — verified live

---

## 2. Drop non-speakable fragments from sentence splitter (tts.py)

**Why:** "Wait... what?" currently splits into `['Wait.', '.', '.', 'what?']`.
The lone-dot fragments are truthy strings, so they get queued and synthesized —
wasted TTS calls and odd audio gaps.

**Fix:** A sentence is only queued if it's >= 2 chars and contains at least one
letter/digit. Lone punctuation fragments are dropped.

**Status:** [x] done — verified live

---

## 3. Keep calibration frames + early-speech protection (asr.py)

**Why:** The 0.5s noise-calibration frames were read and DISCARDED. If the user
starts speaking immediately after the prompt, their first words are cut off.
Worse: if they speak during the whole calibration window, `max(noise)*3` sets the
threshold above their own speech level → nothing is ever detected as speech →
"no speech detected" even though they spoke.

**Fix:** Keep all calibration frames in the recording buffer (leading silence is
already trimmed afterwards). If the calibration window itself contains loud audio
(user talking), fall back to the default `MIN_THRESHOLD` instead of a skewed
noise floor.

**Status:** [x] done — verified live

---

## 4. Per-turn error handling in main loop (main.py)

**Why:** One Groq rate-limit or OpenAI hiccup crashes the whole app out of the
listen loop. A voice assistant should degrade ("something went wrong") and keep
listening.

**Fix:** Wrap each turn in try/except; on failure speak an apology and continue.
`KeyboardInterrupt` (Ctrl+C) exits cleanly instead of dumping a traceback.

**Status:** [x] done — verified live

---

## 5. Always signal the TTS worker, even on mid-stream failure (tts.py)

**Why:** If the agent stream raises mid-iteration, the producer loop dies without
queueing the `None` sentinel, so the worker blocks forever and `speak_stream`
never completes.

**Fix:** `try/finally` around the producer loop that always puts the sentinel.

**Status:** [x] done — verified live

---

## 6. Whisper anti-noise guard: require ~0.3s of real speech (asr.py)

**Why:** A single loud non-speech frame (cough, desk tap) flips `spoken=True`,
records 1.2s of noise, and Whisper can hallucinate text from it ("Thanks for
watching!" is a classic). That garbage then goes to the cleanup LLM and the agent.

**Fix:** Require at least 3 speech frames (~0.3s total above threshold) before
treating the recording as speech; otherwise return None.

**Status:** [x] done — verified live

---

## 7. Warm-load Whisper model at startup (asr.py)

**Why (corrected):** `mlx_whisper` DOES cache the model between calls (ModelHolder
class attribute — my review overstated this as a per-call reload). The real cost
is the FIRST call after app start paying the ~2-5s weight load while the user
waits for their first answer.

**Fix:** Run one warm-up transcription on silence at import time, so the load
happens during app start instead of mid-conversation.

**Status:** [x] done — verified live

---

## 8. Pass audio array straight to Whisper — no temp WAV, no ffmpeg (asr.py)

**Why:** `mlx_whisper.transcribe()` accepts `np.ndarray` directly (verified in
source: `audio: Union[str, np.ndarray, mx.array]`). Currently every turn writes
a temp WAV file and spawns an ffmpeg subprocess to decode it back — pure
overhead (~0.1-0.3s + disk churn) since we already hold the samples in memory.

**Fix:** Convert int16 recording to float32 [-1, 1] and pass it directly.
Removes tempfile/scipy/ffmpeg from the ASR path entirely.

**Status:** [x] done — verified live

---

## 9. "goodbye" quit match too eager (main.py)

**Why:** `"goodbye" in text.lower()` quits the app on any utterance containing
the word — e.g. "how do I say goodbye in French?" kills the session.

**Fix:** Only quit when the utterance IS a goodbye word (goodbye / bye / quit /
exit), after stripping punctuation — not when it merely contains one.

**Status:** [x] done — verified live

---

## 10. Cleanup model failure should degrade, not crash (cleanup.py)

**Why:** Cleanup is an enhancement, not a requirement — the raw Whisper
transcript is still perfectly usable by the agent. If Groq is down or rate
limited, we should send the raw text instead of dying.

**Fix:** try/except in `clean_message()` → on failure, print a note and return
the raw transcript. Also fix the stale docstring (still says "Google STT" —
the ASR is Whisper now).

**Status:** [x] done — verified live

---

## 11. Remove dead code (tts.py, agent.py)

**Why:** Since the streaming switch: `agent_reply()` + `_extract_text()` are
unused (main only calls `agent_reply_stream`); `inbetweener` / `OUTPUT_DIR` /
`tts_out` dir creation in tts.py are leftovers from the afplay file-based
playback. Dead code confuses future edits.

**Fix:** Delete them. (`.env` still has an unused `SEARCH_API_KEY` — left alone,
user's file; can be removed manually.)

**Status:** [x] done — verified live

---

## 12. Flatten mic recording to 1D before Whisper (asr.py)

**Why:** After fix #8 (passing the numpy array straight to Whisper), every real
recording crashed with `[flatten] Shape dimension 250560522000 is outside the
supported range` — `sounddevice` returns frames shaped `(n, channels)` = `(N, 1)`,
and `mlx_whisper` expects flat 1D audio. The temp-WAV path never hit this because
ffmpeg's decode returned 1D. My synthetic tests used 1D zeros, so they missed it —
reproduced live with `(N, 1)` frames, exact same error.

**Fix:** `np.concatenate(frames[start:]).reshape(-1)` in `record_until_silence`.

**Status:** [x] done — verified live: `(41600, 1)` in → `(36800,)` out, Whisper runs clean.

---

## 13. Proper project structure (whole repo)

**Why:** Everything sat flat in the root: pipeline modules, entry point, docs,
tests (ad-hoc inline), no pinned deps, no real README. Flat layout gets messy
fast, inline shell tests can't be re-run, and each module calling
`load_dotenv()` only worked because everything happened to sit next to `.env`.

**Fix:** Package layout with `git mv` (history preserved):

```
main.py                     # entry point (root)
app/                        # the pipeline package
  __init__.py               # no eager imports (submodules load models)
  config.py                 # .env loaded ONCE, explicit path, shared settings
  asr.py cleanup.py agent.py tts.py
tests/test_asr.py            # fake-mic-stream recording tests (no device/API)
tests/test_tts.py            # sentence splitter edge cases
requirements.txt             # pinned to the working venv (pip freeze)
README.md                    # real docs: pipeline, structure, setup, run
```

Also removed dead leftovers (`tts_out/`, `output_0.wav`, `__pycache__/` —
all gitignored anyway). `.env` loading now resolves relative to the package
parent, so the app works no matter which directory it's launched from.

**Status:** [x] done — 6 tts tests + 5 asr tests pass, main.py imports clean.

---

## Verification plan

- [x] Sentence splitter: ellipsis / abbreviations / decimals cases
- [x] Recording: fake-stream test — speech during calibration is kept, noise-only returns None, short blip ignored
- [x] Streaming agent with a forced Tavily search: no JSON leaks into yielded text
- [x] Whisper accepts numpy array (no ffmpeg in path)
- [x] Full imports + one short spoken streaming reply
