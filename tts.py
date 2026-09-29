import os
import queue
import threading

import sounddevice as sd
from kokoro import KPipeline

pipeline = KPipeline(repo_id="hexgrad/Kokoro-82M", lang_code="a")

SAMPLE_RATE = 24000
OUTPUT_DIR = "tts_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

inbetweener = "mmmm"

# --- sentence splitting ---------------------------------------------------

ABBR = {"mr", "mrs", "ms", "dr", "vs", "st", "sr", "jr", "etc", "e.g", "i.e"}


def _find_sentence_end(s: str):
    """Index of the first real sentence-ending char in s, or None."""
    for i, ch in enumerate(s):
        if ch not in ".!?":
            continue
        if ch == ".":
            # decimal number (3.14) -> not a sentence end
            if i > 0 and s[i - 1].isdigit() and i + 1 < len(s) and s[i + 1].isdigit():
                continue
            # abbreviations (Mr., e.g.) and initials (J.) -> not a sentence end
            if i > 0 and s[i - 1].isalpha():
                word = s[max(0, i - 8):i].split()[-1] if s[max(0, i - 8):i].split() else ""
                if len(word) <= 2 or word.lower() in ABBR:
                    continue
        return i
    return None


def _sentence_iter(deltas):
    """Yield complete sentences from a stream of text deltas."""
    buf = ""
    for delta in deltas:
        if not delta:
            continue
        buf += delta
        while True:
            cut = _find_sentence_end(buf)
            if cut is None:
                break
            sentence, buf = buf[: cut + 1], buf[cut + 1 :]
            if sentence.strip():
                yield sentence.strip()
    if buf.strip():  # flush whatever is left at end of stream
        yield buf.strip()


# --- playback -------------------------------------------------------------

def _play(text: str, voice: str):
    """Synthesize and play one piece of text, blocking until done."""
    for _i, _gs, audio in pipeline(text, voice=voice):
        sd.play(audio, SAMPLE_RATE)
        sd.wait()


def speak(text, voice="af_heart"):
    """Speak a fixed string, blocking."""
    if text:
        _play(text, voice)


def speak_stream(text_stream, voice="af_heart", verbose=True):
    """Speak sentences as they stream in; synthesis overlaps generation.

    text_stream: iterable of text deltas (from agent_reply_stream).
    Blocks until every sentence has finished playing.
    """
    q = queue.Queue(maxsize=3)  # backpressure: don't synth the whole reply at once

    def worker():
        while True:
            sentence = q.get()
            try:
                if sentence is None:
                    return
                if verbose:
                    print("  [tts]", sentence)
                _play(sentence, voice)
            except Exception as e:
                print("tts error:", e)
            finally:
                q.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    for sentence in _sentence_iter(deltas=text_stream):
        q.put(sentence)

    q.put(None)
    q.join()  # block until all queued sentences have played
