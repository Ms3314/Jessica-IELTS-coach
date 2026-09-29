import queue
import threading
import time

import numpy as np
import sounddevice as sd
from kokoro import KPipeline

pipeline = KPipeline(repo_id="hexgrad/Kokoro-82M", lang_code="a")

SAMPLE_RATE = 24000

# --- persistent output stream ------------------------------------------------
# sd.play/sd.wait opened+closed a PortAudio stream per audio chunk (~10-16 per
# turn). After a few turns the repeated open/close cycle wedged macOS CoreAudio
# and the next mic open hung forever at 'Speak something...'. We now keep ONE
# output stream for the whole app and write samples into it, tracking when the
# last buffered sample will have played (the drain ledger) so we can wait out
# the tail before the mic starts listening again.

_out_lock = threading.Lock()
_out_stream = None   # single persistent OutputStream, created lazily
_tail_end = 0.0      # monotonic time when the output buffer runs dry


def _get_out_stream():
    global _out_stream
    with _out_lock:
        if _out_stream is None:
            _out_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32"
            )
            _out_stream.start()
        return _out_stream


def _write_audio(audio):
    """Buffer one chunk of samples for playback; update the drain ledger."""
    global _tail_end
    audio = np.asarray(audio, dtype="float32").reshape(-1, 1)
    stream = _get_out_stream()
    with _out_lock:
        stream.write(audio)
        # buffer was empty -> playback starts now; otherwise it chains onto
        # whatever is still queued: both cases are max(now, tail_end) + duration
        _tail_end = max(time.monotonic(), _tail_end) + len(audio) / SAMPLE_RATE


def _drain():
    """Block until everything written has actually finished playing."""
    with _out_lock:
        wait = _tail_end - time.monotonic()
    if wait > 0:
        time.sleep(wait)


# --- sentence splitting ------------------------------------------------------

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


def _speakable(sentence: str) -> bool:
    """True if a fragment is worth synthesizing (not just punctuation noise)."""
    s = sentence.strip()
    return len(s) >= 2 and any(c.isalnum() for c in s)


def _sentence_iter(deltas):
    """Yield complete, speakable sentences from a stream of text deltas."""
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
            if _speakable(sentence):
                yield sentence.strip()
    if _speakable(buf):  # flush whatever is left at end of stream
        yield buf.strip()


# --- playback ----------------------------------------------------------------

def _play(text: str, voice: str):
    """Synthesize one piece of text and buffer it for playback (non-blocking)."""
    for _i, _gs, audio in pipeline(text, voice=voice):
        _write_audio(audio)


def speak(text, voice="af_heart"):
    """Speak a fixed string, blocking until it has finished playing."""
    if text:
        _play(text, voice)
        _drain()


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

    threading.Thread(target=worker, daemon=True).start()

    # try/finally: even if the agent stream dies mid-reply, the worker must
    # get its sentinel or it blocks forever on q.get()
    try:
        for sentence in _sentence_iter(deltas=text_stream):
            q.put(sentence)
    finally:
        q.put(None)

    q.join()  # wait until all sentences are synthesized and buffered
    _drain()  # ...and until the audio tail has actually played out
