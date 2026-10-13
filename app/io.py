"""Pluggable I/O so the SAME conversation engine drives both frontends.

  * AudioSource  — yields recorded mic clips (terminal: sounddevice; web: WS)
  * Sink        — where synthesized TTS audio goes (terminal: speakers; web: WS)
  * EventHandler — state + transcript callbacks (terminal: prints; web: WS events)

The terminal CLI (main.py) wires MicSource + LocalSink + TerminalHandler.
The web server (server.py) wires WebSocketSource + WebSocketSink + WebSocketHandler.
"""
import queue
import time
from abc import ABC, abstractmethod

import numpy as np

# Sentinel returned by a source when the user wants to quit the session.
QUIT = object()

# Sentinel returned by a source when the user paused briefly. The engine should
# surface a silent pause cue to the UI and keep listening — it must NOT
# finalize/transcribe the turn yet.
BACKFILL = object()


# --- interfaces --------------------------------------------------------------

class AudioSource(ABC):
    @abstractmethod
    def listen(self):
        """Block until a recorded clip is ready.

        Returns:
          * np.ndarray  — a recorded clip to transcribe
          * None       — nothing worth transcribing (e.g. silence); keep looping
          * io.BACKFILL — user paused briefly; keep listening (UI may show a cue)
          * io.QUIT    — user requested to quit
        """
        ...


class EventHandler(ABC):
    def on_state(self, state: str):
        """state: idle | listening | transcribing | thinking | speaking | error"""

    def on_user(self, text: str):
        """Final transcript of what the user said."""

    def on_agent(self, sentence: str, idx: int | None = None):
        """One streamed sentence of the agent's reply (for the transcript UI).

        `idx` numbers sentences within the session; the web client reports the
        last one it actually started playing, so an interruption knows what was
        heard.
        """

    def on_pause(self):
        """The user paused mid-thought; the UI may show a subtle cue.

        Unlike on_state, this must not change capture behavior — the mic stays
        open while the user gathers their thoughts, so resumed speech isn't
        dropped. No audio is played for this.
        """

    def on_error(self, message: str):
        """A recoverable error this turn."""

    def on_exam(self, payload: dict):
        """IELTS mock-test progress: phase changes, the Part 2 cue card, timers,
        and the final report. `payload["event"]` says which."""


# --- terminal frontends ------------------------------------------------------

class MicSource(AudioSource):
    """Records from the host microphone via sounddevice (terminal CLI)."""

    def listen(self):
        from app import asr  # imported lazily so the web server can skip the mic

        print("Speak something...")
        audio = asr.record_until_silence(lambda: asr._input)
        if audio is None:
            print("No speech detected, try again.")
        return audio


class TerminalHandler(EventHandler):
    def on_state(self, state):
        if state == "transcribing":
            print("Processing Audio ... (Local Whisper)")

    def on_user(self, text):
        print("You said:", text)

    def on_agent(self, sentence, idx=None):
        # The TTS worker already prints "[tts] <sentence>" in verbose mode.
        pass

    def on_error(self, message):
        print(f"(turn failed: {message})")


# --- web frontends -----------------------------------------------------------

class WebSocketSource(AudioSource):
    """Buffers PCM arriving over a WebSocket and endpoint-detects speech.

    The FastAPI handler pushes raw 16 kHz int16 bytes into `in_q`. `listen()`
    blocks until the recorder has a finished clip, or until QUIT is pushed.
    """

    def __init__(self, in_q: "queue.Queue"):
        from app.recorder import StreamingRecorder

        self._in_q = in_q
        self.recorder = StreamingRecorder()
        # True while a BACKFILL was returned and the same utterance continues.
        self._continuing = False

    def set_pause(self, seconds: float, lo: float = None, hi: float = None):
        """Change how long a quiet spell ends the user's turn (live)."""
        self.recorder.set_pause(seconds, lo, hi)

    def set_max_record(self, seconds: float):
        """Change the cap on one recorded turn."""
        self.recorder.set_max_record(seconds)

    def set_idle_timeout(self, seconds):
        """Give up on a turn after this long with no speech at all."""
        self.recorder.set_idle_timeout(seconds)

    def flush(self):
        """Throw away mic audio queued while we were busy talking."""
        while True:
            try:
                item = self._in_q.get_nowait()
            except queue.Empty:
                return
            if item is QUIT:           # never swallow a shutdown request
                self._in_q.put(QUIT)
                return

    def listen(self):
        # Only start a fresh recording for a new turn. After a BACKFILL the
        # recorder keeps the audio it has already collected.
        if not self._continuing:
            self.recorder.reset()
        self._continuing = False

        while True:
            item = self._in_q.get()
            if item is QUIT:
                return QUIT
            pcm = np.frombuffer(item, dtype=np.int16)
            result = self.recorder.push(pcm)
            if result is BACKFILL:
                self._continuing = True
                return BACKFILL
            if result is not None:
                return result
            if self.recorder.done:
                # _finalize() rejected the clip (too little speech). Reset so
                # the next utterance starts clean; otherwise done=True makes
                # every later push() a no-op and the session stalls.
                self.recorder.reset()
                return None


class WebSocketSink:
    """Forwards synthesized TTS audio (float32 24 kHz) over the WebSocket."""

    SAMPLE_RATE = 24000

    def __init__(self, out_q: "queue.Queue"):
        self._out_q = out_q
        self._tail_end = 0.0  # monotonic time the browser should run out of audio

    def write(self, audio: np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
        self._out_q.put(audio.tobytes())
        # Kokoro synthesizes far faster than real time, so the browser is usually
        # still playing long after the last write. Keep a ledger of that backlog.
        self._tail_end = max(time.monotonic(), self._tail_end) + len(audio) / self.SAMPLE_RATE

    def remaining(self) -> float:
        """Seconds of already-sent audio the browser should still be playing."""
        return max(0.0, self._tail_end - time.monotonic())

    def drain(self):
        # The browser owns playback timing; nothing to wait for here.
        pass


class WebSocketHandler(EventHandler):
    """Serializes engine events into the WebSocket outbound queue."""

    def __init__(self, out_q: "queue.Queue"):
        self._out_q = out_q

    def _send(self, obj):
        self._out_q.put(obj)

    def on_state(self, state):
        self._send({"type": "state", "state": state})

    def on_user(self, text):
        self._send({"type": "user", "text": text})

    def on_agent(self, sentence, idx=None):
        event = {"type": "agent", "text": sentence}
        if idx is not None:
            event["idx"] = idx
        self._send(event)

    def on_exam(self, payload):
        self._send({"type": "exam", **payload})

    def on_pause(self):
        # A dedicated event (not a "state") so the client can show a cue
        # without interpreting it as "stop capturing".
        self._send({"type": "pause"})

    def on_error(self, message):
        self._send({"type": "error", "message": message})
