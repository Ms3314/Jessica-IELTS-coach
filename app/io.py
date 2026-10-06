"""Pluggable I/O so the SAME conversation engine drives both frontends.

  * AudioSource  — yields recorded mic clips (terminal: sounddevice; web: WS)
  * Sink        — where synthesized TTS audio goes (terminal: speakers; web: WS)
  * EventHandler — state + transcript callbacks (terminal: prints; web: WS events)

The terminal CLI (main.py) wires MicSource + LocalSink + TerminalHandler.
The web server (server.py) wires WebSocketSource + WebSocketSink + WebSocketHandler.
"""
import queue
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

    def on_agent(self, sentence: str):
        """One streamed sentence of the agent's reply (for the transcript UI)."""

    def on_pause(self):
        """The user paused mid-thought; the UI may show a subtle cue.

        Unlike on_state, this must not change capture behavior — the mic stays
        open while the user gathers their thoughts, so resumed speech isn't
        dropped. No audio is played for this.
        """

    def on_error(self, message: str):
        """A recoverable error this turn."""


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

    def on_agent(self, sentence):
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

    def __init__(self, out_q: "queue.Queue"):
        self._out_q = out_q

    def write(self, audio: np.ndarray):
        self._out_q.put(np.asarray(audio, dtype=np.float32).tobytes())

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

    def on_agent(self, sentence):
        self._send({"type": "agent", "text": sentence})

    def on_pause(self):
        # A dedicated event (not a "state") so the client can show a cue
        # without interpreting it as "stop capturing".
        self._send({"type": "pause"})

    def on_error(self, message):
        self._send({"type": "error", "message": message})
