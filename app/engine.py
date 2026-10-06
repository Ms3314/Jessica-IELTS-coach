"""Conversation engine shared by the terminal CLI and the web server.

A Session runs the same loop for both frontends:

    listen (mic clip) -> transcribe (Whisper) -> agent (OpenAI + Tavily)
    -> TTS (Kokoro, streamed) -> [repeat]

The actual audio capture, audio playback, and event reporting are injected via
AudioSource / Sink / EventHandler, so the engine itself has no UI or transport
knowledge. This is the single source of truth for conversation flow.
"""
import threading

from app import asr, tts, agent
from app.io import QUIT, BACKFILL

# only an utterance that IS one of these quits the app —
# "how do I say goodbye in French?" must not (substring match did exactly that)
GOODBYE_WORDS = {"goodbye", "good bye", "bye", "bye bye", "quit", "exit"}


def _is_goodbye(text: str) -> bool:
    return text.lower().strip(" .!?") in GOODBYE_WORDS


class Session:
    def __init__(self, source, sink, events, voice: str = "af_heart"):
        self.source = source
        self.sink = sink
        self.events = events
        self.voice = voice
        self.running = True
        # Set by interrupt() when the user talks over a reply (barge-in).
        self._interrupt = threading.Event()

    def stop(self):
        self.running = False

    def interrupt(self):
        """User barged in: drop the rest of the reply and listen again."""
        self._interrupt.set()

    def loop(self):
        """Run turns until the user quits or the source signals QUIT."""
        while self.running:
            try:
                self.events.on_state("listening")
                clip = self.source.listen()
                # A new turn starts now. Ignore any barge-in that arrived while
                # we were idle; it should only cancel a reply in progress.
                self._interrupt.clear()

                if clip is QUIT:
                    break
                if clip is BACKFILL:
                    # The user paused mid-thought. No audio is played — the UI
                    # gets a silent cue — and we keep listening on the SAME
                    # utterance (the source does not reset its recorder), so the
                    # eventual clip still has everything.
                    #
                    # Deliberately no on_state("waiting") here: the web client
                    # stops capturing unless the state is "listening", so the
                    # mic must stay open while the user gathers their thoughts.
                    self.events.on_pause()
                    continue
                if clip is None:
                    # silence / no speech — go straight back to listening
                    self.events.on_state("idle")
                    continue

                self.events.on_state("transcribing")
                text = asr.transcribe(clip)
                if not text:
                    self.events.on_state("idle")
                    continue

                if _is_goodbye(text):
                    self.events.on_state("speaking")
                    tts.speak(
                        "Goodbye! It was nice talking to you.",
                        sink=self.sink,
                        voice=self.voice,
                    )
                    self.running = False
                    break

                self.events.on_user(text)
                self.events.on_state("thinking")
                tts.speak_stream(
                    agent.agent_reply_stream(text),
                    sink=self.sink,
                    voice=self.voice,
                    on_sentence=self.events.on_agent,  # stream transcript to UI
                    cancel=self._interrupt,            # barge-in support
                )
                self.events.on_state("idle")

            except KeyboardInterrupt:
                break
            except Exception as e:
                # one API hiccup (OpenAI/Tavily) must not kill the session —
                # apologize out loud and keep listening
                self.events.on_error(str(e))
                try:
                    tts.speak(
                        "Sorry, something went wrong on my end. Please try again.",
                        sink=self.sink,
                        voice=self.voice,
                    )
                except Exception:
                    pass
