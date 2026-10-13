"""Conversation engine shared by the terminal CLI and the web server.

A Session runs the same loop for both frontends:

    listen (mic clip) -> transcribe (Whisper) -> agent (OpenAI + Tavily)
    -> TTS (Kokoro, streamed) -> [repeat]

The actual audio capture, audio playback, and event reporting are injected via
AudioSource / Sink / EventHandler, so the engine itself has no UI or transport
knowledge. This is the single source of truth for conversation flow.
"""
import logging
import threading
import time

from app import asr, tts, agent
from app.io import QUIT, BACKFILL

logger = logging.getLogger("speakIelts.engine")

# Lazy import: some tests stub app.agent at module level, and sessions imports
# the compiled agent graph. Loading it lazily keeps the engine testable.
_sessions_db = None


def _get_sessions_db():
    global _sessions_db
    if _sessions_db is None:
        from app import sessions as _sessions_db
    return _sessions_db

# only an utterance that IS one of these quits the app —
# "how do I say goodbye in French?" must not (substring match did exactly that)
GOODBYE_WORDS = {"goodbye", "good bye", "bye", "bye bye", "quit", "exit"}

# A cut-off utterance stays open this long: if the user keeps talking after a
# barge-in, their continuation is merged into one message ("I like football …
# and I play every weekend"). Beyond this, it's a new turn.
STITCH_WINDOW_S = 20.0


def _is_goodbye(text: str) -> bool:
    return text.lower().strip(" .!?") in GOODBYE_WORDS


class Session:
    def __init__(
        self,
        source,
        sink,
        events,
        voice: str = "af_heart",
        session_id: str = "default",
        greet: bool = False,
    ):
        self.source = source
        self.sink = sink
        self.events = events
        self.voice = voice
        self.session_id = session_id  # LangGraph thread: the memory of this call
        self.greet = greet             # True when the user just clicked the mic
        self.running = True
        # Set by interrupt() when the user stops a reply (talking over it, or
        # the Stop button). _interrupt_by / _heard_idx describe that interruption.
        self._interrupt = threading.Event()
        self._interrupt_by = "voice"
        self._heard_idx = None
        # Session-wide sentence numbering: the browser reports the last sentence
        # it started playing, which maps back to the text via _turn_sentences.
        self._sentence_counter = 0

    def stop(self):
        self.running = False

    def interrupt(self, by: str = "voice", heard_idx: int | None = None):
        """User stopped the reply: drop the rest of it and listen again.

        by:        "voice" (talked over her) or "button" (pressed Stop).
        heard_idx: index of the last sentence the user's device started playing
                   (-1 = none). None = unknown, assume everything sent was heard.
        """
        self._interrupt_by = by
        self._heard_idx = heard_idx
        self._interrupt.set()

    def loop(self):
        """Run turns until the user quits or the source signals QUIT."""
        # If this Session was created because the user clicked the mic, open
        # with a personalised question before listening.
        if self.greet:
            from app import greeting

            try:
                self.events.on_state("speaking")
                tts.speak(
                    greeting.opening_question(self.session_id),
                    sink=self.sink,
                    voice=self.voice,
                )
                self.events.on_state("idle")
            except Exception as e:
                logger.warning("greeting failed: %s", e)
                self.events.on_error(str(e))
            if not self.running:
                return

        # A barge-in cancels the reply because the user was still talking. Keep
        # that transcript open for a moment so their continuation can be merged
        # into ONE message instead of being answered as a separate turn.
        stitch_text = None
        stitch_deadline = 0.0

        while self.running:
            try:
                self.events.on_state("listening")
                clip = self.source.listen()
                # A new turn starts now. Ignore any barge-in that arrived while
                # we were idle; it should only cancel a reply in progress.
                self._interrupt.clear()
                self._interrupt_by, self._heard_idx = "voice", None

                if clip is QUIT:
                    break
                if clip is BACKFILL:
                    # The user paused mid-thought. No audio is played — the UI
                    # gets a silent cue — and we keep listening on the SAME
                    # utterance (the source does not reset its recorder), so the
                    # eventual clip still has everything. No state change: the
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

                # The user kept talking through a cancelled reply: merge the two
                # transcripts into one continuous utterance.
                if stitch_text and time.monotonic() <= stitch_deadline:
                    text = f"{stitch_text} ... {text}"
                stitch_text = None

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
                _get_sessions_db().bump(self.session_id)
                _get_sessions_db().maybe_auto_rename(self.session_id, text)
                self.events.on_state("thinking")

                # The UI only knows Jessica is speaking once audio actually
                # starts — that's also what arms client-side barge-in, so the
                # user can talk over her and cut the reply short.
                spoke = False
                turn_sentences = {}  # idx -> text of every sentence sent this turn

                def _on_sentence(sentence):
                    nonlocal spoke
                    if not spoke:
                        spoke = True
                        self.events.on_state("speaking")
                    idx = self._sentence_counter
                    self._sentence_counter += 1
                    turn_sentences[idx] = sentence
                    self.events.on_agent(sentence, idx)

                agent_called = False

                if self._interrupt.is_set():
                    # The user was still talking before the reply even started:
                    # skip the LLM call entirely and go listen to them.
                    cancelled = True
                else:
                    # Print the FULL message (stitched, if it was continued)
                    # right before it becomes the LLM's input.
                    logger.info("message to agent: %s", text)
                    agent_called = True
                    cancelled = tts.speak_stream(
                        agent.agent_reply_stream(text, self.session_id),
                        sink=self.sink,
                        voice=self.voice,
                        on_sentence=_on_sentence,
                        cancel=self._interrupt,  # barge-in support
                    )
                if cancelled:
                    by = self._interrupt_by
                    if agent_called:
                        self._record_interruption(turn_sentences, by)
                    # Talking over her means the user was still mid-thought: keep
                    # the utterance open so the continuation is stitched onto it.
                    # The Stop button means "that answer is wrong" — what comes
                    # next is a fresh message, so nothing is stitched.
                    if by != "button":
                        stitch_text = text
                        stitch_deadline = time.monotonic() + STITCH_WINDOW_S
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

    def _record_interruption(self, turn_sentences, by):
        """Tell the agent's memory which part of the reply the user really heard."""
        heard_idx = self._heard_idx
        heard = " ".join(
            sentence
            for idx, sentence in sorted(turn_sentences.items())
            if heard_idx is None or idx <= heard_idx
        )
        try:
            agent.mark_interrupted(self.session_id, heard, by)
        except Exception as e:
            logger.warning("could not record interruption: %s", e)
