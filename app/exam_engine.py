"""Runs one IELTS mock test: the examiner side of the conversation loop.

Same building blocks as app/engine.py (an AudioSource to listen, a Sink to
speak, an EventHandler for the UI) but a different loop: instead of a free chat
it walks through the fixed steps from app/ielts.py. Parts 1 and 3 run against
the clock; the model chooses what to ask. The examiner never reacts to what the
candidate says (no feedback, no barge-in); every answer is just recorded for
analysis once the test is over.
"""
import logging
import time
from typing import Callable, Optional

from app import asr, examiner, ielts, tts
from app.io import BACKFILL, QUIT

logger = logging.getLogger("speakIelts.exam")

IDLE_TIMEOUT_S = 12.0     # silence after a question before moving on
MAX_REPEATS = 2           # "could you repeat that?" is honoured this many times
DEFAULT_PAUSE_S = 1.5
MAX_FAILED_QUESTIONS = 2  # give up on a section after this many model failures in a row


class ExamSession:
    def __init__(
        self,
        source,
        sink,
        events,
        llm,
        voice: str = "af_heart",
        session_id: str = "default",
        steps: Optional[list] = None,
        log: Optional[ielts.ExamLog] = None,
        on_complete: Optional[Callable[[ielts.ExamLog], None]] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.source = source
        self.sink = sink
        self.events = events
        self.llm = llm
        self.voice = voice
        self.session_id = session_id
        self.steps = steps if steps is not None else ielts.build_steps()
        self.log = log or ielts.ExamLog(session_id)
        self.on_complete = on_complete
        self.running = True
        self._sleep = sleep
        self._clock = clock
        self._sentence_counter = 0
        self._card: Optional[examiner.CueCard] = None
        self._history: list[tuple[str, str]] = []      # (question, answer) so far in the exam
        self._part2_answer = ""
        # How long a pause ends a Part 1 / Part 3 answer (the user's setting).
        self.pause_s = DEFAULT_PAUSE_S

    # A real examiner can't be stopped or talked over: the server leaves the
    # reply audio alone when an interrupt arrives for this kind of session.
    interruptible = False

    # --- the engine interface the server expects -------------------------------------
    def stop(self):
        self.running = False

    def interrupt(self, by: str = "voice", heard_idx: Optional[int] = None):
        """A real examiner doesn't get talked over or stopped: ignored in the exam."""

    # --- main loop -------------------------------------------------------------------
    def loop(self):
        started = self._clock()
        self.events.on_exam({
            "event": "start",
            "exam_id": self.log.exam_id,
            "schedule": ielts.schedule(),
        })
        try:
            for step in self.steps:
                if not self.running:
                    break
                self.events.on_exam({
                    "event": "step", "kind": step.kind, "part": step.part,
                    "elapsed_s": round(self._clock() - started, 1),
                })
                try:
                    self._run_step(step)
                except Exception as e:
                    # one hiccup (LLM / speech) must not end the whole test
                    logger.warning("exam step %s failed: %s", step.kind, e, exc_info=True)
                    self.events.on_error(str(e))
        finally:
            self.log.save()
            self.events.on_state("idle")
            self.events.on_exam({
                "event": "complete",
                "exam_id": self.log.exam_id,
                "finished": bool(self.log.finished),
                "answered": sum(1 for t in self.log.turns if t.answer),
            })
            if self.on_complete:
                try:
                    self.on_complete(self.log)
                except Exception:
                    logger.warning("on_complete failed", exc_info=True)

    # --- one step ----------------------------------------------------------------------
    def _run_step(self, step: ielts.Step):
        k = step.kind
        if k == ielts.INTRO:
            line = self._say_line("intro")
            self._ask_turn(1, line)
        elif k == ielts.P1_SECTION:
            self._run_section(step, lambda elapsed, first: examiner.part1_instruction(
                elapsed, step.limit_s, self._history, first))
        elif k == ielts.P2_SETUP:
            self._history = []
            self._card = examiner.generate_cue_card(self.llm)
            self.log.cue_card = self._card.model_dump()
            self._say_line("part2_announce")
            # the card is read out exactly as it is shown on screen
            self.events.on_exam({"event": "cue_card", "card": self.log.cue_card})
            self._say(examiner.cue_card_text(self._card))
        elif k == ielts.P2_PREP:
            self._say_line("part2_prep")
            self.events.on_state("idle")
            self.events.on_exam({"event": "prep", "seconds": ielts.PART2_PREP_S})
            self._wait(ielts.PART2_PREP_S)
        elif k == ielts.P2_TALK:
            self._part2_talk()
        elif k == ielts.P2_ROUND:
            instruction = examiner.part2_round_instruction(self._card, self._part2_answer)
            self._ask_turn(2, self._say_generated(instruction))
        elif k == ielts.P3_INTRO:
            self._history = []
            self._say_line("part3_intro", theme=self._card.theme)
        elif k == ielts.P3_SECTION:
            self._run_section(step, lambda elapsed, first: examiner.part3_instruction(
                self._card, elapsed, step.limit_s, self._history, first))
        elif k == ielts.END:
            self._say_line("end")
            self.log.finished = True

    # --- timed sections (Parts 1 and 3) -------------------------------------------------
    def _run_section(self, step: ielts.Step, instruction_for):
        """Keep asking questions until the part's time is up.

        A question is never started after the limit, but one already under way
        is allowed to finish, as in the real test.
        """
        begun = self._clock()
        self.events.on_exam({"event": "part", "part": step.part, "limit_s": step.limit_s})
        n, failures = 0, 0
        while self.running and self._clock() - begun < step.limit_s:
            n += 1
            question = self._say_generated(instruction_for(self._clock() - begun, n == 1))
            if not question:
                failures += 1
                if failures >= MAX_FAILED_QUESTIONS:
                    logger.warning("part %s: giving up after %d empty questions", step.part, failures)
                    return
                continue
            failures = 0
            self._ask_turn(step.part, question)

    # --- Part 2 long turn ----------------------------------------------------------------
    def _part2_talk(self):
        self._say_line("part2_start")
        turn = self.log.start_turn(2, self._card.title)
        self._configure(
            pause=ielts.PART2_PAUSE_S, max_record=ielts.PART2_RECORD_CAP_S,
            idle=ielts.PART2_IDLE_S, pause_range=(2.0, 6.0),
        )
        self.events.on_exam({"event": "talk", "max_seconds": ielts.PART2_TALK_MAX_S})
        try:
            if self._record_into(turn):
                if turn.duration_s < ielts.PART2_ENOUGH_S and self.running:
                    # too short: invite more, once (the extra clip joins the same turn)
                    self._say_line("part2_more")
                    self._record_into(turn)
                cut_off = turn.duration_s >= ielts.PART2_TALK_MAX_S - 5
                self._say_line("part2_stop" if cut_off else "thanks")
        finally:
            self._configure_default()
        self._part2_answer = turn.answer

    # --- question / answer ---------------------------------------------------------------
    def _ask_turn(self, part: int, question: str):
        """Record the answer to `question` (already spoken) as a new turn."""
        turn = self.log.start_turn(part, question)
        self._configure_default()
        repeats = 0
        while self.running:
            clip_text = self._listen_once()
            if clip_text is None:
                turn.no_answer = not turn.answer
                break
            clip, text = clip_text
            if repeats < MAX_REPEATS and examiner.is_repeat_request(text):
                repeats += 1
                self._say_line("repeat", question=question)
                continue
            self.log.add_clip(turn, clip, text)
            break
        self._history.append((turn.question, turn.answer))
        self.log.save()

    def _record_into(self, turn: ielts.Turn) -> bool:
        """Listen once and add the clip to `turn`. True if something was recorded."""
        got = self._listen_once()
        if got is None:
            return False
        clip, text = got
        self.log.add_clip(turn, clip, text)
        self.log.save()
        return True

    def _listen_once(self):
        """Wait for one answer. Returns (clip, text), or None if nobody spoke."""
        if hasattr(self.source, "flush"):
            self.source.flush()
        silent_windows = 0
        while self.running:
            self.events.on_state("listening")
            clip = self.source.listen()
            if clip is QUIT:
                self.running = False
                return None
            if clip is BACKFILL:
                self.events.on_pause()
                continue
            if clip is not None:
                self.events.on_state("transcribing")
                text = asr.transcribe(clip)
                if text:
                    self.events.on_user(text)
                    return clip, text
            silent_windows += 1
            if silent_windows > ielts.NO_ANSWER_TIMEOUTS:
                self.events.on_state("idle")
                return None
        return None

    # --- speaking --------------------------------------------------------------------------
    def _announce(self, text: str):
        """Tell the UI what is being said (also gives the browser sentence numbers)."""
        idx = self._sentence_counter
        self._sentence_counter += 1
        self.events.on_agent(text, idx)

    def _say(self, text: str) -> str:
        """Speak a finished line."""
        self.events.on_state("speaking")
        self._announce(text)
        tts.speak(text, sink=self.sink, voice=self.voice)
        self._wait_playback()
        self.events.on_state("idle")
        return text

    def _say_line(self, key: str, **ctx) -> str:
        """Have the model write the line for `key` (or fall back), then speak it."""
        self.events.on_state("thinking")
        return self._say(examiner.write_line(self.llm, key, **ctx))

    def _say_generated(self, instruction: str) -> str:
        """Have the model write the next question, speak it as it streams, and
        return the full text ('' if the model failed)."""
        self.events.on_state("thinking")
        parts: list[str] = []

        def deltas():
            for d in examiner.stream_question(self.llm, instruction):
                parts.append(d)
                yield d

        def on_sentence(sentence):
            self.events.on_state("speaking")
            self._announce(sentence)

        try:
            tts.speak_stream(deltas(), sink=self.sink, voice=self.voice, on_sentence=on_sentence)
        except Exception as e:
            logger.warning("question generation failed: %s", e)
            self.events.on_error(str(e))
        self._wait_playback()
        self.events.on_state("idle")
        return "".join(parts).strip()

    # --- helpers ---------------------------------------------------------------------------
    def _wait_playback(self):
        """The server finishes synthesizing long before the browser finishes
        playing. Wait out that backlog so timers and listening start only once
        the candidate has actually heard the whole line."""
        remaining = getattr(self.sink, "remaining", None)
        if remaining is None:
            return
        while self.running:
            left = remaining()
            if left <= 0:
                break
            self._sleep(min(0.25, left))

    def _wait(self, seconds: float):
        """Silent wait (Part 2 preparation) that still reacts to stop()."""
        end = self._clock() + seconds
        while self.running and self._clock() < end:
            self._sleep(min(0.25, max(0.0, end - self._clock())))

    def _configure(self, pause=None, max_record=None, idle=None, pause_range=(None, None)):
        s = self.source
        if pause is not None and hasattr(s, "set_pause"):
            s.set_pause(pause, *pause_range)
        if max_record is not None and hasattr(s, "set_max_record"):
            s.set_max_record(max_record)
        if hasattr(s, "set_idle_timeout"):
            s.set_idle_timeout(idle)

    def _configure_default(self):
        """Part 1 / Part 3 answers: a natural pause ends the turn; silence times out."""
        self._configure(
            pause=self.pause_s, max_record=ielts.ANSWER_RECORD_CAP_S, idle=IDLE_TIMEOUT_S,
            pause_range=(0.8, 4.0),
        )
