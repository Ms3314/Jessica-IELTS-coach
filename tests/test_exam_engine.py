"""ExamSession tests — scripted candidate, fake LLM/TTS/Whisper, fake clock, no devices.

    python tests/test_exam_engine.py
"""
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Stub the heavy modules BEFORE app.exam.engine imports them.
for _name in ("app.speech.asr", "app.speech.tts"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

from app.exam import examiner, ielts  # noqa: E402
import app.exam.engine as ee  # noqa: E402
from app.io import BACKFILL, QUIT  # noqa: E402

SPOKEN = []          # every line the examiner says, in order
NOW = [0.0]          # the fake clock; only listening makes time pass
SECONDS_PER_ANSWER = 40.0


SPEAK_COST = [0.0]   # seconds of the fake clock one spoken line takes


def _fake_speak(text, sink=None, voice=None):
    SPOKEN.append(text)
    NOW[0] += SPEAK_COST[0]


def _fake_speak_stream(stream, sink=None, voice=None, on_sentence=None, **_k):
    text = "".join(stream)
    SPOKEN.append(text)
    NOW[0] += SPEAK_COST[0]
    if on_sentence:
        on_sentence(text)
    return False


ee.tts.speak = _fake_speak
ee.tts.speak_stream = _fake_speak_stream


class Chunk:
    def __init__(self, c):
        self.content = c


class FakeLLM:
    """Numbers its questions; writes a valid line for every kind of line."""

    GOOD = {
        "full name": "Hello, I'm Jessica, your examiner. This is the speaking test. May I have your full name?",
        "one to two minutes": "Next is Part 2. You'll talk for one to two minutes after one minute to prepare.",
        "one minute of preparation": "Your minute to prepare starts now.",
        "invite the candidate to start": "Please begin; you have one to two minutes, and I'll stop you when time is up. One minute is long.",
        "finished their talk": "Thanks.",
        "end of the speaking test": "Thank you, that's the end of the test.",
    }

    def __init__(self, lines=True):
        self.n, self.lines, self.instructions = 0, lines, []

    def stream(self, messages):
        self.instructions.append(messages[1].content)
        self.n += 1
        yield Chunk(f"Question number {self.n}?")

    def invoke(self, messages):
        text = messages[1].content
        self.instructions.append(text)
        if self.lines:
            for needle, line in self.GOOD.items():
                if needle in text:
                    return Chunk(line)
        return Chunk("")                         # an empty reply -> every line falls back

    def with_structured_output(self, _schema):
        class R:
            def invoke(self, _p):
                return examiner.CueCard(title="Describe a book you enjoyed.", points=["a", "b", "c", "d"],
                                        explain="and explain why.", theme="books")
        return R()


class FakeSource:
    def __init__(self, clips):
        self.clips = list(clips)
        self.calls = []

    def listen(self):
        NOW[0] += SECONDS_PER_ANSWER
        return self.clips.pop(0) if self.clips else QUIT

    manual_calls = None

    def listen_manual(self, wait_s=45.0, max_s=90.0, on_start=None, min_s=0.4, limit_s=None):
        # an answer takes its usual time, unless the part's clock cuts it off sooner
        NOW[0] += SECONDS_PER_ANSWER if limit_s is None else min(SECONDS_PER_ANSWER, max(limit_s, 0))
        self.manual_calls = (self.manual_calls or []) + [{"wait_s": wait_s, "max_s": max_s, "limit_s": limit_s}]
        clip = self.clips.pop(0) if self.clips else QUIT
        if on_start and clip is not None and clip is not QUIT and clip is not BACKFILL:
            on_start()
        return None if clip is BACKFILL else clip

    def flush(self):
        self.calls.append("flush")

    def set_pause(self, s, lo=None, hi=None):
        self.calls.append(("pause", s))

    def set_max_record(self, s):
        self.calls.append(("max", s))

    def set_idle_timeout(self, s):
        self.calls.append(("idle", s))


class FakeEvents:
    def __init__(self):
        self.exam, self.users, self.states, self.errors, self.pauses = [], [], [], [], 0

    def on_state(self, s):
        self.states.append(s)

    def on_user(self, t):
        self.users.append(t)

    def on_agent(self, s, idx=None):
        pass

    def on_pause(self):
        self.pauses += 1

    def on_error(self, m):
        self.errors.append(m)

    def on_exam(self, p):
        self.exam.append(p)


def _clip(seconds):
    return np.ones(int(16000 * seconds), dtype=np.int16) * 800


def _steps(p1=100.0, p3=100.0):
    return [
        ielts.Step(ielts.INTRO, 0),
        ielts.Step(ielts.P1_SECTION, 1, p1),
        ielts.Step(ielts.P2_SETUP, 2), ielts.Step(ielts.P2_PREP, 2),
        ielts.Step(ielts.P2_TALK, 2), ielts.Step(ielts.P2_ROUND, 2),
        ielts.Step(ielts.P3_INTRO, 3), ielts.Step(ielts.P3_SECTION, 3, p3),
        ielts.Step(ielts.END, 0),
    ]


def _run(clips, texts=None, steps=None, llm=None, sink=None, on_complete=None, manual=False, length=None):
    SPOKEN.clear()
    NOW[0] = 0.0
    SPEAK_COST[0] = 0.0
    texts = list(texts) if texts is not None else None
    ee.asr.transcribe = lambda _clip: (texts.pop(0) if texts else "an answer")
    tmp = tempfile.TemporaryDirectory()
    log = ielts.ExamLog("s", "e", root=Path(tmp.name))
    src, ev = FakeSource(clips), FakeEvents()
    sess = ee.ExamSession(src, sink or object(), ev, llm or FakeLLM(), steps=steps or _steps(),
                          log=log, on_complete=on_complete, sleep=lambda _s: None, clock=lambda: NOW[0],
                          manual=manual, length=length)
    sess._wait = lambda _s: None      # don't really wait the one-minute preparation
    sess.loop()
    return sess, src, ev, log, tmp


def _full_clips():
    # intro | part 1: 3 answers (40 s each fills 100 s) | part 2 talk (long) | round | part 3: 3 answers
    return [_clip(2)] + [_clip(3)] * 3 + [_clip(100)] + [_clip(3)] + [_clip(4)] * 3


def test_a_timed_section_asks_until_its_time_is_up():
    _, _, _, log, tmp = _run(_full_clips())
    p1 = [t for t in log.turns if t.part == 1][1:]            # skip the name question
    p3 = [t for t in log.turns if t.part == 3]
    # 40 s per answer against a 100 s part: questions start at 0, 40, 80 -> three, never a fourth
    assert len(p1) == 3 and len(p3) == 3
    tmp.cleanup()


def test_no_question_is_started_after_the_time_limit():
    _, _, _, log, tmp = _run(_full_clips(), steps=_steps(p1=30, p3=30))
    assert len([t for t in log.turns if t.part == 1]) == 2    # name + one timed question (40 s > 30 s)
    tmp.cleanup()


def test_the_exam_records_every_answer_in_order_and_finishes():
    _, _, _, log, tmp = _run(_full_clips())
    assert [t.part for t in log.turns][:2] == [1, 1]
    assert [t.part for t in log.turns].count(2) == 2          # the talk and the rounding-off question
    assert log.finished is True and log.cue_card["theme"] == "books"
    assert all((log.dir / f).exists() for t in log.turns for f in t.audio)
    tmp.cleanup()


def test_the_model_chooses_the_questions_and_sees_the_clock_and_history():
    llm = FakeLLM()
    _run(_full_clips(), llm=llm)
    part1 = [i for i in llm.instructions if "Part 1" in i and "Ask the next question" in i]
    assert "first question of Part 1" in part1[0]
    assert "0:00 of about 1:40" in part1[0]
    assert "0:40 of about 1:40" in part1[1] and "Question number 1?" in part1[1]
    assert not any(" topic pool" in i for i in llm.instructions)


def test_examiner_lines_are_written_by_the_model_when_valid():
    _run(_full_clips())
    assert SPOKEN[0].startswith("Hello, I'm Jessica")
    assert examiner.INTRO_LINE not in SPOKEN, "a valid generated line replaces the scripted one"
    assert any("Next is Part 2" in line for line in SPOKEN)
    assert SPOKEN[-1].startswith("Thank you, that's the end")


def test_fallback_lines_are_used_when_the_model_does_not_cooperate():
    _run(_full_clips(), llm=FakeLLM(lines=False))
    assert SPOKEN[0] == examiner.INTRO_LINE
    assert examiner.PART2_ANNOUNCE in SPOKEN and examiner.PART2_PREP_LINE in SPOKEN
    assert SPOKEN[-1] == examiner.END_LINE


def test_the_cue_card_is_read_out_exactly_as_shown():
    _, _, ev, _, tmp = _run(_full_clips())
    card = [e for e in ev.exam if e["event"] == "cue_card"][0]["card"]
    assert examiner.cue_card_text(examiner.CueCard(**card)) in SPOKEN
    tmp.cleanup()


def test_events_give_the_ui_everything_it_needs_for_timers():
    _, _, ev, log, tmp = _run(_full_clips())
    names = [e["event"] for e in ev.exam]
    assert names[0] == "start" and names[-1] == "complete"
    assert ev.exam[0]["schedule"] == ielts.schedule()
    parts = [e for e in ev.exam if e["event"] == "part"]
    assert [(p["part"], p["limit_s"]) for p in parts] == [(1, 100.0), (3, 100.0)]
    assert [e for e in ev.exam if e["event"] == "prep"][0]["seconds"] == ielts.PART2_PREP_S
    assert [e for e in ev.exam if e["event"] == "talk"][0]["max_seconds"] == ielts.PART2_TALK_MAX_S
    assert ev.exam[-1]["finished"] is True
    assert ev.exam[-1]["answered"] == sum(1 for t in log.turns if t.answer) > 0
    steps = [e for e in ev.exam if e["event"] == "step"]
    assert [s["elapsed_s"] for s in steps] == sorted(s["elapsed_s"] for s in steps)
    tmp.cleanup()


def test_part2_is_recorded_with_long_limits_then_restored():
    _, src, _, _, tmp = _run(_full_clips())
    assert ("max", ielts.PART2_RECORD_CAP_S) in src.calls
    last_max = [c for c in src.calls if isinstance(c, tuple) and c[0] == "max"][-1]
    assert last_max == ("max", ielts.ANSWER_RECORD_CAP_S), "limits must go back after Part 2"
    tmp.cleanup()


def test_a_short_part2_talk_gets_one_invitation_to_say_more():
    clips = [_clip(2)] + [_clip(3)] * 3 + [_clip(20), _clip(15)] + [_clip(3)] + [_clip(4)] * 3
    _, _, _, log, tmp = _run(clips, llm=FakeLLM(lines=False))
    assert examiner.PART2_MORE_LINE in SPOKEN
    part2 = [t for t in log.turns if t.part == 2][0]
    assert len(part2.audio) == 2
    tmp.cleanup()


def test_a_long_part2_talk_is_not_asked_for_more_and_gets_a_plain_thanks():
    _run(_full_clips(), llm=FakeLLM(lines=False))
    assert examiner.PART2_MORE_LINE not in SPOKEN and examiner.PART2_STOP_LINE not in SPOKEN
    assert examiner.THANKS_LINE in SPOKEN


def test_a_talk_that_reaches_the_time_limit_is_stopped_politely():
    clips = [_clip(2)] + [_clip(3)] * 3 + [_clip(ielts.PART2_TALK_MAX_S)] + [_clip(3)] + [_clip(4)] * 3
    _run(clips, llm=FakeLLM(lines=False))
    assert examiner.PART2_STOP_LINE in SPOKEN


def test_repeat_request_re_asks_the_question_without_recording_the_request():
    clips = [_clip(1), _clip(1)]
    _, _, _, log, tmp = _run(clips, ["Sorry, could you repeat that?", "Sam"], steps=[ielts.Step(ielts.INTRO, 0)],
                             llm=FakeLLM(lines=False))
    assert any(line.startswith(examiner.REPEAT_LEAD) for line in SPOKEN)
    assert log.turns[0].answer == "Sam" and len(log.turns[0].audio) == 1
    tmp.cleanup()


def test_silence_marks_the_turn_unanswered_and_moves_on():
    clips = [None, None, _clip(3)]
    steps = [ielts.Step(ielts.INTRO, 0), ielts.Step(ielts.P1_SECTION, 1, 10)]
    _, _, _, log, tmp = _run(clips, ["I answered later"], steps=steps)
    assert log.turns[0].no_answer is True and log.turns[0].answer == ""
    assert log.turns[1].answer == "I answered later"
    tmp.cleanup()


def test_backfill_pause_does_not_end_the_answer():
    _, _, ev, log, tmp = _run([BACKFILL, _clip(3)], ["Sam"], steps=[ielts.Step(ielts.INTRO, 0)])
    assert ev.pauses == 1 and log.turns[0].answer == "Sam"
    tmp.cleanup()


def test_quitting_midway_saves_what_was_recorded_and_reports_unfinished():
    done = []
    _, _, ev, log, tmp = _run([_clip(2)], ["Sam"], on_complete=done.append)
    assert log.finished is False and log.turns[0].answer == "Sam"
    assert done == [log], "on_complete must still fire so a partial report can be made"
    assert ev.exam[-1]["finished"] is False
    tmp.cleanup()


def test_the_users_pause_setting_applies_to_part_1_and_3_answers_but_not_the_long_turn():
    SPOKEN.clear()
    NOW[0] = 0.0
    ee.asr.transcribe = lambda _c: "an answer"
    tmp = tempfile.TemporaryDirectory()
    src = FakeSource([_clip(2), _clip(3)])
    sess = ee.ExamSession(src, object(), FakeEvents(), FakeLLM(), steps=[ielts.Step(ielts.INTRO, 0)],
                          log=ielts.ExamLog("s", "e", root=Path(tmp.name)), sleep=lambda _s: None,
                          clock=lambda: NOW[0], manual=False)
    sess.pause_s = 2.2                      # what the server sets from the Settings slider
    sess.loop()
    assert ("pause", 2.2) in src.calls
    src.calls.clear()
    sess._card = examiner.CueCard(title="t", points=["a", "b", "c"], explain="e", theme="x")
    sess._part2_talk()
    assert ("pause", ielts.PART2_PAUSE_S) in src.calls, "Part 2 keeps its own long pause allowance"
    tmp.cleanup()


# --- recording by hand (the real test's default) and the short length ---------------------

def test_by_hand_the_candidate_is_given_an_answer_window_for_every_question():
    _, src, ev, log, tmp = _run(_full_clips(), manual=True)
    windows = [e for e in ev.exam if e["event"] == "answer_window"]
    ends = [e for e in ev.exam if e["event"] == "answer_end"]
    assert len(windows) == len(ends) == len(src.manual_calls) > 5
    assert windows[0]["max_s"] == ielts.MANUAL_ANSWER_CAP_S and windows[0]["wait_s"] == ielts.MANUAL_WAIT_S
    assert log.finished is True and [t.part for t in log.turns][:2] == [1, 1]
    tmp.cleanup()


def test_by_hand_nothing_depends_on_pauses_so_no_voice_detection_settings_are_touched():
    _, src, _, _, tmp = _run(_full_clips(), manual=True)
    assert not [c for c in src.calls if isinstance(c, tuple)], "no pause/limit/idle configuration"
    tmp.cleanup()


def test_by_hand_the_part2_clock_starts_when_the_candidate_starts_speaking():
    _, src, ev, _, tmp = _run(_full_clips(), manual=True)
    names = [e["event"] for e in ev.exam]
    i_talk = names.index("talk")
    # the answer window opens first; the timer event only follows once speech began
    assert names[i_talk - 1] == "answer_window"
    assert ev.exam[i_talk]["max_seconds"] == ielts.PART2_TALK_MAX_S
    part2 = [c for c in src.manual_calls if c["max_s"] == ielts.timing_for("full").part2_talk_s]
    assert len(part2) == 1
    tmp.cleanup()


def test_by_hand_an_unanswered_window_is_a_no_answer_and_the_test_moves_on():
    clips = [None, _clip(3)]
    steps = [ielts.Step(ielts.INTRO, 0), ielts.Step(ielts.P1_SECTION, 1, 10)]
    _, _, _, log, tmp = _run(clips, ["Hello"], steps=steps, manual=True)
    assert log.turns[0].no_answer is True and log.turns[1].answer == "Hello"
    tmp.cleanup()


def test_by_hand_a_repeat_request_still_works():
    _, _, _, log, tmp = _run([_clip(1), _clip(1)], ["Sorry, could you repeat that?", "Sam"],
                             steps=[ielts.Step(ielts.INTRO, 0)], llm=FakeLLM(lines=False), manual=True)
    assert any(line.startswith(examiner.REPEAT_LEAD) for line in SPOKEN)
    assert log.turns[0].answer == "Sam"
    tmp.cleanup()


def test_by_hand_a_short_part2_talk_still_gets_one_invitation_to_say_more():
    clips = [_clip(2)] + [_clip(3)] * 3 + [_clip(20), _clip(15)] + [_clip(3)] + [_clip(4)] * 3
    _, _, _, log, tmp = _run(clips, manual=True, llm=FakeLLM(lines=False))
    assert examiner.PART2_MORE_LINE in SPOKEN
    assert len([t for t in log.turns if t.part == 2][0].audio) == 2
    tmp.cleanup()


def test_the_exam_records_which_length_it_was():
    for length in ("full", "short"):
        _, _, _, log, tmp = _run(_full_clips(), manual=True, length=length)
        assert log.length == length
        tmp.cleanup()


def test_the_short_test_announces_its_own_shorter_times():
    clips = [_clip(2)] + [_clip(3)] * 3 + [_clip(40)] + [_clip(3)] + [_clip(4)] * 3   # 40 s answers, 120 s parts
    _, _, ev, log, tmp = _run(clips, manual=True, length="short", llm=FakeLLM(lines=False),
                              steps=ielts.build_steps("short"))
    assert ev.exam[0]["length"] == "short" and ev.exam[0]["schedule"] == ielts.schedule("short")
    prep = [e for e in ev.exam if e["event"] == "prep"][0]
    assert prep["seconds"] == 30.0
    parts = {e["part"]: e["limit_s"] for e in ev.exam if e["event"] == "part"}
    assert parts == {1: 120.0, 3: 120.0}
    assert any("30 seconds" in line for line in SPOKEN), "the examiner says the short times, not the full ones"
    assert not any("one minute to" in line.replace("up to one minute", "") for line in SPOKEN)
    tmp.cleanup()


def test_a_short_talk_is_judged_against_the_short_time_limit():
    # 35 s is plenty for a 60 s limit (needs >= 30 s), so no invitation to say more
    clips = [_clip(2)] + [_clip(3)] * 3 + [_clip(35)] + [_clip(3)] + [_clip(4)] * 3
    _run(clips, manual=True, length="short", llm=FakeLLM(lines=False), steps=ielts.build_steps("short"))
    assert examiner.PART2_MORE_LINE not in SPOKEN


# --- the clock is hard ----------------------------------------------------------------------

def test_the_part_clock_cuts_off_the_answer_in_progress_and_the_test_moves_on():
    steps = [ielts.Step(ielts.P1_SECTION, 1, 100), ielts.Step(ielts.END, 0)]
    _, src, ev, log, tmp = _run([_clip(30)] * 5, steps=steps, manual=True)
    # answers take 40 s; questions start at 0 s, 40 s and 80 s, with 100, 60 and 20 s left
    assert [round(c["limit_s"]) for c in src.manual_calls] == [100, 60, 20]
    assert len([t for t in log.turns if t.part == 1]) == 3
    assert log.turns[-1].answer, "the answer cut off by the clock is still kept and transcribed"
    assert NOW[0] == 100, "the part ended exactly when its clock did"
    assert log.finished is True, "and the test carried on to the next step"
    tmp.cleanup()


def test_part_three_has_the_same_hard_clock():
    # Part 3 questions are about the Part 2 topic, so the card has to exist first
    steps = [ielts.Step(ielts.P2_SETUP, 2), ielts.Step(ielts.P3_SECTION, 3, 30), ielts.Step(ielts.END, 0)]
    _, src, _, _, tmp = _run([_clip(5)] * 3, steps=steps, manual=True)
    assert round(src.manual_calls[0]["limit_s"]) == 30
    tmp.cleanup()


def test_no_new_question_is_started_in_the_last_seconds_of_a_part():
    steps = [ielts.Step(ielts.P1_SECTION, 1, 85), ielts.Step(ielts.END, 0)]
    _, src, _, log, tmp = _run([_clip(3)] * 5, steps=steps, manual=True)
    # after two 40 s answers only 5 s are left, less than ee.MIN_NEW_QUESTION_S
    assert len(src.manual_calls) == 2 and len(log.turns) == 2
    tmp.cleanup()


def test_a_question_that_eats_the_rest_of_the_part_gets_no_recording_window():
    SPEAK_COST[0] = 0.0
    steps = [ielts.Step(ielts.P1_SECTION, 1, 20), ielts.Step(ielts.END, 0)]

    def run_with_slow_speech():
        import types as _t

        clips = [_clip(3)] * 3
        SPOKEN.clear()
        NOW[0] = 0.0
        ee.asr.transcribe = lambda _c: "an answer"
        SPEAK_COST[0] = 25.0           # asking the question takes longer than the whole part
        tmp = tempfile.TemporaryDirectory()
        log = ielts.ExamLog("s", "e", root=Path(tmp.name))
        src, ev = FakeSource(clips), FakeEvents()
        ee.ExamSession(src, object(), ev, FakeLLM(), steps=steps, log=log, sleep=lambda _s: None,
                       clock=lambda: NOW[0], manual=True).loop()
        SPEAK_COST[0] = 0.0
        return src, ev, log, tmp

    src, ev, log, tmp = run_with_slow_speech()
    assert src.manual_calls is None, "nothing to record when the time is already gone"
    assert not [e for e in ev.exam if e["event"] == "answer_window"]
    assert log.turns[0].no_answer is True
    tmp.cleanup()


def test_part_two_talk_is_ended_by_its_clock_not_by_a_margin():
    _, src, _, _, tmp = _run(_full_clips(), manual=True)
    talk = [c for c in src.manual_calls if c["max_s"] == ielts.PART2_TALK_MAX_S]
    assert len(talk) == 1 and talk[0]["limit_s"] is None, "Part 2 uses the talk length itself as its cap"
    tmp.cleanup()


def test_the_invitation_to_say_more_only_gets_the_time_that_is_left():
    clips = [_clip(2)] + [_clip(3)] * 3 + [_clip(20), _clip(15)] + [_clip(3)] + [_clip(4)] * 3
    _, src, _, _, tmp = _run(clips, manual=True, llm=FakeLLM(lines=False))
    caps = [c["max_s"] for c in src.manual_calls]
    assert ielts.PART2_TALK_MAX_S in caps
    assert ielts.PART2_TALK_MAX_S - 20 in caps, "120 s minus the 20 s already talked"
    tmp.cleanup()


def test_interrupts_are_ignored_in_the_exam():
    sess = ee.ExamSession(FakeSource([]), object(), FakeEvents(), FakeLLM(), steps=[])
    sess.interrupt("button", 3)
    assert sess.running is True and sess.interruptible is False


def test_playback_backlog_is_waited_out_before_listening():
    class Sink:
        def __init__(self):
            self.t = [3.0, 1.5, 0.0]

        def remaining(self):
            return self.t.pop(0) if self.t else 0.0

    slept = []
    sess = ee.ExamSession(FakeSource([]), Sink(), FakeEvents(), FakeLLM(), steps=[],
                          sleep=lambda s: slept.append(s))
    sess._wait_playback()
    assert len(slept) == 2 and all(s > 0 for s in slept)


def test_a_section_gives_up_after_repeated_model_failures():
    class BoomLLM(FakeLLM):
        def stream(self, _m):
            raise RuntimeError("api down")
            yield  # pragma: no cover

    steps = [ielts.Step(ielts.P1_SECTION, 1, 300), ielts.Step(ielts.END, 0)]
    _, _, ev, log, tmp = _run([], steps=steps, llm=BoomLLM())
    assert len(ev.errors) == ee.MAX_FAILED_QUESTIONS, "stops after two failures, not 300 s of retries"
    assert log.finished is True and SPOKEN[-1].startswith(("Thank you", examiner.END_LINE))
    tmp.cleanup()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} exam engine tests passed")
