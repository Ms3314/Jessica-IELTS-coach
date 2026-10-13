"""IELTS exam structure + record tests — no models, no devices.

    python tests/test_ielts.py
"""
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app import ielts


def test_steps_follow_the_real_exam_order():
    kinds = [s.kind for s in ielts.build_steps()]
    assert kinds == [ielts.INTRO, ielts.P1_SECTION, ielts.P2_SETUP, ielts.P2_PREP, ielts.P2_TALK,
                     ielts.P2_ROUND, ielts.P3_INTRO, ielts.P3_SECTION, ielts.END]


def test_parts_one_and_three_are_timed_sections():
    steps = {s.kind: s for s in ielts.build_steps()}
    assert steps[ielts.P1_SECTION].limit_s == ielts.PART1_LIMIT_S > 0
    assert steps[ielts.P3_SECTION].limit_s == ielts.PART3_LIMIT_S > 0
    assert steps[ielts.P1_SECTION].part == 1 and steps[ielts.P3_SECTION].part == 3


def test_nothing_about_topics_or_question_counts_is_hard_coded():
    # the model picks the topics and decides how many questions fit the time
    for gone in ("FIRST_TOPICS", "PART1_TOPIC_POOL", "PART1_QUESTIONS_PER_TOPIC", "PART3_QUESTIONS"):
        assert not hasattr(ielts, gone), gone


def test_schedule_describes_every_part_for_the_ui():
    sched = ielts.schedule()
    assert [p["part"] for p in sched] == [1, 2, 3]
    assert sched[0]["limit_s"] == ielts.PART1_LIMIT_S
    assert sched[1] == {"part": 2, "prep_s": ielts.PART2_PREP_S, "talk_s": ielts.PART2_TALK_MAX_S}
    assert sched[2]["limit_s"] == ielts.PART3_LIMIT_S


def _clip(seconds):
    return (np.ones(int(ielts.SAMPLE_RATE * seconds), dtype=np.int16) * 500)


def test_clip_is_saved_as_a_readable_wav_and_timed():
    with tempfile.TemporaryDirectory() as tmp:
        log = ielts.ExamLog("s1", "e1", root=Path(tmp))
        turn = log.start_turn(1, "What is your name?")
        log.add_clip(turn, _clip(2.0), "My name is Sam")
        path = log.dir / turn.audio[0]
        with wave.open(str(path)) as w:
            assert w.getframerate() == 16000 and w.getnchannels() == 1
            assert w.getnframes() == 32000
        assert abs(turn.duration_s - 2.0) < 1e-6
        assert turn.answer == "My name is Sam"


def test_second_clip_joins_the_same_turn():
    with tempfile.TemporaryDirectory() as tmp:
        log = ielts.ExamLog("s1", "e1", root=Path(tmp))
        turn = log.start_turn(2, "Describe a book")
        log.add_clip(turn, _clip(1.0), "It was a novel")
        log.add_clip(turn, _clip(1.5), "and I loved it")
        assert turn.audio == ["turn01.wav", "turn01_2.wav"]
        assert turn.answer == "It was a novel and I loved it"
        assert abs(turn.duration_s - 2.5) < 1e-6


def test_save_and_load_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        log = ielts.ExamLog("s1", "e1", root=Path(tmp))
        log.cue_card = {"title": "Describe a place"}
        t = log.start_turn(1, "Where do you live?")
        log.add_clip(t, _clip(1.0), "In a city")
        log.finished = True
        log.save()
        loaded = ielts.ExamLog.load("s1", "e1", root=Path(tmp))
        assert loaded.finished and loaded.cue_card == {"title": "Describe a place"}
        assert loaded.turns[0].question == "Where do you live?"
        assert loaded.turns[0].audio == ["turn01.wav"]


def test_transcript_marks_unanswered_questions():
    with tempfile.TemporaryDirectory() as tmp:
        log = ielts.ExamLog("s1", "e1", root=Path(tmp))
        log.start_turn(1, "Q one?")
        t = log.start_turn(1, "Q two?")
        t.answer = "an answer"
        text = log.transcript()
        assert "Candidate: [no answer]" in text and "Candidate: an answer" in text


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} ielts tests passed")
