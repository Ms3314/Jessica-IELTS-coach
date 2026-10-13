"""Exam report tests — fake transcriber and fake model, no audio models, no API.

    python tests/test_exam_report.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.exam import report as er
from app.exam import ielts


def test_round_band_follows_the_ielts_rule():
    assert [er.round_band(x) for x in (6.0, 6.125, 6.25, 6.375, 6.5, 6.75, 6.9)] == \
           [6.0, 6.0, 6.5, 6.5, 6.5, 7.0, 7.0]


def test_clamp_band_survives_nonsense_from_the_model():
    assert er.clamp_band(11) == 9.0 and er.clamp_band(-2) == 0.0
    assert er.clamp_band("seven") == 0.0 and er.clamp_band(6.3) == 6.5


def test_only_real_quotes_survive():
    said = "Well, I think that, um, the city is really quite crowded these days."
    kept = er.verify_quotes(["the city is really quite crowded", "I LOVE living there", "um the city"], said)
    assert kept == ["the city is really quite crowded", "um the city"]


def test_reliability_depends_on_how_much_was_said():
    assert er.reliability(40, 60) == "low"
    assert er.reliability(150, 300) == "medium"
    assert er.reliability(400, 900) == "good"


# --- a fake exam on disk ---------------------------------------------------------------------

def _words(n, start=0.0, step=0.4, text="word"):
    return [{"word": f" {text}{i}", "start": start + i * step, "end": start + i * step + 0.25,
             "probability": 0.9} for i in range(n)]


def _make_log(tmp, answered=4, extra_clip=False):
    log = ielts.ExamLog("s1", "e1", root=Path(tmp))
    log.cue_card = {"title": "Describe a book"}
    for i in range(answered):
        t = log.start_turn(1 + i // 2, f"Question {i}?")
        log.add_clip(t, np.ones(16000 * 3, dtype=np.int16) * 500, f"live answer {i}")
        if extra_clip and i == 0:
            log.add_clip(t, np.ones(16000 * 2, dtype=np.int16) * 500, "more")
    log.start_turn(3, "Unanswered?")
    log.finished = True
    log.save()
    return log


def _fake_transcribe(path):
    return {"text": "I really enjoy reading novels because they take me to other places.",
            "segments": [{"words": _words(40)}]}


def _fake_prosody(paths):
    return {"enough_voiced_speech": True, "pitch_variation_semitones": 3.2}


def test_analysis_combines_clips_and_saves_a_file():
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp, extra_clip=True)
        a = er.analyse_exam(log, _fake_transcribe, _fake_prosody)
        assert (log.dir / "analysis.json").exists()
        assert a["fluency"]["n_words"] == 40 * 5, "four answers plus the extra clip, each 40 words"
        assert a["fluency"]["enough_speech"] is True
        assert a["turns"][0]["answer"].count("I really enjoy") == 2   # both clips of turn 1
        assert a["pronunciation_indicators"]["pitch"]["pitch_variation_semitones"] == 3.2
        assert a["lexical"] is not None and a["grammar"] is not None


def test_a_failed_second_pass_falls_back_to_the_live_transcript():
    def boom(_p):
        raise RuntimeError("model missing")

    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        a = er.analyse_exam(log, boom, _fake_prosody)
        assert a["turns"][0]["answer"] == "live answer 0"
        assert any("timing is missing" in c for c in a["caveats"])
        assert a["fluency"]["enough_speech"] is False


class FakeLLM:
    def __init__(self, report=None, fail=False):
        self.report, self.fail, self.prompt, self.calls = report, fail, None, 0

    def with_structured_output(self, _schema):
        llm = self

        class R:
            def invoke(self, prompt):
                llm.calls += 1
                llm.prompt = prompt
                if llm.fail:
                    raise RuntimeError("api down")
                return llm.report

        return R()


def _criterion(band, quotes=("really enjoy reading novels",)):
    return er.Criterion(band=band, summary="s", strengths=["a"], improvements=["b"], evidence=list(quotes))


def _llm_report(**over):
    base = dict(fluency_coherence=_criterion(6.0), lexical_resource=_criterion(7.0),
                grammatical_range_accuracy=_criterion(6.0), pronunciation=_criterion(8.5),
                top_priorities=["one", "two", "three", "four"], examiner_note="Well done.")
    base.update(over)
    return er.LLMReport(**base)


def _analysis(tmp, **kw):
    log = _make_log(tmp)
    return log, er.analyse_exam(log, _fake_transcribe, _fake_prosody)


def test_overall_band_is_computed_in_code_from_the_four_criteria():
    with tempfile.TemporaryDirectory() as tmp:
        log, a = _analysis(tmp)
        r = er.generate_report(FakeLLM(_llm_report()), log, a)
        # 6.0, 7.0, 6.0 and pronunciation capped from 8.5 to 7.5 -> mean 6.625 -> 6.5
        assert r["pronunciation"]["band"] == 7.5
        assert r["overall_band"] == 6.5
        assert r["top_priorities"] == ["one", "two", "three"]


def test_invented_quotes_are_dropped_and_mentioned():
    with tempfile.TemporaryDirectory() as tmp:
        log, a = _analysis(tmp)
        rep = _llm_report(fluency_coherence=_criterion(6, ("really enjoy reading novels", "I never said this")))
        r = er.generate_report(FakeLLM(rep), log, a)
        assert r["fluency_coherence"]["evidence"] == ["really enjoy reading novels"]
        assert any("could not be found" in c for c in r["caveats"])


def test_the_prompt_carries_descriptors_numbers_and_transcript():
    with tempfile.TemporaryDirectory() as tmp:
        log, a = _analysis(tmp)
        llm = FakeLLM(_llm_report())
        er.generate_report(llm, log, a)
        p = llm.prompt
        assert "FLUENCY AND COHERENCE" in p and "PRONUNCIATION" in p
        assert "words_per_minute" in p and "Question 0?" in p
        assert "You CANNOT hear the audio" in p


def test_little_speech_is_flagged_unreliable():
    with tempfile.TemporaryDirectory() as tmp:
        log, a = _analysis(tmp)
        r = er.generate_report(FakeLLM(_llm_report()), log, a)
        assert r["reliability"] == "low"
        assert any("very little speech" in c for c in r["caveats"])


class Events:
    def __init__(self):
        self.exam = []

    def on_exam(self, p):
        self.exam.append(p)


def test_too_few_answers_means_no_model_call():
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp, answered=2)
        llm, ev = FakeLLM(_llm_report()), Events()
        r = er.run_exam_analysis(log, llm, ev, transcribe=_fake_transcribe, prosody=_fake_prosody)
        assert r["insufficient"] is True and llm.calls == 0
        assert (log.dir / "report.json").exists()
        assert ev.exam[-1]["event"] == "report"


def test_full_run_saves_the_report_and_emits_progress():
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        ev = Events()
        r = er.run_exam_analysis(log, FakeLLM(_llm_report()), ev, transcribe=_fake_transcribe, prosody=_fake_prosody)
        assert [e["event"] for e in ev.exam] == ["analysing", "report"]
        saved = json.loads((log.dir / "report.json").read_text())
        assert saved["overall_band"] == r["overall_band"]


def test_scores_are_handed_to_the_profile_store_and_the_report_says_so():
    saved = []
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        ev = Events()
        r = er.run_exam_analysis(log, FakeLLM(_llm_report()), ev, store=lambda rep, lg: saved.append((rep, lg)) or True,
                                 transcribe=_fake_transcribe, prosody=_fake_prosody)
        assert len(saved) == 1 and saved[0][1] is log and saved[0][0]["overall_band"] == r["overall_band"]
        assert r["profile_saved"] is True
        assert json.loads((log.dir / "report.json").read_text())["profile_saved"] is True


def test_a_failing_store_never_loses_the_report():
    def boom(_r, _l):
        raise RuntimeError("disk full")

    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        r = er.run_exam_analysis(log, FakeLLM(_llm_report()), Events(), store=boom,
                                 transcribe=_fake_transcribe, prosody=_fake_prosody)
        assert r["profile_saved"] is False and "overall_band" in r
        assert (log.dir / "report.json").exists()


def test_nothing_is_stored_when_there_was_no_report():
    saved = []
    with tempfile.TemporaryDirectory() as tmp:
        er.run_exam_analysis(_make_log(tmp, answered=2), FakeLLM(_llm_report()), Events(),
                             store=lambda *a: saved.append(a), transcribe=_fake_transcribe, prosody=_fake_prosody)
        er.run_exam_analysis(_make_log(tmp), FakeLLM(fail=True), Events(),
                             store=lambda *a: saved.append(a), transcribe=_fake_transcribe, prosody=_fake_prosody)
    assert saved == []


def test_a_model_failure_is_reported_not_raised():
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        ev = Events()
        r = er.run_exam_analysis(log, FakeLLM(fail=True), ev, transcribe=_fake_transcribe, prosody=_fake_prosody)
        assert "error" in r and ev.exam[-1]["event"] == "report_failed"
        assert not (log.dir / "report.json").exists(), "no half-made report is saved"


def test_saved_exams_can_be_listed_and_loaded():
    with tempfile.TemporaryDirectory() as tmp:
        log = _make_log(tmp)
        er.run_exam_analysis(log, FakeLLM(_llm_report()), None, transcribe=_fake_transcribe, prosody=_fake_prosody)
        listed = er.list_exams("s1", root=Path(tmp))
        assert listed[0]["exam_id"] == "e1" and listed[0]["has_report"] is True
        full = er.load_exam("s1", "e1", root=Path(tmp))
        assert full["report"]["overall_band"] > 0 and full["analysis"] is not None
        assert er.load_exam("s1", "nope", root=Path(tmp)) is None
        assert er.list_exams("nobody", root=Path(tmp)) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} exam report tests passed")
