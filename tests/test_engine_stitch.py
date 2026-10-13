"""Barge-in stitching tests — no models, no devices needed.

    python tests/test_engine_stitch.py

When the user keeps talking through a reply (barge-in cancels it), the next
transcript must be merged onto the previous one as a single message:

    "I like football" + "and I play every week"
    -> "I like football ... and I play every week"
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub the model-heavy modules BEFORE app.chat.engine imports them, so this test
# stays fast (Whisper/Kokoro/OpenAI are never loaded).
for _name in ("app.speech.asr", "app.speech.tts", "app.llm.agent", "app.chat.sessions"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

# The engine calls sessions.bump() and maybe_auto_rename() after each user turn;
# make them no-ops here.
sys.modules["app.chat.sessions"].bump = lambda _sid: None
sys.modules["app.chat.sessions"].maybe_auto_rename = lambda _sid, _text: None

import app.chat.engine as engine  # noqa: E402
from app.io import QUIT  # noqa: E402


class FakeSource:
    def __init__(self, clips):
        self._clips = list(clips)

    def listen(self):
        return self._clips.pop(0) if self._clips else QUIT


class FakeEvents:
    def __init__(self):
        self.user_texts = []

    def on_state(self, _state):
        pass

    def on_user(self, text):
        self.user_texts.append(text)

    def on_agent(self, _sentence, _idx=None):
        pass

    def on_pause(self):
        pass

    def on_error(self, message):
        raise AssertionError(f"turn failed: {message}")


def _run(clips, texts, cancelled):
    """Drive a Session with scripted clips, transcripts and cancel flags."""
    engine.asr.transcribe = lambda _clip: texts.pop(0)
    engine.agent.agent_reply_stream = lambda text, _sid: iter([f"reply to: {text}"])
    replies = []

    def fake_speak_stream(text_stream, **_kwargs):
        replies.append(next(iter(text_stream)))
        return cancelled.pop(0) if cancelled else False

    engine.tts.speak_stream = fake_speak_stream

    events = FakeEvents()
    engine.Session(FakeSource(clips), sink=object(), events=events).loop()
    return events, replies


def test_finished_reply_is_not_stitched():
    events, replies = _run(
        ["clip1", "clip2"],
        ["hello there", "next question"],
        [False, False],
    )
    assert events.user_texts == ["hello there", "next question"], events.user_texts
    assert replies == ["reply to: hello there", "reply to: next question"], replies


def test_cancelled_reply_stitches_the_continuation():
    events, _ = _run(
        ["clip1", "clip2", "clip3"],
        ["I like football", "because it is fun", "and I play every week"],
        [False, True, False],
    )
    assert events.user_texts == [
        "I like football",
        "because it is fun",
        "because it is fun ... and I play every week",
    ], events.user_texts


def test_barge_in_before_the_reply_skips_synthesis():
    def fake_speak_stream(text_stream, **_kwargs):
        raise AssertionError("the LLM must not be called after a barge-in")

    engine.tts.speak_stream = fake_speak_stream
    engine.agent.agent_reply_stream = lambda text, _sid: iter([text])

    session = engine.Session(FakeSource(["clip1"]), sink=object(), events=FakeEvents())

    def transcribe(_clip):
        session.interrupt()  # user kept talking while Whisper was running
        return "I like football"

    engine.asr.transcribe = transcribe
    session.loop()  # must finish without calling the LLM at all


def test_stitch_window_expiry_starts_a_new_turn():
    saved = engine.STITCH_WINDOW_S
    engine.STITCH_WINDOW_S = -1.0  # force the window closed
    try:
        events, _ = _run(["clip1", "clip2"], ["first", "second"], [True, False])
    finally:
        engine.STITCH_WINDOW_S = saved
    assert events.user_texts == ["first", "second"], events.user_texts


def _interrupted_run(by, heard_idx, sentences, texts):
    """One turn where the reply is cut off; returns (user_texts, recorded calls)."""
    recorded = []
    engine.agent.mark_interrupted = lambda sid, heard, b: recorded.append((sid, heard, b))
    engine.asr.transcribe = lambda _clip: texts.pop(0)
    engine.agent.agent_reply_stream = lambda text, _sid: iter(sentences)
    holder = {}

    def fake_speak_stream(_stream, on_sentence=None, **_kwargs):
        if holder.get("cut"):
            return False  # only the first reply is cut off
        holder["cut"] = True
        for sentence in sentences:  # the engine numbers + reports each sentence
            on_sentence(sentence)
        holder["session"].interrupt(by, heard_idx)  # user stops her mid-reply
        return True

    engine.tts.speak_stream = fake_speak_stream
    events = FakeEvents()
    session = engine.Session(FakeSource(["c1", "c2"]), sink=object(), events=events)
    holder["session"] = session
    session.loop()
    return events.user_texts, recorded


def test_button_stop_does_not_stitch_and_records_the_heard_part():
    users, recorded = _interrupted_run(
        "button", 0, ["Canberra is it.", "It was a compromise."], ["capital?", "no, biggest city"]
    )
    assert users == ["capital?", "no, biggest city"], users  # NOT stitched
    assert recorded == [("default", "Canberra is it.", "button")], recorded


def test_voice_stop_still_stitches_and_records_the_heard_part():
    users, recorded = _interrupted_run(
        "voice", 1, ["One.", "Two.", "Three."], ["I like football", "and I play weekly"]
    )
    assert users == ["I like football", "I like football ... and I play weekly"], users
    assert recorded[0][1:] == ("One. Two.", "voice"), recorded


def test_nothing_heard_records_an_empty_heard_part():
    _, recorded = _interrupted_run("voice", -1, ["Never played."], ["hello", "x"])
    assert recorded[0][1:] == ("", "voice"), recorded


def test_unknown_heard_idx_assumes_everything_sent_was_heard():
    _, recorded = _interrupted_run("button", None, ["A.", "B."], ["hello", "x"])
    assert recorded[0][1:] == ("A. B.", "button"), recorded


def test_interrupt_before_the_llm_runs_does_not_touch_memory():
    recorded = []
    engine.agent.mark_interrupted = lambda *a: recorded.append(a)
    engine.tts.speak_stream = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM"))
    engine.agent.agent_reply_stream = lambda text, _sid: iter([text])
    session = engine.Session(FakeSource(["clip1"]), sink=object(), events=FakeEvents())

    def transcribe(_clip):
        session.interrupt("button")
        return "stop"

    engine.asr.transcribe = transcribe
    session.loop()
    assert recorded == [], "no reply existed, so there is nothing to rewrite"


def test_session_id_is_passed_to_the_agent():
    seen = []

    def fake_reply_stream(_text, thread_id):
        seen.append(thread_id)
        return iter(["ok"])

    engine.asr.transcribe = lambda _clip: "hi"
    engine.agent.agent_reply_stream = fake_reply_stream
    engine.tts.speak_stream = lambda _stream, **_kwargs: False

    session = engine.Session(
        FakeSource(["clip1"]), sink=object(), events=FakeEvents(), session_id="abc123"
    )
    session.loop()
    assert seen == ["abc123"], seen


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} engine stitch tests passed")
