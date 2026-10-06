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

# Stub the model-heavy modules BEFORE app.engine imports them, so this test
# stays fast (Whisper/Kokoro/OpenAI are never loaded).
for _name in ("app.asr", "app.tts", "app.agent", "app.sessions"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

# The engine calls sessions.bump() and maybe_auto_rename() after each user turn;
# make them no-ops here.
sys.modules["app.sessions"].bump = lambda _sid: None
sys.modules["app.sessions"].maybe_auto_rename = lambda _sid, _text: None

import app.engine as engine  # noqa: E402
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

    def on_agent(self, _sentence):
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
