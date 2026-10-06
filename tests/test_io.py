"""I/O plumbing tests — handler serialization, no models needed.

    python tests/test_io.py
"""
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.io import EventHandler, WebSocketHandler


def test_pause_event_is_serialized():
    q = queue.Queue()
    WebSocketHandler(q).on_pause()
    assert q.get() == {"type": "pause"}


def test_pause_default_handler_is_a_noop():
    # handlers (e.g. the terminal) that don't care about pauses must not crash
    EventHandler().on_pause()


def test_state_and_transcript_events_are_serialized():
    q = queue.Queue()
    h = WebSocketHandler(q)
    h.on_state("listening")
    h.on_user("hello")
    h.on_agent("hi")
    h.on_error("oops")
    assert q.get() == {"type": "state", "state": "listening"}
    assert q.get() == {"type": "user", "text": "hello"}
    assert q.get() == {"type": "agent", "text": "hi"}
    assert q.get() == {"type": "error", "message": "oops"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} io tests passed")
