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


def test_exam_events_are_serialized_with_their_name():
    q = queue.Queue()
    WebSocketHandler(q).on_exam({"event": "prep", "seconds": 60})
    assert q.get() == {"type": "exam", "event": "prep", "seconds": 60}


def test_default_handler_ignores_exam_events():
    EventHandler().on_exam({"event": "start"})


def test_sink_tracks_how_much_audio_the_browser_still_has_to_play():
    import numpy as np
    from app.io import WebSocketSink

    sink = WebSocketSink(queue.Queue())
    assert sink.remaining() == 0.0
    sink.write(np.zeros(24000 * 3, dtype=np.float32))   # 3 s of audio, written instantly
    assert 2.5 < sink.remaining() <= 3.0
    sink.write(np.zeros(24000 * 2, dtype=np.float32))   # chains onto the backlog
    assert 4.5 < sink.remaining() <= 5.0


def test_source_flush_drops_stale_audio_but_keeps_a_quit_request():
    import numpy as np
    from app.io import QUIT, WebSocketSource

    q = queue.Queue()
    src = WebSocketSource(q)
    q.put(np.zeros(100, dtype=np.int16).tobytes())
    q.put(np.zeros(100, dtype=np.int16).tobytes())
    src.flush()
    assert q.empty()
    q.put(b"\x00\x00")
    q.put(QUIT)
    src.flush()
    assert q.get_nowait() is QUIT


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} io tests passed")
