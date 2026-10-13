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


def _pcm(seconds):
    import numpy as np

    return (np.ones(int(16000 * seconds), dtype="int16") * 300).tobytes()


def _manual_source(*items):
    from app.io import WebSocketSource

    q = queue.Queue()
    for it in items:
        q.put(it)
    return WebSocketSource(q), q


def test_manual_recording_returns_everything_between_the_first_audio_and_stop():
    from app.io import STOP

    src, _ = _manual_source(_pcm(1.0), _pcm(0.5), STOP)
    clip = src.listen_manual()
    assert abs(len(clip) / 16000 - 1.5) < 1e-6


def test_manual_recording_reports_when_the_speaker_actually_started():
    from app.io import STOP

    started = []
    src, _ = _manual_source(_pcm(1.0), STOP)
    src.listen_manual(on_start=lambda: started.append(True))
    assert started == [True]


def test_manual_recording_finishes_by_itself_at_the_length_limit():
    src, q = _manual_source(*[_pcm(1.0) for _ in range(5)])
    clip = src.listen_manual(max_s=2.0)
    assert len(clip) / 16000 >= 2.0 and q.qsize() >= 2, "stopped at the cap, left the rest queued"


def test_manual_recording_gives_up_if_nobody_starts():
    import time

    src, _ = _manual_source()
    t0 = time.monotonic()
    assert src.listen_manual(wait_s=0.5) is None
    assert time.monotonic() - t0 < 2


def test_a_stop_before_any_audio_is_ignored_and_a_tiny_clip_is_discarded():
    from app.io import STOP

    src, _ = _manual_source(STOP, _pcm(1.0), STOP)
    assert src.listen_manual() is not None, "a stray Stop before the audio doesn't end the wait"
    src2, _ = _manual_source(_pcm(0.1), STOP)
    assert src2.listen_manual(min_s=0.4) is None, "a 0.1 s blip isn't an answer"


def test_manual_recording_passes_a_quit_request_through():
    from app.io import QUIT

    src, _ = _manual_source(_pcm(0.5), QUIT)
    assert src.listen_manual() is QUIT


def test_a_recording_in_progress_is_cut_off_when_the_time_limit_hits():
    import time

    src, _ = _manual_source(_pcm(1.0))      # speech began, nobody ever presses Stop
    t0 = time.monotonic()
    clip = src.listen_manual(wait_s=45, max_s=90, limit_s=0.4)
    assert clip is not None and len(clip) / 16000 >= 1.0, "what was said is kept"
    assert time.monotonic() - t0 < 2, "and the clock ended it, not the 90 s cap"


def test_a_window_nobody_started_closes_at_the_time_limit():
    import time

    src, _ = _manual_source()
    t0 = time.monotonic()
    assert src.listen_manual(wait_s=45, limit_s=0.4) is None
    assert time.monotonic() - t0 < 2


def test_pressing_stop_before_the_limit_still_ends_the_answer_at_once():
    from app.io import STOP
    import time

    src, _ = _manual_source(_pcm(1.0), STOP)
    t0 = time.monotonic()
    assert src.listen_manual(limit_s=30) is not None
    assert time.monotonic() - t0 < 1


def test_automatic_listening_ignores_a_stray_stop_marker():
    import numpy as np
    from app.io import STOP, QUIT

    src, q = _manual_source(STOP, QUIT)
    assert src.listen() is QUIT      # the STOP neither crashes it nor counts as audio


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} io tests passed")
