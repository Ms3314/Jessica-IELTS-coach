"""Streaming recorder endpointing tests — no mic, no models needed.

    python tests/test_recorder.py

VAD is stubbed deterministically so the two-tier silence logic can be tested:
  * 1.5s pause -> BACKFILL (keep the clip open; UI shows a silent cue)
  * 3.0s pause -> finalize the clip

The VAD pattern is a string of 'S' (speech) and 's' (silence), one char per
frame.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import queue

import numpy as np

import app.recorder as rec
from app.io import BACKFILL


def _make_recorder(pattern):
    """A recorder whose VAD follows `pattern` (S=speech, s=silence)."""
    state = {"calls": 0}

    def fake_prob(_chunk):
        i = state["calls"]
        state["calls"] += 1
        ch = pattern[min(i, len(pattern) - 1)]
        return 0.9 if ch == "S" else 0.05

    rec.speech_probability = fake_prob
    return rec.StreamingRecorder()


def _frame():
    return np.ones(rec.FRAME_LEN, dtype=np.int16) * 1000


def _run(recorder, frames):
    """Feed `frames` frames, returning the last non-None result."""
    result = None
    for _ in range(frames):
        r = recorder.push(_frame())
        if r is not None:
            result = r
    return result


def test_backfill_after_first_pause():
    r = _make_recorder("S" * 5 + "s" * 15)
    assert _run(r, 5 + rec.BACKFILL_SILENCE) is BACKFILL
    assert r.done is False, "backfill must NOT finalize the clip"


def test_finalize_after_second_pause():
    r = _make_recorder("S" * 5 + "s" * 30)
    assert _run(r, 5 + rec.BACKFILL_SILENCE) is BACKFILL
    clip = _run(r, rec.FINAL_SILENCE - rec.BACKFILL_SILENCE)
    assert isinstance(clip, np.ndarray), "3.0s of silence must finalize the clip"
    assert clip.ndim == 1, "clip must be flat 1D audio"
    assert len(clip) > 0


def test_backfill_fires_once_per_pause():
    r = _make_recorder("S" * 3 + "s" * (rec.BACKFILL_SILENCE + 5))
    assert _run(r, 3 + rec.BACKFILL_SILENCE) is BACKFILL
    # more silence in the SAME pause must not acknowledge again
    assert _run(r, 5) is None
    assert r.backfill_fired is True


def test_resume_after_backfill_then_finalize():
    # The happy path: speech -> pause -> resume -> stop -> full clip.
    pattern = "S" * 5 + "s" * 15 + "S" * 5 + "s" * 30
    r = _make_recorder(pattern)
    assert _run(r, 5 + rec.BACKFILL_SILENCE) is BACKFILL
    _run(r, 5)  # user resumes speaking before the final threshold
    clip = _run(r, rec.FINAL_SILENCE)
    assert isinstance(clip, np.ndarray), "resumed speech must still finalize"
    # the clip must include the ORIGINAL speech, not just the resumption
    assert len(clip) / rec.SAMPLE_RATE > 1.5, "earlier words must not be lost"


def test_new_pause_can_backfill_again():
    # speech, pause->backfill, resume, pause->backfill again
    pattern = "S" * 3 + "s" * 16 + "S" * 5 + "s" * 15
    r = _make_recorder(pattern)
    assert _run(r, 3 + rec.BACKFILL_SILENCE) is BACKFILL
    _run(r, 1)                     # one more silent frame in the same pause
    _run(r, 5)                     # user resumes speaking
    assert r.backfill_fired is False
    assert r.silent == 0
    assert _run(r, rec.BACKFILL_SILENCE) is BACKFILL


def test_cough_without_pause_is_not_backfilled():
    # only 2 speech frames: below MIN_SPEECH, so no pause cue for a blip
    r = _make_recorder("S" * 2 + "s" * rec.BACKFILL_SILENCE)
    assert _run(r, 2 + rec.BACKFILL_SILENCE) is None
    assert r.backfill_fired is False


def test_rejected_clip_does_not_stall_source():
    # Regression: a rejected blip used to leave the recorder done=True, which
    # made WebSocketSource.listen() swallow every later frame until QUIT.
    from app.io import WebSocketSource

    # one char per VAD call: blip (rejected), then a real utterance
    pattern = (
        "S" * 2
        + "s" * rec.FINAL_SILENCE
        + "S" * 5
        + "s" * (2 * rec.FINAL_SILENCE)
    )
    state = {"calls": 0}

    def fake_prob(_chunk):
        i = state["calls"]
        state["calls"] += 1
        ch = pattern[min(i, len(pattern) - 1)]
        return 0.9 if ch == "S" else 0.05

    rec.speech_probability = fake_prob

    q = queue.Queue()
    source = WebSocketSource(q)

    def feed(n):
        for _ in range(n):
            q.put(np.ones(rec.FRAME_LEN, dtype=np.int16).tobytes())

    # blip: 2 speech frames is below MIN_SPEECH -> finalize rejects it
    feed(2 + rec.FINAL_SILENCE)
    assert source.listen() is None, "blip must be rejected, not returned as a clip"
    assert source.recorder.done is False, "recorder must be reset after rejection"

    # a real utterance afterwards must still be captured and finalized
    feed(5 + 2 * rec.FINAL_SILENCE)
    result = source.listen()
    if result is BACKFILL:
        result = source.listen()
    assert isinstance(result, np.ndarray), "later speech must still be recorded"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} recorder tests passed")
