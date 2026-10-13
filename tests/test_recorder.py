"""Streaming recorder endpointing tests — no mic, no models needed.

    python tests/test_recorder.py

VAD is stubbed deterministically so the two-tier silence logic can be tested:
  * 1.5s pause -> BACKFILL (keep the clip open; UI shows a silent cue)
  * 2.5s pause -> finalize the clip

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


def _plain():
    """Switch off the loudness gate + flat-noise rule: the older tests drive the
    recorder with constant fake audio and only care about the VAD/silence logic."""
    rec.NOISE_GATE_RATIO = 0.0
    rec.FLAT_CV = 0.0
    # the original 1.5 s / 2.5 s pauses these tests' patterns were written for
    rec.BACKFILL_SILENCE = 15
    rec.FINAL_SILENCE = 25


def _frame_of_call(n):
    """Frame in which the n-th VAD chunk call happens (chunks are 512 samples,
    frames 1600, and a chunk is processed once its last sample has arrived)."""
    return ((n + 1) * rec.CHUNK_SIZE - 1) // rec.FRAME_LEN


def _make_recorder(pattern):
    """A recorder whose VAD follows `pattern` (S=speech, s=silence)."""
    state = {"calls": 0}

    def fake_prob(_chunk):
        i = _frame_of_call(state["calls"])
        state["calls"] += 1
        ch = pattern[min(i, len(pattern) - 1)]
        return 0.9 if ch == "S" else 0.05

    rec.speech_probability = fake_prob
    rec.reset_states = lambda: None  # no real model in these tests
    _plain()
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
    assert isinstance(clip, np.ndarray), "2.5s of silence must finalize the clip"
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
        i = _frame_of_call(state["calls"])
        state["calls"] += 1
        ch = pattern[min(i, len(pattern) - 1)]
        return 0.9 if ch == "S" else 0.05

    rec.speech_probability = fake_prob
    rec.reset_states = lambda: None
    _plain()

    q = queue.Queue()
    source = WebSocketSource(q)

    def feed(n):
        for _ in range(n):
            q.put((np.ones(rec.FRAME_LEN, dtype=np.int16) * 1000).tobytes())

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


def _prob_recorder(probs):
    """Recorder whose VAD returns `probs[i]` for the i-th chunk (last repeats)."""
    state = {"calls": 0}

    def fake_prob(_chunk):
        p = probs[min(state["calls"], len(probs) - 1)]
        state["calls"] += 1
        return p

    rec.speech_probability = fake_prob
    rec.reset_states = lambda: None
    _plain()
    return rec.StreamingRecorder()


def test_flapping_around_threshold_does_not_end_speech():
    # Real speech: start confidently, then wobble between 0.4 and 0.6. The old
    # single 0.5 threshold saw this as speech/silence flapping; hysteresis holds.
    probs = [0.9] * 6 + [0.6, 0.4] * 40
    r = _prob_recorder(probs)
    for _ in range(20):
        assert r.push(_frame()) is None
    assert r.silent == 0, "wobble above the end threshold must not count as silence"


def test_brief_noise_burst_never_starts_speech():
    # Fewer than START_CHUNKS confident chunks in a row is not an utterance.
    probs = ([0.9] * (rec.START_CHUNKS - 1) + [0.1] * 4) * 10
    r = _prob_recorder(probs)
    for _ in range(15):
        r.push(_frame())
    assert r.speech_frames == 0


def test_chunks_are_fed_contiguously():
    # every 512-sample chunk of the stream reaches the VAD (none skipped)
    seen = []
    rec.speech_probability = lambda c: seen.append(len(c)) or 0.1
    rec.reset_states = lambda: None
    _plain()
    r = rec.StreamingRecorder()
    for _ in range(10):
        r.push(_frame())
    assert len(seen) == (10 * rec.FRAME_LEN) // rec.CHUNK_SIZE
    assert set(seen) == {rec.CHUNK_SIZE}


# --- loudness gate / noise floor / stuck-in-noise ------------------------------
# These use real-looking audio (random noise at a chosen RMS) and a VAD that
# ALWAYS says "speech", i.e. the worst case: the VAD is fooled by the noise.

_rng = np.random.default_rng(0)


def _noise(rms):
    """One frame of random noise with roughly the given RMS (0..1 scale)."""
    x = _rng.normal(0, rms * 32768, rec.FRAME_LEN)
    return np.clip(x, -32768, 32767).astype(np.int16)


def _voice(base=0.12):
    """A frame with speech-like, varying loudness."""
    return _noise(base * _rng.uniform(0.3, 1.7))


def _gated_recorder():
    rec.speech_probability = lambda _c: 0.9   # VAD fooled: everything is "speech"
    rec.reset_states = lambda: None
    rec.NOISE_GATE_RATIO = 2.5
    rec.FLAT_CV = 0.15
    return rec.StreamingRecorder()


def _push(r, frames):
    result = None
    for f in frames:
        out = r.push(f)
        if out is not None:
            result = out
    return result


def test_floor_is_calibrated_from_a_quiet_room():
    r = _gated_recorder()
    r.vad_flags  # (exists)
    rec.speech_probability = lambda _c: 0.05   # nobody talking
    _push(r, [_noise(0.004) for _ in range(rec.CAL_FRAMES + 2)])
    assert r.noise_floor is not None and r.noise_floor < 0.01
    assert r._gate() >= rec.ABS_MIN_LEVEL


def test_quiet_noise_the_vad_mistakes_for_speech_is_ignored():
    r = _gated_recorder()
    rec.speech_probability = lambda _c: 0.05
    _push(r, [_noise(0.004) for _ in range(rec.CAL_FRAMES)])   # calibrate
    rec.speech_probability = lambda _c: 0.9                    # VAD now fooled
    _push(r, [_noise(0.006) for _ in range(20)])               # quiet hiss
    assert r.speech_frames == 0, "hiss barely above the floor must not count"


def test_real_speech_above_the_floor_still_counts():
    r = _gated_recorder()
    rec.speech_probability = lambda _c: 0.05
    _push(r, [_noise(0.004) for _ in range(rec.CAL_FRAMES)])
    rec.speech_probability = lambda _c: 0.9
    _push(r, [_voice() for _ in range(8)])
    assert r.speech_frames >= 6


def test_steady_loud_noise_is_learned_and_does_not_hold_the_turn_open():
    r = _gated_recorder()
    rec.speech_probability = lambda _c: 0.05
    _push(r, [_noise(0.004) for _ in range(rec.CAL_FRAMES)])
    rec.speech_probability = lambda _c: 0.9
    # a loud, perfectly steady fan: far above the floor, VAD says speech
    _push(r, [_noise(0.08) for _ in range(rec.FLAT_WINDOW + 5)])
    assert r.speech_frames == 0, "steady noise must be taken back as non-speech"
    assert r.noise_floor > 0.05, "its level becomes the new floor"
    # ...and a person talking over it (clearly louder, varying) is still heard
    _push(r, [_voice(0.4) for _ in range(6)])
    assert r.speech_frames >= 4


def test_varying_loudness_is_not_mistaken_for_noise():
    r = _gated_recorder()
    rec.speech_probability = lambda _c: 0.05
    _push(r, [_noise(0.004) for _ in range(rec.CAL_FRAMES)])
    rec.speech_probability = lambda _c: 0.9
    _push(r, [_voice() for _ in range(rec.FLAT_WINDOW + 10)])  # long, natural speech
    assert r.speech_frames >= rec.FLAT_WINDOW, "long varied speech must be kept"


def test_noise_then_silence_finalizes_a_real_utterance_promptly():
    # speech, then steady noise the VAD thinks is speech: the turn must END
    # (silence counted) instead of running to the 15 s cap.
    r = _gated_recorder()
    rec.speech_probability = lambda _c: 0.05
    _push(r, [_noise(0.004) for _ in range(rec.CAL_FRAMES)])
    rec.speech_probability = lambda _c: 0.9
    frames = [_voice() for _ in range(10)] + [_noise(0.08) for _ in range(rec.FLAT_WINDOW + 5)]
    result = _push(r, frames)
    assert isinstance(result, np.ndarray), "turn must finalize despite the steady noise"
    assert len(result) / rec.SAMPLE_RATE < 8, "and long before the safety cap"


def test_calibration_during_speech_cannot_lock_the_user_out():
    # if the user is already talking while the room is measured, the floor is
    # capped so their speech is still above the gate afterwards
    r = _gated_recorder()
    rec.speech_probability = lambda _c: 0.05  # VAD sees no speech -> frames feed calibration
    _push(r, [_voice(0.3) for _ in range(rec.CAL_FRAMES)])
    assert r.noise_floor <= rec.MAX_CAL_FLOOR
    rec.speech_probability = lambda _c: 0.9
    _push(r, [_voice(0.3) for _ in range(6)])
    assert r.speech_frames >= 4


def test_floor_survives_reset_between_utterances():
    r = _gated_recorder()
    rec.speech_probability = lambda _c: 0.05
    _push(r, [_noise(0.004) for _ in range(rec.CAL_FRAMES)])
    floor = r.noise_floor
    r.reset()
    assert r.noise_floor == floor


# --- configurable pause ---------------------------------------------------------

def _silent_after_speech(r, silent_frames):
    """Feed a loud 'speech' burst, then quiet frames; return the last result."""
    rec.speech_probability = lambda c: 0.9 if float(np.abs(c).mean()) > 0.02 else 0.05
    rec.reset_states = lambda: None
    loud = (np.ones(rec.FRAME_LEN, dtype=np.int16) * 4000)
    quiet = np.zeros(rec.FRAME_LEN, dtype=np.int16)
    result = None
    for f in [loud] * 6 + [quiet] * silent_frames:
        out = r.push(f)
        if out is not None:
            result = out
    return result


def test_default_pause_is_shorter_than_the_old_2_5_seconds():
    assert rec.backfill_for(rec.config.FINAL_SILENCE_S) < rec.config.FINAL_SILENCE_S
    assert rec.config.FINAL_SILENCE_S <= 1.5, "the default wait should be well under 2.5 s"


def test_set_pause_changes_when_the_turn_ends():
    _plain()
    rec.NOISE_GATE_RATIO = 0.0
    short, long_ = rec.StreamingRecorder(), rec.StreamingRecorder()
    short.set_pause(1.0)
    long_.set_pause(3.0)
    # 12 quiet frames (1.2 s): past the short pause, not the long one
    assert isinstance(_silent_after_speech(short, 12), np.ndarray)
    out = _silent_after_speech(long_, 12)
    assert not isinstance(out, np.ndarray), "a 3 s pause must not end the turn at 1.2 s"


def test_set_pause_is_clamped_to_a_sane_range():
    r = rec.StreamingRecorder()
    r.set_pause(0.0)
    assert r.final_silence * rec.FRAME_MS / 1000 >= rec.MIN_PAUSE_S
    r.set_pause(999)
    assert r.final_silence * rec.FRAME_MS / 1000 <= rec.MAX_PAUSE_S


def test_pause_cue_always_comes_before_the_end():
    for s in (0.8, 1.0, 1.2, 2.0, 2.5, 4.0):
        r = rec.StreamingRecorder()
        r.set_pause(s)
        assert 1 <= r.backfill_silence < r.final_silence, s


def test_pause_setting_survives_between_utterances():
    r = rec.StreamingRecorder()
    r.set_pause(2.0)
    before = (r.backfill_silence, r.final_silence)
    r.reset()
    assert (r.backfill_silence, r.final_silence) == before


def test_max_record_is_configurable_per_recorder():
    r = _prob_recorder([0.9])          # VAD says speech forever
    _plain()
    r = rec.StreamingRecorder()
    r.set_max_record(30)
    assert r.max_frames == 300
    out = None
    for _ in range(299):
        out = r.push(_frame())
    assert out is None, "must keep recording past the old 15 s cap"
    out = r.push(_frame())
    assert isinstance(out, np.ndarray), "and stop at the configured cap"


def test_idle_timeout_gives_up_when_nobody_speaks():
    rec.speech_probability = lambda _c: 0.05
    rec.reset_states = lambda: None
    _plain()
    r = rec.StreamingRecorder()
    r.set_idle_timeout(2.0)
    results = [r.push(_frame()) for _ in range(25)]
    assert r.done is True, "no speech for 2 s must end the (empty) recording"
    assert not any(isinstance(x, np.ndarray) for x in results)


def test_idle_timeout_does_not_cut_off_someone_who_is_speaking():
    rec.speech_probability = lambda _c: 0.9
    rec.reset_states = lambda: None
    _plain()
    r = rec.StreamingRecorder()
    r.set_idle_timeout(2.0)
    for _ in range(40):
        r.push(_frame())
    assert r.done is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} recorder tests passed")
