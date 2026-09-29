"""Recording logic tests — run with a FAKE mic stream, no device needed.

    python tests/test_asr.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.asr import SAMPLE_RATE, record_until_silence


class FakeStream:
    """Mimics sounddevice.InputStream: .read(n) -> (frames, overflowed)."""

    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return f, False


def _frames(speech_seq, silence_seq):
    frame = int(SAMPLE_RATE * 0.1)
    tone = (np.sin(2 * np.pi * 300 * np.arange(frame) / SAMPLE_RATE) * 3000).astype("int16")
    quiet = np.zeros(frame, dtype="int16")
    return [tone if s else quiet for s in speech_seq] + [quiet] * silence_seq


def test_speech_is_captured_with_tail():
    # 0.6s of speech (calibration included) then silence
    out = record_until_silence(lambda: FakeStream(_frames([1] * 6, 15)))
    assert out is not None, "speech must be captured"
    dur = len(out) / SAMPLE_RATE
    assert 1.4 < dur < 2.4, f"unexpected duration {dur:.1f}s"


def test_speech_during_calibration_is_not_lost():
    # speaking starts at frame 0 (inside the calibration window)
    out = record_until_silence(lambda: FakeStream(_frames([1] * 8, 15)))
    assert out is not None
    assert len(out) / SAMPLE_RATE > 1.6, "first words must be kept"


def test_noise_only_returns_none():
    out = record_until_silence(lambda: FakeStream(_frames([], 30)))
    assert out is None


def test_cough_blip_is_ignored():
    # only 2 loud frames (~0.2s) — below the min-speech guard
    seq = [0] * 8 + [1, 1] + [0] * 20
    frame = int(SAMPLE_RATE * 0.1)
    tone = (np.sin(2 * np.pi * 300 * np.arange(frame) / SAMPLE_RATE) * 3000).astype("int16")
    quiet = np.zeros(frame, dtype="int16")
    frames = [tone if s else quiet for s in seq]
    out = record_until_silence(lambda: FakeStream(frames))
    assert out is None, "a 2-frame blip must not be sent to Whisper"


def test_output_is_flat_1d():
    # sounddevice frames are (n, channels); whisper needs flat audio
    frame = int(SAMPLE_RATE * 0.1)
    tone = (np.sin(2 * np.pi * 300 * np.arange(frame) / SAMPLE_RATE) * 3000).astype("int16")
    out = record_until_silence(
        lambda: FakeStream([tone.reshape(-1, 1)] * 8 + [np.zeros((frame, 1), dtype="int16")] * 15)
    )
    assert out is not None and out.ndim == 1, "must return flat 1D audio"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} asr tests passed")
