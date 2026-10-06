"""Recording logic tests — fake mic stream, deterministic VAD, no models.

    python tests/test_asr.py

The real Silero model is stateful, and a 300 Hz test tone is borderline for
it, which made these tests flaky. Here a loudness check stands in for the VAD
(test_vad.py covers the model itself), and Whisper is stubbed out.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Stub the model-heavy modules BEFORE app.asr imports them.
_vad = types.ModuleType("app.vad")
_vad.SAMPLE_RATE = 16000
_vad.CHUNK_SIZE = 512
_vad.speech_probability = (
    lambda chunk: 0.9 if np.sqrt(np.mean(np.square(chunk))) > 0.02 else 0.05
)
sys.modules.setdefault("app.vad", _vad)

_whisper = types.ModuleType("mlx_whisper")  # only needed by the warm-load
_whisper.transcribe = lambda *a, **k: {"text": ""}
sys.modules.setdefault("mlx_whisper", _whisper)

from app.asr import (  # noqa: E402
    SAMPLE_RATE,
    FRAME_MS,
    CALIBRATION_S,
    MIN_SPEECH_S,
    record_until_silence,
)

FRAME = int(SAMPLE_RATE * FRAME_MS / 1000)
CAL_FRAMES = int(CALIBRATION_S * 1000 / FRAME_MS)
MIN_SPEECH = int(MIN_SPEECH_S * 1000 / FRAME_MS)
LEAD_IN_FRAMES = 1 + CAL_FRAMES  # one dropped frame + the kept lead-in

TONE = (np.sin(2 * np.pi * 300 * np.arange(FRAME) / SAMPLE_RATE) * 3000).astype("int16")
QUIET = np.zeros(FRAME, dtype="int16")


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
    return [TONE if s else QUIET for s in speech_seq] + [QUIET] * silence_seq


def test_speech_is_captured_with_tail():
    # speech covers the lead-in and a few frames beyond it, then silence
    out = record_until_silence(
        lambda: FakeStream(_frames([1] * (LEAD_IN_FRAMES + MIN_SPEECH + 1), 15))
    )
    assert out is not None, "speech must be captured"
    dur = len(out) / SAMPLE_RATE
    assert 1.5 < dur < 2.6, f"unexpected duration {dur:.1f}s"


def test_speech_during_lead_in_is_not_lost():
    # speaking starts at frame 0, inside the lead-in window
    out = record_until_silence(
        lambda: FakeStream(_frames([1] * (LEAD_IN_FRAMES + MIN_SPEECH), 15))
    )
    assert out is not None
    assert len(out) / SAMPLE_RATE > 1.6, "first words must be kept"


def test_noise_only_returns_none():
    out = record_until_silence(lambda: FakeStream(_frames([], 30)))
    assert out is None


def test_cough_blip_is_ignored():
    # only 2 loud frames (~0.2s) — below the min-speech guard
    seq = [0] * 8 + [1, 1] + [0] * 20
    out = record_until_silence(lambda: FakeStream(_frames(seq, 0)))
    assert out is None, "a 2-frame blip must not be sent to Whisper"


def test_output_is_flat_1d():
    # sounddevice frames are (n, channels); whisper needs flat audio
    frames = [TONE.reshape(-1, 1)] * (LEAD_IN_FRAMES + MIN_SPEECH)
    frames += [np.zeros((FRAME, 1), dtype="int16")] * 15
    out = record_until_silence(lambda: FakeStream(frames))
    assert out is not None and out.ndim == 1, "must return flat 1D audio"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} asr tests passed")
