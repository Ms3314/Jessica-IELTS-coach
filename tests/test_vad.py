"""VAD tests — no mic needed, just numpy arrays.

    python tests/test_vad.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.vad import is_speech, speech_probability, CHUNK_SIZE, SAMPLE_RATE


def test_silence_is_not_speech():
    silence = np.zeros(CHUNK_SIZE, dtype=np.float32)
    assert not is_speech(silence), "silence must not be speech"
    assert speech_probability(silence) < 0.5


def test_quiet_noise_is_not_speech():
    noise = (np.random.randn(CHUNK_SIZE) * 0.05).astype(np.float32)
    assert not is_speech(noise), "quiet noise must not be speech"


def test_loud_noise_is_not_speech():
    # random noise at high amplitude — VAD should reject it (not speech-like)
    noise = (np.random.randn(CHUNK_SIZE) * 0.5).astype(np.float32)
    assert speech_probability(noise) < 0.5, "random noise is not speech"


def test_accepts_numpy_int16():
    # sounddevice returns int16 — VAD must handle it without crashing
    chunk = (np.random.randn(CHUNK_SIZE) * 100).astype(np.int16)
    prob = speech_probability(chunk)
    assert 0.0 <= prob <= 1.0, "probability must be in [0, 1]"


def test_accepts_numpy_float64():
    # dtype conversion must work transparently
    chunk = np.random.randn(CHUNK_SIZE).astype(np.float64) * 0.1
    prob = speech_probability(chunk)
    assert 0.0 <= prob <= 1.0


def test_accepts_list_input():
    # plain Python list should also work
    chunk = [0.0] * CHUNK_SIZE
    assert not is_speech(chunk)


def test_probability_is_stable():
    # the model has internal state (RNN), so consecutive calls differ slightly,
    # but for the same quiet input, both should be low
    chunk = (np.random.randn(CHUNK_SIZE) * 0.05).astype(np.float32)
    p1 = speech_probability(chunk)
    p2 = speech_probability(chunk)
    assert p1 < 0.5 and p2 < 0.5, "quiet input should give low probability"


def test_chunk_size_is_512():
    # the model requires exactly 512 samples at 16kHz — guard against accidental changes
    assert CHUNK_SIZE == 512
    assert SAMPLE_RATE == 16000


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} vad tests passed")
