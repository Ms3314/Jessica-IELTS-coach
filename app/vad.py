"""Silero VAD — voice activity detection for the voice pipeline.

Replaces the RMS-energy threshold in asr.py with a neural network that
actually understands what speech sounds like.  Benefits:

  * Far fewer false triggers from keyboard taps, coughs, background noise
  * More precise speech/silence boundaries (faster response)
  * Runs on CPU in ~1ms per 32ms chunk — negligible overhead

The model expects exactly 512-sample chunks at 16kHz (32ms each).
It returns a speech probability in [0, 1].
"""

import sys
import torch
import numpy as np

# --- constants ----------------------------------------------------------------
SAMPLE_RATE = 16000
CHUNK_SIZE = 512          # samples per chunk (32ms at 16kHz)

# Speech threshold can be overridden via environment variable for testing
import os
SPEECH_THRESHOLD = float(os.getenv("VAD_SPEECH_THRESHOLD", "0.5"))


# --- model loading ------------------------------------------------------------

def _load_model():
    """Load Silero VAD via torch hub. Cached after first download."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    # utils: [get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks]
    return model


print("Loading Silero VAD model...")
_model = _load_model()
# warm up: one inference so the first real call isn't slow
_model(torch.zeros(CHUNK_SIZE, dtype=torch.float32), SAMPLE_RATE)
print("Silero VAD ready.")


# --- public API ---------------------------------------------------------------

def is_speech(audio_chunk) -> bool:
    """Return True if this 512-sample chunk contains speech.

    audio_chunk: 1D array/tensor of exactly 512 float32 samples at 16kHz.
    Accepts numpy arrays, torch tensors, or lists.
    """
    if not isinstance(audio_chunk, torch.Tensor):
        audio_chunk = torch.tensor(audio_chunk, dtype=torch.float32)
    if audio_chunk.dtype != torch.float32:
        audio_chunk = audio_chunk.float()

    prob = _model(audio_chunk, SAMPLE_RATE).item()
    return prob >= SPEECH_THRESHOLD


def speech_probability(audio_chunk) -> float:
    """Return the raw speech probability for a 512-sample chunk.

    Useful for debugging / tuning the threshold.
    """
    if not isinstance(audio_chunk, torch.Tensor):
        audio_chunk = torch.tensor(audio_chunk, dtype=torch.float32)
    if audio_chunk.dtype != torch.float32:
        audio_chunk = audio_chunk.float()

    return _model(audio_chunk, SAMPLE_RATE).item()
