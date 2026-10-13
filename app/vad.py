"""Silero VAD — voice activity detection for the voice pipeline.

The model expects exactly 512-sample chunks at 16 kHz (32 ms each) and
returns a speech probability in [0, 1]. Callers pick their own threshold.
The model is stateful, so chunks must be fed contiguously and in order.
"""
import numpy as np
import torch

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # samples per chunk (32ms at 16kHz)


def _load_model():
    """Load Silero VAD via torch hub (cached after the first download)."""
    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    return model


print("Loading Silero VAD model...")
_model = _load_model()
_model(torch.zeros(CHUNK_SIZE, dtype=torch.float32), SAMPLE_RATE)  # warm up
print("Silero VAD ready.")


def reset_states() -> None:
    """Forget the model's memory of earlier audio (call between utterances).

    Silero is stateful: it uses the audio it just heard to judge the next
    chunk, so a new utterance must not inherit the previous one's context.
    """
    _model.reset_states()


def speech_probability(audio_chunk) -> float:
    """Speech probability in [0, 1] for one 512-sample chunk at 16 kHz.

    Accepts numpy arrays, torch tensors, or lists of any numeric dtype.
    """
    if not isinstance(audio_chunk, torch.Tensor):
        audio_chunk = torch.tensor(audio_chunk, dtype=torch.float32)
    if audio_chunk.dtype != torch.float32:
        audio_chunk = audio_chunk.float()

    return _model(audio_chunk, SAMPLE_RATE).item()
