"""Streaming speech recorder for the web (WebSocket) audio path.

The terminal path records straight from the mic via sounddevice
(`asr.record_until_silence`). The web path instead receives raw 16 kHz PCM
in small chunks over a WebSocket; this class ingests those chunks, runs the
same Silero VAD + silence-endpointing logic, and emits a finished clip the
moment the user stops talking — so the server can transcribe it with Whisper.

It reuses `app.vad.speech_probability`, exactly like the terminal recorder, so
behaviour stays consistent across both frontends.
"""
import logging
import numpy as np

from app.io import BACKFILL
from app.vad import speech_probability, CHUNK_SIZE, SAMPLE_RATE as VAD_SR

logger = logging.getLogger("speakIelts.recorder")

# We expect 16 kHz mono int16 from the browser; VAD wants 16 kHz too.
SAMPLE_RATE = 16000
FRAME_MS = 100            # process audio in 100 ms frames
BACKFILL_SILENCE_S = 1.5  # first pause: notify the UI, keep listening
FINAL_SILENCE_S = 3.0     # second pause: user is done, finalize the clip
MIN_SPEECH_S = 0.3        # require this much real speech (a cough isn't speech)
MAX_RECORD_S = 15.0       # safety cap so a noisy room can't record forever
PRE_ROLL_S = 0.3          # keep this much audio before speech started

FRAME_LEN = int(SAMPLE_RATE * FRAME_MS / 1000)                  # 1600 samples
BACKFILL_SILENCE = int(BACKFILL_SILENCE_S * 1000 / FRAME_MS)    # 15 frames
FINAL_SILENCE = int(FINAL_SILENCE_S * 1000 / FRAME_MS)          # 30 frames
MIN_SPEECH = int(MIN_SPEECH_S * 1000 / FRAME_MS)                # 3 frames
MAX_FRAMES = int(MAX_RECORD_S * 1000 / FRAME_MS)                # 150 frames
PRE_ROLL = int(PRE_ROLL_S * 1000 / FRAME_MS)                    # 3 frames


class StreamingRecorder:
    """Feed 16 kHz mono int16 PCM via `push`; get a clip back when speech ends.

    `push(chunk)` returns:
      * None            — not enough audio yet (keep feeding)
      * io.BACKFILL     — user paused; keep this clip open (silent UI cue only)
      * np.ndarray      — a finished, trimmed clip (int16, 1D) to transcribe
    Call `reset()` before each new utterance (the engine does this in listen()).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._pending = np.zeros(0, dtype=np.int16)
        self.frames = []        # list of int16 frames (each FRAME_LEN long)
        self.vad_flags = []     # VAD speech decision per frame
        self.speech_frames = 0  # running count of speech frames (no re-summing)
        self.silent = 0         # consecutive silent frames after speech began
        self.backfill_fired = False  # already acknowledged this pause
        self.done = False       # a clip has been emitted; stop accepting

    def push(self, pcm: np.ndarray):
        pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
        if self._pending.size:
            pcm = np.concatenate([self._pending, pcm])
        self._pending = pcm

        while len(self._pending) >= FRAME_LEN and not self.done:
            frame = self._pending[:FRAME_LEN].copy()
            self._pending = self._pending[FRAME_LEN:]
            result = self._feed_frame(frame)
            if result is not None:
                return result
        return None

    def _feed_frame(self, frame):
        self.frames.append(frame)

        # VAD on the first 512-sample chunk (same as the terminal recorder)
        prob = speech_probability(frame[:CHUNK_SIZE].astype(np.float32) / 32768.0)
        speaking = prob > 0.5
        self.vad_flags.append(speaking)

        if speaking:
            self.speech_frames += 1
            self.silent = 0
            self.backfill_fired = False  # a new pause may be acknowledged again
        else:
            self.silent += 1

            # First pause threshold: notify the UI and keep listening. Do NOT
            # finalize — the user may just be thinking mid-answer. Require a
            # minimum amount of real speech so a cough/blip isn't announced.
            if (
                self.silent >= BACKFILL_SILENCE
                and self.speech_frames >= MIN_SPEECH
                and not self.backfill_fired
            ):
                self.backfill_fired = True
                logger.info(
                    "recorder: BACKFILL after %.1fs of silence",
                    self.silent * FRAME_MS / 1000,
                )
                return BACKFILL

            # Second pause threshold: the user is done. (_finalize() itself
            # rejects clips with too little speech, as before.)
            if self.silent >= FINAL_SILENCE and self.speech_frames > 0:
                return self._finalize()

        if len(self.frames) >= MAX_FRAMES:
            return self._finalize()
        return None

    def _finalize(self):
        self.done = True
        # not enough real speech: a blip / cough / noise isn't worth Whisper
        if self.speech_frames < MIN_SPEECH:
            logger.info(
                "recorder: clip REJECTED (%d/%d speech frames < min %d) — no utterance",
                self.speech_frames,
                len(self.vad_flags),
                MIN_SPEECH,
            )
            return None
        first = next(i for i, s in enumerate(self.vad_flags) if s)
        start = max(0, first - PRE_ROLL)
        # reshape(-1): frames may be (n, channels); Whisper wants flat 1D audio
        clip = np.concatenate(self.frames[start:]).reshape(-1)
        logger.info(
            "recorder: clip FINALIZED (%d samples, ~%.2fs, %d/%d speech frames)",
            clip.shape[0],
            clip.shape[0] / SAMPLE_RATE,
            self.speech_frames,
            len(self.vad_flags),
        )
        return clip
