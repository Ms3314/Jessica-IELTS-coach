"""Streaming speech recorder for the web (WebSocket) audio path.

The terminal path records straight from the mic via sounddevice
(`asr.record_until_silence`). The web path instead receives raw 16 kHz PCM
in small chunks over a WebSocket; this class ingests those chunks, runs the
same Silero VAD + silence-endpointing logic, and emits a finished clip the
moment the user stops talking — so the server can transcribe it with Whisper.

It reuses `app.speech.vad.speech_probability`, exactly like the terminal recorder, so
behaviour stays consistent across both frontends.
"""
import logging
import numpy as np

from app import config
from app.io import BACKFILL
from app.speech.vad import speech_probability, reset_states, CHUNK_SIZE

logger = logging.getLogger("speakIelts.recorder")

# We expect 16 kHz mono int16 from the browser; VAD wants 16 kHz too.
SAMPLE_RATE = 16000
FRAME_MS = 100            # process audio in 100 ms frames
FINAL_SILENCE_S = config.FINAL_SILENCE_S  # second pause: finalize the clip (stitching recovers)
MIN_PAUSE_S, MAX_PAUSE_S = 0.8, 4.0       # the range the user may choose


def backfill_for(final_s: float) -> float:
    """First pause (UI cue only, keep listening): shortly before the clip ends."""
    return max(0.6, min(1.5, final_s - 0.5))


BACKFILL_SILENCE_S = backfill_for(FINAL_SILENCE_S)
MIN_SPEECH_S = 0.3        # require this much real speech (a cough isn't speech)
MAX_RECORD_S = config.MAX_RECORD_S  # safety cap so a noisy room can't record forever
PRE_ROLL_S = 0.3          # keep this much audio before speech started

# Hysteresis: it takes a confident, sustained burst to START counting as speech,
# but only a clear drop to STOP. Probabilities in between keep the current state,
# so values wobbling around one threshold no longer flip speech/silence.
START_PROB = 0.5          # chunk prob needed to (begin to) count as speech
END_PROB = 0.35           # chunk prob below which speech is considered over
START_CHUNKS = 3          # consecutive 32 ms chunks above START_PROB to begin

# Loudness gate: the VAD alone also fires on TV/music/chatter, so a frame only
# counts as the user's speech if it is ALSO clearly louder than the room.
NOISE_GATE_RATIO = config.NOISE_GATE_RATIO  # speech must exceed floor * this
ABS_MIN_LEVEL = 0.006     # never count anything quieter than this (RMS, 0..1)
CAL_FRAMES = 5            # frames (0.5 s) used to measure the room at first
MAX_CAL_FLOOR = 0.02      # a talkative calibration can't push the floor above this
FLOOR_FALL = 0.5          # floor follows a quieter room down quickly...
FLOOR_RISE = 0.05         # ...and creeps up slowly (so speech can't drag it up)

# "Stuck in noise": speech is dynamic (syllables rise and fall); a fan, hum or
# steady music is flat. If a long run of "speech" frames barely varies in
# loudness, treat it as noise, learn its level as the new floor, and move on.
FLAT_WINDOW = 30          # frames (3 s) of continuous speech to examine
FLAT_CV = 0.15            # loudness std/mean below this = flat = noise

FRAME_LEN = int(SAMPLE_RATE * FRAME_MS / 1000)                  # 1600 samples
BACKFILL_SILENCE = int(BACKFILL_SILENCE_S * 1000 / FRAME_MS)    # frames
FINAL_SILENCE = int(FINAL_SILENCE_S * 1000 / FRAME_MS)          # frames
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
        # The noise floor describes the ROOM, not one utterance, so it survives
        # reset(). None until CAL_FRAMES frames have been heard.
        self.noise_floor = None
        self._cal_levels = []
        # Pause lengths (frames) are per recorder so the user can change them live.
        self.backfill_silence = BACKFILL_SILENCE
        self.final_silence = FINAL_SILENCE
        self.max_frames = MAX_FRAMES
        self.idle_frames = None  # frames of pure silence before giving up (None: use max)
        self.reset()

    def set_idle_timeout(self, seconds) -> None:
        """Give up on a turn where nothing at all was said for this long."""
        self.idle_frames = None if seconds is None else max(1, round(seconds * 1000 / FRAME_MS))

    def set_max_record(self, seconds: float) -> None:
        """Cap on one recorded turn (exam Part 2 needs far more than a chat turn)."""
        self.max_frames = max(1, round(seconds * 1000 / FRAME_MS))

    def set_pause(self, seconds: float, lo: float = None, hi: float = None) -> None:
        """How long a quiet spell must last before the clip is finalized.

        lo/hi override the user-facing range (the exam picks its own).
        """
        seconds = min(hi or MAX_PAUSE_S, max(lo or MIN_PAUSE_S, float(seconds)))
        self.final_silence = max(1, round(seconds * 1000 / FRAME_MS))
        self.backfill_silence = max(
            1, round(backfill_for(seconds) * 1000 / FRAME_MS)
        )

    def reset(self):
        self._pending = np.zeros(0, dtype=np.int16)
        self.frames = []        # list of int16 frames (each FRAME_LEN long)
        self.vad_flags = []     # speech decision per frame (VAD AND loud enough)
        self.levels = []        # loudness (RMS, 0..1) per frame
        self.speech_frames = 0  # running count of speech frames (no re-summing)
        self.silent = 0         # consecutive silent frames after speech began
        self.backfill_fired = False  # already acknowledged this pause
        self.done = False       # a clip has been emitted; stop accepting
        # VAD runs on EVERY contiguous 512-sample chunk (Silero is stateful),
        # so frame audio is queued here and consumed chunk by chunk.
        self._vad_pending = np.zeros(0, dtype=np.int16)
        self._in_speech = False   # hysteresis state
        self._start_run = 0       # consecutive chunks above START_PROB
        reset_states()            # new utterance: forget the previous audio

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

    def _gate(self) -> float:
        """Minimum loudness for a frame to count as speech right now."""
        floor = self.noise_floor
        if floor is None:
            return ABS_MIN_LEVEL
        return max(ABS_MIN_LEVEL, floor * NOISE_GATE_RATIO)

    def _track_floor(self, level):
        """Measure the room at the start, then follow it while nobody speaks."""
        if self.noise_floor is None:
            self._cal_levels.append(level)
            if len(self._cal_levels) >= CAL_FRAMES:
                self.noise_floor = min(min(self._cal_levels), MAX_CAL_FLOOR)
                logger.info("recorder: noise floor calibrated to %.4f", self.noise_floor)
            return
        # only called on frames that were NOT speech
        rate = FLOOR_FALL if level < self.noise_floor else FLOOR_RISE
        self.noise_floor += (level - self.noise_floor) * rate

    def _is_flat_noise(self) -> bool:
        """True if the last FLAT_WINDOW frames were all 'speech' but steady."""
        n = FLAT_WINDOW
        if len(self.vad_flags) < n or not all(self.vad_flags[-n:]):
            return False
        window = np.asarray(self.levels[-n:])
        mean = float(window.mean())
        return mean > 0 and float(window.std()) / mean < FLAT_CV

    def _feed_frame(self, frame):
        self.frames.append(frame)
        level = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
        self.levels.append(level)

        # VAD says "speech-like"; the gate says "and clearly louder than the room"
        speaking = self._vad_frame(frame) and level >= self._gate()
        self.vad_flags.append(speaking)

        if speaking and self._is_flat_noise():
            # A steady, non-varying "speech" run is noise the VAD mistook for a
            # voice. Learn its level as the new floor and take those frames back.
            n = FLAT_WINDOW
            mean = float(np.mean(self.levels[-n:]))
            self.noise_floor = max(self.noise_floor or 0.0, mean)
            self.vad_flags[-n:] = [False] * n
            self.speech_frames = max(0, self.speech_frames - (n - 1))
            self.silent = n - 1  # these frames are silence, not speech
            speaking = False
            logger.info("recorder: steady noise at %.4f ignored (floor raised)", mean)

        if not speaking:
            self._track_floor(level)

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
                self.silent >= self.backfill_silence
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
            if self.silent >= self.final_silence and self.speech_frames > 0:
                return self._finalize()

        if len(self.frames) >= self.max_frames:
            return self._finalize()
        if (
            self.idle_frames is not None
            and self.speech_frames == 0
            and len(self.frames) >= self.idle_frames
        ):
            return self._finalize()  # nobody spoke: rejected as "no utterance"
        return None

    def _vad_frame(self, frame) -> bool:
        """Run the VAD over all of this frame's audio; True if it's speech.

        The frame is not a multiple of 512 samples, so leftovers carry over to
        the next frame and the model always sees an unbroken audio stream.
        """
        self._vad_pending = np.concatenate([self._vad_pending, frame])
        while len(self._vad_pending) >= CHUNK_SIZE:
            chunk = self._vad_pending[:CHUNK_SIZE]
            self._vad_pending = self._vad_pending[CHUNK_SIZE:]
            prob = speech_probability(chunk.astype(np.float32) / 32768.0)
            if self._in_speech:
                if prob < END_PROB:
                    self._in_speech = False
                    self._start_run = 0
            else:
                self._start_run = self._start_run + 1 if prob >= START_PROB else 0
                if self._start_run >= START_CHUNKS:
                    self._in_speech = True
        return self._in_speech

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
