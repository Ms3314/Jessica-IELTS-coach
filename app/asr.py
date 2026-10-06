import sys
import numpy as np
import sounddevice as sd
import mlx_whisper
from app.vad import is_speech, speech_probability, CHUNK_SIZE


MODEL = "mlx-community/whisper-large-v3-turbo"

SAMPLE_RATE = 16000
FRAME_MS = 100           # read audio in 100ms chunks
SILENCE_LIMIT_S = 1.2    # stop after this much silence once speech started
MAX_RECORD_S = 15.0      # safety cap so a noisy mic can't record forever
PRE_ROLL_S = 0.3         # keep this much audio before speech started
CALIBRATION_S = 0.5      # measure background noise for this long before listening
NOISE_FACTOR = 3.0       # speech threshold = noise floor * this
MIN_THRESHOLD = 350.0    # rms floor so quiet mics still trigger
MAX_THRESHOLD = 4000.0   # rms ceiling so loud rooms don't lock out speech
MIN_SPEECH_S = 0.3       # require this much actual speech so a cough doesn't trigger Whisper


def _rms(frame) -> float:
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))


def _visualize_audio(frame, vad_prob, is_speaking):
    """Show a real-time audio level bar with speech indicator.
    
    Updates in-place on the same line for smooth animation.
    """
    # Calculate audio level (0-40 chars wide)
    rms = _rms(frame)
    # Normalize: 0-5000 RMS maps to 0-40 bars
    level = min(40, int(rms / 125))
    
    # Build the visualization
    bar = "█" * level + "░" * (40 - level)
    
    # Speech indicator
    if is_speaking:
        indicator = "🎤"
        color = "\033[92m"  # green
    elif vad_prob > 0.3:
        indicator = "👂"
        color = "\033[93m"  # yellow
    else:
        indicator = "⏺"
        color = "\033[90m"  # gray
    
    reset = "\033[0m"
    
    # Print in-place (carriage return, no newline)
    sys.stdout.write(f"\r{color}{indicator}{reset} [{bar}] {vad_prob:.2f}")
    sys.stdout.flush()


class ReusableInputStream:
    """Keeps ONE InputStream open for the whole app; start/stop per turn.

    Opening/closing a PortAudio stream every turn eventually wedges macOS
    CoreAudio (the freeze at 'Speak something...'). start()/stop() are cheap
    and don't renegotiate the device. stop() also discards audio that
    trickled in while Jessica was talking, so the mic never transcribes
    her own voice from the speakers.
    """

    def __init__(self):
        self._stream = None

    def __enter__(self):
        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16"
            )
        self._stream.start()
        return self._stream

    def __exit__(self, *a):
        self._stream.stop()
        return False


_input = ReusableInputStream()


def record_until_silence(open_stream):
    """Record from `open_stream()` until ~1s of silence follows speech.

    open_stream: callable returning a context manager whose .read(n) -> (data, overflowed),
    e.g. sounddevice.InputStream. Returns an int16 ndarray, or None if no speech.

    Calibration frames are KEPT (leading silence is trimmed afterwards) so the
    first words aren't lost if the user starts speaking immediately.
    """
    frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)
    silence_limit = int(SILENCE_LIMIT_S * 1000 / FRAME_MS)
    max_frames = int(MAX_RECORD_S * 1000 / FRAME_MS)
    cal_frames = int(CALIBRATION_S * 1000 / FRAME_MS)
    pre_roll = int(PRE_ROLL_S * 1000 / FRAME_MS)
    min_speech_frames = int(MIN_SPEECH_S * 1000 / FRAME_MS)

    frames, rms_vals = [], []

    with open_stream() as stream:
        # drop one frame right after (re)start: anything stale buffered while
        # we were speaking must never reach Whisper
        stream.read(frame_len)

        # first half second: learn the noise floor (frames are kept, not thrown away)
        for _ in range(cal_frames):
            data, _ = stream.read(frame_len)
            frames.append(data.copy())
            rms_vals.append(_rms(data))
            # Show visualization during calibration
            vad_prob = speech_probability(data[:CHUNK_SIZE].flatten().astype(np.float32) / 32768.0)
            _visualize_audio(data, vad_prob, False)

        noise_floor = max(rms_vals)
        if noise_floor > 2 * MIN_THRESHOLD:
            # user was already talking during calibration -> noise floor is skewed,
            # fall back to the default threshold instead of locking out their speech
            threshold = MIN_THRESHOLD
        else:
            threshold = min(MAX_THRESHOLD, max(MIN_THRESHOLD, noise_floor * NOISE_FACTOR))

        silent = 0
        vad_flags = []  # Track VAD speech detection per frame
        while len(frames) < max_frames:
            data, _ = stream.read(frame_len)
            frames.append(data.copy())
            rms_vals.append(_rms(data))

            # Use VAD for speech detection
            vad_prob = speech_probability(data[:CHUNK_SIZE].flatten().astype(np.float32) / 32768.0)
            is_speaking = vad_prob > 0.5
            vad_flags.append(is_speaking)

            if is_speaking:
                silent = 0
            else:
                silent += 1
                if silent >= silence_limit and any(vad_flags):
                    break  # speech happened, then silence -> done

            # Show real-time visualization
            _visualize_audio(data, vad_prob, is_speaking)

    # Clear the visualization line
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

    # Count speech frames using VAD (more accurate than RMS)
    speech_frames = sum(vad_flags)

    # not enough real speech: a blip/cough isn't worth sending to Whisper
    # (it can hallucinate text out of pure noise)
    if speech_frames < min_speech_frames:
        return None

    # trim leading silence but keep a little pre-roll before the first speech frame
    first = next(i for i, s in enumerate(vad_flags) if s)
    start = max(0, first - pre_roll)
    # reshape(-1): sounddevice frames are (n, channels) — whisper needs flat 1D audio
    return np.concatenate(frames[start:]).reshape(-1)


def transcribe(audio: np.ndarray):
    """Run Whisper on a 16 kHz mono int16 clip and return the text (or None).

    Shared by both frontends: the terminal records from the mic, the web
    frontend receives the clip over a WebSocket, but both end up here.
    """
    if audio is None or len(audio) == 0:
        return None

    # pass samples straight to Whisper as float32 [-1, 1] — no temp WAV,
    # no ffmpeg decode subprocess (mlx_whisper accepts numpy arrays directly)
    result = mlx_whisper.transcribe(
        audio.astype(np.float32) / 32768.0,
        path_or_hf_repo=MODEL,
        language="en",
        temperature=0,
    )

    text = result["text"].strip()
    return text or None


def asr():
    print("Speak something...")

    audio = record_until_silence(lambda: _input)

    if audio is None:
        print("No speech detected, try again.")
        return None

    print(f"Processing Audio ... (Local Whisper) [{len(audio) / SAMPLE_RATE:.1f}s of audio]")

    text = transcribe(audio)

    if text:
        print("You said:", text)
        return text

    print("Sorry, I couldn't understand that.")
    return None


def _warm_load():
    """Load Whisper weights once at import so the first question isn't slow.

    mlx_whisper caches the model after the first call (ModelHolder); this
    just moves that first call to app startup instead of mid-conversation.
    """
    print("Loading Whisper model...")
    try:
        mlx_whisper.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            path_or_hf_repo=MODEL,
            language="en",
            temperature=0,
        )
    except Exception as e:
        print(f"(model will load on first use: {e})")


_warm_load()


if __name__ == "__main__":
    asr()
