import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import mlx_whisper
import tempfile
import os


MODEL = "mlx-community/whisper-large-v3-turbo"

SAMPLE_RATE = 16000
FRAME_MS = 100          # read audio in 100ms chunks
SILENCE_LIMIT_S = 1.2   # stop after this much silence once speech started
MAX_RECORD_S = 15.0     # safety cap so a noisy mic can't record forever
PRE_ROLL_S = 0.3        # keep this much audio before speech started
CALIBRATION_S = 0.5     # measure background noise for this long before listening
NOISE_FACTOR = 3.0      # speech threshold = noise floor * this
MIN_THRESHOLD = 350.0   # rms floor so quiet mics still trigger
MAX_THRESHOLD = 4000.0  # rms ceiling so loud rooms don't lock out speech


def _rms(frame) -> float:
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))


def record_until_silence(open_stream):
    """Record from `open_stream()` until ~1s of silence follows speech.

    open_stream: callable returning a context manager whose .read(n) -> (data, overflowed),
    e.g. sounddevice.InputStream. Returns an int16 ndarray, or None if no speech.
    """
    frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)
    silence_limit = int(SILENCE_LIMIT_S * 1000 / FRAME_MS)
    max_frames = int(MAX_RECORD_S * 1000 / FRAME_MS)
    pre_roll = int(PRE_ROLL_S * 1000 / FRAME_MS)

    with open_stream() as stream:
        # calibrate on the first half second to learn the noise floor
        noise = []
        for _ in range(int(CALIBRATION_S * 1000 / FRAME_MS)):
            data, _ = stream.read(frame_len)
            noise.append(_rms(data))
        threshold = min(MAX_THRESHOLD, max(MIN_THRESHOLD, max(noise) * NOISE_FACTOR))

        frames = []
        speech_rms = []
        spoken = False
        silent = 0

        while len(frames) < max_frames:
            data, _ = stream.read(frame_len)
            frames.append(data.copy())
            r = _rms(data)
            speech_rms.append(r > threshold)
            if r > threshold:
                spoken = True
                silent = 0
            elif spoken:
                silent += 1
                if silent >= silence_limit:
                    break

    if not spoken:
        return None

    # trim leading silence but keep a little pre-roll before the first speech frame
    first = next(i for i, s in enumerate(speech_rms) if s)
    start = max(0, first - pre_roll)
    return np.concatenate(frames[start:])


def asr():
    print("Speak something...")

    audio = record_until_silence(
        lambda: sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        )
    )

    if audio is None:
        print("No speech detected, try again.")
        return None

    print(f"Processing Audio ... (Local Whisper) [{len(audio) / SAMPLE_RATE:.1f}s of audio]")

    # Save temporary WAV
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp:
        temp_path = temp.name

    wav.write(temp_path, SAMPLE_RATE, audio)

    try:
        # Run Whisper locally
        result = mlx_whisper.transcribe(
            temp_path,
            path_or_hf_repo=MODEL,
            language="en",
            temperature=0
        )

        text = result["text"].strip()

        if text:
            print("You said:", text)
            return text

        print("Sorry, I couldn't understand that.")
        return None

    finally:
        os.remove(temp_path)


if __name__ == "__main__":
    asr()
