"""Speech in and out. Importing asr or tts loads its model, so import only what you use.

  vad.py       - Silero VAD: is this 32 ms chunk speech?
  recorder.py  - web path: PCM chunks in, a finished clip out when the user stops talking
  asr.py       - Whisper transcription (+ the terminal mic recorder)
  tts.py       - Kokoro speech, streamed sentence by sentence
"""
