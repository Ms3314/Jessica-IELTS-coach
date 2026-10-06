"""TalkingAngela — a local-first voice assistant.

Pipeline: mic -> Whisper (local, MLX) -> agent (OpenAI + Tavily)
          -> streaming TTS (Kokoro, sentence-by-sentence).

Intentionally no eager imports here: importing a submodule (asr, tts, ...)
loads its model (Whisper warm-load, Kokoro pipeline), so only import what
you actually use.
"""
