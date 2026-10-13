"""speakIelts — a local-first voice assistant.

Pipeline: mic -> Whisper (local, MLX) -> agent (OpenAI + Tavily)
          -> streaming TTS (Kokoro, sentence-by-sentence).

Intentionally no eager imports here: importing a submodule (asr, tts, ...)
loads its model (Whisper warm-load, Kokoro pipeline), so only import what
you actually use.

  config.py, settings_store.py  - configuration and saved API keys
  io.py      - AudioSource / Sink / EventHandler: how an engine hears, speaks, reports
  speech/    - VAD, recording, Whisper, Kokoro
  llm/       - the agent and its memory
  chat/      - free conversation (Session)
  exam/      - the IELTS mock test (ExamSession) and its report
  api/       - REST routes and the /ws WebSocket for the web UI

See ARCHITECTURE.md for the full map.
"""
