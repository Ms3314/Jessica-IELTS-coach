# voice pipeline: mic -> STT (Whisper local) -> agent (OpenAI) -> TTS (Kokoro)
#
# Terminal frontend. The conversation flow lives in app/chat/engine.Session; this
# file just wires the host-mic source, the local-speaker sink, and the
# terminal printer, then runs the loop. The web frontend (server.py) drives
# the very same engine over a WebSocket.
from app.io import MicSource, TerminalHandler
from app.speech.tts import LocalSink
from app.chat.engine import Session


def main():
    print("Say 'goodbye' to quit.")
    session = Session(MicSource(), LocalSink(), TerminalHandler())
    session.loop()


if __name__ == "__main__":
    main()
