# voice pipeline: mic -> STT (Whisper local) -> cleanup (Groq) -> agent (OpenAI) -> TTS (Kokoro)
from app.asr import asr
from app.cleanup import clean_message
from app.agent import agent_reply_stream
from app.tts import speak, speak_stream

# only an utterance that IS one of these quits the app —
# "how do I say goodbye in French?" must not (substring match did exactly that)
GOODBYE_WORDS = {"goodbye", "good bye", "bye", "bye bye", "quit", "exit"}


def _is_goodbye(text: str) -> bool:
    return text.lower().strip(" .!?") in GOODBYE_WORDS


def main():
    print("Say 'goodbye' to quit.")
    while True:
        try:
            # 1. listen
            text = asr()
            if not text:
                continue

            # 2. quit command
            if _is_goodbye(text):
                speak("Goodbye! It was nice talking to you.")
                break

            # 3. clean the raw STT text into a proper message
            cleaned = clean_message(text)
            if cleaned == "[unclear]":
                print("Cleanup failed: transcript too garbled, ask user to repeat.")
                speak("Sorry, I didn't catch that. Could you say it again?")
                continue
            print("Cleaned message:", cleaned)

            # 4. stream agent reply while speaking it
            print("Agent:", end=" ", flush=True)
            speak_stream(agent_reply_stream(cleaned))
            print()  # newline after streaming

        except KeyboardInterrupt:
            print()
            break  # Ctrl+C exits cleanly instead of dumping a traceback
        except Exception as e:
            # one API hiccup (Groq/OpenAI/Tavily) must not kill the session —
            # apologize out loud and keep listening
            print(f"\n(turn failed: {e})")
            try:
                speak("Sorry, something went wrong on my end. Please try again.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
