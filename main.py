# voice pipeline: mic -> STT -> cleanup LLM -> agent -> TTS
from asr import asr
from cleanup import clean_message
from agent import agent_reply_stream
from tts import speak, speak_stream


def main():
    print("Say 'goodbye' to quit.")
    while True:
        # 1. listen
        text = asr()
        if not text:
            continue

        # 2. quit command
        if "goodbye" in text.lower():
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


if __name__ == "__main__":
    main()
