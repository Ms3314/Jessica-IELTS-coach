"""Cleans raw STT output into a proper message before it reaches the agent.

Google STT returns raw speech: filler words ("uhm", "like"), homophone errors
("WhatsApp" instead of "what's up"), missing punctuation. This layer turns
that into clean text so the agent isn't confused by transcription noise.
"""
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Groq's free, fast model — good enough for text cleanup, near-zero latency
_cleanup_model = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
)

CLEANUP_PROMPT = """You clean up speech-to-text transcripts before they are sent to an AI assistant.

Rules:
- Fix obvious transcription errors and homophones ("WhatsApp" -> "what's up" when it fits the sentence).
- Remove filler words (uhm, uh, like, you know, hmm) unless they carry meaning.
- Add punctuation and capitalization.
- If the transcript is too garbled to understand, return: [unclear]
- NEVER answer the message. NEVER add new content. Output ONLY the cleaned message, nothing else.

Transcript: {transcript}"""


def clean_message(raw_text: str) -> str:
    """Convert a raw STT transcript into a clean user message."""
    response = _cleanup_model.invoke(
        CLEANUP_PROMPT.format(transcript=raw_text)
    )
    return response.content.strip()
