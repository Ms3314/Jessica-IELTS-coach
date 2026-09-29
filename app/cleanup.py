"""Cleans raw STT output into a proper message before it reaches the agent.

Whisper returns raw speech: filler words ("uhm", "like"), homophone errors
("WhatsApp" instead of "what's up"), odd punctuation. This layer turns that
into clean text so the agent isn't confused by transcription noise.

Cleanup is an enhancement, not a requirement: if the cleanup model is
unreachable, the raw transcript is passed through unchanged.
"""
from langchain_groq import ChatGroq

from .config import GROQ_MODEL

# Groq's free, fast model — good enough for text cleanup, near-zero latency
_cleanup_model = ChatGroq(
    model=GROQ_MODEL,
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
    """Convert a raw STT transcript into a clean user message.

    Falls back to the raw transcript if the cleanup model fails — a slightly
    messy question is much better than a dead conversation.
    """
    try:
        response = _cleanup_model.invoke(
            CLEANUP_PROMPT.format(transcript=raw_text)
        )
        return response.content.strip() or raw_text
    except Exception as e:
        print(f"(cleanup unavailable, using raw transcript: {e})")
        return raw_text
