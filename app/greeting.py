"""Opening questions Jessica asks at the start of a call."""
import random

from app import sessions as sessions_db
from app.agent import agent, model, text_from_content

_STARTERS = [
    "Hi, I'm Jessica. What would you like to talk about today?",
    "Hey there. What's on your mind?",
    "Hello. Is there something I can help you figure out?",
    "Hi. What are you working on right now?",
    "Hey. Tell me about your day.",
    "Hi, I'm Jessica. Got a question for me?",
    "Hello. What should we chat about?",
]


def _has_messages(session_id: str) -> bool:
    state = agent.get_state({"configurable": {"thread_id": session_id}})
    if not state:
        return False
    return bool(state.values.get("messages"))


def opening_question(session_id: str) -> str:
    """Return the first thing Jessica says when the user starts a call.

    New sessions get a random starter. Returning sessions get a natural
    follow-up based on the cached conversation summary.
    """
    if not _has_messages(session_id):
        return random.choice(_STARTERS)

    summary = sessions_db.get_summary(session_id)
    if not summary:
        summary = sessions_db.summarize(session_id)
        sessions_db.set_summary(session_id, summary)

    prompt = (
        "You are Jessica, a friendly assistant. The following is a summary of "
        "your ongoing conversation with the user. Ask one natural follow-up question "
        "to continue the conversation. Keep it short and conversational.\n\n"
        f"Summary: {summary}\n\n"
        "Follow-up question:"
    )
    return text_from_content(model.invoke(prompt).content)
