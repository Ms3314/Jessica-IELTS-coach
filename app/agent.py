"""Jessica's brain: a LangChain agent that streams reply text.

Conversation memory is persisted in SQLite, one LangGraph thread per voice
session. The browser sends its session id when it opens the socket, so
Quit/Start keeps the same memory and "New session" starts a fresh thread.
"""
import sqlite3

from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import OPENAI_MODEL, PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "checkpoints.db"

# gpt-5.6 needs the Responses API for function tools (chat/completions rejects them)
model = ChatOpenAI(model=OPENAI_MODEL, use_responses_api=True)

search = TavilySearch(
    max_results=3,
    topic="general",
)


SYSTEM_PROMPT = """You are Jessica, a friendly and knowledgeable assistant.

Your job:
- Answer the user's questions directly and naturally, like a smart friend talking out loud.
- Just answer what they asked. A short follow-up question is okay only if it clearly helps.

Voice rules (this reply will be converted to speech by a TTS engine):
- Keep replies SHORT: 2-4 sentences max. The user has to listen to all of it.
- No markdown, no bullet points, no emoji, no lists. Plain conversational sentences only.
- Spell out numbers and symbols the way you would say them out loud.

If a question needs facts you are unsure about, use the search tool first."""


# One connection for the whole process; SqliteSaver serializes access with an
# internal lock, and engine threads may call it concurrently.
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_checkpointer = SqliteSaver(_conn)
_checkpointer.setup()  # create the tables on first run

agent = create_agent(
    model=model,
    tools=[search],
    system_prompt=SYSTEM_PROMPT,
    name="jessica_assistant",
    checkpointer=_checkpointer,
)


def _text_deltas(content):
    """Yield text-only deltas from an AIMessageChunk content (str or block list).

    gpt-5.6-luna via the Responses API emits reasoning blocks alongside text
    blocks; only 'text' blocks should ever reach the TTS.
    """
    if isinstance(content, str):
        if content:
            yield content
        return
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    yield text


def text_from_content(content) -> str:
    """Extract plain text from a complete AIMessage.content value.

    Non-streaming invokes can return a string or a list of content blocks
    (e.g. from the Responses API). This normalises both to a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def agent_reply_stream(user_text: str, session_id: str):
    """Send a user message on the session's thread, yield text deltas.

    Only AIMessageChunk text is yielded. ToolMessage results (raw Tavily JSON)
    and HumanMessage must never be spoken — they also have `.content`, so an
    isinstance check on the chunk type is the guard.
    """
    config = {"configurable": {"thread_id": session_id}}
    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        config=config,
        stream_mode="messages",
    ):
        if not isinstance(chunk, AIMessageChunk):
            continue
        yield from _text_deltas(chunk.content)
