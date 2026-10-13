"""Jessica's brain: a LangChain agent that streams reply text.

Conversation memory is persisted in SQLite, one LangGraph thread per voice
session. The browser sends its session id when it opens the socket, so
Quit/Start keeps the same memory and "New session" starts a fresh thread.
"""
import queue
import sqlite3
import threading

from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import OPENAI_MODEL, PROJECT_ROOT
from .interrupts import ReplyTracker, rewrite_last_reply
from .llmtext import _text_deltas, text_from_content  # noqa: F401 (re-exported)

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

If a question needs facts you are unsure about, use the search tool first.

Joined speech: the user's message may be several spoken pieces joined with "...", because they
kept talking or paused mid-thought. Treat it as one train of thought and respond to the whole
thing, not just the last words. Spoken text can also contain transcription mistakes or stray
background phrases, so don't dwell on odd fragments. Only say goodbye if the whole message is a
goodbye, not when "bye" merely appears at the end of a longer message.

Interruptions: if one of your earlier replies ends with <interrupted by="..." />, the user
cut you off, so they only heard the text BEFORE the tag (the tag alone means they heard
nothing). Do not resume or repeat that old answer. Respond to what they say next, and treat
a correction like "no, I meant ..." as replacing your earlier answer. Never write or say
the tag yourself."""


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


_tracker = ReplyTracker()
_END = object()


def agent_reply_stream(user_text: str, session_id: str):
    """Send a user message on the session's thread, yield text deltas.

    Only AIMessageChunk text is yielded. ToolMessage results (raw Tavily JSON)
    and HumanMessage must never be spoken — they also have `.content`, so an
    isinstance check on the chunk type is the guard.

    The LangGraph stream runs in a background thread. Abandoning a LangGraph
    stream blocks until the whole reply has been generated, which would freeze
    the voice loop right after the user interrupts. Here the consumer can walk
    away immediately; the worker just finishes in the background (see
    `mark_interrupted`). The run is registered before this returns, so an
    interruption can always find it.
    """
    run = _tracker.begin(session_id)  # waits if a cut-off reply is still settling
    out: "queue.Queue" = queue.Queue()
    config = {"configurable": {"thread_id": session_id}}

    def worker():
        try:
            for chunk, _metadata in agent.stream(
                {"messages": [{"role": "user", "content": user_text}]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk):
                    for delta in _text_deltas(chunk.content):
                        out.put(delta)
        except Exception as e:  # surfaced to the consumer below
            out.put(e)
        finally:
            out.put(_END)
            _tracker.finish(run)

    threading.Thread(target=worker, daemon=True).start()

    def deltas():
        while True:
            item = out.get()
            if item is _END:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    return deltas()


def apply_keys(openai_api_key: str = "", tavily_api_key: str = "") -> None:
    """Use new API keys right away (the model and search tool are already built)."""
    from .llm_keys import swap_openai_credentials, swap_tavily_key

    if openai_api_key:
        fresh = ChatOpenAI(model=OPENAI_MODEL, use_responses_api=True, api_key=openai_api_key)
        swap_openai_credentials(model, fresh)
    if tavily_api_key:
        swap_tavily_key(search, tavily_api_key)


def mark_interrupted(session_id: str, heard_text: str, by: str) -> None:
    """Record that the user cut the latest reply off.

    Returns immediately. Once the (still generating) reply finishes, its saved
    text is replaced with only `heard_text` plus an `<interrupted by=... />`
    marker, and the session's next turn waits until that rewrite is done.
    """
    _tracker.rewrite_after_run(
        session_id, lambda: rewrite_last_reply(agent, session_id, heard_text, by)
    )
