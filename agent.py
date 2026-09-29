import os
import uuid

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# change model via .env -> OPENAI_MODEL=...
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

# gpt-5.6 needs the Responses API for function tools (chat/completions rejects them)
model = ChatOpenAI(model=DEFAULT_MODEL, use_responses_api=True)

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


agent = create_agent(
    model=model,
    tools=[search],
    system_prompt=SYSTEM_PROMPT,
    name="jessica_assistant",
    checkpointer=InMemorySaver(),  # keeps conversation history per thread_id
)

_config = {"configurable": {"thread_id": str(uuid.uuid4())}}


def _extract_text(content) -> str:
    """Pull plain text out of a message content (str or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(p for p in parts if p).strip()
    return str(content)


def agent_reply(user_text: str) -> str:
    """Send a (already cleaned) user message to the agent, return its reply text."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config=_config,
    )
    return _extract_text(result["messages"][-1].content)


def agent_reply_stream(user_text: str):
    """Send a user message to the agent, yield text deltas as they stream in.

    Filters out reasoning blocks (gpt-5.6-luna emits those) and tool calls.
    Only yields actual text deltas for TTS.
    """
    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        config=_config,
        stream_mode="messages",
    ):
        # AIMessageChunk from Responses API has content as list of blocks
        if not hasattr(chunk, "content"):
            continue
        content = chunk.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        yield text
        elif isinstance(content, str) and content:
            yield content
