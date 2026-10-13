"""Turn a saved conversation into plain messages the UI can show."""
from langchain_core.messages import AIMessage, HumanMessage

from app.interrupts import strip_tags
from app.llmtext import text_from_content


def to_chat_messages(messages) -> list[dict]:
    """[{role: "user"|"assistant", text, interrupted}] from LangChain messages.

    Tool calls and tool results are internal and skipped. A reply the user cut off
    keeps only what was heard, flagged `interrupted`, and the raw marker the model
    sees is never exposed.
    """
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            text = text_from_content(m.content).strip()
            if text:
                out.append({"role": "user", "text": text, "interrupted": False})
        elif isinstance(m, AIMessage) and not m.tool_calls:
            raw = text_from_content(m.content)
            text = strip_tags(raw).strip()
            interrupted = text != raw.strip()
            if text or interrupted:
                out.append({"role": "assistant", "text": text, "interrupted": interrupted})
    return out
