"""Helpers for reading text out of LangChain message content (str or block list)."""


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
