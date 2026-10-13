"""Saved-conversation formatting + live key swap tests — no network, no models.

    python tests/test_history.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.history import to_chat_messages
from app.llm_keys import swap_openai_credentials, swap_tavily_key


def test_a_plain_conversation_becomes_alternating_messages():
    msgs = [HumanMessage("Hello"), AIMessage("Hi, how can I help?"), HumanMessage("What is DSA?"),
            AIMessage("Data structures and algorithms.")]
    out = to_chat_messages(msgs)
    assert [(m["role"], m["text"]) for m in out] == [
        ("user", "Hello"), ("assistant", "Hi, how can I help?"),
        ("user", "What is DSA?"), ("assistant", "Data structures and algorithms."),
    ]
    assert not any(m["interrupted"] for m in out)


def test_tool_calls_and_their_results_are_hidden():
    msgs = [HumanMessage("latest news?"),
            AIMessage("", tool_calls=[{"name": "tavily_search", "args": {"query": "news"}, "id": "t1"}]),
            ToolMessage('{"results": [...]}', tool_call_id="t1"),
            AIMessage("Here is the news.")]
    out = to_chat_messages(msgs)
    assert [m["text"] for m in out] == ["latest news?", "Here is the news."]


def test_an_interrupted_reply_shows_what_was_heard_and_hides_the_marker():
    out = to_chat_messages([AIMessage('Canberra is the capital. <interrupted by="voice" />')])
    assert out == [{"role": "assistant", "text": "Canberra is the capital.", "interrupted": True}]
    assert "interrupted by" not in out[0]["text"]


def test_a_reply_cut_off_before_anything_was_heard_is_still_recorded():
    out = to_chat_messages([AIMessage('<interrupted by="button" />')])
    assert out == [{"role": "assistant", "text": "", "interrupted": True}]


def test_reasoning_blocks_are_not_shown():
    blocks = [{"type": "reasoning", "summary": []}, {"type": "text", "text": "The answer."}]
    assert to_chat_messages([AIMessage(blocks)])[0]["text"] == "The answer."


def test_empty_messages_and_empty_history_are_skipped():
    assert to_chat_messages([]) == []
    assert to_chat_messages([HumanMessage("   "), AIMessage("")]) == []


def test_swapping_the_openai_key_updates_the_live_client():
    import os

    os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder-for-test")
    from langchain_openai import ChatOpenAI

    live = ChatOpenAI(model="gpt-5.6-luna", use_responses_api=True, api_key="sk-old-0000")
    fresh = ChatOpenAI(model="gpt-5.6-luna", use_responses_api=True, api_key="sk-new-1111")
    swap_openai_credentials(live, fresh)
    assert live.openai_api_key.get_secret_value() == "sk-new-1111"
    assert live.root_client.api_key == "sk-new-1111"
    assert live.root_async_client.api_key == "sk-new-1111"
    assert live.model_name == "gpt-5.6-luna", "everything else about the model is untouched"


def test_swapping_the_search_key():
    class Wrapper:
        tavily_api_key = None

    class Tool:
        api_wrapper = Wrapper()

    assert swap_tavily_key(Tool(), "tvly-new-9999") is True
    assert Tool.api_wrapper.tavily_api_key.get_secret_value() == "tvly-new-9999"
    assert swap_tavily_key(object(), "x") is False, "an unexpected tool shape must not crash"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} history tests passed")
