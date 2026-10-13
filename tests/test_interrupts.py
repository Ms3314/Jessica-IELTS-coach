"""Interrupted-reply memory tests — fake slow model, in-memory DB, no API calls.

    python tests/test_interrupts.py
"""
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.sqlite import SqliteSaver

from app.interrupts import (
    ReplyTracker,
    interrupted_text,
    rewrite_last_reply,
    strip_tags,
)

WORDS = "Canberra is the capital. It was a compromise. Sydney is bigger.".split()


class SlowModel(BaseChatModel):
    delay: float = 0.05

    @property
    def _llm_type(self):
        return "slow"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=" ".join(WORDS)))]
        )

    def _stream(self, messages, stop=None, run_manager=None, **kw):
        for w in WORDS:
            time.sleep(self.delay)
            yield ChatGenerationChunk(message=AIMessageChunk(content=w + " "))


def _graph():
    cp = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    cp.setup()
    return create_agent(model=SlowModel(), tools=[], system_prompt="x", checkpointer=cp)


def _ai_texts(graph, sid):
    state = graph.get_state({"configurable": {"thread_id": sid}})
    return [m.content for m in state.values["messages"] if isinstance(m, AIMessage)]


def _ask(graph, sid, text="q"):
    list(graph.stream({"messages": [{"role": "user", "content": text}]},
                      config={"configurable": {"thread_id": sid}}))


def test_strip_tags():
    assert strip_tags('Hi <interrupted by="voice" /> there') == "Hi  there"
    assert strip_tags("a <INTERRUPTED/> b") == "a  b"
    assert strip_tags("no tag here") == "no tag here"


def test_interrupted_text_variants():
    assert interrupted_text("Canberra is it.", "voice") == 'Canberra is it. <interrupted by="voice" />'
    assert interrupted_text("", "button") == '<interrupted by="button" />'
    # a model echoing the tag into the heard text must not double it
    assert interrupted_text('x <interrupted by="voice" />', "voice").count("<interrupted") == 1


def test_rewrite_replaces_full_reply_with_heard_part():
    g = _graph()
    _ask(g, "s1")
    assert _ai_texts(g, "s1")[-1].startswith("Canberra is the capital. It was")
    rewrite_last_reply(g, "s1", "Canberra is the capital.", "button")
    texts = _ai_texts(g, "s1")
    assert texts == ['Canberra is the capital. <interrupted by="button" />'], texts


def test_rewrite_only_touches_the_latest_reply():
    g = _graph()
    _ask(g, "s2", "first")
    first = _ai_texts(g, "s2")[0]
    _ask(g, "s2", "second")
    rewrite_last_reply(g, "s2", "", "voice")
    texts = _ai_texts(g, "s2")
    assert texts[0] == first, "earlier replies must be untouched"
    assert texts[1] == '<interrupted by="voice" />'


def test_rewrite_with_no_saved_reply_is_a_noop():
    g = _graph()
    rewrite_last_reply(g, "empty", "x", "voice")  # must not raise or invent messages
    state = g.get_state({"configurable": {"thread_id": "empty"}})
    assert not (state.values.get("messages") if state else None)


def test_tracker_next_turn_waits_for_rewrite():
    tracker = ReplyTracker(settle_timeout=5)
    order = []
    run = tracker.begin("t")

    def worker():                      # the abandoned agent stream, still running
        time.sleep(0.2)
        order.append("generation finished")
        tracker.finish(run)

    threading.Thread(target=worker, daemon=True).start()
    tracker.rewrite_after_run("t", lambda: order.append("rewritten"))

    t0 = time.time()
    tracker.begin("t")                 # next turn must block until rewrite done
    order.append("next turn began")
    assert order == ["generation finished", "rewritten", "next turn began"], order
    assert time.time() - t0 >= 0.15


def test_tracker_no_interrupt_does_not_block():
    tracker = ReplyTracker(settle_timeout=5)
    run = tracker.begin("u")
    tracker.finish(run)
    t0 = time.time()
    tracker.begin("u")
    assert time.time() - t0 < 0.1


def test_tracker_rewrite_after_run_already_finished():
    tracker = ReplyTracker(settle_timeout=5)
    run = tracker.begin("v")
    tracker.finish(run)                # reply completed before the interrupt landed
    done = queue.Queue()
    tracker.rewrite_after_run("v", lambda: done.put("ok"))
    assert done.get(timeout=2) == "ok"
    tracker.begin("v")                 # and the session settles afterwards


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} interrupts tests passed")
