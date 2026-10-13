"""Interrupted replies: keep the agent's memory honest when the user cuts it off.

When the user stops Jessica mid-reply (Stop button or by talking over her), two
things are true that the conversation memory doesn't know on its own:

  * the agent keeps generating after we stop *listening* to it — abandoning a
    LangGraph stream blocks until the whole reply is produced and saved, and
  * the saved reply is the FULL text, including the part the user never heard.

So after an interruption the last assistant message is rewritten to contain only
what was actually heard, followed by an explicit marker the model can read:

    The capital of Australia is Canberra. <interrupted by="voice" />

This module is deliberately free of model imports so it can be tested without
OpenAI/Whisper/Kokoro. It has no knowledge of the agent beyond the graph object
passed in.
"""
import logging
import re
import threading

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger("speakIelts.interrupts")

# How long the next turn waits for a cut-off reply to finish generating and be
# rewritten before giving up and carrying on anyway.
SETTLE_TIMEOUT_S = 60.0

# Matches the marker (and any stray open/close variants a model might echo).
_TAG_RE = re.compile(r"<\s*/?\s*interrupted\b[^>]*>", re.IGNORECASE)


def strip_tags(text: str) -> str:
    """Remove interruption markers so they are never spoken or shown."""
    return _TAG_RE.sub("", text)


def interrupted_text(heard: str, by: str) -> str:
    """The text that replaces a cut-off reply in memory: heard part + marker."""
    tag = f'<interrupted by="{by}" />'
    heard = strip_tags(heard).strip()
    return f"{heard} {tag}" if heard else tag


def rewrite_last_reply(graph, session_id: str, heard: str, by: str) -> None:
    """Replace the latest assistant reply of a thread with heard text + marker.

    Targets the last AI message (one without tool calls) after the last human
    message. Reusing its id makes the graph's message reducer replace it instead
    of appending a second one.
    """
    config = {"configurable": {"thread_id": session_id}}
    state = graph.get_state(config)
    messages = state.values.get("messages", []) if state else []
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )
    target = None
    for m in reversed(messages[last_human + 1 :]):
        if isinstance(m, AIMessage) and not m.tool_calls:
            target = m
            break
    if target is None:
        return  # nothing was saved for this turn; there is nothing to correct
    graph.update_state(
        config,
        {"messages": [AIMessage(content=interrupted_text(heard, by), id=target.id)]},
    )


class _Run:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.done = threading.Event()  # the agent stream has fully finished
        self.rewrite = False           # a post-interrupt rewrite owns "settled"


class ReplyTracker:
    """Serialises agent runs per session around interruptions.

    A cut-off run keeps generating in the background. Until it has finished and
    its reply has been rewritten, the session is "unsettled": the next
    `begin()` for that session waits, so a new turn never races the rewrite.
    """

    def __init__(self, settle_timeout: float = SETTLE_TIMEOUT_S):
        self._timeout = settle_timeout
        self._lock = threading.Lock()
        self._settled: dict[str, threading.Event] = {}
        self._runs: dict[str, _Run] = {}

    def _event(self, session_id: str) -> threading.Event:
        ev = self._settled.get(session_id)
        if ev is None:
            ev = threading.Event()
            ev.set()
            self._settled[session_id] = ev
        return ev

    def begin(self, session_id: str) -> _Run:
        """Start a run: wait until the previous one settled, then register."""
        with self._lock:
            ev = self._event(session_id)
        if not ev.wait(self._timeout):
            logger.warning("session %s never settled; continuing anyway", session_id)
        with self._lock:
            ev.clear()
            run = _Run(session_id)
            self._runs[session_id] = run
        return run

    def finish(self, run: _Run) -> None:
        """Called by the worker when the agent stream ends (normally or not)."""
        run.done.set()
        with self._lock:
            if not run.rewrite:
                self._event(run.session_id).set()

    def rewrite_after_run(self, session_id: str, fn) -> None:
        """Run `fn` (the memory rewrite) in the background once the current run
        has finished, without blocking the caller."""
        with self._lock:
            run = self._runs.get(session_id)
            if run is None or run.rewrite:
                return
            run.rewrite = True
            ev = self._event(session_id)
            ev.clear()

        def work():
            try:
                run.done.wait(self._timeout)
                fn()
            except Exception:
                logger.warning("interrupt rewrite failed for %s", session_id, exc_info=True)
            finally:
                ev.set()

        threading.Thread(target=work, daemon=True).start()
