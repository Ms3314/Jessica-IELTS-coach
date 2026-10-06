"""Session metadata and summaries stored alongside the LangGraph checkpoint DB.

The checkpointer keeps the conversation messages; this module keeps a small
metadata table (name, colour, summary, last-used time) so the UI can list
sessions, let the user rename and colour-code them, and request a short summary
of any thread.
"""
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.agent import agent, model, text_from_content
from app.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "checkpoints.db"

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_lock = threading.Lock()

_DEFAULT_COLORS = [
    "#3b82f6",  # blue
    "#8b5cf6",  # violet
    "#06b6d4",  # cyan
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#ec4899",  # pink
    "#ef4444",  # red
]

_DEFAULT_ICONS = [
    "Bot",
    "Sparkles",
    "Zap",
    "MessageCircle",
    "Brain",
    "Lightbulb",
    "Star",
    "Heart",
    "Moon",
    "Sun",
    "Cloud",
    "Flame",
    "Gem",
    "Rocket",
    "Target",
    "Compass",
    "Crown",
    "Diamond",
    "Feather",
    "Ghost",
    "Globe",
    "Key",
    "Leaf",
    "Map",
    "Music",
    "Palette",
    "Shield",
    "Smile",
    "Snowflake",
    "Umbrella",
    "Wallet",
    "Watch",
]


def _init() -> None:
    with _lock:
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_meta (
                id TEXT PRIMARY KEY,
                name TEXT,
                color TEXT,
                icon TEXT,
                summary TEXT,
                updated_at REAL
            )
            """
        )
        # Add icon column to tables created before this migration.
        try:
            _conn.execute("ALTER TABLE session_meta ADD COLUMN icon TEXT")
            _conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


_init()


@dataclass
class SessionMeta:
    id: str
    name: str
    color: str
    icon: str
    summary: Optional[str]
    updated_at: float


def _default_name(session_id: str) -> str:
    return f"Session {session_id[:6]}"


def _default_color() -> str:
    return random.choice(_DEFAULT_COLORS)


def _default_icon() -> str:
    return random.choice(_DEFAULT_ICONS)


def touch(session_id: str) -> None:
    """Ensure a metadata row exists for the session (does not reorder it)."""
    with _lock:
        cur = _conn.execute(
            "SELECT 1 FROM session_meta WHERE id = ?", (session_id,)
        )
        if cur.fetchone() is None:
            _conn.execute(
                """
                INSERT INTO session_meta (id, name, color, icon, summary, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    _default_name(session_id),
                    _default_color(),
                    _default_icon(),
                    None,
                    time.time(),
                ),
            )
            _conn.commit()


def bump(session_id: str) -> None:
    """Bump the session to the top of the list after real conversation activity."""
    with _lock:
        _conn.execute(
            "UPDATE session_meta SET updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        _conn.commit()


def list_sessions() -> list[SessionMeta]:
    """Return sessions ordered by most recently used first."""
    with _lock:
        rows = _conn.execute(
            """
            SELECT id, name, color, icon, summary, updated_at
            FROM session_meta
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [
        SessionMeta(
            id=r[0],
            name=r[1] or _default_name(r[0]),
            color=r[2] or _DEFAULT_COLORS[0],
            icon=r[3] or _DEFAULT_ICONS[0],
            summary=r[4],
            updated_at=r[5],
        )
        for r in rows
    ]


def update(
    session_id: str,
    name: Optional[str] = None,
    color: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """Rename, recolour or change the icon of a session."""
    parts = []
    values = []
    if name is not None:
        parts.append("name = ?")
        values.append(name)
    if color is not None:
        parts.append("color = ?")
        values.append(color)
    if icon is not None:
        parts.append("icon = ?")
        values.append(icon)
    if not parts:
        return
    values.append(session_id)
    with _lock:
        _conn.execute(
            f"UPDATE session_meta SET {', '.join(parts)} WHERE id = ?", tuple(values)
        )
        _conn.commit()


def get_summary(session_id: str) -> Optional[str]:
    """Return the cached summary, if any."""
    with _lock:
        row = _conn.execute(
            "SELECT summary FROM session_meta WHERE id = ?", (session_id,)
        ).fetchone()
    return row[0] if row else None


def set_summary(session_id: str, summary: str) -> None:
    """Cache a summary and bump the session to the top of the list."""
    with _lock:
        _conn.execute(
            """
            UPDATE session_meta
            SET summary = ?, updated_at = ?
            WHERE id = ?
            """,
            (summary, time.time(), session_id),
        )
        _conn.commit()


def maybe_auto_rename(session_id: str, first_message: str) -> None:
    """Give the session a short 2-3 word title based on the first user message."""
    with _lock:
        row = _conn.execute(
            "SELECT name FROM session_meta WHERE id = ?", (session_id,)
        ).fetchone()
    current = row[0] if row else ""
    if current and not current.startswith("Session "):
        return  # user or a previous run already renamed it

    prompt = (
        "Give this conversation a very short 2-3 word title. "
        "Reply with only the title, no punctuation or explanation.\n\n"
        f'First message: "{first_message}"'
    )
    title = text_from_content(model.invoke(prompt).content).strip().strip('"').strip("'")
    if title:
        update(session_id, name=title)


def delete(session_id: str) -> None:
    """Delete metadata and the matching LangGraph thread checkpoints."""
    with _lock:
        _conn.execute("DELETE FROM session_meta WHERE id = ?", (session_id,))
        _conn.commit()
    try:
        agent.checkpointer.delete({"configurable": {"thread_id": session_id}})
    except Exception:
        # If the checkpointer doesn't expose delete or the thread is missing,
        # the metadata row is already gone — that's the important part.
        pass


def summarize(session_id: str) -> str:
    """Ask the model to summarize the conversation thread."""
    state = agent.get_state({"configurable": {"thread_id": session_id}})
    messages = state.values.get("messages", []) if state else []
    if not messages:
        return "No conversation yet."

    lines = []
    for m in messages:
        role = getattr(m, "type", "unknown")
        content = getattr(m, "content", "")
        if role == "human":
            lines.append(f"User: {content}")
        elif role == "ai":
            lines.append(f"Jessica: {content}")
    transcript = "\n".join(lines)

    prompt = (
        "Summarize the following conversation between a user and Jessica, "
        "a friendly assistant. Keep it to one short paragraph. Mention the "
        "topics discussed and any key advice given.\n\n"
        f"{transcript}"
    )
    return text_from_content(model.invoke(prompt).content)
