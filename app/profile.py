"""Your profile: the scores of every mock test you have taken, kept on this computer.

One small SQLite file (data/profile.db). A score is added automatically when a
test's report is made, so progress can be tracked over time. Deleting a test's
session removes its scores too.
"""
import json
import logging
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger("speakIelts.profile")

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "profile.db"
DEFAULT_EXAMS_DIR = Path(__file__).resolve().parent.parent / "data" / "exams"

CRITERIA = ("fluency", "lexical", "grammar", "pronunciation")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS exam_scores (
    session_id    TEXT NOT NULL,
    exam_id       TEXT NOT NULL,
    taken_at      REAL NOT NULL,
    length        TEXT NOT NULL,
    finished      INTEGER NOT NULL,
    overall       REAL NOT NULL,
    fluency       REAL,
    lexical       REAL,
    grammar       REAL,
    pronunciation REAL,
    reliability   TEXT,
    PRIMARY KEY (session_id, exam_id)
)
"""

_REPORT_KEYS = {
    "fluency": "fluency_coherence",
    "lexical": "lexical_resource",
    "grammar": "grammatical_range_accuracy",
    "pronunciation": "pronunciation",
}


def _connect(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def record_exam(report: dict, log, path: Path = DEFAULT_PATH) -> bool:
    """Save a test's scores. Returns False (and saves nothing) for a report with no
    scores, such as 'not enough answered'. Taking the same test again replaces it."""
    overall = report.get("overall_band")
    if not isinstance(overall, (int, float)):
        return False
    row = {
        "session_id": log.session_id,
        "exam_id": log.exam_id,
        "taken_at": float(getattr(log, "started_at", None) or time.time()),
        "length": getattr(log, "length", None) or "full",
        # the test's own record says whether it ran to the end; the report is only a copy
        "finished": 1 if getattr(log, "finished", report.get("finished", False)) else 0,
        "overall": float(overall),
        "reliability": report.get("reliability"),
    }
    for col, key in _REPORT_KEYS.items():
        band = (report.get(key) or {}).get("band")
        row[col] = float(band) if isinstance(band, (int, float)) else None
    conn = _connect(path)
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO exam_scores "
                "(session_id, exam_id, taken_at, length, finished, overall, fluency, lexical, grammar, "
                "pronunciation, reliability) VALUES (:session_id, :exam_id, :taken_at, :length, :finished, "
                ":overall, :fluency, :lexical, :grammar, :pronunciation, :reliability)",
                row,
            )
    finally:
        conn.close()
    return True


def list_scores(path: Path = DEFAULT_PATH) -> list[dict]:
    """Every saved test, newest first."""
    if not Path(path).exists():
        return []
    conn = _connect(path)
    try:
        rows = conn.execute("SELECT * FROM exam_scores ORDER BY taken_at DESC").fetchall()
    finally:
        conn.close()
    return [dict(r) | {"finished": bool(r["finished"])} for r in rows]


def delete_session(session_id: str, path: Path = DEFAULT_PATH) -> None:
    """Forget the scores of a deleted session."""
    if not Path(path).exists():
        return
    conn = _connect(path)
    try:
        with conn:
            conn.execute("DELETE FROM exam_scores WHERE session_id = ?", (session_id,))
    finally:
        conn.close()


def _avg(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 1) if values else None


def summarize(rows: list[dict]) -> dict:
    """Headline numbers and a trend from saved tests (newest first)."""
    if not rows:
        return {"tests": 0, "latest": None, "best": None, "average": None, "change": None,
                "criteria_average": {c: None for c in CRITERIA}, "trend": []}
    overall = [r["overall"] for r in rows]
    return {
        "tests": len(rows),
        "latest": overall[0],
        "best": max(overall),
        "average": _avg(overall),
        # how the latest test compares with the one before it
        "change": round(overall[0] - overall[1], 1) if len(overall) > 1 else None,
        "criteria_average": {c: _avg([r[c] for r in rows if r[c] is not None]) for c in CRITERIA},
        "trend": [r["overall"] for r in reversed(rows)],     # oldest to newest, for a chart
    }


def sync_from_exams(path: Path = DEFAULT_PATH, exams_dir: Path = DEFAULT_EXAMS_DIR) -> int:
    """Add any finished report that is not in the profile yet.

    Scores are normally saved the moment a report is made. This catches the rest:
    tests taken before the profile existed, or while it could not be written.
    Existing rows are never touched. Returns how many were added.
    """
    exams_dir = Path(exams_dir)
    if not exams_dir.is_dir():
        return 0
    known = {(r["session_id"], r["exam_id"]) for r in list_scores(path)}
    added = 0
    for report_file in sorted(exams_dir.glob("*/*/report.json")):
        exam_dir = report_file.parent
        key = (exam_dir.parent.name, exam_dir.name)
        if key in known:
            continue
        try:
            report = json.loads(report_file.read_text())
            if not isinstance(report.get("overall_band"), (int, float)):
                continue                       # "not enough answered" and failed reports have no scores
            meta = {}
            if (exam_dir / "exam.json").exists():
                meta = json.loads((exam_dir / "exam.json").read_text())
            log = SimpleNamespace(
                session_id=key[0],
                exam_id=key[1],
                started_at=meta.get("started_at") or report_file.stat().st_mtime,
                length=meta.get("length", "full"),     # tests saved before lengths existed were full
                finished=meta.get("finished", report.get("finished", False)),
            )
            if record_exam(report, log, path):
                added += 1
        except (OSError, ValueError):
            logger.warning("skipping unreadable test %s", exam_dir, exc_info=True)
    return added


def overview(path: Path = DEFAULT_PATH, exams_dir: Path = DEFAULT_EXAMS_DIR) -> dict:
    sync_from_exams(path, exams_dir)
    rows = list_scores(path)
    return {"summary": summarize(rows), "history": rows}
