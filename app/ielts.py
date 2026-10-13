"""IELTS Speaking mock test: the fixed structure of the exam, and its record.

The exam is deterministic code, not a free-running chat: this module decides
WHAT happens and WHEN (which part, how many questions, how long Part 2 lasts);
the language model (app/examiner.py) only phrases the questions.

    intro -> Part 1 (3 familiar topics) -> Part 2 (cue card: 1 min prep,
    up to 2 min talk, one rounding-off question) -> Part 3 (abstract
    discussion) -> end

Every answer is saved (audio + transcript) so it can be analysed after the
test, when speed no longer matters.
"""
import json
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from app import config
from app.config import PROJECT_ROOT

SAMPLE_RATE = 16000
EXAMS_DIR = PROJECT_ROOT / "data" / "exams"

# --- exam shape ----------------------------------------------------------------
# Parts 1 and 3 are run against the clock, like the real test; the examiner (the
# model) chooses the topics and questions on its own. Only the structure and the
# timings are fixed, and the timings come in two lengths: the full test, and a
# shorter one for a quick practice.
OVERHEAD_S = 60.0              # introduction + closing, on top of the four timed parts


def _describe_prep(seconds: float) -> tuple[str, str]:
    """How the examiner says the preparation time: (phrase, word that proves it)."""
    if seconds == 60:
        return "one minute", "minute"
    if seconds % 60 == 0:
        n = int(seconds // 60)
        return f"{n} minutes", "minute"
    return f"{int(seconds)} seconds", "seconds"


def _describe_talk(seconds: float) -> tuple[str, str]:
    if seconds == 120:
        return "one to two minutes", "minute"
    if seconds == 60:
        return "up to one minute", "minute"
    if seconds % 60 == 0:
        return f"up to {int(seconds // 60)} minutes", "minute"
    return f"up to {int(seconds)} seconds", "seconds"


@dataclass(frozen=True)
class Timing:
    """How long each part of one test may take."""
    name: str
    part1_s: float
    part2_prep_s: float
    part2_talk_s: float
    part3_s: float

    @property
    def enough_s(self) -> float:
        """A Part 2 talk shorter than this earns one invitation to say more."""
        return self.part2_talk_s / 2

    @property
    def record_cap_s(self) -> float:
        """Recording limit for the Part 2 talk: the time limit plus a small margin."""
        return self.part2_talk_s + 5

    @property
    def total_s(self) -> float:
        return self.part1_s + self.part2_prep_s + self.part2_talk_s + self.part3_s + OVERHEAD_S

    def phrases(self) -> dict:
        """The wording of this test's times, for the examiner's Part 2 lines."""
        prep, prep_word = _describe_prep(self.part2_prep_s)
        talk, talk_word = _describe_talk(self.part2_talk_s)
        return {"prep_phrase": prep, "prep_word": prep_word, "talk_phrase": talk, "talk_word": talk_word}


TIMINGS = {
    "full": Timing("full", config.IELTS_PART1_S, 60.0, 120.0, config.IELTS_PART3_S),   # about 14 minutes
    "short": Timing("short", 120.0, 30.0, 60.0, 120.0),                                # about 6 minutes
}
DEFAULT_LENGTH = "full"


def timing_for(length: str | None) -> Timing:
    """The timings for a length name; anything unknown means the full test."""
    return TIMINGS.get(length or DEFAULT_LENGTH, TIMINGS[DEFAULT_LENGTH])


# The full test's values, kept as plain constants for code that doesn't pick a length.
PART1_LIMIT_S = TIMINGS["full"].part1_s
PART3_LIMIT_S = TIMINGS["full"].part3_s
PART2_PREP_S = TIMINGS["full"].part2_prep_s
PART2_TALK_MAX_S = TIMINGS["full"].part2_talk_s
PART2_RECORD_CAP_S = TIMINGS["full"].record_cap_s
PART2_PAUSE_S = 4.0            # a long pause is allowed mid-monologue (automatic recording only)
PART2_ENOUGH_S = TIMINGS["full"].enough_s
PART2_IDLE_S = 20.0            # silence after "please start" before giving up (automatic recording)
ANSWER_RECORD_CAP_S = 45.0     # Part 1 / Part 3 answers (automatic recording)
MANUAL_ANSWER_CAP_S = 90.0     # Part 1 / Part 3 answers when the candidate records by hand
MANUAL_WAIT_S = 45.0           # how long to wait for the candidate to press Record
NO_ANSWER_TIMEOUTS = 1         # silent recording windows before moving on

# Step kinds, in the order they happen.
INTRO = "intro"
P1_SECTION = "p1_section"  # timed question-and-answer loop
P2_SETUP = "p2_setup"      # announce Part 2, present the cue card
P2_PREP = "p2_prep"        # one silent minute
P2_TALK = "p2_talk"        # the long turn
P2_ROUND = "p2_round"      # one short rounding-off question
P3_INTRO = "p3_intro"
P3_SECTION = "p3_section"  # timed discussion loop
END = "end"


@dataclass
class Step:
    kind: str
    part: int                       # 0 = intro/end, 1, 2, 3
    limit_s: float = 0.0            # time budget for the timed sections


def build_steps(length: str | None = None) -> list[Step]:
    """The whole test, in order, with the time budgets of the chosen length."""
    t = timing_for(length)
    return [
        Step(INTRO, 0),
        Step(P1_SECTION, 1, t.part1_s),
        Step(P2_SETUP, 2), Step(P2_PREP, 2), Step(P2_TALK, 2), Step(P2_ROUND, 2),
        Step(P3_INTRO, 3),
        Step(P3_SECTION, 3, t.part3_s),
        Step(END, 0),
    ]


def schedule(length: str | None = None) -> list[dict]:
    """The timings of the chosen length, in the form the UI wants."""
    t = timing_for(length)
    return [
        {"part": 1, "limit_s": t.part1_s},
        {"part": 2, "prep_s": t.part2_prep_s, "talk_s": t.part2_talk_s},
        {"part": 3, "limit_s": t.part3_s},
    ]


def recent_cue_titles(root: Path = EXAMS_DIR, limit: int = 10) -> list[str]:
    """Titles of the cue cards from the most recent saved tests (newest first), so a new
    test can avoid repeating them."""
    found = []
    for f in Path(root).glob("*/*/exam.json"):
        try:
            data = json.loads(f.read_text())
            title = (data.get("cue_card") or {}).get("title")
            if title:
                found.append((data.get("started_at", 0), title))
        except Exception:
            continue
    return [t for _, t in sorted(found, reverse=True)[:limit]]


# --- the record of one exam -----------------------------------------------------

@dataclass
class Turn:
    n: int
    part: int
    question: str
    answer: str = ""                # live (fast) transcription
    audio: list[str] = field(default_factory=list)   # wav file names, in order
    duration_s: float = 0.0         # recorded audio, including pauses
    no_answer: bool = False


def write_wav(path: Path, clip: np.ndarray) -> None:
    """Save a 16 kHz mono int16 clip as a standard WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.asarray(clip, dtype=np.int16).tobytes())


class ExamLog:
    """Everything recorded during one exam, saved under data/exams/<session>/<id>/."""

    def __init__(self, session_id: str, exam_id: Optional[str] = None, root: Path = EXAMS_DIR):
        self.session_id = session_id
        self.exam_id = exam_id or time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / session_id / self.exam_id
        self.turns: list[Turn] = []
        self.cue_card: Optional[dict] = None
        self.started_at = time.time()
        self.finished = False
        self.length = DEFAULT_LENGTH     # "full" or "short"

    # -- recording ---------------------------------------------------------------
    def start_turn(self, part: int, question: str) -> Turn:
        turn = Turn(n=len(self.turns) + 1, part=part, question=question)
        self.turns.append(turn)
        return turn

    def add_clip(self, turn: Turn, clip: np.ndarray, text: str) -> None:
        """Attach one recorded clip (and its live transcript) to a turn."""
        name = f"turn{turn.n:02d}" + (f"_{len(turn.audio) + 1}" if turn.audio else "") + ".wav"
        write_wav(self.dir / name, clip)
        turn.audio.append(name)
        turn.duration_s += len(clip) / SAMPLE_RATE
        turn.answer = f"{turn.answer} {text}".strip()

    # -- persistence -------------------------------------------------------------
    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "exam.json"
        path.write_text(json.dumps({
            "session_id": self.session_id,
            "exam_id": self.exam_id,
            "started_at": self.started_at,
            "finished": self.finished,
            "length": self.length,
            "cue_card": self.cue_card,
            "turns": [asdict(t) for t in self.turns],
        }, indent=2))
        return path

    @classmethod
    def load(cls, session_id: str, exam_id: str, root: Path = EXAMS_DIR) -> "ExamLog":
        data = json.loads((Path(root) / session_id / exam_id / "exam.json").read_text())
        log = cls(session_id, exam_id, root)
        log.started_at = data["started_at"]
        log.finished = data["finished"]
        log.length = data.get("length", DEFAULT_LENGTH)     # tests saved before lengths existed were full
        log.cue_card = data["cue_card"]
        log.turns = [Turn(**t) for t in data["turns"]]
        return log

    def transcript(self) -> str:
        """The examiner/candidate dialogue as plain text (for the report prompt)."""
        lines = []
        for t in self.turns:
            lines.append(f"Examiner (Part {t.part}): {t.question}")
            lines.append(f"Candidate: {t.answer or '[no answer]'}")
        return "\n".join(lines)
