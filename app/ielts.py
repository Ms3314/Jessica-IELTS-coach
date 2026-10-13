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
# timings are fixed.
PART1_LIMIT_S = config.IELTS_PART1_S
PART3_LIMIT_S = config.IELTS_PART3_S
PART2_PREP_S = 60.0            # silent thinking time with the cue card
PART2_TALK_MAX_S = 120.0       # the examiner stops the candidate here
PART2_RECORD_CAP_S = 125.0     # recorder cap: max talk time + a small margin
PART2_PAUSE_S = 4.0            # a long pause is allowed mid-monologue
PART2_ENOUGH_S = 60.0          # shorter than this and the examiner asks for more
PART2_IDLE_S = 20.0            # silence after "please start" before giving up
ANSWER_RECORD_CAP_S = 45.0     # Part 1 / Part 3 answers
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


def build_steps() -> list[Step]:
    """The whole test, in order."""
    return [
        Step(INTRO, 0),
        Step(P1_SECTION, 1, PART1_LIMIT_S),
        Step(P2_SETUP, 2), Step(P2_PREP, 2), Step(P2_TALK, 2), Step(P2_ROUND, 2),
        Step(P3_INTRO, 3),
        Step(P3_SECTION, 3, PART3_LIMIT_S),
        Step(END, 0),
    ]


def schedule() -> list[dict]:
    """The timings, in the form the UI wants (sent when the test starts)."""
    return [
        {"part": 1, "limit_s": PART1_LIMIT_S},
        {"part": 2, "prep_s": PART2_PREP_S, "talk_s": PART2_TALK_MAX_S},
        {"part": 3, "limit_s": PART3_LIMIT_S},
    ]


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
