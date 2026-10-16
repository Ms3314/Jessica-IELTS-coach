"""The post-test report: analyse the saved answers, then estimate IELTS bands.

    saved audio --careful re-transcription--> words + timestamps
        --> fluency / vocabulary / grammar / pitch numbers   (app/exam/analysis.py)
        --> ONE structured model call: numbers + transcript + band descriptors
        --> checked, completed in code, saved as report.json

The model never hears audio; it judges numbers and text. Two things are done in
code instead of trusting the model: the overall band (the average of the four
criteria, rounded to the nearest half band, as in the real test) and quote
checking (every quote the model offers as evidence must really appear in what
the candidate said, otherwise it is dropped).

These are ESTIMATES. Pronunciation especially is inferred from indicators, not
assessed by ear.
"""
import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.exam import analysis as ea
from app.exam.ielts import EXAMS_DIR, ExamLog

logger = logging.getLogger("speakIelts.report")

# Analyses running right now in this server, so a page opened later can tell
# "still working" apart from "never started / the server restarted in between".
_running: set[tuple[str, str]] = set()
_running_lock = threading.Lock()


def is_analysing(session_id: str, exam_id: str) -> bool:
    with _running_lock:
        return (session_id, exam_id) in _running

MIN_ANSWERED_TURNS = 3


# --- what the model must return --------------------------------------------------------

class Criterion(BaseModel):
    band: float = Field(description="Estimated band 0-9 in steps of 0.5")
    summary: str = Field(description="Two or three sentences explaining the band")
    strengths: list[str] = Field(description="One to three specific strengths")
    improvements: list[str] = Field(description="One to three specific, actionable improvements")
    evidence: list[str] = Field(
        description="Two to four SHORT verbatim quotes from the candidate that support the judgement"
    )


class LLMReport(BaseModel):
    fluency_coherence: Criterion
    lexical_resource: Criterion
    grammatical_range_accuracy: Criterion
    pronunciation: Criterion
    top_priorities: list[str] = Field(description="The three most valuable things to practise, most important first")
    examiner_note: str = Field(description="A short, encouraging closing paragraph addressed to the candidate")


CRITERIA = ("fluency_coherence", "lexical_resource", "grammatical_range_accuracy", "pronunciation")

# Condensed paraphrase of the public IELTS Speaking band descriptors.
BAND_DESCRIPTORS = """\
FLUENCY AND COHERENCE
9: fluent; rare repetition/self-correction; hesitation only to think of ideas; fully coherent, topics fully developed.
8: fluent; occasional repetition/self-correction; hesitation mostly about content; topics developed coherently.
7: speaks at length without noticeable effort; some language-related hesitation/repetition; flexible use of connectives and discourse markers.
6: willing to speak at length but loses coherence at times from repetition, self-correction or hesitation; connectives used but not always well.
5: keeps going but with repetition, self-correction or slow speech; overuses some connectives; simple speech is fluent, complex speech causes problems.
4: noticeable pauses, slow speech, frequent repetition/self-correction; links basic sentences with simple connectives; coherence breaks down.
LEXICAL RESOURCE
9: full flexibility and precision on all topics; natural, accurate idiom.
8: wide resource used readily to convey precise meaning; skilful use of uncommon/idiomatic items with occasional slips; paraphrases effectively.
7: uses vocabulary flexibly on varied topics; some uncommon/idiomatic items with awareness of style and collocation (some inappropriate choices); paraphrases effectively.
6: enough vocabulary to talk at length and be clear despite some inappropriacy; usually paraphrases successfully.
5: manages familiar and unfamiliar topics but with limited flexibility; paraphrase attempts have mixed success.
4: basic meaning on familiar topics only; frequent word-choice errors; rarely paraphrases.
GRAMMATICAL RANGE AND ACCURACY
9: full flexibility and accuracy across structures; rare slips.
8: wide range used flexibly; most sentences error-free; occasional non-systematic errors.
7: range of complex structures with some flexibility; frequent error-free sentences though some mistakes persist.
6: mix of short and complex forms with limited flexibility; errors frequent in complex structures but rarely impede understanding.
5: basic forms fairly accurate; limited complex structures, usually with errors that may cause some difficulty.
4: basic forms with reasonable accuracy; very few complex structures; frequent errors that can cause misunderstanding.
PRONUNCIATION
9: full range of features precisely and subtly; effortless to understand.
8: wide range of features with sustained flexible use; occasional lapses; easy to understand; accent has minimal effect.
7: all positive features of band 6 and some of band 8.
6: a range of features with variable control; generally understood though some mispronunciation reduces clarity at times.
5: all positive features of band 4 and some of band 6.
4: limited range of features; frequent mispronunciation causes difficulty for the listener.
"""

PRONUNCIATION_CAUTION = """\
You CANNOT hear the audio. For pronunciation you only have indicators (word-recognition confidence and pitch
variation). Be conservative: unless the indicators clearly point elsewhere, estimate close to the other three
criteria, never above band 7.5, and say plainly in the summary that this is an estimate from indicators. Low
recognition confidence can come from rare words or noise, not only unclear speech. An accent is not an error;
only comment on intelligibility, intonation and stress. When you mention this limit to the candidate
(including in the examiner note), say pronunciation was "estimated from speech indicators", never that the
audio was missing: their answers WERE recorded and measured."""


# --- helpers -----------------------------------------------------------------------------

def round_band(x: float) -> float:
    """Nearest half band, rounding .25 up to .5 and .75 up to the next whole band (IELTS rule)."""
    return math.floor(x * 2 + 0.5) / 2


def clamp_band(x) -> float:
    try:
        return min(9.0, max(0.0, round_band(float(x))))
    except (TypeError, ValueError):
        return 0.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", text.lower())).strip()


def verify_quotes(quotes: list[str], candidate_text: str) -> list[str]:
    """Keep only quotes that genuinely occur in what the candidate said."""
    haystack = _norm(candidate_text)
    return [q for q in quotes if _norm(q) and _norm(q) in haystack]


def reliability(speech_seconds: float, n_words: int) -> str:
    if speech_seconds < 90 or n_words < 150:
        return "low"
    if speech_seconds < 240 or n_words < 400:
        return "medium"
    return "good"


# --- analysis ------------------------------------------------------------------------------

def analyse_exam(
    log: ExamLog,
    transcribe: Callable = ea.transcribe_precise,
    prosody: Callable = ea.prosody_metrics,
) -> dict:
    """Careful second pass over every saved answer. Returns (and saves) the analysis."""
    raws, wordlists, per_turn, caveats = [], [], [], []
    for turn in log.turns:
        texts = []
        for name in turn.audio:
            path = log.dir / name
            try:
                result = transcribe(path)
            except Exception as e:  # fall back to the live transcript, without word timing
                logger.warning("precise transcription failed for %s: %s", name, e)
                caveats.append(f"Detailed timing is missing for answer {turn.n}.")
                texts.append(turn.answer)
                continue
            words = ea.flatten_words(result)
            wordlists.extend(words)
            raws.append(ea.fluency_raw(words))     # per clip: pauses never span two clips
            texts.append((result.get("text") or "").strip())
        per_turn.append({"n": turn.n, "part": turn.part, "question": turn.question,
                         "answer": " ".join(t for t in texts if t) or turn.answer,
                         "duration_s": round(turn.duration_s, 1)})

    candidate_text = " ".join(t["answer"] for t in per_turn)
    fluency = ea.derive_fluency(ea.combine_fluency(raws)) if raws else {"enough_speech": False}
    wavs = [log.dir / n for t in log.turns for n in t.audio]
    try:
        pitch = prosody(wavs)
    except Exception as e:
        logger.warning("prosody analysis failed: %s", e)
        pitch = {"enough_voiced_speech": False}
        caveats.append("Pitch analysis was unavailable.")

    analysis = {
        "turns": per_turn,
        "candidate_text": candidate_text,
        "fluency": fluency,
        "lexical": ea.lexical_metrics(candidate_text),
        "grammar": ea.grammar_stats(candidate_text),
        "pronunciation_indicators": {
            "word_confidence": ea.confidence_summary(wordlists),
            "pitch": pitch,
        },
        "caveats": caveats,
    }
    (log.dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    return analysis


# --- the report ------------------------------------------------------------------------------

def build_prompt(log: ExamLog, analysis: dict) -> str:
    dialogue = []
    for t in analysis["turns"]:
        dialogue.append(f"[Part {t['part']}] Examiner: {t['question']}\nCandidate: {t['answer'] or '[no answer]'}")
    metrics = {k: analysis[k] for k in ("fluency", "lexical", "grammar", "pronunciation_indicators")}
    return (
        "You are an experienced IELTS Speaking examiner marking a MOCK test. Estimate a band score "
        "(0-9, steps of 0.5) for each of the four criteria, using the band descriptors, the transcript "
        "and the measured indicators below. Be honest and calibrated: do not inflate bands, and do not "
        "invent problems the evidence does not show.\n\n"
        f"BAND DESCRIPTORS\n{BAND_DESCRIPTORS}\n"
        f"{PRONUNCIATION_CAUTION}\n\n"
        "Evidence rules: every item in `evidence` must be a short VERBATIM quote from the candidate's "
        "own words in the transcript. Improvements must be specific and actionable (name the habit, show a "
        "better way to say it). Fluency numbers are measured from timestamps and are reliable; vocabulary "
        "numbers are word-frequency statistics; grammar numbers are rough pattern counts, so judge "
        "accuracy from the transcript itself (the transcript is automatic and may have tidied up errors).\n\n"
        f"MEASURED INDICATORS (JSON)\n{json.dumps(metrics, indent=2)}\n\n"
        f"TRANSCRIPT\n" + "\n\n".join(dialogue)
    )


def generate_report(llm, log: ExamLog, analysis: dict) -> dict:
    """One structured model call, then checked and completed in code."""
    raw = llm.with_structured_output(LLMReport).invoke(build_prompt(log, analysis))
    if isinstance(raw, dict):
        raw = LLMReport(**raw)
    candidate_text = analysis["candidate_text"]

    report: dict = {}
    dropped = 0
    for key in CRITERIA:
        c = getattr(raw, key)
        kept = verify_quotes(c.evidence, candidate_text)
        dropped += len(c.evidence) - len(kept)
        report[key] = {
            "band": clamp_band(c.band),
            "summary": c.summary,
            "strengths": c.strengths,
            "improvements": c.improvements,
            "evidence": kept,
        }
    # pronunciation is inferred from indicators, so it is capped and labelled as such
    report["pronunciation"]["band"] = min(report["pronunciation"]["band"], 7.5)
    report["pronunciation"]["basis"] = "estimated from indicators (word confidence, pitch); not heard by an examiner"

    bands = [report[k]["band"] for k in CRITERIA]
    fl = analysis["fluency"]
    caveats = list(analysis.get("caveats", []))
    rel = reliability(fl.get("speech_seconds", 0.0), fl.get("n_words", 0))
    if rel == "low":
        caveats.append("There was very little speech to judge, so these bands are unreliable.")
    if dropped:
        caveats.append(f"{dropped} quote(s) that could not be found in your answers were left out.")
    caveats.append("Bands are practice estimates, not official scores; pronunciation is the least certain.")

    report.update({
        "overall_band": round_band(sum(bands) / len(bands)),
        "reliability": rel,
        "top_priorities": raw.top_priorities[:3],
        "examiner_note": raw.examiner_note,
        "caveats": caveats,
        "exam_id": log.exam_id,
        "finished": log.finished,
    })
    return report


# --- orchestration -----------------------------------------------------------------------------

def run_exam_analysis(log: ExamLog, llm, events=None, store=None, **analysis_kwargs) -> dict:
    """Analyse a finished (or abandoned) exam and save the report. Never raises.

    `store(report, log)`, if given, saves the scores to the user's profile.
    """
    key = (log.session_id, log.exam_id)
    with _running_lock:
        if key in _running:                       # already being analysed: don't start a twin
            return {"error": "already analysing", "exam_id": log.exam_id}
        _running.add(key)
    try:
        return _analyse_and_save(log, llm, events, store, **analysis_kwargs)
    finally:
        with _running_lock:
            _running.discard(key)


def _analyse_and_save(log: ExamLog, llm, events, store, **analysis_kwargs) -> dict:
    def emit(payload):
        if events is not None:
            try:
                events.on_exam(payload)
            except Exception:
                pass

    answered = sum(1 for t in log.turns if t.answer)
    if answered < MIN_ANSWERED_TURNS:
        result = {"insufficient": True, "exam_id": log.exam_id,
                  "message": "The test ended before enough questions were answered to give feedback."}
        (log.dir / "report.json").write_text(json.dumps(result, indent=2))
        emit({"event": "report", "report": result})
        return result
    emit({"event": "analysing"})
    try:
        analysis = analyse_exam(log, **analysis_kwargs)
        report = generate_report(llm, log, analysis)
        report["profile_saved"] = False
        if store is not None:
            try:
                report["profile_saved"] = bool(store(report, log))
            except Exception:
                logger.warning("could not save the scores to the profile", exc_info=True)
        (log.dir / "report.json").write_text(json.dumps(report, indent=2))
        (log.dir / "analysis_error.json").unlink(missing_ok=True)
        emit({"event": "report", "report": report})
        return report
    except Exception as e:
        logger.warning("exam report failed", exc_info=True)
        failed = {"error": str(e), "exam_id": log.exam_id}
        try:
            (log.dir / "analysis_error.json").write_text(json.dumps(failed, indent=2))   # a later visit can say why
        except OSError:
            pass
        emit({"event": "report_failed", "message": str(e)})
        return failed


# --- reading saved exams (for the REST API) ------------------------------------------------------

def list_exams(session_id: str, root: Path = EXAMS_DIR) -> list[dict]:
    base = Path(root) / session_id
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir(), reverse=True):
        meta = d / "exam.json"
        if meta.exists():
            data = json.loads(meta.read_text())
            out.append({"exam_id": d.name, "started_at": data["started_at"], "finished": data["finished"],
                        "turns": len(data["turns"]), "has_report": (d / "report.json").exists()})
    return out


def load_exam(session_id: str, exam_id: str, root: Path = EXAMS_DIR) -> Optional[dict]:
    d = Path(root) / session_id / exam_id
    if not (d / "exam.json").exists():
        return None
    out = json.loads((d / "exam.json").read_text())
    for name in ("analysis", "report"):
        p = d / f"{name}.json"
        out[name] = json.loads(p.read_text()) if p.exists() else None
    err = d / "analysis_error.json"
    out["analysis_error"] = json.loads(err.read_text()).get("error") if err.exists() and out["report"] is None else None
    out["analysing"] = is_analysing(session_id, exam_id)
    return out
