"""Turn recorded exam answers into numbers the report can be built from.

Everything here is measured AFTER the test from the saved audio, so speed does
not matter and nothing runs in the live conversation. The functions that do the
maths are pure (words in, numbers out) so they can be tested with synthetic
data; only `transcribe_precise` and `prosody_metrics` touch real audio/models.

These are indicators, not an examiner's judgement:
  * fluency  - reliable: comes straight from word timestamps
  * lexical  - fairly reliable: word-frequency statistics
  * grammar  - rough counts only; the LLM judges accuracy from the transcript
  * prosody / word confidence - pronunciation INDICATORS, not a score
"""
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Prompting Whisper with fillers makes it keep "um/uh" instead of tidying them away.
FILLER_PROMPT = "Um, so, uh, I think, you know, like, well, er, I mean, hmm, let me think."

PAUSE_MIN_S = 0.3        # a gap shorter than this is just articulation
PAUSE_SHORT_S = 0.5
PAUSE_LONG_S = 1.0
PAUSE_VERY_LONG_S = 2.0

HESITATIONS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm", "mmm", "eh"}
DISCOURSE_FILLERS = ("you know", "i mean", "kind of", "sort of", "like")


@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float = 1.0

    @property
    def norm(self) -> str:
        return re.sub(r"[^a-z']", "", self.word.lower())


def flatten_words(result: dict) -> list[Word]:
    """Words (with timestamps and confidence) out of an mlx_whisper result."""
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(Word(w["word"], float(w["start"]), float(w["end"]), float(w.get("probability", 1.0))))
    return words


# --- fluency ------------------------------------------------------------------------

def fluency_raw(words: list[Word]) -> dict:
    """Raw counts and times for one answer (sum these across answers, then `derive`)."""
    raw = {
        "n_words": 0, "speech_time": 0.0, "pause_time": 0.0,
        "pauses_short": 0, "pauses_long": 0, "pauses_very_long": 0,
        "mid_clause_pauses": 0, "longest_pause": 0.0,
        "hesitations": 0, "discourse_fillers": 0, "repetitions": 0,
        "runs": 0,
    }
    if not words:
        return raw
    raw["n_words"] = len(words)
    raw["speech_time"] = max(0.0, words[-1].end - words[0].start)
    raw["runs"] = 1

    for prev, nxt in zip(words, words[1:]):
        gap = nxt.start - prev.end
        if gap < PAUSE_MIN_S:
            continue
        raw["pause_time"] += gap
        raw["longest_pause"] = max(raw["longest_pause"], gap)
        if gap >= PAUSE_VERY_LONG_S:
            raw["pauses_very_long"] += 1
        if gap >= PAUSE_LONG_S:
            raw["pauses_long"] += 1
        if gap >= PAUSE_SHORT_S:
            raw["pauses_short"] += 1
            raw["runs"] += 1
            if not prev.word.strip().endswith((".", "?", "!", ",", ";", ":")):
                raw["mid_clause_pauses"] += 1   # hesitating mid-phrase, not between ideas

    norms = [w.norm for w in words]
    raw["hesitations"] = sum(1 for n in norms if n in HESITATIONS)
    text = " " + " ".join(norms) + " "
    raw["discourse_fillers"] = sum(text.count(f" {f} ") for f in DISCOURSE_FILLERS)
    # "the the" / "I think I think": immediate repetition of a word or two-word phrase
    reps = sum(1 for a, b in zip(norms, norms[1:]) if a and a == b)
    reps += sum(1 for i in range(len(norms) - 3)
                if norms[i] != norms[i + 1] and norms[i:i + 2] == norms[i + 2:i + 4])
    raw["repetitions"] = reps
    return raw


def combine_fluency(raws: Iterable[dict]) -> dict:
    total = {}
    for r in raws:
        for k, v in r.items():
            total[k] = max(total.get(k, 0.0), v) if k == "longest_pause" else total.get(k, 0) + v
    return total


def derive_fluency(raw: dict) -> dict:
    """Human-meaningful fluency measures from the raw counts."""
    n, t, p = raw.get("n_words", 0), raw.get("speech_time", 0.0), raw.get("pause_time", 0.0)
    if n == 0 or t <= 0:
        return {"n_words": n, "enough_speech": False}
    articulating = max(t - p, 1e-6)
    per100 = lambda x: round(100.0 * x / n, 1)  # noqa: E731
    return {
        "n_words": n,
        "enough_speech": n >= 30,
        "speech_seconds": round(t, 1),
        "words_per_minute": round(60.0 * n / t, 1),
        "articulation_rate_wpm": round(60.0 * n / articulating, 1),   # speed while actually talking
        "pause_share": round(p / t, 3),                                # fraction of time spent paused
        "pauses_over_0_5s": raw["pauses_short"],
        "pauses_over_1s": raw["pauses_long"],
        "pauses_over_2s": raw["pauses_very_long"],
        "mid_clause_pauses": raw["mid_clause_pauses"],
        "longest_pause_s": round(raw["longest_pause"], 1),
        "mean_run_length_words": round(n / max(raw["runs"], 1), 1),    # words between pauses
        "hesitations_per_100_words": per100(raw["hesitations"]),
        "discourse_fillers_per_100_words": per100(raw["discourse_fillers"]),
        "repetitions_per_100_words": per100(raw["repetitions"]),
    }


# --- vocabulary ----------------------------------------------------------------------

_TOKEN = re.compile(r"[a-zA-Z']+")

# Function words are allowed to repeat; basic content words ("good", "thing") are what
# examiners notice a candidate leaning on.
_FUNCTION_WORDS = frozenset(
    "the a an and or but so of in on at to for with from by as is are was were be been am do does did "
    "have has had it its this that these those i you he she we they me him her us them my your his our their "
    "not no yes if then than there here what who which when where how also just very really".split()
)


def _zipf(word: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(word, "en")


def mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio: vocabulary variety that doesn't punish long answers."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return sum(ratios) / len(ratios)


def lexical_metrics(text: str) -> dict:
    tokens = [t.lower().strip("'") for t in _TOKEN.findall(text)]
    tokens = [t for t in tokens if t]
    if len(tokens) < 20:
        return {"n_tokens": len(tokens), "enough_speech": False}
    zipfs = {t: _zipf(t) for t in set(tokens)}
    less_common = [t for t in tokens if 0 < zipfs[t] < 4.0]       # beyond everyday words
    rare = [t for t in tokens if 0 < zipfs[t] < 3.0]
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    # content words repeated a lot
    overused = sorted(((t, c) for t, c in counts.items() if c >= 4 and t not in _FUNCTION_WORDS),
                      key=lambda x: -x[1])[:8]
    notable = sorted({t for t in tokens if 1.5 <= zipfs[t] < 3.8 and len(t) > 4})[:20]
    return {
        "n_tokens": len(tokens),
        "enough_speech": True,
        "distinct_words": len(set(tokens)),
        "mattr": round(mattr(tokens), 3),
        "less_common_word_share": round(len(less_common) / len(tokens), 3),
        "rare_word_share": round(len(rare) / len(tokens), 3),
        "average_word_length": round(sum(map(len, tokens)) / len(tokens), 2),
        "overused_words": [{"word": w, "count": c} for w, c in overused],
        "notable_words_used": notable,
    }


# --- grammar (rough counts; the LLM judges accuracy) -----------------------------------

_SUBORDINATORS = ("because", "although", "though", "which", "who", "whom", "whose", "that", "if", "when",
                  "while", "whereas", "since", "unless", "until", "after", "before", "so that")
_PERFECT = re.compile(r"\b(have|has|had|having)\s+(\w+ed|been|done|gone|seen|made|taken|known|become|had)\b", re.I)
_PASSIVE = re.compile(r"\b(is|are|was|were|be|been|being)\s+(\w+ed|made|done|given|taken|seen|known|built|written)\b", re.I)
_CONDITIONAL = re.compile(r"\bif\b[^.?!]*\b(would|will|could|might|had|were)\b", re.I)
_MODALS = re.compile(r"\b(would|could|should|might|must|may|can't|couldn't|wouldn't)\b", re.I)


def grammar_stats(text: str) -> dict:
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if len(s.split()) >= 2]
    if len(sentences) < 3:
        return {"n_sentences": len(sentences), "enough_speech": False}
    lens = [len(s.split()) for s in sentences]
    complex_n = sum(1 for s in sentences if any(re.search(rf"\b{m}\b", s, re.I) for m in _SUBORDINATORS))
    return {
        "enough_speech": True,
        "n_sentences": len(sentences),
        "average_sentence_length": round(sum(lens) / len(lens), 1),
        "complex_sentence_share": round(complex_n / len(sentences), 2),
        "conditionals": len(_CONDITIONAL.findall(text)),
        "perfect_tenses": len(_PERFECT.findall(text)),
        "passives": len(_PASSIVE.findall(text)),
        "modal_verbs": len(_MODALS.findall(text)),
        "note": "rough pattern counts; sentence boundaries come from the transcript's punctuation",
    }


# --- pronunciation indicators -----------------------------------------------------------

def confidence_summary(words: list[Word], threshold: float = 0.5) -> dict:
    """How sure the recogniser was of each word. Low-confidence words MAY be unclear
    speech, but can also be names, rare words or noise."""
    real = [w for w in words if w.norm and w.norm not in HESITATIONS]
    if not real:
        return {"n_words": 0}
    low = [w for w in real if w.prob < threshold]
    counts: dict[str, int] = {}
    for w in low:
        counts[w.norm] = counts.get(w.norm, 0) + 1
    return {
        "n_words": len(real),
        "mean_confidence": round(float(np.mean([w.prob for w in real])), 3),
        "low_confidence_share": round(len(low) / len(real), 3),
        "low_confidence_words": sorted(counts, key=lambda k: -counts[k])[:15],
    }


def semitones(f0_hz: np.ndarray) -> np.ndarray:
    return 12.0 * np.log2(np.asarray(f0_hz, dtype=float) / 100.0)


def pitch_summary(f0_hz: np.ndarray) -> dict:
    """Intonation indicators from voiced pitch samples (Hz)."""
    f0 = np.asarray(f0_hz, dtype=float)
    f0 = f0[(f0 > 50) & (f0 < 600)]
    if len(f0) < 50:
        return {"enough_voiced_speech": False}
    st = semitones(f0)
    lo, hi = np.percentile(st, [5, 95])
    return {
        "enough_voiced_speech": True,
        "median_pitch_hz": round(float(np.median(f0)), 1),
        "pitch_range_semitones": round(float(hi - lo), 1),
        "pitch_variation_semitones": round(float(np.std(st)), 2),
        "note": "variation under ~2 semitones usually sounds monotone; 3-5 is typical of expressive speech",
    }


def prosody_metrics(wav_paths: Iterable[Path]) -> dict:
    """Pitch statistics across all answers (uses Praat via parselmouth)."""
    import parselmouth

    chunks = []
    for path in wav_paths:
        sound = parselmouth.Sound(str(path))
        pitch = sound.to_pitch(pitch_floor=75, pitch_ceiling=500)
        values = pitch.selected_array["frequency"]
        chunks.append(values[values > 0])
    if not chunks:
        return {"enough_voiced_speech": False}
    return pitch_summary(np.concatenate(chunks))


# --- second-pass transcription --------------------------------------------------------------

def transcribe_precise(wav_path: Path, model: Optional[str] = None) -> dict:
    """Careful re-transcription of a saved answer, with word timestamps and confidence."""
    import os

    import mlx_whisper
    import soundfile as sf

    from app.speech import asr

    audio, rate = sf.read(str(wav_path), dtype="float32")
    if rate != 16000:
        raise ValueError(f"expected 16 kHz audio, got {rate}")
    return mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model or os.getenv("EXAM_ASR_MODEL", asr.MODEL),
        language="en",
        word_timestamps=True,
        initial_prompt=FILLER_PROMPT,
        condition_on_previous_text=False,
        temperature=0,
    )
