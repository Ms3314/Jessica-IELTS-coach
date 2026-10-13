"""Exam measurement maths — synthetic words/audio, no models.

    python tests/test_exam_analysis.py
"""
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app import exam_analysis as ea
from app.exam_analysis import Word


def _words(spec):
    """spec: list of (text, start, end[, prob])."""
    return [Word(*s) for s in spec]


def test_flatten_words_reads_whisper_segments():
    result = {"segments": [{"words": [{"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.9},
                                      {"word": " there", "start": 0.5, "end": 0.9, "probability": 0.8}]}]}
    words = ea.flatten_words(result)
    assert [w.word for w in words] == [" Hello", " there"] and words[1].prob == 0.8
    assert ea.flatten_words({}) == []


def test_pauses_are_counted_by_length_and_position():
    words = _words([
        ("I", 0.0, 0.2), ("went", 0.25, 0.5),      # gap 0.05: normal
        ("to", 1.1, 1.3),                          # gap 0.6 after "went" (mid-clause)
        ("school.", 1.35, 1.8),
        ("Then", 3.3, 3.5),                        # gap 1.5 after a full stop (between ideas)
        ("I", 6.1, 6.2),                           # gap 2.6 mid-clause
    ])
    raw = ea.fluency_raw(words)
    assert raw["pauses_short"] == 3 and raw["pauses_long"] == 2 and raw["pauses_very_long"] == 1
    assert raw["mid_clause_pauses"] == 2, "only the pauses NOT after sentence/clause punctuation"
    assert abs(raw["longest_pause"] - 2.6) < 1e-6
    assert abs(raw["pause_time"] - (0.6 + 1.5 + 2.6)) < 1e-6


def test_small_gaps_are_not_pauses():
    raw = ea.fluency_raw(_words([("a", 0, 0.2), ("b", 0.4, 0.6), ("c", 0.8, 1.0)]))
    assert raw["pause_time"] == 0 and raw["pauses_short"] == 0


def test_speed_and_pause_share():
    # 60 words over 30 s with 6 s of pauses
    words = [Word("w", i * 0.5, i * 0.5 + 0.2) for i in range(60)]
    raw = ea.fluency_raw(words)
    f = ea.derive_fluency(raw)
    assert abs(f["words_per_minute"] - 60 * 60 / raw["speech_time"]) < 0.2
    assert f["articulation_rate_wpm"] > f["words_per_minute"], "excluding pauses must be faster"
    assert 0 < f["pause_share"] < 1


def test_fillers_and_repetitions_are_detected():
    words = _words([(w, i * 0.3, i * 0.3 + 0.2) for i, w in enumerate(
        ["um", "I", "think", "I", "think", "that", "the", "the", "film", "was", "uh", "like", "you", "know", "good"])])
    raw = ea.fluency_raw(words)
    assert raw["hesitations"] == 2
    assert raw["discourse_fillers"] == 2            # "like" and "you know"
    assert raw["repetitions"] == 2                  # "I think I think" and "the the"


def test_too_little_speech_is_flagged_not_scored():
    assert ea.derive_fluency(ea.fluency_raw([]))["enough_speech"] is False
    few = ea.derive_fluency(ea.fluency_raw(_words([("hi", 0, 0.3), ("there", 0.4, 0.8)])))
    assert few["enough_speech"] is False


def test_combine_sums_counts_and_keeps_the_longest_pause():
    a = {"n_words": 10, "longest_pause": 1.0, "pause_time": 2.0}
    b = {"n_words": 5, "longest_pause": 3.0, "pause_time": 1.0}
    c = ea.combine_fluency([a, b])
    assert c == {"n_words": 15, "longest_pause": 3.0, "pause_time": 3.0}


PLAIN = ("I like going to the park with my friends because it is a nice place to relax and we always "
         "have a good time there when the weather is good and I think it is very good for us")
RICH = ("I am passionate about exploring botanical gardens because their serene atmosphere offers a "
        "remarkable respite from the relentless pace of urban life and fosters genuine camaraderie")


def test_richer_vocabulary_scores_higher_on_frequency_measures():
    plain, rich = ea.lexical_metrics(PLAIN), ea.lexical_metrics(RICH)
    assert plain["enough_speech"] and rich["enough_speech"]
    assert rich["less_common_word_share"] > plain["less_common_word_share"]
    assert rich["average_word_length"] > plain["average_word_length"]


def test_repetitive_speech_has_a_lower_variety_score():
    repeat = ea.lexical_metrics(("good thing good place good time " * 10).strip())
    varied = ea.lexical_metrics(RICH + " " + PLAIN)
    assert repeat["mattr"] < varied["mattr"]
    assert any(o["word"] == "good" for o in repeat["overused_words"])


def test_short_text_is_not_scored_for_vocabulary():
    assert ea.lexical_metrics("I like it.")["enough_speech"] is False


def test_grammar_stats_count_structures():
    text = ("If I had more time, I would travel more. I have lived here for years. "
            "The bridge was built in 1990 and it is used daily. I think it is nice because it is quiet.")
    g = ea.grammar_stats(text)
    assert g["enough_speech"] and g["n_sentences"] == 4
    assert g["conditionals"] == 1 and g["perfect_tenses"] >= 1 and g["passives"] >= 2
    assert g["complex_sentence_share"] >= 0.5
    assert ea.grammar_stats("Yes. No.")["enough_speech"] is False


def test_confidence_summary_lists_the_shaky_words_once_each():
    words = _words([("clear", 0, 1, 0.95), ("thoroughly", 1, 2, 0.2), ("thoroughly", 2, 3, 0.3),
                    ("um", 3, 4, 0.1), ("fine", 4, 5, 0.9)])
    s = ea.confidence_summary(words)
    assert s["n_words"] == 4, "hesitation sounds are not judged"
    assert s["low_confidence_words"] == ["thoroughly"]
    assert abs(s["low_confidence_share"] - 0.5) < 1e-9


def test_flat_pitch_reads_as_monotone_and_varied_pitch_does_not():
    flat = ea.pitch_summary(np.full(500, 120.0) + np.random.default_rng(0).normal(0, 0.5, 500))
    varied = ea.pitch_summary(100 + 100 * np.abs(np.sin(np.linspace(0, 12, 500))))
    assert flat["pitch_variation_semitones"] < 0.2
    assert varied["pitch_variation_semitones"] > 3
    assert varied["pitch_range_semitones"] > flat["pitch_range_semitones"]
    assert ea.pitch_summary(np.array([100.0, 110.0]))["enough_voiced_speech"] is False


def test_prosody_reads_a_real_wav_file():
    sr, secs, f = 16000, 2.0, 150.0
    t = np.arange(int(sr * secs)) / sr
    tone = (0.4 * np.sin(2 * np.pi * f * t) * 32767).astype(np.int16)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tone.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
            w.writeframes(tone.tobytes())
        out = ea.prosody_metrics([path])
    assert out["enough_voiced_speech"] and abs(out["median_pitch_hz"] - f) < 8
    assert out["pitch_variation_semitones"] < 1.0, "a steady tone has no intonation"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} exam analysis tests passed")
