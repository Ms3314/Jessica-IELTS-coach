"""Sentence splitter tests — fast, no devices or models needed.

    python tests/test_tts.py

(Importing app.tts does load the Kokoro pipeline — that's ~10s
of model load; the split logic itself is pure string handling.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tts import _sentence_iter


def test_basic_split():
    out = list(_sentence_iter(iter(["Hello there. How are you?"])))
    assert out == ["Hello there.", "How are you?"], out


def test_abbreviations_do_not_split():
    out = list(_sentence_iter(iter(["Mr. Smith arrived. He was on time."])))
    assert out == ["Mr. Smith arrived.", "He was on time."], out


def test_decimals_do_not_split():
    out = list(_sentence_iter(iter(["The price is $3.14. That is cheap."])))
    assert out == ["The price is $3.14.", "That is cheap."], out


def test_ellipsis_fragments_are_dropped():
    # "Wait... what?" must not queue lone "." fragments for synthesis
    out = list(_sentence_iter(iter(["Wait... what? Really!"])))
    assert out == ["Wait.", "what?", "Really!"], out


def test_deltas_accumulate_across_chunks():
    # a sentence arriving one character at a time must still split
    text = "Hi there. Second sentence."
    out = list(_sentence_iter(iter(list(text))))
    assert out == ["Hi there.", "Second sentence."], out


def test_tail_is_flushed():
    # no final punctuation: the remainder must still be spoken
    out = list(_sentence_iter(iter(["One sentence. And a trailing fragment"])))
    assert out == ["One sentence.", "And a trailing fragment"], out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} tts tests passed")
