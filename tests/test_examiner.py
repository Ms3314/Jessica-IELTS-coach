"""Examiner prompt/question tests — fake LLM, no API calls.

    python tests/test_examiner.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.exam import examiner


class Chunk:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Stands in for ChatOpenAI: streams chunks, invokes, structured output."""

    def __init__(self, chunks=None, card=None, fail=False):
        self.chunks, self.card, self.fail = chunks or [], card, fail
        self.last_messages = None

    def stream(self, messages):
        self.last_messages = messages
        for c in self.chunks:
            yield Chunk(c)

    def invoke(self, messages):
        self.last_messages = messages
        return Chunk("".join(c if isinstance(c, str) else "" for c in self.chunks))

    def with_structured_output(self, _schema):
        llm = self

        class Runner:
            def invoke(self, _prompt):
                if llm.fail:
                    raise RuntimeError("boom")
                return llm.card

        return Runner()


def test_repeat_requests_are_recognised_only_when_short():
    for text in ["Sorry, could you repeat that?", "Pardon?", "Can you say that again", "sorry?"]:
        assert examiner.is_repeat_request(text), text
    for text in ["I live in a city and I repeat the same routine every day because I like it a lot",
                 "My name is Sam", "", None]:
        assert not examiner.is_repeat_request(text), text


def test_cue_card_from_the_model_is_used():
    card = examiner.CueCard(title="Describe a book you enjoyed.", points=["what", "when", "who", "why"],
                            explain="and explain why.", theme="books")
    assert examiner.generate_cue_card(FakeLLM(card=card)) is card


def test_cue_card_falls_back_when_the_model_fails_or_is_empty():
    assert examiner.generate_cue_card(FakeLLM(fail=True), random.Random(0)) in examiner._FALLBACK_CARDS
    bad = examiner.CueCard(title=" ", points=["a"], explain="x", theme="")
    assert examiner.generate_cue_card(FakeLLM(card=bad), random.Random(0)) in examiner._FALLBACK_CARDS


class _Spy:
    """Records the prompt the model was given for the cue card."""
    def __init__(self, card):
        self.card, self.prompts = card, []

    def with_structured_output(self, _schema):
        spy = self

        class Runner:
            def invoke(self, prompt):
                spy.prompts.append(prompt)
                return spy.card
        return Runner()


def test_cue_card_prompt_lists_part_1_topics_and_recent_cards():
    card = examiner.CueCard(title="Describe a book you enjoyed.", points=["a", "b", "c", "d"],
                            explain="and explain why.", theme="books")
    spy = _Spy(card)
    examiner.generate_cue_card(spy, random.Random(1), avoid=["Describe a useful object."],
                               covered=["Do you enjoy cooking?"])
    prompt = spy.prompts[0]
    assert "Do you enjoy cooking?" in prompt and "Describe a useful object." in prompt


def test_fallback_card_skips_recent_titles():
    avoid = [c.title for c in examiner._FALLBACK_CARDS[:-1]]
    got = examiner.generate_cue_card(FakeLLM(fail=True), random.Random(0), avoid=avoid)
    assert got is examiner._FALLBACK_CARDS[-1]


def test_cue_card_is_read_out_naturally():
    card = examiner.CueCard(title="Describe a place.", points=["where it is,", "when you went"],
                            explain="and explain why you liked it.", theme="places")
    text = examiner.cue_card_text(card)
    assert text == "Describe a place. You should say: where it is, when you went, and explain why you liked it."


def test_part1_lets_the_model_choose_topics_and_tracks_the_clock():
    first = examiner.part1_instruction(0, 300, [], first=True)
    assert "first question of Part 1" in first and "0:00 of about 5:00" in first
    assert "Choose the topics yourself" in first
    later = examiner.part1_instruction(125, 300, [("Do you work?", "Yes, as a nurse")], first=False)
    assert "2:05 of about 5:00" in later and "Do you work?" in later and "as a nurse" in later
    assert "first question of Part 1" not in later


def test_part3_gets_more_abstract_as_time_passes_and_uses_the_theme():
    card = examiner.CueCard(title="t", points=["a", "b", "c"], explain="e", theme="books and reading")
    early = examiner.part3_instruction(card, 10, 300, [], first=True)
    late = examiner.part3_instruction(card, 250, 300, [("q", "a")], first=False)
    assert "books and reading" in early and "accessible and general" in early
    assert "abstract" in late and "the transition has already been said" not in late


# --- model-written lines (with fallbacks) --------------------------------------------

def test_every_line_has_an_intent_and_facts_it_must_convey():
    for key, (intent, must, proof, fallback) in examiner.LINES.items():
        assert intent and must, key
        assert fallback is not None or key in ("part3_intro", "repeat"), key
        text = fallback if fallback else examiner.fallback_line(key, theme="books", question="Q?")
        assert examiner.line_ok(key, text), f"the fallback for {key} must satisfy its own check"


def test_line_instruction_includes_context_and_the_must_haves():
    instr = examiner.line_instruction("part3_intro", theme="books and reading")
    assert "books and reading" in instr and "Part 3 is a discussion" in instr
    assert "vary the phrasing" in instr.lower()
    assert "Of course" not in instr, "no scripted wording is handed to the model"
    assert "Q1?" in examiner.line_instruction("repeat", question="Q1?")


def test_part2_lines_use_the_times_of_the_chosen_length():
    from app.exam import ielts

    short = ielts.timing_for("short").phrases()
    for key in ("part2_announce", "part2_prep", "part2_start"):
        text = examiner.fallback_line(key, **short)
        assert examiner.line_ok(key, text, **short), key
        assert "one minute to" not in text.replace("up to one minute", ""), "no full-test times in a short test"
    assert "30 seconds" in examiner.fallback_line("part2_prep", **short)
    assert "up to one minute" in examiner.fallback_line("part2_start", **short)
    assert "30 seconds" in examiner.line_instruction("part2_announce", **short)
    assert "up to one minute" in examiner.line_instruction("part2_announce", **short)


def test_a_line_quoting_the_wrong_length_is_rejected_and_replaced():
    from app.exam import ielts

    short = ielts.timing_for("short").phrases()
    llm = FakeLLM(chunks=["You have one minute to prepare."])      # right for full, wrong for short
    assert examiner.write_line(llm, "part2_prep", **short) == examiner.fallback_line("part2_prep", **short)
    assert examiner.write_line(llm, "part2_prep") == "You have one minute to prepare."


def test_the_greeting_is_told_the_time_of_day():
    instr = examiner.line_instruction("intro")
    assert any(f"currently {p}" in instr for p in ("morning", "afternoon", "evening"))


def test_a_good_generated_line_is_used():
    llm = FakeLLM(chunks=["Hello, I am Jessica. May I have your full name, please?"])
    assert examiner.write_line(llm, "intro") == "Hello, I am Jessica. May I have your full name, please?"


def test_a_line_missing_a_required_fact_falls_back():
    llm = FakeLLM(chunks=["Here is your topic."])          # forgot to mention the minute
    assert examiner.write_line(llm, "part2_announce") == examiner.PART2_ANNOUNCE
    assert examiner.write_line(FakeLLM(chunks=["  "]), "intro") == examiner.INTRO_LINE


def test_a_model_failure_falls_back_instead_of_raising():
    class Boom(FakeLLM):
        def invoke(self, _m):
            raise RuntimeError("api down")

    assert examiner.write_line(Boom(), "end") == examiner.END_LINE
    assert examiner.write_line(Boom(), "part3_intro", theme="music").endswith("related to this.")
    assert "music" in examiner.write_line(Boom(), "part3_intro", theme="music")
    assert examiner.write_line(Boom(), "repeat", question="Do you work?") == "Of course. Do you work?"


def test_lines_without_a_required_context_value_fall_back_cleanly():
    # part3_intro without a theme must not crash the exam
    assert examiner.write_line(FakeLLM(chunks=["x"]), "part3_intro")


def test_stream_question_yields_only_text_blocks():
    llm = FakeLLM(chunks=["What ", [{"type": "reasoning", "summary": []}, {"type": "text", "text": "is your name?"}]])
    assert "".join(examiner.stream_question(llm, "instr")) == "What is your name?"
    assert llm.last_messages[0].content == examiner.EXAMINER_SYSTEM
    assert llm.last_messages[1].content == "instr"


def test_ask_returns_one_string():
    assert examiner.ask(FakeLLM(chunks=["Do you ", "work?"]), "instr") == "Do you work?"


def test_examiner_rules_forbid_feedback():
    assert "Never comment on the quality" in examiner.EXAMINER_SYSTEM


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} examiner tests passed")
