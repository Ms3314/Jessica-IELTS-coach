"""The IELTS examiner's voice: how each line is asked for, and written.

app/exam/ielts.py fixes the STRUCTURE and TIMINGS of the test; everything the
examiner says is written by the language model each time, so no two tests sound
alike. For every line the model is given an intent ("greet the candidate and ask
their name") plus the facts the line must convey ("one minute to prepare"). If
the model is unavailable, or leaves a required fact out, a plain fallback line
is spoken instead, so the exam can always continue.

Nothing here imports the agent or any speech model, so it can be tested with a
fake `llm` (anything with .stream / .invoke / .with_structured_output).
"""
import logging
import random
import re
import time
from typing import Iterator, Optional

from pydantic import BaseModel, Field

from app.llm.llmtext import _text_deltas, text_from_content

logger = logging.getLogger(__name__)

EXAMINER_SYSTEM = """You are Jessica, an IELTS Speaking examiner running a mock test. Your words are
read aloud by a text-to-speech voice.

Rules:
- Ask exactly ONE question per turn, in one or two short sentences.
- Never comment on the quality of an answer, never praise or correct it, and never give hints
  or tips. A neutral "Thank you" or "I see" is fine, but do not use it every time.
- Do not repeat a question you already asked, and do not ask about something the candidate has
  already covered in full.
- Use plain spoken English: no markdown, no lists, no emoji, no stage directions.
- If the candidate asks you something about the test itself, stay in role and just continue.
- Output ONLY what the examiner says next."""

# --- fallback lines ---------------------------------------------------------------
# Only used when the model can't write a line (or omits something it must say).

INTRO_LINE = (
    "Good day. My name is Jessica, and I will be your examiner today. "
    "This is the speaking test. First, can you tell me your full name, please?"
)
# How the times are worded in the full test; a shorter test supplies its own (see ielts.Timing.phrases).
_FULL_PHRASES = {"prep_phrase": "one minute", "prep_word": "minute", "talk_phrase": "one to two minutes", "talk_word": "minute"}

_PART2_ANNOUNCE_T = (
    "Thank you. Now, I am going to give you a topic, and I would like you to talk about it "
    "for {talk_phrase}. Before you talk, you will have {prep_phrase} to think about what "
    "you are going to say. You can make some notes if you wish. Here is your topic."
)
_PART2_PREP_T = "You have {prep_phrase} to prepare. Your time starts now."
_PART2_START_T = (
    "All right? Remember, you have {talk_phrase} for this, so do not worry if I stop you. "
    "I will tell you when the time is up. Please start speaking now."
)
PART2_ANNOUNCE = _PART2_ANNOUNCE_T.format(**_FULL_PHRASES)
PART2_PREP_LINE = _PART2_PREP_T.format(**_FULL_PHRASES)
PART2_START_LINE = _PART2_START_T.format(**_FULL_PHRASES)
PART2_MORE_LINE = "Can you tell me a little more about that?"
PART2_STOP_LINE = "Thank you. I will stop you there."
THANKS_LINE = "Thank you."
END_LINE = "Thank you. That is the end of the speaking test."
REPEAT_LEAD = "Of course."

# key -> (what the line is for, facts it must convey, words that prove it did, fallback)
LINES = {
    "intro": (
        "Greet the candidate appropriately for the time of day (it is currently {period}), introduce "
        "yourself as Jessica, their examiner, say this is the speaking test, and ask for their full name.",
        ["your name is Jessica", "this is the speaking test", "ask for their full name"],
        ["name"], INTRO_LINE,
    ),
    "part2_announce": (
        "Part 1 is over. Introduce Part 2: you will give them a topic to talk about for {talk_phrase}.",
        ["a topic is coming", "they will talk for {talk_phrase}", "they get {prep_phrase} to prepare first",
         "they may make notes", "end by saying you are giving them the topic now"],
        ["{prep_word}"], PART2_ANNOUNCE,
    ),
    "part2_prep": (
        "Tell the candidate their {prep_phrase} of preparation starts now.",
        ["{prep_phrase} to prepare", "it starts now"],
        ["{prep_word}"], PART2_PREP_LINE,
    ),
    "part2_start": (
        "The preparation time is over. Invite the candidate to start talking now.",
        ["they should talk for {talk_phrase}", "you may stop them when time is up", "ask them to begin"],
        ["{talk_word}"], PART2_START_LINE,
    ),
    "part2_more": (
        "The candidate stopped talking early. Invite them, once, to say a bit more.",
        ["ask them to say more about the topic"], [], PART2_MORE_LINE,
    ),
    "part2_stop": (
        "The speaking time is up. Politely stop the candidate.",
        ["thank them", "say you will stop them there"], [], PART2_STOP_LINE,
    ),
    "thanks": (
        "The candidate has finished their talk. Thank them very briefly.", ["a brief thank you"], [], THANKS_LINE,
    ),
    "part3_intro": (
        "Part 2 is over. Introduce Part 3: you will now ask some more general questions linked to the Part 2 "
        "theme, which was \"{theme}\".",
        ["Part 3 is a discussion", "it is related to the theme"], [], None,  # fallback built from the theme
    ),
    "end": (
        "The test is over. Thank the candidate and tell them this is the end of the speaking test.",
        ["thank them", "the speaking test is finished"], ["test"], END_LINE,
    ),
    "repeat": (
        "The candidate asked you to repeat. Briefly acknowledge, then repeat this question (you may rephrase it "
        "slightly): {question}",
        ["a very short acknowledgement", "the question again"], [], None,  # fallback built from the question
    ),
}


def part3_intro_fallback(theme: str) -> str:
    return (
        f"Thank you. We have been talking about {theme.strip()}, and now I would like to ask you "
        "some more general questions related to this."
    )


def _period_of_day() -> str:
    hour = time.localtime().tm_hour
    return "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"


_TEMPLATES = {
    "part2_announce": _PART2_ANNOUNCE_T,
    "part2_prep": _PART2_PREP_T,
    "part2_start": _PART2_START_T,
}


def _context(**ctx) -> dict:
    """Everything a line's wording may refer to (times default to the full test's)."""
    return {**_FULL_PHRASES, **ctx}


def line_instruction(key: str, **ctx) -> str:
    """The instruction that asks the model for one examiner line."""
    ctx = _context(**ctx)
    ctx.setdefault("period", _period_of_day())
    intent, must, _proof, _fallback = LINES[key]
    return (
        "Write ONE short examiner utterance. "
        + intent.format(**ctx)
        + "\nIt must convey: " + "; ".join(m.format(**ctx) for m in must) + "."
        + "\nUse natural, formal, neutral wording and vary the phrasing from a standard script. "
        "Keep it to one to three short sentences. Do not ask anything else."
    )


def line_ok(key: str, text: str, **ctx) -> bool:
    """A generated line is accepted only if it is non-empty and contains its required words."""
    text = (text or "").strip()
    if not text:
        return False
    ctx = _context(**ctx)
    return all(word.format(**ctx) in text.lower() for word in LINES[key][2])


def fallback_line(key: str, **ctx) -> str:
    ctx = _context(**ctx)
    if key in _TEMPLATES:
        return _TEMPLATES[key].format(**ctx)
    if key == "part3_intro":
        return part3_intro_fallback(ctx.get("theme", "that topic"))
    if key == "repeat":
        return f"{REPEAT_LEAD} {ctx.get('question', '')}".strip()
    return LINES[key][3]


def write_line(llm, key: str, **ctx) -> str:
    """The examiner's line for `key`, written by the model, or the fallback."""
    try:
        text = ask(llm, line_instruction(key, **ctx))
        if line_ok(key, text, **ctx):
            return text
    except Exception:
        pass
    return fallback_line(key, **ctx)


_REPEAT_RE = re.compile(
    r"\b(repeat|say (that|it) again|come again|what was the question|"
    r"didn'?t (hear|catch)|can you (say|ask) (that|it) again)\b"
    r"|^\W*(sorry|pardon|pardon me|what|excuse me)\W*$",
    re.IGNORECASE,
)


def is_repeat_request(text: str) -> bool:
    """True when a SHORT answer is really "could you say that again?"."""
    text = (text or "").strip()
    return 0 < len(text.split()) <= 8 and bool(_REPEAT_RE.search(text))


# --- Part 2 cue card -------------------------------------------------------------

class CueCard(BaseModel):
    title: str = Field(description='Starts with "Describe", e.g. "Describe a book that you enjoyed reading."')
    points: list[str] = Field(description="Exactly four short 'You should say' prompts, e.g. 'what the book was'.")
    explain: str = Field(description='The final prompt, starting "and explain", e.g. "and explain why you enjoyed it."')
    theme: str = Field(description='Two to four words naming the general theme, e.g. "books and reading".')


_FALLBACK_CARDS = [
    CueCard(
        title="Describe a place you have visited that you really liked.",
        points=["where the place is", "when you went there", "what you did there"],
        explain="and explain why you liked it so much.",
        theme="places and travel",
    ),
    CueCard(
        title="Describe a person who has influenced you.",
        points=["who the person is", "how you know them", "what they are like"],
        explain="and explain how they have influenced you.",
        theme="people and influence",
    ),
    CueCard(
        title="Describe a useful object that you own.",
        points=["what it is", "how long you have had it", "how you use it"],
        explain="and explain why it is useful to you.",
        theme="objects and technology",
    ),
    CueCard(
        title="Describe a time when you helped someone.",
        points=["who you helped", "what the problem was", "how you helped"],
        explain="and explain how you felt afterwards.",
        theme="helping others",
    ),
    CueCard(
        title="Describe a film or programme that you enjoyed watching.",
        points=["what it was", "when you watched it", "what it was about"],
        explain="and explain why you enjoyed it.",
        theme="films and television",
    ),
    CueCard(
        title="Describe a celebration that you remember well.",
        points=["what the celebration was", "who was there", "what happened"],
        explain="and explain why you remember it.",
        theme="celebrations",
    ),
]


def generate_cue_card(llm, rng: Optional[random.Random] = None, avoid: Optional[list[str]] = None,
                      covered: Optional[list[str]] = None) -> CueCard:
    """Ask the model to choose and write a fresh cue card; fall back to a stock one on failure.

    `avoid` holds recent card titles and `covered` the questions already asked in Part 1, so
    the model picks a topic that has not come up."""
    rng = rng or random.Random()
    avoid = [a for a in (avoid or []) if a][:12]
    covered = [c for c in (covered or []) if c][:10]
    prompt = (
        "You are an IELTS examiner. Choose a topic yourself and write one Part 2 cue card in the "
        "style of real IELTS cards. Pick something specific and a little unexpected (a person, place, "
        "event, experience, skill, media, possession or habit), not the most obvious topic.\n"
    )
    if covered:
        prompt += "These were already discussed in Part 1, so do not use the same subject:\n"
        prompt += "\n".join(f"  * {c}" for c in covered) + "\n"
    if avoid:
        prompt += "These cards were used in earlier tests; choose something clearly different:\n"
        prompt += "\n".join(f"  * {a}" for a in avoid) + "\n"
    prompt += f"Variety seed: {rng.randint(0, 10**6)}."
    try:
        card = llm.with_structured_output(CueCard).invoke(prompt)
        if isinstance(card, dict):
            card = CueCard(**card)
        if card.title.strip() and len(card.points) >= 3 and card.theme.strip():
            return card
    except Exception:
        logger.warning("cue card generation failed; using a stock card", exc_info=True)
    pool = [c for c in _FALLBACK_CARDS if c.title not in avoid] or _FALLBACK_CARDS
    return rng.choice(pool)


def cue_card_text(card: CueCard) -> str:
    """The cue card as the examiner reads it out."""
    points = ", ".join(p.strip().rstrip(",.") for p in card.points)
    return f"{card.title.strip()} You should say: {points}, {card.explain.strip()}"


# --- question instructions ---------------------------------------------------------

def _history_block(history: list[tuple[str, str]], limit: int = 8) -> str:
    if not history:
        return "(nothing asked yet)"
    lines = []
    for q, a in history[-limit:]:
        lines.append(f"Examiner: {q}\nCandidate: {a or '[no answer]'}")
    return "\n".join(lines)


def _clock(elapsed_s: float, limit_s: float) -> str:
    return f"{int(elapsed_s // 60)}:{int(elapsed_s % 60):02d} of about {int(limit_s // 60)}:{int(limit_s % 60):02d}"


def part1_instruction(elapsed_s: float, limit_s: float, history, first: bool) -> str:
    opening = (
        "This is the first question of Part 1. Begin with a brief natural lead-in saying that you will now "
        "ask some questions about the candidate, then ask your first question."
        if first
        else "Do not add a lead-in unless you are changing topic, in which case use a short, natural one."
    )
    return (
        f"IELTS Speaking Part 1 (the interview). Time used: {_clock(elapsed_s, limit_s)}. Ask the next question.\n"
        "- Choose the topics yourself. Start with the candidate's home, work or studies, using anything they "
        "have already told you. After three or four questions on one topic, move to a different everyday "
        "topic (for example free time, food, travel, technology, friends, routines).\n"
        "- Part 1 questions are simple, personal and concrete (habits, preferences, past and present) and can "
        "be answered in a few sentences. Vary the question types.\n"
        f"- {opening}\n\nConversation so far:\n" + _history_block(history)
    )


def part2_round_instruction(card: CueCard, answer: str) -> str:
    return (
        "IELTS Speaking Part 2 has just ended. Ask ONE short rounding-off question related to the "
        f"candidate's talk (topic: \"{card.title}\"), answerable in one or two sentences, for example a "
        "'Do you think...?' or 'Would you...?' question.\n\nWhat the candidate said:\n"
        + (answer or "[nothing]")
    )


def part3_instruction(card: CueCard, elapsed_s: float, limit_s: float, history, first: bool) -> str:
    progress = min(1.0, elapsed_s / limit_s) if limit_s else 1.0
    depth = (
        "relatively accessible and general"
        if progress < 0.3
        else "more abstract: society, trends, causes and effects, comparisons, the future"
    )
    return (
        f"IELTS Speaking Part 3 (the discussion). Time used: {_clock(elapsed_s, limit_s)}. The Part 2 theme was "
        f"\"{card.theme}\". Ask the next question: a discussion question on that theme (or an idea that grew out "
        f"of the candidate's last answer) that is {depth}.\n"
        "- Ask for opinions and reasons about people or society in general, not personal experience as in Part 1. "
        "Follow up on something the candidate just said when that is natural.\n"
        "- Do not add a lead-in" + ("; the transition has already been said." if first else ".")
        + "\n\nConversation so far:\n" + _history_block(history)
    )


# --- generation ---------------------------------------------------------------------

def stream_question(llm, instruction: str) -> Iterator[str]:
    """Yield the examiner's next utterance as text deltas, ready for TTS."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(content=EXAMINER_SYSTEM), HumanMessage(content=instruction)]
    for chunk in llm.stream(messages):
        yield from _text_deltas(chunk.content)


def ask(llm, instruction: str) -> str:
    """The examiner's next utterance as one string (no streaming)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(content=EXAMINER_SYSTEM), HumanMessage(content=instruction)]
    return text_from_content(llm.invoke(messages).content).strip()
