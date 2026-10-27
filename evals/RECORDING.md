# Recording the test set (Phase 2)

Ten short mock tests at levels you choose on purpose: **3 weak, 3 medium, 4 strong**.
They are the answer key for Phase 3: a weak test should score below a strong one.

## Before each test

- Choose the **short** test (about 6 minutes) on the exam start screen.
- Quiet room, same microphone every time.
- The app picks a different cue card each time (it avoids recent ones), so topics vary on their own.

## How to speak each level

| Level | How to answer |
|---|---|
| Weak | One or two short sentences per question, long pauses, basic repeated words, a few grammar slips |
| Medium | Complete answers, some hesitation, limited vocabulary |
| Strong | Long, fluent answers (use the full time in Part 2), varied vocabulary, examples, linking words |

**Weak tests must still answer at least 3 questions.** With fewer, the app reports
"insufficient" and the test can't be scored. Weak tests will often show reliability "low"
(under 90 s of speech or 150 words). That is expected; leave them in.

## After each test

1. Wait until the report appears in the app. That is when `analysis.json` is saved.
2. Run `python evals/score_eval.py init-labels`, which adds the new test to `evals/labels.json`.
3. Set its `level` to `weak`, `medium` or `strong`, and add the cue card topic in `note`.
4. Run `python evals/score_eval.py status` to check it shows **Scoreable: yes** and the level.

The old tests already in `labels.json` can stay at `null`; unlabelled tests are skipped.

## Then Phase 3

```bash
python evals/score_eval.py run --provider openai --yes    # ~50 calls, about 20 s each
python evals/score_eval.py report --provider openai
python evals/score_eval.py run --provider ollama          # free but slow, about 95 minutes
python evals/score_eval.py report --provider ollama
```
