# Jessica Evals Plan

## Goal and test set

Five evals, about 7 hours, all runnable on your Mac. They show that Jessica is fast, stable and ranks answers sensibly. They do not show that her bands match official IELTS scores.

### What each eval lets you claim

| Eval | You can say | You cannot say |
| --- | --- | --- |
| 1. Latency | Per-stage and total response time, p50 and p95, on your Mac | That it is fast on other hardware |
| 2. Consistency | How much the score moves when the same test is re-scored | That the score is correct |
| 3. Ordering | Weak tests score below strong tests | That the gaps match real IELTS gaps |
| 4. OpenAI vs local | How a 4B local model differs from the OpenAI model on the same tests | That either is accurate |
| 5. Quote grounding | How often cited quotes really appear in the transcript | That each quote supports the advice |

### Folder layout

Keep results in the repo, but not your recordings.

```
evals/
  data/labels.json      # exam id -> intended level (weak / medium / strong)
  latency.py            # reads the timing log, prints p50 / p95 per stage
  consistency.py        # re-scores each exam N times, prints spread
  ordering.py           # compares mean scores with intended levels
  compare_models.py     # OpenAI vs Ollama on the same exams
  results/              # JSON outputs and results.md
```

### Build the test set (about 1.5 hours)

1. Run 10 short mock tests (about 6 minutes each) in the app: **3 weak, 3 medium, 4 strong**.
2. Speak each level on purpose:
   - **Weak:** one or two short sentences, long pauses, repeated basic words, a few grammar slips.
   - **Medium:** complete answers, some hesitation, limited vocabulary range.
   - **Strong:** extended answers of 1-2 minutes in Part 2, fluent, varied vocabulary, examples and linking words.
3. Each test is saved under `data/exams/<session>/<exam>/` as WAV files plus `exam.json` and `report.json`. Copy the ten exam folders somewhere safe and list each id with its intended level in `labels.json`.
4. Use at least four different cue cards and topics, so the set is not one topic repeated.

One speaker faking levels is a real limit. It is fine for crude ordering, so say so in the README. If a friend can record two or three tests, add them and note it.

## Eval 1: latency

Measure the time from end of speech to first audio, split by stage, over 20 turns per LLM backend. My guess is that Whisper and the LLM's first sentence dominate, but the data decides.

### Where to put timestamps

Confirm the exact functions in the repo; the names below come from `ARCHITECTURE.md`.

1. **t0:** the recorder decides you finished speaking (`speech/recorder.py`, or `record_until_silence` in `speech/asr.py` for the terminal).
2. **t1a and t1b:** just before and after `transcribe` in `speech/asr.py`.
3. **t2:** the first text delta from `agent_reply_stream` in `llm/agent.py`.
4. **t3:** the first complete sentence handed to TTS (`chat/engine.py` and `speak_stream` in `speech/tts.py`).
5. **t4:** the first TTS audio chunk written to the sink.

Stage times: ASR = t1b - t1a. LLM first token = t2 - t1b. LLM first sentence = t3 - t1b. TTS first audio = t4 - t3. Total = t4 - t0.

Your pause setting adds a delay before t0 that you chose, not one the system measured. Report it as a separate configuration value.

### Logging snippet

```python
import json, os, time

def mark(turn_id, name, path="evals/results/latency.jsonl"):
    if os.environ.get("EVALS_LATENCY") != "1":
        return
    with open(path, "a") as f:
        f.write(json.dumps({"turn": turn_id, "event": name,
                            "t": time.perf_counter()}) + "\n")
```

Call `mark(turn, "t0")`, `mark(turn, "t1a")` and so on at the five points. Then `latency.py` groups events by turn, computes the stage durations and prints percentiles.

### Protocol

- Discard the first turn after startup, since the first call to each model is slower.
- Run 20 turns per backend, OpenAI first, then Ollama, using the same 20 spoken prompts of 5 to 15 seconds each, read from a list.
- Use a quiet room, close heavy apps and plug the Mac in.
- Record the chip, macOS version, model names and whether you were on Wi-Fi. The Mac was only tested on an M4 per your README.
- Report p50, p95, min and max.

### Results table to fill

| Stage | OpenAI p50 | OpenAI p95 | Ollama p50 | Ollama p95 |
| --- | --- | --- | --- | --- |
| Whisper transcription |  |  |  |  |
| LLM first sentence |  |  |  |  |
| TTS first audio |  |  |  |  |
| Total, end of speech to first audio |  |  |  |  |

Add one line below it with the stage that takes the largest share and by how much. That line is the finding, and it decides what you fix later.

## Evals 2 and 3: consistency and ordering

Re-score each saved test 5 times to measure how much the band moves, then check that weak tests score below strong ones. Both run without a microphone, because the audio is already saved.

### Consistency protocol

1. Find the function that runs the post-test analysis. `ARCHITECTURE.md` names `run_exam_analysis` in `app/exam/report.py`, and your UI has an "Analyse again" action, so re-running on an existing exam folder should be possible. Confirm how.
2. For each of the 10 exams, re-run the LLM scoring step 5 times on the same saved transcript and measured numbers. Save every output to `evals/results/runs/<exam>/<run>.json`.
3. On 3 exams, also re-run the whole pipeline twice (re-transcribe, analyse, score) to confirm the measured numbers come out identical.
4. Write down the model name and the temperature used. If temperature is above 0, say so.

| Metric | How to compute it | What it tells you |
| --- | --- | --- |
| Std dev per band | Standard deviation across the 5 runs, for fluency, vocabulary, grammar, pronunciation and overall | How noisy each band is |
| Range of the overall band | Max minus min across 5 runs | Worst-case wobble for one test |
| Share of exams with identical overall | Exams with range 0, out of 10 | How often it is fully stable |
| Measured numbers identical | Compare pace, pauses, word counts across re-runs | Whether all the noise comes from the LLM |

There is no official pass mark. Report the numbers as they are. A large spread is a finding you can then fix.

### Ordering protocol

1. Give each test its intended level: weak = 1, medium = 2, strong = 3.
2. Average the overall band across the 5 runs for each exam.
3. **Pairwise accuracy:** take every pair of tests with different levels (33 pairs for a 3, 3, 4 split). Count how often the higher level gets the higher mean band. Report ties separately.
4. **Rank correlation:** compute Spearman correlation between level and mean band with `scipy.stats.spearmanr`.
5. Repeat for each band, because pronunciation may order worse than fluency.

### Script sketch

Adjust the key to match your real `report.json`.

```python
import glob, json, statistics as st
from collections import defaultdict

runs = defaultdict(list)
for p in glob.glob("evals/results/runs/*/*.json"):
    exam = p.split("/")[-2]
    runs[exam].append(json.load(open(p))["overall_band"])

for exam, v in sorted(runs.items()):
    print(exam, round(st.mean(v), 2), round(st.pstdev(v), 2), max(v) - min(v))
```

### Results table to fill

| Exam | Intended level | Mean overall | Range over 5 runs | Fluency | Vocabulary | Grammar | Pronunciation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e01 | weak |  |  |  |  |  |  |

Add one row per exam. Then state pairwise accuracy and the Spearman value under the table.

### Pitfalls

- Medium and strong may overlap, because you are one speaker faking levels. Report it honestly.
- If you re-score with a cached transcript, you are not testing Whisper's variation. The 3-exam full re-run covers that.

## Evals 4 and 5, plus optional extras

Run the same 10 tests through OpenAI and your local Ollama model, and check how many cited quotes really appear in the transcript. The README currently says the local 4B model's scoring is untested, so this fills a stated gap.

### Eval 4: OpenAI vs local

1. Set `LLM_PROVIDER` to `openai`, then to `ollama` (default `qwen3.5:4b`), per the README. Record both model names.
2. Re-score the same 10 exams 5 times per backend, using the saved transcript and measured numbers so only the LLM call differs. That is about 100 calls.
3. Run 10 calls first and check your OpenAI usage dashboard before running the rest. Use a cheaper OpenAI model for the repeats if the cost looks high.
4. Count a failed or malformed report as a failure. Do not drop it.

| Metric | OpenAI | Ollama |
| --- | --- | --- |
| Mean overall band across exams |  |  |
| Mean absolute difference between the two, paired by exam |  |  |
| Average range over 5 runs |  |  |
| Pairwise ordering accuracy |  |  |
| Malformed report rate |  |  |
| Quote rejection rate |  |  |
| Seconds per report |  |  |
| Cost per report |  |  |

### Eval 5: quote grounding

1. Your pipeline already checks each quote against the transcript. Log the verified and rejected counts for every report.
2. Quote rejection rate = rejected quotes divided by all quotes, per model.
3. Hand-check 20 verified quotes: does the quote actually support the advice next to it? Report the share that does. A string match proves the words exist, not that the advice is right.

### Optional: transcription accuracy (about 1.5 hours)

1. Record 15 clips of 20 to 30 seconds. Read a known text aloud for some, and speak naturally and type the true transcript afterwards for others.
2. Run Whisper on each clip and compute word error rate with the `jiwer` package after lowercasing and removing punctuation.
3. Report the mean and range, and note that it is one speaker and one microphone.

### Optional: OpenTelemetry and Grafana (about 2 hours, only after the tables)

1. Add the OpenTelemetry SDK and an OTLP exporter. Create one span per stage with the backend and model as attributes.
2. Run a collector, a trace store and Grafana in Docker Compose, with latency histograms for the dashboard.
3. Build one panel for p50 and p95 per stage and keep one screenshot of a trace for a slow turn.
4. Describe it as instrumentation of a local app, not production monitoring. It is only worth including if it shows you something the plain table did not.

## Fix, deliverables and schedule

Pick one fix that the numbers point to, measure it again, and write down before and after. That single change is what turns a set of tables into a case study.

### One fix to make (about 1 hour)

| If the data shows | Try | Re-measure |
| --- | --- | --- |
| The LLM's first sentence dominates latency | Shorten the system prompt, cap the first sentence, or use a smaller model for chat | p50 and p95 for that stage |
| Scores wobble a lot between runs | Set temperature to 0 if allowed, add band descriptors to the prompt, or take the median of 3 calls | Range and std dev over 5 runs |
| The local model returns malformed reports | Add structured-output validation and one retry | Malformed report rate |

### Deliverables

- The `evals/` folder with scripts and a `results.md`.
- An **Evaluation** section in the README with three tables (latency, consistency and ordering, OpenAI vs local) and a short limits paragraph.
- A closing slide or pinned comment on your demo video with the headline numbers.

### Schedule

| Block | Task | Hours |
| --- | --- | --- |
| 1 | Record the 10 test exams and write `labels.json` | 1.5 |
| 2 | Add timestamps, run 2 x 20 turns, make the latency table | 1.5 |
| 3 | Consistency and ordering runs and tables | 1.5 |
| 4 | OpenAI vs local comparison and quote grounding | 1 |
| 5 | One fix and re-measure | 1 |
| 6 | README write-up | 1 |

Total is about 7 hours. If you are short on time, drop block 5 first, then the quote hand-check, then the full-pipeline re-run on 3 exams.

### How to word the results

- Say: "Practice band estimates. On 10 self-recorded tests by one speaker, scores varied by at most X across 5 runs and ranked weak, medium and strong correctly in N of 33 pairs."
- Do not say the bands are accurate or validated.
- State the model names, the machine and the date next to every number.

### Resume bullet templates (fill with real numbers only)

- Measured end-to-end voice latency per stage (p50 and p95) and reduced time to first audio from X s to Y s by changing Z.
- Evaluated IELTS band scoring on 10 recorded tests: overall band varied by at most X across 5 runs, and ordered weak, medium and strong tests correctly in N of 33 pairs.
- Compared GPT and a local 4B model on the same tests: mean band difference X, malformed reports A% vs B%.
