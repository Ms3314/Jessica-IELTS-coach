"""Scoring evals for the IELTS report: how stable are the bands, and do they rank sensibly?

Works on exams already saved by the app (data/exams/<session>/<exam>/analysis.json), so it
needs no microphone and never re-runs Whisper. Each "run" is ONE call of the report's model
step on the saved transcript and measured numbers, exactly as app/exam/report.py makes it.

    python evals/score_eval.py init-labels                 # list saved exams in evals/labels.json
    python evals/score_eval.py status                      # which tests are scoreable, labelled, run
    python evals/score_eval.py run --provider openai --all # score every saved exam 5 times
    python evals/score_eval.py report --provider openai    # tables -> evals/results/results_openai.md

Fill in `level` (weak / medium / strong) for each exam in evals/labels.json. `run` scores
only labelled exams unless you pass --all; `report` always ranks only labelled ones.
"""
import argparse
import json
import statistics as st
import sys
import time
from datetime import date
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402  (loads .env and the keys saved in Settings)
from app.exam import report as rep  # noqa: E402
from app.exam.ielts import EXAMS_DIR, ExamLog  # noqa: E402

EVALS = ROOT / "evals"
LABELS = EVALS / "labels.json"
RESULTS_DIR = EVALS / "results"
RUNS = RESULTS_DIR / "runs"

LEVELS = {"weak": 1, "medium": 2, "strong": 3}
METRICS = (*rep.CRITERIA, "overall")
SHORT = {"fluency_coherence": "Fluency", "lexical_resource": "Vocabulary",
         "grammatical_range_accuracy": "Grammar", "pronunciation": "Pronunciation", "overall": "Overall"}


# --- exams and labels ---------------------------------------------------------------------

def saved_exams() -> list[str]:
    """'session/exam' for every saved test that has an analysis.json."""
    return [f"{f.parent.parent.name}/{f.parent.name}" for f in sorted(EXAMS_DIR.glob("*/*/analysis.json"))]


def load_labels() -> dict:
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def labelled(labels: dict) -> dict[str, str]:
    return {k: v["level"] for k, v in labels.items() if v.get("level") in LEVELS}


def run_dir(provider: str, key: str) -> Path:
    return RUNS / provider / key.replace("/", "__")


# --- one scoring call -----------------------------------------------------------------------

def build_llm(provider: str):
    """The same model settings as app/llm/agent.py, without importing the agent (which opens
    the chat database and builds the search tool)."""
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, reasoning=False,
                          keep_alive="30m", num_ctx=8192), config.OLLAMA_MODEL
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.OPENAI_MODEL, use_responses_api=True), config.OPENAI_MODEL


def score_once(llm, log: ExamLog, analysis: dict) -> dict:
    """Ask the model for a report and record what came back. A failure is a result, not an error."""
    started = time.perf_counter()
    try:
        raw = llm.with_structured_output(rep.LLMReport).invoke(rep.build_prompt(log, analysis))
        if isinstance(raw, dict):
            raw = rep.LLMReport(**raw)
        bands, quotes = {}, {}
        for key in rep.CRITERIA:
            c = getattr(raw, key)
            kept = rep.verify_quotes(c.evidence, analysis["candidate_text"])
            bands[key] = rep.clamp_band(c.band)
            quotes[key] = {"kept": len(kept), "dropped": len(c.evidence) - len(kept),
                           "rejected": [q for q in c.evidence if q not in kept]}   # kept so you can read why they failed
        bands["pronunciation"] = min(bands["pronunciation"], 7.5)      # the app caps it the same way
        bands["overall"] = rep.round_band(sum(bands[k] for k in rep.CRITERIA) / len(rep.CRITERIA))
        result = {"ok": True, "bands": bands, "quotes": quotes}
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


# --- commands ---------------------------------------------------------------------------------

def cmd_init_labels(_args):
    labels = load_labels()
    for key in saved_exams():
        labels.setdefault(key, {"level": None, "note": ""})
    LABELS.write_text(json.dumps(labels, indent=2) + "\n")
    print(f"{len(labels)} exams in {LABELS.relative_to(ROOT)}. Set each `level` to weak / medium / strong.")


def cmd_status(_args):
    """Every saved test: can it be scored, is it labelled, and how many runs each model has."""
    labels = load_labels()
    rows = []
    for meta in sorted(EXAMS_DIR.glob("*/*/exam.json")):
        d = meta.parent
        key = f"{d.parent.name}/{d.name}"
        exam = json.loads(meta.read_text())
        answered = sum(1 for t in exam["turns"] if t.get("answer"))
        if (d / "analysis.json").exists():
            fl = json.loads((d / "analysis.json").read_text())["fluency"]
            speech = f"{fl.get('speech_seconds', 0):.0f}s / {fl.get('n_words', 0)} words"
            usable = "yes"
        else:
            speech, usable = "-", "no (too few answers)" if answered < rep.MIN_ANSWERED_TURNS else "no (not analysed yet)"
        report = json.loads((d / "report.json").read_text()) if (d / "report.json").exists() else {}
        runs = " ".join(f"{p} {len(list(run_dir(p, key).glob('*.json')))}" for p in ("openai", "ollama"))
        rows.append([f"{d.parent.name[:8]}/{d.name}", exam.get("length", "full"), answered, speech,
                     report.get("reliability", "-"), usable, (labels.get(key) or {}).get("level") or "-", runs])
    print(table(["Exam", "Length", "Answered", "Speech", "Reliability", "Scoreable", "Level", "Runs"], rows))
    lv = list(labelled(labels).values())
    print(f"\nLabelled: {lv.count('weak')} weak, {lv.count('medium')} medium, {lv.count('strong')} strong "
          f"(target 3 / 3 / 4)")


def cmd_run(args):
    labels = load_labels()
    wanted = saved_exams() if args.all else [k for k in labelled(labels) if k in saved_exams()]
    if args.only:
        wanted = [k for k in wanted if args.only in k]
    if not wanted:
        sys.exit("No exams to score. Label some in evals/labels.json, or pass --all.")

    todo = [(k, i) for k in wanted for i in range(1, args.runs + 1) if not (run_dir(args.provider, k) / f"{i:02d}.json").exists()]
    llm, model = build_llm(args.provider)
    print(f"{len(wanted)} exams x {args.runs} runs, {len(todo)} calls left to make with {args.provider} ({model}).")
    if args.provider == "openai" and len(todo) > 10 and not args.yes:
        if input("That is a paid API. Continue? [y/N] ").strip().lower() != "y":
            sys.exit("Stopped. Try --only <part of an exam id> to start with one exam.")

    for n, (key, i) in enumerate(todo, 1):
        session_id, exam_id = key.split("/")
        analysis = json.loads((EXAMS_DIR / key / "analysis.json").read_text())
        result = score_once(llm, ExamLog.load(session_id, exam_id), analysis)
        result.update(exam=key, provider=args.provider, model=model, run=i)
        out = run_dir(args.provider, key)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{i:02d}.json").write_text(json.dumps(result, indent=2))
        shown = f"overall {result['bands']['overall']}" if result["ok"] else f"FAILED {result['error'][:60]}"
        print(f"[{n}/{len(todo)}] {key[:8]}../{exam_id} run {i}: {shown} ({result['seconds']}s)")


# --- the report -------------------------------------------------------------------------------

def load_runs(provider: str) -> dict[str, list[dict]]:
    runs = {}
    for f in sorted((RUNS / provider).glob("*/*.json")) if (RUNS / provider).exists() else []:
        r = json.loads(f.read_text())
        runs.setdefault(r["exam"], []).append(r)
    return runs


def pairwise(means: dict[str, dict], levels: dict[str, str], metric: str) -> tuple[int, int, int]:
    """(correct, tied, wrong) over every pair of exams whose intended levels differ."""
    right = tied = wrong = 0
    for a, b in combinations(means, 2):
        la, lb = LEVELS[levels[a]], LEVELS[levels[b]]
        if la == lb:
            continue
        (lo, hi) = (a, b) if la < lb else (b, a)
        if means[hi][metric] > means[lo][metric]:
            right += 1
        elif means[hi][metric] == means[lo][metric]:
            tied += 1
        else:
            wrong += 1
    return right, tied, wrong


def spearman(means: dict[str, dict], levels: dict[str, str], metric: str):
    from scipy.stats import spearmanr

    x = [LEVELS[levels[k]] for k in means]
    y = [means[k][metric] for k in means]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None                      # one level only, or every score identical: no correlation to compute
    return round(float(spearmanr(x, y).statistic), 2)


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def build_report(provider: str) -> str:
    runs = load_runs(provider)
    if not runs:
        sys.exit(f"No runs found for {provider}. Run `run --provider {provider}` first.")
    levels = labelled(load_labels())
    all_runs = [r for rs in runs.values() for r in rs]
    ok_runs = {k: [r for r in rs if r["ok"]] for k, rs in runs.items()}
    model = all_runs[0]["model"]

    out = [f"## {provider} ({model}), {date.today()}", "",
           "Temperature: not set by the app, so the provider default applies.", ""]

    # consistency: how much the same test's band moves between runs
    rows, stds = [], {m: [] for m in METRICS}
    for key, rs in sorted(ok_runs.items()):
        if not rs:
            continue
        row = [key.split("/")[1], levels.get(key, "-"), len(rs)]
        overall = [r["bands"]["overall"] for r in rs]
        row += [round(st.mean(overall), 2), round(max(overall) - min(overall), 1)]
        for m in rep.CRITERIA:
            vals = [r["bands"][m] for r in rs]
            sd = st.pstdev(vals)
            stds[m].append(sd)
            row.append(round(sd, 2))
        stds["overall"].append(st.pstdev(overall))
        rows.append(row)
    out += ["### Consistency (same saved test, scored repeatedly)", "",
            table(["Exam", "Level", "Runs", "Mean overall", "Overall range"] + [f"{SHORT[m]} sd" for m in rep.CRITERIA], rows), ""]
    identical = sum(1 for r in rows if r[4] == 0)
    out += [f"- Exams whose overall band never moved: {identical} of {len(rows)}",
            "- Average std dev per band: " + ", ".join(f"{SHORT[m]} {round(st.mean(v), 2)}" for m, v in stds.items() if v), ""]

    # reliability of the model call itself
    quotes = [q for r in all_runs if r["ok"] for q in r["quotes"].values()]
    kept, dropped = sum(q["kept"] for q in quotes), sum(q["dropped"] for q in quotes)
    failed = sum(1 for r in all_runs if not r["ok"])
    out += ["### Reliability", "",
            f"- Calls: {len(all_runs)}; malformed or failed: {failed} ({100 * failed / len(all_runs):.0f}%)",
            f"- Quotes: {kept + dropped} offered; {dropped} not found in the transcript "
            f"({100 * dropped / max(kept + dropped, 1):.0f}% rejected)",
            f"- Average seconds per report: {st.mean(r['seconds'] for r in all_runs):.1f}", ""]

    # ordering: do weak tests score below strong ones
    means = {k: {m: st.mean(r["bands"][m] for r in rs) for m in METRICS} for k, rs in ok_runs.items() if rs and k in levels}
    out += ["### Ordering (weak < medium < strong)", ""]
    if len(set(levels[k] for k in means)) < 2:
        out += ["Needs at least two different levels labelled in evals/labels.json.", ""]
    else:
        counts = {lv: sum(1 for k in means if levels[k] == lv) for lv in LEVELS}
        out += [f"Exams ranked: {len(means)} ({counts['weak']} weak, {counts['medium']} medium, {counts['strong']} strong)", "", table(
            ["Band", "Correct pairs", "Tied", "Wrong", "Spearman"],
            [[SHORT[m], f"{pairwise(means, levels, m)[0]}", pairwise(means, levels, m)[1], pairwise(means, levels, m)[2],
              spearman(means, levels, m)] for m in METRICS]), ""]
        out += [table(["Level", "Mean overall"],
                      [[lv, round(st.mean(means[k]["overall"] for k in means if levels[k] == lv), 2)]
                       for lv in LEVELS if counts[lv]]), ""]
    return "\n".join(out)


def cmd_report(args):
    text = build_report(args.provider)
    print(text)
    out = RESULTS_DIR / f"results_{args.provider}.md"
    out.write_text(text + "\n")
    print(f"\nSaved to {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-labels").set_defaults(fn=cmd_init_labels)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("run")
    p.add_argument("--provider", choices=("openai", "ollama"), default="openai")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--all", action="store_true", help="score every saved exam, labelled or not")
    p.add_argument("--only", help="score only exams whose 'session/exam' id contains this text")
    p.add_argument("--yes", action="store_true", help="don't ask before a large paid run")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--provider", choices=("openai", "ollama"), default="openai")
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
