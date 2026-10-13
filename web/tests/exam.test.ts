// Run with:  node --test web/tests/        (Node 22+ runs TypeScript directly)
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  INITIAL_EXAM,
  examReducer,
  formatClock,
  remainingS,
  stepStatus,
  type ExamAction,
  type ExamEvent,
  type ExamState,
} from "../src/lib/exam.ts";

const ev = (event: ExamEvent, now = 0): ExamAction => ({ type: "event", event, now });
const run = (actions: ExamAction[], from: ExamState = INITIAL_EXAM) => actions.reduce(examReducer, from);

test("start activates the exam and remembers its id and clock", () => {
  const s = run([ev({ event: "start", exam_id: "e1" }, 1000)]);
  assert.equal(s.active, true);
  assert.equal(s.examId, "e1");
  assert.equal(s.startedAt, 1000);
});

test("steps move the phase and part", () => {
  const s = run([
    ev({ event: "start", exam_id: "e1" }),
    ev({ event: "step", kind: "p1_section", part: 1 }),
  ]);
  assert.equal(s.phase, "part1");
  assert.equal(s.part, 1);
});

test("a part event starts that part's countdown", () => {
  const s = run([
    ev({ event: "start" }),
    ev({ event: "step", kind: "p1_section", part: 1 }),
    ev({ event: "part", part: 1, limit_s: 300 }, 5000),
  ]);
  assert.deepEqual(s.timer, { kind: "part", totalS: 300, startedAt: 5000 });
});

test("the Part 1 timer survives until the next step, which clears it", () => {
  const afterSetup = run([
    ev({ event: "start" }),
    ev({ event: "step", kind: "p1_section", part: 1 }),
    ev({ event: "part", part: 1, limit_s: 300 }),
    ev({ event: "step", kind: "p2_setup", part: 2 }),
  ]);
  assert.equal(afterSetup.timer, null);
  assert.equal(afterSetup.part, 2);
});

test("Part 2 has a prep countdown, then a talk countdown, and keeps the cue card", () => {
  let s = run([
    ev({ event: "start" }),
    ev({ event: "step", kind: "p2_setup", part: 2 }),
    ev({ event: "cue_card", card: { title: "Describe a book", points: ["a"], explain: "and explain", theme: "books" } }),
    ev({ event: "step", kind: "p2_prep", part: 2 }),
    ev({ event: "prep", seconds: 60 }, 100),
  ]);
  assert.equal(s.timer?.kind, "prep");
  assert.equal(s.timer?.totalS, 60);
  s = run([ev({ event: "step", kind: "p2_talk", part: 2 }), ev({ event: "talk", max_seconds: 120 }, 70000)], s);
  assert.equal(s.timer?.kind, "talk");
  assert.equal(s.card?.title, "Describe a book");
  assert.equal(s.phase, "part2_talk");
});

test("complete ends the call and waits for the report", () => {
  const s = run([ev({ event: "start" }), ev({ event: "complete", finished: true })]);
  assert.equal(s.active, false);
  assert.equal(s.complete, true);
  assert.equal(s.finished, true);
  assert.equal(s.analysing, true);
  assert.equal(s.timer, null);
});

test("the report arrives over the socket or by polling; the first one wins either way", () => {
  const report = { overall_band: 6.5 };
  const viaSocket = run([ev({ event: "start" }), ev({ event: "complete", finished: true }), ev({ event: "report", report })]);
  assert.equal(viaSocket.report?.overall_band, 6.5);
  assert.equal(viaSocket.analysing, false);
  const viaPoll = run([ev({ event: "start" }), ev({ event: "complete", finished: true }), { type: "report", report }]);
  assert.deepEqual(viaPoll.report, report);
});

test("hanging up early marks the exam over but unfinished", () => {
  const s = run([ev({ event: "start" }), ev({ event: "step", kind: "p1_section", part: 1 }), { type: "ended" }]);
  assert.equal(s.active, false);
  assert.equal(s.complete, true);
  assert.equal(s.finished, false);
});

test("hanging up with no exam running changes nothing", () => {
  assert.equal(examReducer(INITIAL_EXAM, { type: "ended" }), INITIAL_EXAM);
});

test("a failed report is surfaced, not swallowed", () => {
  const s = run([ev({ event: "start" }), ev({ event: "complete" }), ev({ event: "report_failed", message: "api down" })]);
  assert.equal(s.reportError, "api down");
  assert.equal(s.analysing, false);
});

test("a new test starts from a clean slate", () => {
  const dirty = run([ev({ event: "start", exam_id: "old" }), ev({ event: "complete", finished: true }), { type: "report", report: { overall_band: 7 } }]);
  const fresh = run([ev({ event: "start", exam_id: "new" }, 9)], dirty);
  assert.equal(fresh.report, null);
  assert.equal(fresh.complete, false);
  assert.equal(fresh.examId, "new");
  assert.equal(examReducer(dirty, { type: "reset" }), INITIAL_EXAM);
});

test("unknown events are ignored", () => {
  const s = run([ev({ event: "start" })]);
  assert.equal(examReducer(s, ev({ event: "something_new" })), s);
});

test("remaining time counts down and never goes below zero", () => {
  const t = { kind: "part" as const, totalS: 300, startedAt: 1000 };
  assert.equal(remainingS(t, 1000), 300);
  assert.equal(remainingS(t, 61000), 240);
  assert.equal(remainingS(t, 999999), 0);
});

test("clock formatting rounds partial seconds up so it never shows 0:00 early", () => {
  assert.equal(formatClock(300), "5:00");
  assert.equal(formatClock(59.2), "1:00");
  assert.equal(formatClock(61), "1:01");
  assert.equal(formatClock(0), "0:00");
  assert.equal(formatClock(-4), "0:00");
});

test("stepper status", () => {
  const mid = { part: 2 as const, complete: false };
  assert.equal(stepStatus(1, mid), "done");
  assert.equal(stepStatus(2, mid), "active");
  assert.equal(stepStatus(3, mid), "upcoming");
  assert.equal(stepStatus(3, { part: 3, complete: true }), "done");
});
