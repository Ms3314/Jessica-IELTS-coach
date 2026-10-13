import assert from "node:assert/strict";
import { test } from "node:test";
import { groupByDay, isTestSession, matchesSearch, sessionMode } from "../src/lib/history.ts";
import type { SessionMeta } from "../src/lib/api.ts";

const NOW = new Date(2026, 9, 13, 15, 0, 0).getTime(); // 13 Oct 2026, 15:00 local
const at = (daysAgo: number, hour = 12) => new Date(2026, 9, 13 - daysAgo, hour, 0, 0).getTime() / 1000;
const s = (id: string, name: string, updated_at: number, summary: string | null = null): SessionMeta => ({
  id, name, color: "#000", summary, updated_at,
});

test("sessions are grouped by day, newest first inside each group", () => {
  const groups = groupByDay(
    [s("a", "Old", at(30)), s("b", "Today early", at(0, 8)), s("c", "Yesterday", at(1)), s("d", "Today late", at(0, 14)), s("e", "Last week", at(4))],
    NOW
  );
  assert.deepEqual(groups.map((g) => g.label), ["Today", "Yesterday", "Previous 7 days", "Earlier"]);
  assert.deepEqual(groups[0].items.map((i) => i.id), ["d", "b"]);
});

test("empty groups are not shown", () => {
  assert.deepEqual(groupByDay([s("a", "x", at(0, 9))], NOW).map((g) => g.label), ["Today"]);
  assert.deepEqual(groupByDay([], NOW), []);
});

test("a session just after midnight is yesterday, not today", () => {
  const justBeforeMidnight = new Date(2026, 9, 12, 23, 59, 0).getTime() / 1000;
  assert.equal(groupByDay([s("a", "x", justBeforeMidnight)], NOW)[0].label, "Yesterday");
});

test("mock tests are recognised by their name", () => {
  assert.equal(isTestSession({ name: "IELTS mock test 13 Oct 22:30" }), true);
  assert.equal(isTestSession({ name: "Talking about DSA" }), false);
  assert.equal(sessionMode({ name: "IELTS mock test 1" }), "ielts");
  assert.equal(sessionMode({ name: "Session abc" }), "chat");
});

test("search matches names and summaries, ignoring case and spaces", () => {
  assert.equal(matchesSearch({ name: "Talking about DSA", summary: null }, "  dsa "), true);
  assert.equal(matchesSearch({ name: "Session 1", summary: "We discussed compilers" }, "COMPILER"), true);
  assert.equal(matchesSearch({ name: "Session 1", summary: null }, "compiler"), false);
  assert.equal(matchesSearch({ name: "x", summary: null }, ""), true);
});
