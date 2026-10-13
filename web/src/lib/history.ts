// Helpers for the history sidebar: what kind of item is it, and how to group by day.
import type { Mode, SessionMeta } from "./api";

/** Mock tests are named by the server when they start. */
export const isTestSession = (s: Pick<SessionMeta, "name">): boolean => s.name.startsWith("IELTS mock test");

export const sessionMode = (s: Pick<SessionMeta, "name">): Mode => (isTestSession(s) ? "ielts" : "chat");

export interface HistoryGroup {
  label: string;
  items: SessionMeta[];
}

const DAY = 86400;

/** "Today", "Yesterday", "Previous 7 days", "Earlier" (newest first inside each). */
export function groupByDay(sessions: SessionMeta[], nowMs: number = Date.now()): HistoryGroup[] {
  const now = new Date(nowMs);
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
  const buckets: [string, (t: number) => boolean][] = [
    ["Today", (t) => t >= startOfToday],
    ["Yesterday", (t) => t >= startOfToday - DAY && t < startOfToday],
    ["Previous 7 days", (t) => t >= startOfToday - 7 * DAY && t < startOfToday - DAY],
    ["Earlier", () => true],
  ];
  const groups: HistoryGroup[] = buckets.map(([label]) => ({ label, items: [] }));
  for (const s of [...sessions].sort((a, b) => b.updated_at - a.updated_at)) {
    const i = buckets.findIndex(([, test]) => test(s.updated_at));
    groups[i].items.push(s);
  }
  return groups.filter((g) => g.items.length > 0);
}

export function matchesSearch(s: Pick<SessionMeta, "name" | "summary">, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return s.name.toLowerCase().includes(q) || (s.summary ?? "").toLowerCase().includes(q);
}

export function timeLabel(epochSeconds: number, nowMs: number = Date.now()): string {
  const d = new Date(epochSeconds * 1000);
  const today = new Date(nowMs);
  const sameDay = d.toDateString() === today.toDateString();
  return sameDay
    ? d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })
    : d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
