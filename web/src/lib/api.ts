// Everything the UI asks the server for, in one place.

export type Mode = "chat" | "ielts";

export interface SessionMeta {
  id: string;
  name: string;
  color: string;
  summary: string | null;
  updated_at: number; // epoch seconds
}

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  interrupted: boolean;
}

export interface ExamListItem {
  exam_id: string;
  started_at: number;
  finished: boolean;
  turns: number;
  has_report: boolean;
}

export interface ScoreRow {
  session_id: string;
  exam_id: string;
  taken_at: number; // epoch seconds
  length: "full" | "short";
  finished: boolean;
  overall: number;
  fluency: number | null;
  lexical: number | null;
  grammar: number | null;
  pronunciation: number | null;
  reliability: string | null;
}

export interface ProfileSummary {
  tests: number;
  latest: number | null;
  best: number | null;
  average: number | null;
  change: number | null; // latest minus the test before it
  criteria_average: Record<"fluency" | "lexical" | "grammar" | "pronunciation", number | null>;
  trend: number[]; // overall bands, oldest to newest
}

export interface ProfileData {
  summary: ProfileSummary;
  history: ScoreRow[];
}

export interface KeyStatus {
  set: boolean;
  source: "settings" | "environment" | null;
  hint: string | null;
}

export interface ServerSettings {
  keys: { openai_api_key: KeyStatus; tavily_api_key: KeyStatus };
  model: string;
  provider: string;
}

async function getJson<T>(url: string, fallback: T): Promise<T> {
  try {
    const res = await fetch(url);
    return res.ok ? ((await res.json()) as T) : fallback;
  } catch {
    return fallback;
  }
}

const id = encodeURIComponent;

export const fetchSessions = () => getJson<SessionMeta[]>("/sessions", []);
export const fetchMessages = (sessionId: string) =>
  getJson<ChatMessage[]>(`/sessions/${id(sessionId)}/messages`, []);
export const fetchExams = (sessionId: string) => getJson<ExamListItem[]>(`/exams/${id(sessionId)}`, []);
export const fetchExam = (sessionId: string, examId: string) =>
  getJson<any | null>(`/exams/${id(sessionId)}/${id(examId)}`, null);
export const fetchProfile = () => getJson<ProfileData | null>("/profile", null);
export const fetchServerSettings = () => getJson<ServerSettings | null>("/settings", null);

export async function saveKeys(body: {
  openai_api_key?: string;
  tavily_api_key?: string;
}): Promise<{ ok: boolean; error?: string; note?: string | null }> {
  try {
    const res = await fetch("/settings/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    return res.ok ? { ok: true, note: data.note } : { ok: false, error: data.error ?? "Could not save the key." };
  } catch {
    return { ok: false, error: "Could not reach the server." };
  }
}

export async function updateSession(sessionId: string, patch: { name?: string; color?: string }) {
  await fetch(`/sessions/${id(sessionId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

export async function deleteSession(sessionId: string) {
  await fetch(`/sessions/${id(sessionId)}`, { method: "DELETE" });
}
