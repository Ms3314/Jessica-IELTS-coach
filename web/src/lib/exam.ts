// Client-side state for the IELTS mock test: what part we're in, which timer is
// running, the cue card, and the report. It is driven entirely by the "exam"
// events the server sends, so the server stays the source of truth for the test
// flow and the UI is just a view of it. Pure functions only (no React) so it is
// easy to test.

export interface CueCard {
  title: string;
  points: string[];
  explain: string;
  theme: string;
}

export interface Criterion {
  band: number;
  summary: string;
  strengths: string[];
  improvements: string[];
  evidence: string[];
  basis?: string;
}

export interface Report {
  exam_id?: string;
  insufficient?: boolean;
  message?: string;
  error?: string;
  overall_band?: number;
  reliability?: "low" | "medium" | "good";
  fluency_coherence?: Criterion;
  lexical_resource?: Criterion;
  grammatical_range_accuracy?: Criterion;
  pronunciation?: Criterion;
  top_priorities?: string[];
  examiner_note?: string;
  caveats?: string[];
  finished?: boolean;
  /** the scores were saved to the user's profile (progress over time) */
  profile_saved?: boolean;
}

export type Phase =
  | "intro"
  | "part1"
  | "part2_intro"
  | "part2_prep"
  | "part2_talk"
  | "part2_round"
  | "part3_intro"
  | "part3"
  | "end";

export type Part = 0 | 1 | 2 | 3;

/** A countdown. `startedAt` is the client clock (ms) when the server announced it. */
export interface Timer {
  kind: "part" | "prep" | "talk";
  totalS: number;
  startedAt: number;
}

/** The server is waiting for the candidate to record an answer. */
export interface AnswerWindow {
  maxS: number; // the recording ends by itself at this length
  waitS: number; // and the question is skipped if they never start within this long
  openedAt: number; // client clock (ms)
}

export interface ExamState {
  active: boolean; // a mock-test call is running
  length: string | null; // "full" | "short"
  answerWindow: AnswerWindow | null;
  examId: string | null;
  phase: Phase;
  part: Part;
  timer: Timer | null;
  card: CueCard | null;
  startedAt: number | null; // client clock (ms), for the total elapsed time
  complete: boolean; // the test is over (finished or ended early)
  finished: boolean; // ...and it ran to the end
  analysing: boolean;
  report: Report | null;
  reportError: string | null;
}

export const INITIAL_EXAM: ExamState = {
  active: false,
  length: null,
  answerWindow: null,
  examId: null,
  phase: "intro",
  part: 0,
  timer: null,
  card: null,
  startedAt: null,
  complete: false,
  finished: false,
  analysing: false,
  report: null,
  reportError: null,
};

/** Shape of the server's {"type":"exam", ...} events (only the fields we read). */
export interface ExamEvent {
  event: string;
  exam_id?: string;
  kind?: string;
  part?: number;
  limit_s?: number;
  seconds?: number;
  max_seconds?: number;
  max_s?: number;
  wait_s?: number;
  length?: string;
  card?: CueCard;
  finished?: boolean;
  report?: Report;
  message?: string;
}

export type ExamAction =
  | { type: "event"; event: ExamEvent; now: number }
  | { type: "ended" } // the user hung up
  | { type: "report"; report: Report }
  | { type: "reportError"; message: string }
  | { type: "retryReport" } // analyse again: wait for a report once more
  | { type: "reset" };

const PHASE_FOR_STEP: Record<string, Phase> = {
  intro: "intro",
  p1_section: "part1",
  p2_setup: "part2_intro",
  p2_prep: "part2_prep",
  p2_talk: "part2_talk",
  p2_round: "part2_round",
  p3_intro: "part3_intro",
  p3_section: "part3",
  end: "end",
};

export function examReducer(state: ExamState, action: ExamAction): ExamState {
  switch (action.type) {
    case "reset":
      return INITIAL_EXAM;

    case "ended":
      // hung up before the server could say so: the test is over, unfinished
      if (!state.active || state.complete) return state;
      return { ...state, active: false, complete: true, timer: null, answerWindow: null };

    case "report":
      return { ...state, report: action.report, analysing: false, reportError: null };

    case "reportError":
      return { ...state, analysing: false, reportError: action.message };

    case "retryReport":
      return { ...state, analysing: true, reportError: null, report: null };

    case "event": {
      const e = action.event;
      switch (e.event) {
        case "start":
          return {
            ...INITIAL_EXAM,
            active: true,
            examId: e.exam_id ?? null,
            length: e.length ?? null,
            startedAt: action.now,
          };
        case "step": {
          const phase = PHASE_FOR_STEP[e.kind ?? ""] ?? state.phase;
          // timers belong to a step; entering a step without one clears it
          const keepsTimer = phase === "part1" || phase === "part3";
          return {
            ...state,
            phase,
            part: ((e.part ?? 0) as Part) || state.part,
            timer: keepsTimer ? state.timer : null,
            answerWindow: null,
          };
        }
        case "answer_window":
          return {
            ...state,
            answerWindow: { maxS: e.max_s ?? 0, waitS: e.wait_s ?? 0, openedAt: action.now },
          };
        case "answer_end":
          return { ...state, answerWindow: null };
        case "part":
          return {
            ...state,
            timer: { kind: "part", totalS: e.limit_s ?? 0, startedAt: action.now },
          };
        case "cue_card":
          return { ...state, card: e.card ?? null };
        case "prep":
          return {
            ...state,
            timer: { kind: "prep", totalS: e.seconds ?? 0, startedAt: action.now },
          };
        case "talk":
          return {
            ...state,
            timer: { kind: "talk", totalS: e.max_seconds ?? 0, startedAt: action.now },
          };
        case "complete":
          return {
            ...state,
            active: false,
            complete: true,
            finished: Boolean(e.finished),
            timer: null,
            answerWindow: null,
            phase: "end",
            analysing: state.report === null,
          };
        case "analysing":
          return { ...state, analysing: true };
        case "report":
          return { ...state, report: e.report ?? null, analysing: false, reportError: null };
        case "report_failed":
          return { ...state, analysing: false, reportError: e.message ?? "The report could not be made." };
        default:
          return state;
      }
    }
  }
}

// --- helpers for the views -----------------------------------------------------------

/** Seconds left on a timer at client time `now` (never negative). */
export function remainingS(timer: Timer, now: number): number {
  return Math.max(0, timer.totalS - (now - timer.startedAt) / 1000);
}

export function formatClock(seconds: number): string {
  const s = Math.max(0, Math.ceil(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export type StepStatus = "done" | "active" | "upcoming";

/** Where each of Parts 1-3 stands, for the stepper. */
export function stepStatus(part: Part, state: Pick<ExamState, "part" | "complete">): StepStatus {
  if (state.complete) return "done";
  if (state.part === part) return "active";
  return state.part > part ? "done" : "upcoming";
}

export const PART_TITLES: Record<1 | 2 | 3, { name: string; blurb: string; duration: string }> = {
  1: { name: "Part 1", blurb: "Introduction & interview", duration: "about 5 min" },
  2: { name: "Part 2", blurb: "Long turn", duration: "1 min to prepare + up to 2 min talking" },
  3: { name: "Part 3", blurb: "Discussion", duration: "about 5 min" },
};

export const PHASE_LABEL: Record<Phase, string> = {
  intro: "Introduction",
  part1: "Interview",
  part2_intro: "Your topic",
  part2_prep: "Preparation time",
  part2_talk: "Your talk",
  part2_round: "Follow-up question",
  part3_intro: "Moving on",
  part3: "Discussion",
  end: "Finished",
};

export const BAND_CRITERIA: { key: keyof Report; title: string }[] = [
  { key: "fluency_coherence", title: "Fluency & Coherence" },
  { key: "lexical_resource", title: "Lexical Resource" },
  { key: "grammatical_range_accuracy", title: "Grammatical Range & Accuracy" },
  { key: "pronunciation", title: "Pronunciation" },
];
