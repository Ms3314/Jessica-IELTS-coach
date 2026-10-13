import { useCallback, useState } from "react";

export interface Settings {
  /**
   * Let Jessica be cut off just by talking over her. Off by default: loud
   * rooms can trigger it by accident. The Stop button always works.
   */
  voiceInterrupt: boolean;
  /**
   * Seconds of quiet before Jessica decides you have finished and answers.
   * Shorter = faster replies; longer = more room to pause and think.
   */
  pauseS: number;
  /** Which mock test to take: the full one (about 14 min) or a short one (about 6 min). */
  examLength: ExamLength;
}

export type ExamLength = "full" | "short";

export const PAUSE_RANGE = { min: 0.8, max: 3.0, step: 0.1 } as const;

export const DEFAULT_SETTINGS: Settings = {
  voiceInterrupt: false,
  pauseS: 1.2,
  examLength: "full",
};

const KEY = "speakielts.settings";

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const saved = { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
      // guard against a hand-edited or stale value
      const p = Number(saved.pauseS);
      saved.pauseS = Number.isFinite(p)
        ? Math.min(PAUSE_RANGE.max, Math.max(PAUSE_RANGE.min, p))
        : DEFAULT_SETTINGS.pauseS;
      if (saved.examLength !== "full" && saved.examLength !== "short") saved.examLength = DEFAULT_SETTINGS.examLength;
      return saved;
    }
  } catch {
    /* storage blocked or corrupt: fall back to defaults */
  }
  return DEFAULT_SETTINGS;
}

/** App settings, persisted in this browser's localStorage. */
export function useSettings() {
  const [settings, setSettings] = useState<Settings>(load);

  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      try {
        localStorage.setItem(KEY, JSON.stringify(next));
      } catch {
        /* not persisted; still applies for this visit */
      }
      return next;
    });
  }, []);

  return { settings, update };
}
