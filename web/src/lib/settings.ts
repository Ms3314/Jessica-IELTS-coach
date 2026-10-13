import { useCallback, useState } from "react";

export interface Settings {
  /**
   * Let Jessica be cut off just by talking over her. Off by default: loud
   * rooms can trigger it by accident. The Stop button always works.
   */
  voiceInterrupt: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  voiceInterrupt: false,
};

const KEY = "speakielts.settings";

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
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
