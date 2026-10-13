// The live conversation shown during a call: what you said and what Jessica said,
// built from the server's "user" and "agent" events. Pure functions (no React).

export interface Line {
  id: number;
  role: "user" | "assistant";
  text: string;
  interrupted?: boolean;
}

let nextId = 1;
const line = (role: Line["role"], text: string): Line => ({ id: nextId++, role, text });

/**
 * Add something the user said. When they keep talking after a cut-off reply the
 * server sends the earlier words again with the new ones joined on, so a message
 * that contains the previous one replaces it instead of appearing twice.
 */
export function addUserText(lines: Line[], text: string): Line[] {
  // the user's latest message, provided nothing but a cut-off reply of hers came after it
  let i = lines.length - 1;
  while (i >= 0 && lines[i].role === "assistant" && lines[i].interrupted) i--;
  const last = i >= 0 ? lines[i] : undefined;
  if (last && last.role === "user" && text.includes(last.text.replace(/\s*\.\.\.\s*$/, ""))) {
    return [...lines.slice(0, i), { ...last, text }, ...lines.slice(i + 1)];
  }
  return [...lines, line("user", text)];
}

/** Add one spoken sentence of Jessica's. `newLine` starts a new message. */
export function addAgentSentence(lines: Line[], sentence: string, newLine: boolean): Line[] {
  const last = lines[lines.length - 1];
  if (!newLine && last && last.role === "assistant") {
    return [...lines.slice(0, -1), { ...last, text: `${last.text} ${sentence}` }];
  }
  return [...lines, line("assistant", sentence)];
}

/** The user cut Jessica off: flag her latest message. */
export function markInterrupted(lines: Line[]): Line[] {
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].role === "assistant") {
      if (lines[i].interrupted) return lines;
      return [...lines.slice(0, i), { ...lines[i], interrupted: true }, ...lines.slice(i + 1)];
    }
  }
  return lines;
}
