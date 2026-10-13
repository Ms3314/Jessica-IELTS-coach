import assert from "node:assert/strict";
import { test } from "node:test";
import { addAgentSentence, addUserText, markInterrupted, type Line } from "../src/lib/transcript.ts";

const texts = (lines: Line[]) => lines.map((l) => `${l.role}:${l.text}`);

test("a conversation builds up in order", () => {
  let t: Line[] = [];
  t = addAgentSentence(t, "Hi, I'm Jessica.", true);
  t = addUserText(t, "Hello");
  t = addAgentSentence(t, "How are you?", true);
  assert.deepEqual(texts(t), ["assistant:Hi, I'm Jessica.", "user:Hello", "assistant:How are you?"]);
});

test("sentences of one reply join into a single message", () => {
  let t: Line[] = [];
  t = addAgentSentence(t, "First sentence.", true);
  t = addAgentSentence(t, "Second sentence.", false);
  assert.deepEqual(texts(t), ["assistant:First sentence. Second sentence."]);
});

test("a new reply starts a new message even right after another reply", () => {
  let t = addAgentSentence([], "One.", true);
  t = addAgentSentence(t, "Two.", true);
  assert.equal(t.length, 2);
});

test("after the user speaks, Jessica's next sentence is a new message regardless of the flag", () => {
  let t = addAgentSentence([], "Question?", true);
  t = addUserText(t, "Answer");
  t = addAgentSentence(t, "Follow-up.", false);
  assert.deepEqual(texts(t), ["assistant:Question?", "user:Answer", "assistant:Follow-up."]);
});

test("a stitched continuation replaces the earlier message instead of repeating it", () => {
  let t: Line[] = [];
  t = addUserText(t, "I like football");
  t = addAgentSentence(t, "Nice.", true);
  t = markInterrupted(t);
  t = addUserText(t, "I like football ... and I play weekly");
  assert.deepEqual(texts(t).filter((x) => x.startsWith("user")), ["user:I like football ... and I play weekly"]);
});

test("unrelated user messages are kept separate", () => {
  let t = addUserText([], "First thing");
  t = addUserText(t, "Something else entirely");
  assert.equal(t.length, 2);
});

test("marking an interruption flags only the latest assistant message", () => {
  let t = addAgentSentence([], "Old reply.", true);
  t = addUserText(t, "ok");
  t = addAgentSentence(t, "New reply.", true);
  t = markInterrupted(t);
  assert.equal(t[0].interrupted, undefined);
  assert.equal(t[2].interrupted, true);
  assert.equal(markInterrupted(t), t, "flagging twice changes nothing");
  assert.deepEqual(markInterrupted([]), []);
});

test("each message gets its own stable id", () => {
  const t = addAgentSentence(addUserText([], "a"), "b", true);
  assert.notEqual(t[0].id, t[1].id);
});
