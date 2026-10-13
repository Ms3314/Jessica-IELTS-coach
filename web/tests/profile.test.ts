import assert from "node:assert/strict";
import { test } from "node:test";
import { changeLabel, trendChart } from "../src/lib/profile.ts";

test("no tests, no chart", () => {
  const c = trendChart([], 400, 200);
  assert.deepEqual(c.points, []);
  assert.equal(c.path, "");
});

test("points run left to right in order and higher scores are higher on the chart", () => {
  const c = trendChart([5.5, 6.0, 7.0], 400, 200);
  assert.equal(c.points.length, 3);
  assert.ok(c.points[0].x < c.points[1].x && c.points[1].x < c.points[2].x);
  assert.ok(c.points[2].y < c.points[1].y && c.points[1].y < c.points[0].y, "SVG y grows downwards");
});

test("the points stay inside the padded box", () => {
  const c = trendChart([4.5, 9, 6], 400, 200, 24);
  for (const p of c.points) {
    assert.ok(p.x >= 24 && p.x <= 376, `x ${p.x}`);
    assert.ok(p.y >= 24 && p.y <= 176, `y ${p.y}`);
  }
});

test("a single test is a single centred point with no line", () => {
  const c = trendChart([6.5], 400, 200);
  assert.equal(c.points.length, 1);
  assert.equal(c.points[0].x, 200);
  assert.equal(c.path, "");
});

test("the vertical range hugs the data but is never narrower than two bands or outside 0-9", () => {
  const near = trendChart([6.0, 6.0], 400, 200);
  assert.ok(near.max - near.min >= 2);
  const top = trendChart([9, 9], 400, 200);
  assert.ok(top.max <= 9 && top.max - top.min >= 2);
  const bottom = trendChart([0.5], 400, 200);
  assert.ok(bottom.min >= 0);
});

test("the path starts with a move and has one segment per further point", () => {
  const c = trendChart([5, 6, 7, 8], 400, 200);
  assert.ok(c.path.startsWith("M"));
  assert.equal(c.path.split("L").length - 1, 3);
});

test("axis ticks are whole bands from the low end to the high end", () => {
  const c = trendChart([5.5, 7.5], 400, 200);
  assert.deepEqual(c.ticks.map((t) => t.value), Array.from({ length: c.max - c.min + 1 }, (_, i) => c.min + i));
});

test("change labels", () => {
  assert.equal(changeLabel(null), null);
  assert.equal(changeLabel(0), "no change");
  assert.equal(changeLabel(0.5), "+0.5");
  assert.equal(changeLabel(-1), "−1.0");
});
