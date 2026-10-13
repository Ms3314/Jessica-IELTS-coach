// Geometry for the progress chart on the profile screen: pure functions, no React.

export interface ChartPoint {
  x: number;
  y: number;
  value: number;
}

export interface TrendChart {
  points: ChartPoint[];
  /** SVG path through the points ("" when there are fewer than two) */
  path: string;
  /** the band range shown on the vertical axis */
  min: number;
  max: number;
  /** whole bands to label on the axis */
  ticks: { value: number; y: number }[];
}

/**
 * Lay out overall bands (oldest to newest) in a width x height box with padding.
 * The vertical range hugs the data, always at least two bands tall and within 0-9,
 * so small changes stay visible.
 */
export function trendChart(values: number[], width: number, height: number, pad = 24): TrendChart {
  if (values.length === 0) return { points: [], path: "", min: 0, max: 9, ticks: [] };

  let min = Math.max(0, Math.floor(Math.min(...values) - 0.5));
  let max = Math.min(9, Math.ceil(Math.max(...values) + 0.5));
  if (max - min < 2) {
    // widen towards whichever end has room
    if (max + (2 - (max - min)) <= 9) max = min + 2;
    else min = max - 2;
  }

  const innerW = width - 2 * pad;
  const innerH = height - 2 * pad;
  const yFor = (v: number) => pad + innerH * (1 - (v - min) / (max - min));
  const xFor = (i: number) => (values.length === 1 ? width / 2 : pad + (innerW * i) / (values.length - 1));

  const points = values.map((value, i) => ({ x: xFor(i), y: yFor(value), value }));
  const path = points.length > 1 ? points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ") : "";
  const ticks: { value: number; y: number }[] = [];
  for (let v = min; v <= max; v++) ticks.push({ value: v, y: yFor(v) });
  return { points, path, min, max, ticks };
}

/** "+0.5", "−1.0", "no change" for the badge next to the latest score. */
export function changeLabel(change: number | null): string | null {
  if (change === null) return null;
  if (change === 0) return "no change";
  return `${change > 0 ? "+" : "−"}${Math.abs(change).toFixed(1)}`;
}
