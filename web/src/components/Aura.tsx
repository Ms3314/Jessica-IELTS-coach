import { useEffect, useRef, type RefObject } from "react";
import type { AudioEngine } from "../lib/audio";
import type { EngineState } from "../lib/useVoice";

interface Props {
  engineRef: RefObject<AudioEngine | null>;
  state: EngineState;
}

// A LiveKit-Aura-style voice visual, recreated locally with canvas 2D: a
// glowing core plus three organically wobbling rings whose radius, speed, and
// brightness follow the live mic/TTS levels and the engine state. (The real
// LiveKit component needs a LiveKit room; we don't.)

const COLORS: Record<EngineState, [number, number, number]> = {
  idle: [91, 141, 239],
  listening: [57, 217, 138],
  transcribing: [242, 193, 78],
  thinking: [242, 161, 84],
  speaking: [240, 110, 169],
  error: [239, 87, 87],
};

export default function Aura({ engineRef, state }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let raf = 0;
    let level = 0.1;
    let color: [number, number, number] = [
      COLORS.idle[0],
      COLORS.idle[1],
      COLORS.idle[2],
    ];
    const started = performance.now();

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
    };
    resize();
    window.addEventListener("resize", resize);

    const tick = (now: number) => {
      const engine = engineRef.current;
      const s = stateRef.current;
      const t = ((now - started) / 1000) * (reduceMotion ? 0.25 : 1);

      // Target intensity from state + live audio level.
      const mic = engine?.mic ?? 0;
      const mouth = engine?.mouth ?? 0;
      let target: number;
      switch (s) {
        case "listening":
          target = 0.25 + mic * 0.95;
          break;
        case "speaking":
          target = 0.32 + mouth * 0.9;
          break;
        case "thinking":
        case "transcribing":
          target = 0.34 + 0.12 * Math.sin(t * 2.4);
          break;
        case "error":
          target = 0.4;
          break;
        default:
          target = 0.1 + 0.05 * Math.sin(t * 1.1);
      }
      level += (target - level) * 0.075;

      // Smooth color between states.
      const c = COLORS[s];
      color = [
        color[0] + (c[0] - color[0]) * 0.06,
        color[1] + (c[1] - color[1]) * 0.06,
        color[2] + (c[2] - color[2]) * 0.06,
      ];

      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const base = Math.min(w, h) * 0.155; // core radius
      const rgb = `${Math.round(color[0])}, ${Math.round(color[1])}, ${Math.round(
        color[2]
      )}`;

      ctx.globalCompositeOperation = "lighter";

      // Broad, very soft ambient glow.
      const glowR = base * (2.1 + level * 0.45);
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR);
      glow.addColorStop(0, `rgba(${rgb}, ${0.06 + level * 0.08})`);
      glow.addColorStop(0.55, `rgba(${rgb}, ${0.04 + level * 0.06})`);
      glow.addColorStop(1, `rgba(${rgb}, 0)`);
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, glowR, 0, Math.PI * 2);
      ctx.fill();

      // Glowing core.
      const coreR = base * (1.2 + level * 0.45);
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
      core.addColorStop(0, `rgba(${rgb}, ${0.32 + level * 0.38})`);
      core.addColorStop(0.55, `rgba(${rgb}, ${0.1 + level * 0.18})`);
      core.addColorStop(1, `rgba(${rgb}, 0)`);
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
      ctx.fill();

      // Bright inner rim.
      ctx.beginPath();
      ctx.arc(cx, cy, base * (0.9 + level * 0.14), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${rgb}, ${0.35 + level * 0.4})`;
      ctx.lineWidth = 2;
      ctx.shadowColor = `rgba(${rgb}, 0.9)`;
      ctx.shadowBlur = base * (0.3 + level * 0.45);
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Organically wobbling glow rings.
      const ring = (
        radius: number,
        amp: number,
        speed: number,
        phase: number,
        width: number,
        alpha: number
      ) => {
        ctx.beginPath();
        const N = 96;
        for (let i = 0; i <= N; i++) {
          const a = (i / N) * Math.PI * 2;
          const wob =
            Math.sin(a * 3 + t * speed + phase) * amp +
            Math.sin(a * 5 - t * speed * 0.7 + phase * 1.7) * amp * 0.6 +
            Math.sin(a * 8 + t * speed * 1.4) * amp * 0.35;
          const r = radius + wob * (0.5 + level);
          const x = cx + Math.cos(a) * r;
          const y = cy + Math.sin(a) * r;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = `rgba(${rgb}, ${alpha})`;
        ctx.lineWidth = width;
        ctx.shadowColor = `rgba(${rgb}, 0.9)`;
        ctx.shadowBlur = base * (0.16 + level * 0.24);
        ctx.stroke();
        ctx.shadowBlur = 0;
      };

      ring(base * (1.55 + level * 0.3), base * (0.06 + level * 0.09), 0.9, 0, 2.2, 0.45);
      ring(base * (1.95 + level * 0.4), base * (0.08 + level * 0.11), 0.6, 2.1, 1.5, 0.26);
      ring(base * (2.3 + level * 0.45), base * (0.1 + level * 0.13), 0.4, 4.2, 1.0, 0.14);

      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [engineRef]);

  return <canvas ref={canvasRef} className="aura-canvas" aria-hidden="true" />;
}
