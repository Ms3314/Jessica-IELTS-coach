import { useEffect, useRef, type RefObject } from "react";
import type { AudioEngine } from "../lib/audio";
import type { EngineState } from "../lib/useVoice";

interface Props {
  engineRef: RefObject<AudioEngine | null>;
  state: EngineState;
  color: string; // hex colour for this session
}

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  const bigint = parseInt(clean, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return [r, g, b];
}

// A LiveKit-Aura-style voice visual, recreated locally with canvas 2D: a
// glowing core plus three organically wobbling rings whose radius, speed, and
// brightness follow the live mic/TTS levels. The colour is constant per
// session (set by the user in the sidebar) instead of flipping between user-
// turn and AI-turn hues.
export default function Aura({ engineRef, state, color }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;
  const colorRef = useRef(hexToRgb(color));
  colorRef.current = hexToRgb(color);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let raf = 0;
    let level = 0.1;
    let currentColor: [number, number, number] = [colorRef.current[0], colorRef.current[1], colorRef.current[2]];
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

      const mic = engine?.mic ?? 0;
      const mouth = engine?.mouth ?? 0;
      let target: number;
      switch (s) {
        case "listening":
          target = 0.25 + mic * 0.95;
          break;
        case "speaking":
          target = 0.35 + mouth * 1.15;
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
      level += (target - level) * 0.12;

      // Smoothly blend toward the session colour (instant on first frame).
      const c = colorRef.current;
      currentColor = [
        currentColor[0] + (c[0] - currentColor[0]) * 0.06,
        currentColor[1] + (c[1] - currentColor[1]) * 0.06,
        currentColor[2] + (c[2] - currentColor[2]) * 0.06,
      ];

      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const base = Math.min(w, h) * 0.155;
      const motion = t * (1 + level * 1.1);
      const rgb = `${Math.round(currentColor[0])}, ${Math.round(
        currentColor[1]
      )}, ${Math.round(currentColor[2])}`;

      ctx.globalCompositeOperation = "lighter";

      const glowR = base * (2.1 + level * 0.45);
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR);
      glow.addColorStop(0, `rgba(${rgb}, ${0.06 + level * 0.08})`);
      glow.addColorStop(0.55, `rgba(${rgb}, ${0.04 + level * 0.06})`);
      glow.addColorStop(1, `rgba(${rgb}, 0)`);
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, glowR, 0, Math.PI * 2);
      ctx.fill();

      const coreR = base * (1.2 + level * 0.45);
      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, coreR);
      core.addColorStop(0, `rgba(${rgb}, ${0.32 + level * 0.38})`);
      core.addColorStop(0.55, `rgba(${rgb}, ${0.1 + level * 0.18})`);
      core.addColorStop(1, `rgba(${rgb}, 0)`);
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, base * (0.9 + level * 0.14), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${rgb}, ${0.35 + level * 0.4})`;
      ctx.lineWidth = 2;
      ctx.shadowColor = `rgba(${rgb}, 0.9)`;
      ctx.shadowBlur = base * (0.3 + level * 0.45);
      ctx.stroke();
      ctx.shadowBlur = 0;

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
            Math.sin(a * 3 + motion * speed + phase) * amp +
            Math.sin(a * 5 - motion * speed * 0.7 + phase * 1.7) * amp * 0.6 +
            Math.sin(a * 8 + motion * speed * 1.4) * amp * 0.35;
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

      ring(base * (1.55 + level * 0.5), base * (0.06 + level * 0.16), 0.9, 0, 2.2, 0.45);
      ring(base * (1.9 + level * 0.6), base * (0.075 + level * 0.18), 0.6, 2.1, 1.5, 0.26);
      ring(base * (2.2 + level * 0.55), base * (0.09 + level * 0.2), 0.4, 4.2, 1.0, 0.14);

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
