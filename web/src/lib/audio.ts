// AudioEngine: the browser side of the voice pipeline.
//
//  * Captures the mic, resamples to 16 kHz, and emits Int16 PCM for the server
//    (which runs Whisper + the agent + Kokoro). Only emits while `capturing`.
//  * Plays the 24 kHz float32 TTS PCM streamed back from the server.
//  * Exposes live `mouth` and `mic` levels (0..1) for the avatar to read in its
//    own animation frame loop (no React re-render per frame).
//  * Exposes `visualSource`, a FrameSource<VisualFrame> for the audiocn bar
//    visualizer.
//
// A single AudioContext is shared by capture and playback. It must be created
// from a user gesture, so `init()` is called from the Start button. Mic capture
// uses an AudioWorklet (off the main thread) instead of the deprecated
// ScriptProcessorNode.

import { createFrameEmitter } from "@/lib/audio/frame-source";
import type { FrameEmitter } from "@/lib/audio/frame-source";
import type { VisualFrame } from "@/lib/audio/types";

const TARGET_RATE = 16000; // mic -> server (Whisper expects 16 kHz)
const TTS_RATE = 24000; // server -> browser (Kokoro outputs 24 kHz)
const VISUAL_BANDS = 24; // number of bars in the visualizer

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private micStream: MediaStream | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private analyser: AnalyserNode | null = null;
  private micAnalyser: AnalyserNode | null = null;
  private silentGain: GainNode | null = null;
  private levelGen = 0; // invalidates old rAF level loops after stop()/init()

  private ttsQueue: Float32Array[] = [];
  ttsPlaying = false;
  private currentSource: AudioBufferSourceNode | null = null;

  // Live levels, updated every animation frame, read by the avatar.
  mouth = 0; // 0 closed .. 1 open (driven by TTS playback)
  mic = 0; // 0 quiet .. 1 loud (driven by mic input)

  capturing = false;

  // Push visual frames to this source for the bar visualizer.
  visualSource: FrameEmitter<VisualFrame> = createFrameEmitter<VisualFrame>();

  // Called with an Int16Array of 16 kHz mono PCM to forward to the server.
  onPCM: (pcm: Int16Array) => void = () => {};

  async init(): Promise<void> {
    if (this.ctx) {
      await this.ctx.resume();
      return;
    }
    const ctx = new AudioContext();
    this.ctx = ctx;
    try {
      // Playback analyser (for lipsync). TTS sources connect here -> speakers.
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.connect(ctx.destination);
      this.analyser = analyser;

      // Load the mic-capture worklet.
      const workletUrl = `${import.meta.env.BASE_URL}worklets/recorder-processor.js`;
      await ctx.audioWorklet.addModule(workletUrl);

      // Mic capture.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
        },
      });
      this.micStream = stream;
      const source = ctx.createMediaStreamSource(stream);

      // Mic analyser for the visualizer.
      const micAnalyser = ctx.createAnalyser();
      micAnalyser.fftSize = 1024;
      source.connect(micAnalyser);
      this.micAnalyser = micAnalyser;

      const worklet = new AudioWorkletNode(ctx, "recorder-processor");
      // Mute any worklet output (it's silent anyway) before reaching the speakers.
      const silent = ctx.createGain();
      silent.gain.value = 0;
      worklet.port.onmessage = (e: MessageEvent<Float32Array>) => {
        const chunk = e.data;
        this.mic = Math.min(1, rms(chunk) * 3);
        if (!this.capturing) return;
        const resampled = resample(chunk, ctx.sampleRate, TARGET_RATE);
        this.onPCM(floatTo16(resampled));
      };
      source.connect(worklet);
      worklet.connect(silent);
      silent.connect(ctx.destination);
      this.workletNode = worklet;
      this.silentGain = silent;

      this.startLevelLoop();
    } catch (err) {
      // Leave no half-initialized state behind, so a later Start retries
      // cleanly (e.g. after the user denies the mic the first time).
      this.stop();
      throw err;
    }
  }

  setCapturing(active: boolean) {
    this.capturing = active;
  }

  /** Queue one TTS chunk (Float32Array, 24 kHz mono) for playback. */
  playTTS(f32: Float32Array) {
    this.ttsQueue.push(f32);
    if (!this.ttsPlaying) this.playNext();
  }

  /** Stop the reply immediately (barge-in) and drop anything queued. */
  stopPlayback() {
    this.ttsQueue = [];
    this.ttsPlaying = false;
    try {
      this.currentSource?.stop();
    } catch {
      /* already stopped */
    }
    this.currentSource = null;
  }

  private playNext() {
    if (!this.ctx || !this.analyser) return;
    const f32 = this.ttsQueue.shift();
    if (!f32 || f32.length === 0) {
      this.ttsPlaying = false;
      return;
    }
    this.ttsPlaying = true;
    const buf = this.ctx.createBuffer(1, f32.length, TTS_RATE);
    buf.getChannelData(0).set(f32);
    const node = this.ctx.createBufferSource();
    node.buffer = buf;
    node.connect(this.analyser);
    node.onended = () => {
      if (this.currentSource === node) this.currentSource = null;
      this.playNext();
    };
    this.currentSource = node;
    node.start();
  }

  private startLevelLoop() {
    if (!this.analyser) return;
    const gen = ++this.levelGen;
    const timeData = new Uint8Array(this.analyser.fftSize);
    const freqData = new Uint8Array(this.analyser.frequencyBinCount);
    const micFreqData = new Uint8Array(this.micAnalyser?.frequencyBinCount ?? 0);
    const tick = () => {
      if (gen !== this.levelGen) return; // stopped or re-initialized
      const analyser = this.analyser;
      if (analyser) {
        analyser.getByteTimeDomainData(timeData);
        let sum = 0;
        for (let i = 0; i < timeData.length; i++) {
          const v = (timeData[i] - 128) / 128;
          sum += v * v;
        }
        const r = Math.sqrt(sum / timeData.length);
        // Smooth toward the new amplitude so the mouth moves naturally.
        const target = Math.min(1, r * 3.2);
        this.mouth = this.mouth * 0.55 + target * 0.45;
      }

      // Push a visual frame to the bar visualizer.
      const activeAnalyser = this.ttsPlaying
        ? this.analyser
        : this.micAnalyser;
      const activeFreq = this.ttsPlaying ? freqData : micFreqData;
      if (activeAnalyser && activeFreq.length > 0) {
        activeAnalyser.getByteFrequencyData(activeFreq);
        this.visualSource.emit({
          bands: frequencyBands(activeFreq, VISUAL_BANDS),
          history: new Float32Array(0),
          historyStart: 0,
          historyLength: 0,
          peakDb: 0,
        });
      }

      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  stop() {
    this.levelGen++; // kill the level loop
    this.capturing = false;
    this.stopPlayback();
    this.workletNode?.disconnect();
    this.workletNode?.port.close();
    this.workletNode = null;
    this.silentGain?.disconnect();
    this.silentGain = null;
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.micStream = null;
    this.analyser?.disconnect();
    this.analyser = null;
    this.micAnalyser?.disconnect();
    this.micAnalyser = null;
    this.ctx?.close();
    this.ctx = null;
    this.mouth = 0;
    this.mic = 0;
  }
}

// --- DSP helpers ------------------------------------------------------------

function rms(x: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  return Math.sqrt(sum / x.length);
}

function resample(input: Float32Array, inRate: number, outRate: number): Float32Array {
  const ratio = inRate / outRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = pos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

function floatTo16(x: Float32Array): Int16Array {
  const out = new Int16Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const s = Math.max(-1, Math.min(1, x[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/** Resample raw FFT bins into a smaller number of frequency bands (0..1). */
function frequencyBands(data: Uint8Array, bandCount: number): Float32Array {
  const bands = new Float32Array(bandCount);
  const binCount = data.length;
  for (let i = 0; i < bandCount; i++) {
    const start = Math.floor(binCount * Math.pow(i / bandCount, 1.4));
    const end = Math.floor(binCount * Math.pow((i + 1) / bandCount, 1.4));
    let sum = 0;
    let n = 0;
    for (let j = start; j < end && j < binCount; j++) {
      sum += data[j] / 255;
      n++;
    }
    bands[i] = n > 0 ? sum / n : 0;
  }
  return bands;
}
