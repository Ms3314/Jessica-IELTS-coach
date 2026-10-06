// AudioWorklet processor: forwards mic input (channel 0) to the main thread in
// 4096-sample Float32 blocks. Runs off the main thread (unlike ScriptProcessor).
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(4096);
    this._len = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const ch = input[0];
      for (let i = 0; i < ch.length; i++) {
        this._buf[this._len++] = ch[i];
        if (this._len >= this._buf.length) {
          // Post a fresh buffer so it isn't detached/reused by the audio thread.
          this.port.postMessage(this._buf);
          this._buf = new Float32Array(4096);
          this._len = 0;
        }
      }
    }
    return true; // keep the processor alive
  }
}

registerProcessor("recorder-processor", RecorderProcessor);
