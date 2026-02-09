/**
 * Audio playback with AudioWorklet (preferred) and ScriptProcessorNode fallback.
 * Receives 16kHz float32 chunks, upsamples to native rate, plays via ring buffer.
 */
class AudioPlayback {
    constructor() {
        this.audioCtx = null;
        this.workletNode = null;
        this.scriptNode = null;
        this.onLevel = null;  // callback(0-1 level)
        this.running = false;
        this._useWorklet = false;

        // Fallback ring buffer (for ScriptProcessorNode path)
        this._ringBuffer = null;
        this._ringWrite = 0;
        this._ringRead = 0;
        this._ringAvailable = 0;
        this._nativeSR = 48000;
        this._sourceSR = 16000;
    }

    async start() {
        this.audioCtx = new AudioContext();
        this._nativeSR = this.audioCtx.sampleRate;

        this._useWorklet = !!(this.audioCtx.audioWorklet);
        if (this._useWorklet) {
            try {
                await this.audioCtx.audioWorklet.addModule('/static/worklets/playback-processor.js');
                this.workletNode = new AudioWorkletNode(this.audioCtx, 'playback-processor');
                this.workletNode.connect(this.audioCtx.destination);
            } catch (e) {
                console.warn('Playback AudioWorklet failed, using ScriptProcessor:', e);
                this._useWorklet = false;
                this._startScriptProcessor();
            }
        } else {
            this._startScriptProcessor();
        }

        this.running = true;
    }

    _startScriptProcessor() {
        const bufferSize = 4096;
        // Ring buffer: ~2 seconds at native SR
        const ringSize = Math.ceil(this._nativeSR * 2);
        this._ringBuffer = new Float32Array(ringSize);
        this._ringWrite = 0;
        this._ringRead = 0;
        this._ringAvailable = 0;

        this.scriptNode = this.audioCtx.createScriptProcessor(bufferSize, 0, 1);
        this.scriptNode.onaudioprocess = (e) => {
            const output = e.outputBuffer.getChannelData(0);
            for (let i = 0; i < output.length; i++) {
                if (this._ringAvailable > 0) {
                    output[i] = this._ringBuffer[this._ringRead];
                    this._ringRead = (this._ringRead + 1) % this._ringBuffer.length;
                    this._ringAvailable--;
                } else {
                    output[i] = 0; // silence on underrun
                }
            }
        };
        this.scriptNode.connect(this.audioCtx.destination);
    }

    _upsampleAndEnqueue(samples16k) {
        // Linear interpolation from 16kHz to native SR
        const ratio = this._nativeSR / this._sourceSR;
        const outLen = Math.ceil(samples16k.length * ratio);
        const ring = this._ringBuffer;
        const ringLen = ring.length;

        for (let i = 0; i < outLen; i++) {
            const srcIdx = i / ratio;
            const idx0 = Math.floor(srcIdx);
            const idx1 = Math.min(idx0 + 1, samples16k.length - 1);
            const frac = srcIdx - idx0;
            const sample = samples16k[idx0] * (1 - frac) + samples16k[idx1] * frac;

            ring[this._ringWrite] = sample;
            this._ringWrite = (this._ringWrite + 1) % ringLen;
            this._ringAvailable++;

            // Prevent overflow: drop oldest
            if (this._ringAvailable >= ringLen) {
                this._ringRead = (this._ringRead + 1) % ringLen;
                this._ringAvailable = ringLen - 1;
            }
        }
    }

    enqueue(float32Array) {
        if (!this.running) return;

        // Calculate output level
        if (this.onLevel) {
            const rms = Math.sqrt(float32Array.reduce((s, v) => s + v * v, 0) / float32Array.length);
            this.onLevel(Math.min(1, rms * 5));
        }

        if (this._useWorklet && this.workletNode) {
            this.workletNode.port.postMessage(
                { type: 'audio', samples: float32Array },
                [float32Array.buffer]
            );
        } else if (this._ringBuffer) {
            this._upsampleAndEnqueue(float32Array);
        }
    }

    clear() {
        if (this._useWorklet && this.workletNode) {
            this.workletNode.port.postMessage({ type: 'clear' });
        } else if (this._ringBuffer) {
            this._ringWrite = 0;
            this._ringRead = 0;
            this._ringAvailable = 0;
        }
    }

    stop() {
        this.running = false;
        if (this.workletNode) {
            try { this.workletNode.port.postMessage({ type: 'stop' }); } catch (e) {}
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        if (this.scriptNode) {
            this.scriptNode.disconnect();
            this.scriptNode = null;
        }
        this._ringBuffer = null;
        if (this.audioCtx) {
            this.audioCtx.close();
            this.audioCtx = null;
        }
    }
}
