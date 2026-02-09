/**
 * AudioWorklet processor for output playback.
 * Receives 16kHz audio chunks, upsamples to native rate, outputs via ring buffer.
 */
class PlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sourceSR = 16000;
        this.nativeSR = sampleRate;
        this.ratio = this.nativeSR / this.sourceSR;

        // Ring buffer (holds ~2 seconds at native SR)
        this.bufferSize = Math.ceil(this.nativeSR * 2);
        this.ringBuffer = new Float32Array(this.bufferSize);
        this.writePos = 0;
        this.readPos = 0;
        this.available = 0;
        this.running = true;

        this.port.onmessage = (e) => {
            if (e.data.type === 'audio') {
                this._enqueue(e.data.samples);
            } else if (e.data.type === 'stop') {
                this.running = false;
            } else if (e.data.type === 'clear') {
                this.writePos = 0;
                this.readPos = 0;
                this.available = 0;
            }
        };
    }

    _enqueue(samples16k) {
        // Upsample from 16kHz to native SR using linear interpolation
        const ratio = this.ratio;
        const outLen = Math.ceil(samples16k.length * ratio);

        for (let i = 0; i < outLen; i++) {
            const srcIdx = i / ratio;
            const idx0 = Math.floor(srcIdx);
            const idx1 = Math.min(idx0 + 1, samples16k.length - 1);
            const frac = srcIdx - idx0;
            const sample = samples16k[idx0] * (1 - frac) + samples16k[idx1] * frac;

            this.ringBuffer[this.writePos] = sample;
            this.writePos = (this.writePos + 1) % this.bufferSize;
            this.available++;

            // Prevent overflow
            if (this.available >= this.bufferSize) {
                this.readPos = (this.readPos + 1) % this.bufferSize;
                this.available = this.bufferSize - 1;
            }
        }
    }

    process(inputs, outputs) {
        if (!this.running) return false;
        const output = outputs[0];
        if (!output || !output[0]) return true;

        const channel = output[0];
        for (let i = 0; i < channel.length; i++) {
            if (this.available > 0) {
                channel[i] = this.ringBuffer[this.readPos];
                this.readPos = (this.readPos + 1) % this.bufferSize;
                this.available--;
            } else {
                channel[i] = 0; // silence on underrun
            }
        }

        return true;
    }
}

registerProcessor('playback-processor', PlaybackProcessor);
