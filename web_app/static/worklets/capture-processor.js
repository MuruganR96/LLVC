/**
 * AudioWorklet processor for mic capture.
 * Accumulates samples at native sample rate, decimates to 16kHz,
 * and posts chunks of the configured size to the main thread.
 */
class CaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.targetSR = 16000;
        this.nativeSR = sampleRate; // provided by AudioWorklet global
        this.ratio = this.nativeSR / this.targetSR;
        this.chunkLen = options.processorOptions?.chunkLen || 208;
        this.buffer = new Float32Array(0);
        this.running = true;

        this.port.onmessage = (e) => {
            if (e.data.type === 'setChunkLen') {
                this.chunkLen = e.data.chunkLen;
                this.buffer = new Float32Array(0);
            } else if (e.data.type === 'stop') {
                this.running = false;
            }
        };
    }

    process(inputs) {
        if (!this.running) return false;
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const samples = input[0]; // Float32, native SR, 128 samples typically

        // Simple decimation: pick every ratio-th sample
        // For 48kHz -> 16kHz, ratio = 3, so pick every 3rd sample
        const decimated = this._decimate(samples);

        // Append to buffer
        const newBuf = new Float32Array(this.buffer.length + decimated.length);
        newBuf.set(this.buffer);
        newBuf.set(decimated, this.buffer.length);
        this.buffer = newBuf;

        // Emit full chunks
        while (this.buffer.length >= this.chunkLen) {
            const chunk = this.buffer.slice(0, this.chunkLen);
            this.port.postMessage({ type: 'chunk', audio: chunk }, [chunk.buffer]);
            this.buffer = this.buffer.slice(this.chunkLen);
        }

        return true;
    }

    _decimate(samples) {
        // Simple decimation with averaging for anti-aliasing
        const ratio = Math.round(this.ratio);
        if (ratio <= 1) return samples;

        const outLen = Math.floor(samples.length / ratio);
        const out = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) {
            let sum = 0;
            for (let j = 0; j < ratio; j++) {
                sum += samples[i * ratio + j];
            }
            out[i] = sum / ratio;
        }
        return out;
    }
}

registerProcessor('capture-processor', CaptureProcessor);
