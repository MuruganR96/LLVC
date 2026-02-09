/**
 * Mic audio capture with AudioWorklet (preferred) and ScriptProcessorNode fallback.
 * Captures from microphone, resamples to 16kHz, emits chunks.
 */
class AudioCapture {
    constructor() {
        this.audioCtx = null;
        this.stream = null;
        this.workletNode = null;
        this.scriptNode = null;
        this.sourceNode = null;
        this.onChunk = null;     // callback(Float32Array)
        this.onLevel = null;     // callback(0-1 level)
        this.chunkLen = 208;
        this.running = false;
        this._buffer = new Float32Array(0);
        this._useWorklet = false;
    }

    async start(chunkLen) {
        this.chunkLen = chunkLen || this.chunkLen;
        this._buffer = new Float32Array(0);

        this.stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
            }
        });

        this.audioCtx = new AudioContext();
        this.sourceNode = this.audioCtx.createMediaStreamSource(this.stream);

        // Try AudioWorklet first, fall back to ScriptProcessorNode
        this._useWorklet = !!(this.audioCtx.audioWorklet);
        if (this._useWorklet) {
            try {
                await this.audioCtx.audioWorklet.addModule('/static/worklets/capture-processor.js');
                this._startWorklet();
            } catch (e) {
                console.warn('AudioWorklet failed, falling back to ScriptProcessor:', e);
                this._useWorklet = false;
                this._startScriptProcessor();
            }
        } else {
            this._startScriptProcessor();
        }

        this.running = true;
    }

    _startWorklet() {
        this.workletNode = new AudioWorkletNode(this.audioCtx, 'capture-processor', {
            processorOptions: { chunkLen: this.chunkLen },
        });

        this.workletNode.port.onmessage = (e) => {
            if (e.data.type === 'chunk') {
                this._emitChunk(e.data.audio);
            }
        };

        this.sourceNode.connect(this.workletNode);
        // Connect then disconnect so worklet stays alive without outputting raw mic
        this.workletNode.connect(this.audioCtx.destination);
        this.workletNode.disconnect(this.audioCtx.destination);
    }

    _startScriptProcessor() {
        // ScriptProcessorNode fallback — bufferSize 4096 is a safe default
        const bufferSize = 4096;
        this.scriptNode = this.audioCtx.createScriptProcessor(bufferSize, 1, 1);
        const nativeSR = this.audioCtx.sampleRate;
        const targetSR = 16000;
        const ratio = Math.round(nativeSR / targetSR);

        this.scriptNode.onaudioprocess = (e) => {
            if (!this.running) return;
            const input = e.inputBuffer.getChannelData(0);

            // Decimate: average every `ratio` samples
            const decimatedLen = Math.floor(input.length / ratio);
            const decimated = new Float32Array(decimatedLen);
            for (let i = 0; i < decimatedLen; i++) {
                let sum = 0;
                for (let j = 0; j < ratio; j++) {
                    sum += input[i * ratio + j];
                }
                decimated[i] = sum / ratio;
            }

            // Accumulate in buffer
            const newBuf = new Float32Array(this._buffer.length + decimated.length);
            newBuf.set(this._buffer);
            newBuf.set(decimated, this._buffer.length);
            this._buffer = newBuf;

            // Emit full chunks
            while (this._buffer.length >= this.chunkLen) {
                const chunk = this._buffer.slice(0, this.chunkLen);
                this._buffer = this._buffer.slice(this.chunkLen);
                this._emitChunk(chunk);
            }

            // Output silence (don't play back raw mic)
            const output = e.outputBuffer.getChannelData(0);
            for (let i = 0; i < output.length; i++) output[i] = 0;
        };

        this.sourceNode.connect(this.scriptNode);
        this.scriptNode.connect(this.audioCtx.destination);
    }

    _emitChunk(chunk) {
        if (this.onChunk) this.onChunk(chunk);
        if (this.onLevel) {
            const rms = Math.sqrt(chunk.reduce((s, v) => s + v * v, 0) / chunk.length);
            this.onLevel(Math.min(1, rms * 5));
        }
    }

    setChunkLen(chunkLen) {
        this.chunkLen = chunkLen;
        this._buffer = new Float32Array(0);
        if (this._useWorklet && this.workletNode) {
            this.workletNode.port.postMessage({ type: 'setChunkLen', chunkLen });
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
        if (this.sourceNode) {
            this.sourceNode.disconnect();
            this.sourceNode = null;
        }
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
        if (this.audioCtx) {
            this.audioCtx.close();
            this.audioCtx = null;
        }
    }
}
