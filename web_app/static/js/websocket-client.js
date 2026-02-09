/**
 * WebSocket client for real-time audio streaming.
 * Protocol: text frames (JSON metrics) + binary frames (PCM float32 audio).
 */
class WebSocketClient {
    constructor() {
        this.ws = null;
        this.onMetrics = null;   // callback(metricsObj)
        this.onAudio = null;     // callback(Float32Array)
        this.onControl = null;   // callback(msgObj)
        this.onError = null;     // callback(errorMsg)
        this.onClose = null;     // callback()
        this._pendingMetrics = null;
    }

    connect(path) {
        return new Promise((resolve, reject) => {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const url = `${protocol}//${location.host}${path}`;
            this.ws = new WebSocket(url);
            this.ws.binaryType = 'arraybuffer';

            this.ws.onopen = () => resolve();

            this.ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'audio_chunk') {
                        // Store metrics; next binary message is the audio
                        this._pendingMetrics = msg;
                    } else if (msg.type === 'error') {
                        if (this.onError) this.onError(msg.message);
                    } else {
                        if (this.onControl) this.onControl(msg);
                    }
                } else {
                    // Binary audio frame
                    const float32 = new Float32Array(event.data);
                    if (this._pendingMetrics && this.onMetrics) {
                        this.onMetrics(this._pendingMetrics);
                    }
                    if (this.onAudio) {
                        this.onAudio(float32);
                    }
                    this._pendingMetrics = null;
                }
            };

            this.ws.onerror = (err) => {
                if (this.onError) this.onError('WebSocket error');
                reject(err);
            };

            this.ws.onclose = () => {
                if (this.onClose) this.onClose();
            };
        });
    }

    sendAudio(float32Array) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(float32Array.buffer);
        }
    }

    sendControl(action, extra) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ action, ...extra }));
        }
    }

    sendText(obj) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(obj));
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    get isConnected() {
        return this.ws && this.ws.readyState === WebSocket.OPEN;
    }
}
