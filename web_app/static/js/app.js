/**
 * Main application controller for LLVC Real-Time Voice Conversion.
 * Ties together: model selection, mic capture, file upload, WebSocket streaming,
 * memory chart, waveform visualization, and metrics display.
 */
document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let currentChunkLen = 832;
    let isModelLoaded = false;
    let isMicStreaming = false;
    let isFileStreaming = false;
    let uploadedFileId = null;
    let convertedChunks = [];
    let convertedBlobUrl = null;

    // --- Components ---
    const memoryChart = new MemoryChart('memory-chart');
    const wsClient = new WebSocketClient();
    const audioCapture = new AudioCapture();
    const audioPlayback = new AudioPlayback();

    // --- DOM Elements ---
    const modelSelect = document.getElementById('model-select');
    const loadModelBtn = document.getElementById('load-model-btn');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const modelInfo = document.getElementById('model-info');
    const modelParamsTag = document.getElementById('model-params-tag');
    const modelSrTag = document.getElementById('model-sr-tag');
    const modelLoadTag = document.getElementById('model-load-tag');

    const micStartBtn = document.getElementById('mic-start-btn');
    const micStopBtn = document.getElementById('mic-stop-btn');
    const chunkFactorSlider = document.getElementById('chunk-factor-slider');
    const chunkFactorVal = document.getElementById('chunk-factor-val');
    const chunkSizeDisplay = document.getElementById('chunk-size-display');
    const chunkMsDisplay = document.getElementById('chunk-ms-display');
    const inputLevel = document.getElementById('input-level');
    const outputLevel = document.getElementById('output-level');

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileChooseBtn = document.getElementById('file-choose-btn');
    const fileInfoDiv = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileDuration = document.getElementById('file-duration');
    const fileConvertBtn = document.getElementById('file-convert-btn');
    const fileStopBtn = document.getElementById('file-stop-btn');
    const fileProgressContainer = document.getElementById('file-progress-container');
    const fileProgress = document.getElementById('file-progress');
    const filePlayer = document.getElementById('file-player');
    const fileAudio = document.getElementById('file-audio');
    const fileDownloadBtn = document.getElementById('file-download-btn');
    const fileReconvertBtn = document.getElementById('file-reconvert-btn');

    const metricRtf = document.getElementById('metric-rtf');
    const metricLatency = document.getElementById('metric-latency');
    const metricChunks = document.getElementById('metric-chunks');
    const metricRss = document.getElementById('metric-rss');

    const clearChartBtn = document.getElementById('clear-chart-btn');
    const waveformCanvas = document.getElementById('waveform-canvas');
    const waveformCtx = waveformCanvas.getContext('2d');

    // Waveform buffers
    let inputWaveform = new Float32Array(0);
    let outputWaveform = new Float32Array(0);
    const WAVEFORM_LEN = 4000;

    // --- Initialize ---
    loadModelList();

    // --- Model Selection ---
    async function loadModelList() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            modelSelect.innerHTML = '<option value="">-- Select Speaker --</option>';
            for (const [key, info] of Object.entries(data.models)) {
                const opt = document.createElement('option');
                opt.value = key;
                opt.textContent = info.name;
                opt.title = info.description;
                if (key === data.current) opt.selected = true;
                modelSelect.appendChild(opt);
            }
            loadModelBtn.disabled = false;
            if (data.is_ready) {
                setStatus('ready', `Model loaded: ${data.current}`);
                isModelLoaded = true;
                enableControls();
            }
        } catch (e) {
            setStatus('error', 'Failed to fetch models');
        }
    }

    loadModelBtn.addEventListener('click', async () => {
        const key = modelSelect.value;
        if (!key) return;
        setStatus('loading', 'Loading model...');
        loadModelBtn.disabled = true;
        try {
            const res = await fetch('/api/models/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_key: key }),
            });
            const data = await res.json();
            if (data.status === 'loaded') {
                isModelLoaded = true;
                currentChunkLen = data.chunk_len;
                updateChunkDisplay();
                setStatus('ready', `Loaded: ${data.name}`);
                modelInfo.style.display = 'flex';
                modelParamsTag.textContent = `${data.params_M}M params`;
                modelSrTag.textContent = `${data.sr} Hz`;
                modelLoadTag.textContent = `${data.load_time_ms}ms load`;
                enableControls();
            } else {
                setStatus('error', data.error || 'Load failed');
            }
        } catch (e) {
            setStatus('error', 'Load failed: ' + e.message);
        }
        loadModelBtn.disabled = false;
    });

    function setStatus(state, text) {
        statusDot.className = 'status-dot ' + state;
        statusText.textContent = text;
    }

    function enableControls() {
        micStartBtn.disabled = !isModelLoaded;
        fileConvertBtn.disabled = !isModelLoaded || !uploadedFileId;
    }

    // --- Chunk Factor ---
    chunkFactorSlider.addEventListener('input', () => {
        const factor = parseInt(chunkFactorSlider.value);
        chunkFactorVal.textContent = factor;
        currentChunkLen = 208 * factor;
        updateChunkDisplay();
        audioCapture.setChunkLen(currentChunkLen);
        if (wsClient.isConnected) {
            wsClient.sendControl('set_chunk_factor', { value: factor });
        }
    });

    function updateChunkDisplay() {
        chunkSizeDisplay.textContent = currentChunkLen;
        chunkMsDisplay.textContent = (currentChunkLen / 16).toFixed(0);
    }

    // --- Metrics ---
    function updateMetrics(metrics) {
        if (metrics.rtf !== undefined) metricRtf.textContent = metrics.rtf.toFixed(3);
        if (metrics.inference_time_ms !== undefined) metricLatency.textContent = metrics.inference_time_ms.toFixed(1) + 'ms';
        if (metrics.chunk_index !== undefined) metricChunks.textContent = metrics.chunk_index;
        if (metrics.memory && metrics.memory.rss_mb !== undefined) {
            metricRss.textContent = metrics.memory.rss_mb.toFixed(1) + ' MB';
        }
        if (metrics.memory) {
            memoryChart.addDataPoint(metrics.chunk_index, metrics.memory);
        }
    }

    clearChartBtn.addEventListener('click', () => memoryChart.clear());

    // --- Waveform ---
    function appendWaveform(inputChunk, outputChunk) {
        inputWaveform = concatFloat32(inputWaveform, inputChunk, WAVEFORM_LEN);
        outputWaveform = concatFloat32(outputWaveform, outputChunk, WAVEFORM_LEN);
        drawWaveform();
    }

    function concatFloat32(existing, newData, maxLen) {
        const combined = new Float32Array(existing.length + newData.length);
        combined.set(existing);
        combined.set(newData, existing.length);
        if (combined.length > maxLen) {
            return combined.slice(combined.length - maxLen);
        }
        return combined;
    }

    function drawWaveform() {
        const w = waveformCanvas.width;
        const h = waveformCanvas.height;
        waveformCtx.fillStyle = '#12141c';
        waveformCtx.fillRect(0, 0, w, h);

        // Center line
        waveformCtx.strokeStyle = '#2a2d3a';
        waveformCtx.beginPath();
        waveformCtx.moveTo(0, h / 2);
        waveformCtx.lineTo(w, h / 2);
        waveformCtx.stroke();

        drawWaveformLine(inputWaveform, '#4f6ef7', 0.3, h * 0.25);
        drawWaveformLine(outputWaveform, '#2ea043', 0.8, h * 0.75);
    }

    function drawWaveformLine(data, color, alpha, yCenter) {
        if (data.length < 2) return;
        const w = waveformCanvas.width;
        const h = waveformCanvas.height;
        const step = Math.max(1, Math.floor(data.length / w));
        waveformCtx.strokeStyle = color;
        waveformCtx.globalAlpha = alpha;
        waveformCtx.lineWidth = 1;
        waveformCtx.beginPath();
        for (let i = 0; i < w; i++) {
            const idx = Math.floor(i * data.length / w);
            const val = data[idx] || 0;
            const y = yCenter - val * (h * 0.2);
            if (i === 0) waveformCtx.moveTo(i, y);
            else waveformCtx.lineTo(i, y);
        }
        waveformCtx.stroke();
        waveformCtx.globalAlpha = 1;
    }

    // --- Mic Streaming ---
    micStartBtn.addEventListener('click', startMic);
    micStopBtn.addEventListener('click', stopMic);

    async function startMic() {
        if (!isModelLoaded || isMicStreaming) return;
        try {
            micStartBtn.disabled = true;
            setStatus('loading', 'Starting mic...');

            // Start playback first
            await audioPlayback.start();

            // Connect WebSocket
            await wsClient.connect('/ws/stream');

            // Set chunk factor on server
            const factor = parseInt(chunkFactorSlider.value);
            wsClient.sendControl('set_chunk_factor', { value: factor });

            // Wait for config_updated
            await new Promise(resolve => {
                const origHandler = wsClient.onControl;
                wsClient.onControl = (msg) => {
                    if (msg.type === 'config_updated') {
                        currentChunkLen = msg.chunk_len;
                        updateChunkDisplay();
                        resolve();
                    }
                    if (origHandler) origHandler(msg);
                };
                // Timeout fallback
                setTimeout(resolve, 500);
            });

            // Wire up WebSocket callbacks
            let lastInputChunk = null;
            wsClient.onMetrics = (metrics) => updateMetrics(metrics);
            wsClient.onAudio = (float32) => {
                audioPlayback.enqueue(new Float32Array(float32));
                if (lastInputChunk) {
                    appendWaveform(lastInputChunk, float32);
                    lastInputChunk = null;
                }
            };
            wsClient.onError = (msg) => setStatus('error', msg);
            wsClient.onClose = () => {
                if (isMicStreaming) stopMic();
            };

            // Start mic capture
            audioCapture.onChunk = (chunk) => {
                lastInputChunk = chunk;
                wsClient.sendAudio(new Float32Array(chunk));
            };
            audioCapture.onLevel = (level) => {
                inputLevel.style.width = (level * 100) + '%';
            };
            audioPlayback.onLevel = (level) => {
                outputLevel.style.width = (level * 100) + '%';
            };

            await audioCapture.start(currentChunkLen);

            isMicStreaming = true;
            micStopBtn.disabled = false;
            setStatus('ready', 'Streaming...');
        } catch (e) {
            setStatus('error', 'Mic error: ' + e.message);
            micStartBtn.disabled = false;
        }
    }

    function stopMic() {
        isMicStreaming = false;
        audioCapture.stop();
        audioPlayback.stop();
        wsClient.disconnect();
        micStartBtn.disabled = !isModelLoaded;
        micStopBtn.disabled = true;
        inputLevel.style.width = '0%';
        outputLevel.style.width = '0%';
        setStatus('ready', 'Mic stopped');
    }

    // --- File Upload ---
    fileChooseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    async function handleFileSelect(file) {
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);

        setStatus('loading', 'Uploading...');
        try {
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();
            uploadedFileId = data.file_id;
            fileName.textContent = data.filename;
            fileDuration.textContent = data.duration_s.toFixed(2);
            fileInfoDiv.style.display = 'block';
            dropZone.style.display = 'none';
            enableControls();
            setStatus('ready', 'File uploaded');
        } catch (e) {
            setStatus('error', 'Upload failed: ' + e.message);
        }
    }

    fileConvertBtn.addEventListener('click', startFileConversion);
    fileStopBtn.addEventListener('click', stopFileConversion);

    async function startFileConversion() {
        if (!uploadedFileId || !isModelLoaded || isFileStreaming) return;
        isFileStreaming = true;
        fileConvertBtn.disabled = true;
        fileStopBtn.disabled = false;
        fileProgressContainer.style.display = 'block';
        fileProgress.style.width = '0%';
        filePlayer.style.display = 'none';
        convertedChunks = [];
        revokeConvertedBlob();

        try {
            await audioPlayback.start();

            const fileWs = new WebSocketClient();
            await fileWs.connect('/ws/file-stream');

            fileWs.onMetrics = (metrics) => {
                updateMetrics(metrics);
                if (metrics.total_chunks) {
                    const pct = (metrics.chunk_index / metrics.total_chunks) * 100;
                    fileProgress.style.width = pct + '%';
                }
            };
            fileWs.onAudio = (float32) => {
                const chunk = new Float32Array(float32);
                // Copy for replay buffer BEFORE enqueue (enqueue may transfer/detach the buffer)
                convertedChunks.push(new Float32Array(chunk));
                audioPlayback.enqueue(chunk);
                appendWaveform(new Float32Array(float32.length).fill(0), float32);
            };
            fileWs.onControl = (msg) => {
                if (msg.type === 'file_complete') {
                    fileProgress.style.width = '100%';
                    setStatus('ready', 'Conversion complete');
                    setTimeout(() => {
                        audioPlayback.stop();
                        isFileStreaming = false;
                        fileConvertBtn.disabled = !isModelLoaded;
                        fileStopBtn.disabled = true;
                        buildConvertedAudioPlayer();
                    }, 2000);
                } else if (msg.type === 'file_start') {
                    setStatus('ready', 'Converting...');
                }
            };
            fileWs.onError = (msg) => setStatus('error', msg);

            fileWs.sendText({ file_id: uploadedFileId });
            window._fileWs = fileWs;
        } catch (e) {
            setStatus('error', 'Conversion error: ' + e.message);
            isFileStreaming = false;
            fileConvertBtn.disabled = !isModelLoaded;
            fileStopBtn.disabled = true;
        }
    }

    function stopFileConversion() {
        isFileStreaming = false;
        if (window._fileWs) {
            window._fileWs.disconnect();
            window._fileWs = null;
        }
        audioPlayback.stop();
        fileConvertBtn.disabled = !isModelLoaded;
        fileStopBtn.disabled = true;
        setStatus('ready', 'Conversion stopped');
        if (convertedChunks.length > 0) {
            buildConvertedAudioPlayer();
        }
    }

    // --- Converted audio player ---
    function revokeConvertedBlob() {
        if (convertedBlobUrl) {
            URL.revokeObjectURL(convertedBlobUrl);
            convertedBlobUrl = null;
        }
    }

    function buildConvertedAudioPlayer() {
        if (convertedChunks.length === 0) return;

        // Merge all chunks into one Float32Array
        let totalLen = 0;
        for (const c of convertedChunks) totalLen += c.length;
        const merged = new Float32Array(totalLen);
        let offset = 0;
        for (const c of convertedChunks) {
            merged.set(c, offset);
            offset += c.length;
        }

        // Encode as 16-bit PCM WAV at 16kHz
        const wavBlob = encodeWav(merged, 16000);
        revokeConvertedBlob();
        convertedBlobUrl = URL.createObjectURL(wavBlob);
        fileAudio.src = convertedBlobUrl;
        filePlayer.style.display = 'block';
    }

    function encodeWav(samples, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const byteRate = sampleRate * numChannels * bitsPerSample / 8;
        const blockAlign = numChannels * bitsPerSample / 8;
        const dataSize = samples.length * blockAlign;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        // RIFF header
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeString(view, 8, 'WAVE');
        // fmt chunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        // data chunk
        writeString(view, 36, 'data');
        view.setUint32(40, dataSize, true);

        // Write PCM samples (float32 -> int16)
        let pos = 44;
        for (let i = 0; i < samples.length; i++) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            s = s < 0 ? s * 0x8000 : s * 0x7FFF;
            view.setInt16(pos, s, true);
            pos += 2;
        }
        return new Blob([buffer], { type: 'audio/wav' });
    }

    function writeString(view, offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    fileDownloadBtn.addEventListener('click', () => {
        if (!convertedBlobUrl) return;
        const a = document.createElement('a');
        a.href = convertedBlobUrl;
        const srcName = fileName.textContent || 'audio';
        const baseName = srcName.replace(/\.[^.]+$/, '');
        a.download = baseName + '_converted.wav';
        a.click();
    });

    fileReconvertBtn.addEventListener('click', () => {
        filePlayer.style.display = 'none';
        fileAudio.pause();
        fileAudio.src = '';
        revokeConvertedBlob();
        convertedChunks = [];
        startFileConversion();
    });

    // --- Initial waveform draw ---
    drawWaveform();
});
