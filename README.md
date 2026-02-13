# Optimized-LLVC: Low-Latency Real-Time Voice Conversion

A production-ready, CPU-optimized fork of [Koe AI's LLVC](https://koe.ai/papers/llvc.pdf) with a real-time web interface for streaming voice conversion.

**Paper:** [Low-latency Real-time Voice Conversion on CPU](https://arxiv.org/abs/2311.00873)
**Demo samples:** https://koeai.github.io/llvc-demo/
**Original repo:** https://github.com/KoeAI/llvc

---

## What's New in This Fork

- **Real-time web UI** -- FastAPI + WebSocket app for live mic-to-speaker voice conversion and file upload conversion, all running on CPU
- **Dynamic speaker selection** -- auto-discovers `.pth` speaker checkpoints from the model directory; add new speakers by dropping in a checkpoint file
- **Performance tuning** -- default chunk factor of 4 (832 samples / 52ms) achieves RTF ~1.92 on CPU (92% headroom for real-time)
- **Memory monitoring** -- live RSS + tracemalloc charts per inference frame via Chart.js
- **Benchmark script** -- automated latency, RTF, and memory profiling for streaming and non-streaming modes
- **Converted audio replay** -- after file conversion, play back, download as WAV, or re-convert with different settings

---

## Architecture

LLVC uses a causal encoder-decoder with a cached dilated convolution prenet, designed for chunk-based streaming on CPU:

```
Input Audio (16kHz) --> CachedConvNet Prenet (12 layers, dilated causal conv)
                                |
                        --> Encoder (8 attention layers, dim=512)
                                |
                        --> Decoder (1 causal attention layer, dim=256)
                                |
                        --> Output Audio (16kHz)
```

| Parameter       | Value     |
|-----------------|-----------|
| Sample rate     | 16,000 Hz |
| Latency (L)     | 16 samples (1ms) |
| Chunk size      | 208 x chunk_factor samples |
| Default chunk   | 832 samples (52ms) at chunk_factor=4 |
| Parameters      | ~3.26M    |
| Discriminator   | HiFi-GAN (8 periods) |

---

## Project Structure

```
Optimized-LLVC/
|-- model.py                  # LLVC model architecture (Net class)
|-- train.py                  # Distributed training with AMP + TensorBoard
|-- infer.py                  # CLI inference (streaming & non-streaming)
|-- utils.py                  # Checkpoint loading, audio utils, MFCC
|-- benchmark.py              # Latency / RTF / memory profiling
|-- dataset.py                # Training dataset (aligned audio pairs)
|-- mel_processing.py         # Mel spectrogram generation
|-- cached_convnet.py         # Causal convolution with state caching
|-- discriminators.py         # Multi-scale discriminators
|-- hfg_disc.py               # HiFi-GAN discriminators
|-- compare_infer.py          # Side-by-side RVC / QuickVC / LLVC comparison
|-- eval.py                   # Speaker similarity + WVMOS evaluation
|-- download_models.py        # Download pretrained models from HuggingFace
|
|-- experiments/
|   |-- llvc/config.json      # Standard LLVC config
|   |-- llvc_hfg/config.json  # LLVC + HiFi-GAN discriminator (default)
|   +-- llvc_nc/config.json   # Non-causal variant
|
|-- web_app/                  # Real-time web interface
|   |-- app.py                # FastAPI backend (REST + WebSocket)
|   |-- audio_processor.py    # Streaming inference engine
|   |-- memory_monitor.py     # psutil + tracemalloc monitor
|   |-- templates/
|   |   +-- index.html        # Single-page UI
|   +-- static/
|       |-- js/
|       |   |-- app.js              # Main controller
|       |   |-- websocket-client.js # WebSocket audio transport
|       |   |-- audio-capture.js    # Mic input (AudioWorklet)
|       |   |-- audio-playback.js   # Speaker output
|       |   +-- memory-chart.js     # Chart.js memory visualization
|       |-- css/style.css           # Dark theme UI
|       +-- worklets/
|           |-- capture-processor.js
|           +-- playback-processor.js
|
|-- quantization/                # Int8 QAT + ONNX export pipeline
|   |-- __init__.py              # Public API exports
|   |-- qat_config.py            # QConfig, observer specs, module targeting
|   |-- qat_model.py             # QATNet, FakeQuantizedConvTranspose1d
|   |-- qat_train.py             # QAT fine-tuning script (CLI)
|   |-- onnx_wrapper.py          # ONNX-exportable streaming wrapper
|   |-- export_onnx.py           # ONNX export: fp32 + int8 QDQ (CLI)
|   |-- onnx_inference.py        # ONNX Runtime streaming engine
|   |-- validate.py              # Validation + benchmarking (CLI)
|   +-- requirements.txt         # onnx, onnxruntime
|
|-- minimal_rvc/              # RVC voice conversion module
|-- minimal_quickvc/          # QuickVC voice conversion module
|-- test_wavs/                # 10 LibriSpeech test audio files
|-- requirements.txt
|-- eval_requirements.txt
+-- LICENSE                   # MIT
```

---

## Setup

### Prerequisites

- Python 3.11+
- PyTorch + torchaudio ([install guide](https://pytorch.org/get-started/locally/))

### Installation

```bash
git clone https://github.com/MuruganR96/LLVC.git
cd LLVC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install PyTorch (CPU example)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Web app additional dependencies
pip install fastapi uvicorn python-multipart jinja2

# Download pretrained models
python download_models.py
```

### Speaker Checkpoints

Place `.pth` checkpoint files in:
```
llvc_models/models/checkpoints/llvc_hfg/
```

The web app automatically discovers all `.pth` files in this directory and presents them as speaker options. Filenames become display names (underscores replaced with spaces).

---

## Web App (Real-Time Voice Conversion)

### Start the Server

```bash
python -m uvicorn web_app.app:app --host 0.0.0.0 --port 8765
```

Open http://localhost:8765 in your browser.

### Features

**Speaker Selection**
- Select from auto-discovered speaker checkpoints
- Load model with one click; see parameter count, sample rate, and load time

**Microphone Mode**
- Real-time mic-to-speaker voice conversion via WebSocket
- Adjustable chunk factor (1-10) for latency vs. quality tradeoff
- Input/output audio level meters

**File Upload Mode**
- Drag-and-drop or browse for audio files
- Streaming conversion with progress bar
- After conversion: play back with HTML5 audio controls, download as WAV, or re-convert

**Performance Metrics**
- Real-time RTF (Real-Time Factor), chunk latency, chunks processed
- RSS memory usage display
- Live Chart.js graph: RSS total, RSS delta, tracemalloc current/peak per frame

### Web App Architecture

```
Browser                          Server (FastAPI)
+-------------------+            +--------------------+
| AudioWorklet      |--WebSocket-| /ws/stream         |
| (mic capture)     |  (float32) |                    |
|                   |<-WebSocket-| StreamingInference  |
| AudioWorklet      |  (float32) | Engine             |
| (playback)        |            |                    |
+-------------------+            +--------------------+
| File Upload       |--POST------| /api/upload        |
|                   |--WebSocket-| /ws/file-stream    |
| Audio Player      |<-WebSocket-|                    |
+-------------------+            +--------------------+
```

---

## CLI Inference

### Single File / Folder

```bash
python infer.py -p llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth \
                -c experiments/llvc_hfg/config.json \
                -f test_wavs/ \
                -o converted_out/
```

### Streaming Mode

Add `-s` to simulate real-time streaming. Use `-n` to set chunk factor (higher = more latency, better RTF):

```bash
python infer.py -s -n 4 \
                -p llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth \
                -c experiments/llvc_hfg/config.json \
                -f test_wavs/ \
                -o converted_out/
```

### Compare with RVC / QuickVC

```bash
python compare_infer.py
```

---

## Benchmarking

```bash
python benchmark.py
```

Outputs a JSON report to `benchmark_out/` with:
- Model loading time and memory delta
- Per-file streaming latency, RTF, and memory usage
- Per-file non-streaming latency, RTF, and memory usage

---

## Training

1. Prepare a dataset with aligned audio pairs (`*_original.wav` and `*_converted.wav`) in `train/`, `dev/`, and `val/` subdirectories

2. Create or edit a config:
   ```bash
   cp experiments/llvc_hfg/config.json experiments/my_run/config.json
   # Edit data.dir, batch_size, etc.
   ```

3. Train:
   ```bash
   python train.py -d experiments/my_run
   ```

4. Monitor with TensorBoard:
   ```bash
   tensorboard --logdir experiments/my_run/logs
   ```

Checkpoints are saved every 5,000 steps to `experiments/my_run/`.

### Dataset Creation

To create a training dataset from LibriSpeech using RVC conversion:

1. Download `dev-clean.tar.gz` and `train-clean-360.tar.gz` from https://www.openslr.org/12

2. Generate aligned pairs:
```bash
python -m minimal_rvc._infer_folder \
    --train_set_path "LibriSpeech/train-clean-360" \
    --dev_set_path "LibriSpeech/dev-clean" \
    --out_path "f_8312_ls360" \
    --flatten \
    --model_path "llvc_models/models/rvc/f_8312_32k-325.pth" \
    --model_name "f_8312" \
    --target_sr 16000 \
    --f0_method "rmvpe" \
    --val_percent 0.02 \
    --random_seed 42 \
    --f0_up_key 12
```

---

## Evaluation

Evaluation requires a separate Python 3.9 environment due to dependency conflicts:

```bash
conda create -n llvc-eval python=3.9
conda activate llvc-eval
pip install -r eval_requirements.txt
```

1. Convert test audio using `infer.py`
2. Run evaluation:
```bash
python eval.py --converted_dir converted_out/ --ground_truth_dir test_wavs/
```

Metrics: ResemblySpeaker similarity score and WVMOS (Voice MOS).

---

## Model Variants

| Variant   | Config                             | Discriminator | Notes                    |
|-----------|------------------------------------|---------------|--------------------------|
| llvc      | `experiments/llvc/config.json`     | RVC           | Standard LLVC            |
| llvc_hfg  | `experiments/llvc_hfg/config.json` | HiFi-GAN      | Default, best quality    |
| llvc_nc   | `experiments/llvc_nc/config.json`  | HiFi-GAN      | Non-causal variant       |

---

## API Reference

| Method | Endpoint           | Description                        |
|--------|--------------------|------------------------------------|
| GET    | `/`                | Web UI                             |
| GET    | `/api/models`      | List available speaker checkpoints |
| POST   | `/api/models/load` | Load a speaker model by key        |
| POST   | `/api/upload`      | Upload audio file for conversion   |
| GET    | `/api/status`      | Server and model status            |
| WS     | `/ws/stream`       | Real-time mic streaming            |
| WS     | `/ws/file-stream`  | File conversion streaming          |

---

## Quantization & ONNX Export (`quantization/`)

A complete int8 Quantization Aware Training (QAT) and ONNX export pipeline. All code lives in the `quantization/` folder -- no existing files are modified.

### Overview

| | PyTorch fp32 | ONNX fp32 | ONNX int8 (after QAT) |
|---|---|---|---|
| **Weights dtype** | float32 | float32 | int8 |
| **Weight memory** | ~13 MB | 11.8 MB | ~3 MB (target) |
| **Mean chunk latency** | 24.85 ms | 2.84 ms | < 2 ms (target) |
| **RTF (real-time factor)** | 2.09x | 18.29x | > 20x (target) |
| **Max diff vs PyTorch** | -- | 1.08e-06 | QAT-dependent |
| **Framework needed** | PyTorch | ONNX Runtime only | ONNX Runtime only |

### Quantization Folder Structure

```
quantization/
    __init__.py               # Public API: get_qat_qconfig, prepare_qat_model,
    |                         #   convert_to_quantized, OnnxStreamingWrapper,
    |                         #   OnnxStreamingInferenceEngine
    qat_config.py             # QConfig, observer specs, module targeting
    qat_model.py              # QATNet wrapper, FakeQuantizedConvTranspose1d,
    |                         #   prepare_qat_model(), convert_to_quantized()
    qat_train.py              # Adversarial QAT fine-tuning (CLI)
    onnx_wrapper.py           # ONNX-exportable streaming wrapper
    export_onnx.py            # ONNX export: float32 and int8 QDQ (CLI)
    onnx_inference.py         # ONNX Runtime streaming engine (no PyTorch needed)
    validate.py               # PyTorch vs ONNX validation + benchmarks (CLI)
    requirements.txt          # onnx>=1.14.0, onnxruntime>=1.16.0
```

---

### Original Model Architecture (layer-by-layer)

The LLVC `Net` class in `model.py` has this inference data flow during streaming (`pad=False`):

```
Input audio [1, 1, 864]            (832 chunk + 32 lookahead samples)
        |
        v
+-- CachedConvNet Prenet ---------- 12 layers, skip_connection='add'
|   Layer i:
|     ctx_slice = ctx[..., start:end]       (2 samples context per layer)
|     conv_in = cat(ctx_slice, x)           (causal: past context + current)
|     ctx[..., start:end] = conv_in[-2:]    ** IN-PLACE buffer update **
|     x = ResidualBlock(conv_in)            (filter Conv1d + gate Conv1d -> tanh*sigmoid)
|       filter: Conv1d(1, 1, k=3, d=1)
|       gate:   Conv1d(1, 1, k=3, d=1)
|       BatchNorm1d(1), Dropout1d, LeakyReLU(inplace=True)
|   After 12 layers: x = input_audio + convnet_output
|
|   State: convnet_pre_ctx [1, 1, 24]      (12 layers x 2 samples each)
+-------------------------------------------
        |
        v  [1, 1, 864]
+-- in_conv ----------------------------
|   Conv1d(1, 512, kernel=48, stride=16, bias=False)
|   ReLU
+---------------------------------------
        |
        v  [1, 512, 52]                    T_enc = (864-48)/16 + 1 = 52
+-- label_embedding --------------------
|   Linear(1, 512) -> LayerNorm(512) -> ReLU -> Linear(512, 512) -> LayerNorm(512) -> ReLU
|   Input: torch.zeros(B, 1)               ** dynamic torch.zeros call **
|   Output: [1, 512] constant label vector
+---------------------------------------
        |
        v
+-- MaskNet ----------------------------
|
|   +-- DilatedCausalConvEncoder ------- 8 layers, channels=512
|   |   Layer i (dilation = 2^i):
|   |     ctx_slice = enc_buf[..., start:end]
|   |     dcc_in = cat(ctx_slice, x)
|   |     enc_buf[..., start:end] = dcc_in[-buf_len:]  ** IN-PLACE **
|   |     x = x + DepthwiseSeparableConv(dcc_in)
|   |       depthwise: Conv1d(512, 512, k=3, groups=512, dilation=2^i)
|   |       LayerNormPermuted(512), ReLU
|   |       pointwise: Conv1d(512, 512, k=1)
|   |       LayerNormPermuted(512), ReLU
|   |
|   |   State: enc_buf [1, 512, 510]       ((k-1)*(2^8 - 1) = 510)
|   +-----------------------------------
|           |
|           v  e = [1, 512, 52]
|
|   Label integration: l = label.unsqueeze(2) * e  -> [1, 512, 52]
|
|   +-- Projection e2d -----------------
|   |   proj_e2d_e: Conv1d(512, 256, k=1, groups=256) + ReLU  -> e_proj [1, 256, 52]
|   |   proj_e2d_l: Conv1d(512, 256, k=1, groups=256) + ReLU  -> m_proj [1, 256, 52]
|   +-----------------------------------
|           |
|           v
|   +-- CausalTransformerDecoder ------- 1 layer, dim=256, ctx=13, chunk=13
|   |   mod_pad: T=52, 52%13=0, so mod=0, no padding needed
|   |
|   |   mem context: cat(dec_buf[:,0,:,:], mem) -> [1, 65, 256]
|   |   dec_buf[:,0,:,:] = mem[:,-13:,:]        ** IN-PLACE **
|   |
|   |   _causal_unfold via nn.Unfold:          ** ONNX BLOCKER: nn.Unfold **
|   |     window_size = ctx_len + chunk_size = 26
|   |     4 windows: [0:26], [13:39], [26:52], [39:65]
|   |     -> mem_ctx [4, 26, 256]
|   |   Add PositionalEncoding
|   |
|   |   tgt context: cat(dec_buf[:,1,:,:], tgt) -> [1, 65, 256]
|   |   dec_buf[:,1,:,:] = tgt[:,-13:,:]        ** IN-PLACE **
|   |   _causal_unfold tgt -> tgt_ctx [4, 26, 256]
|   |   Add PositionalEncoding
|   |
|   |   tgt = zeros_like(tgt_ctx)[:,-13:,:]    ** dynamic torch.zeros **
|   |   for j in range(ceil(4/1000)):           ** DYNAMIC LOOP **
|   |     CausalTransformerDecoderLayer:
|   |       self_attn:      MultiheadAttention(256, 8 heads)
|   |       cross_attn:     MultiheadAttention(256, 8 heads)
|   |       FFN: Linear(256, 512) -> ReLU -> Dropout -> Linear(512, 256)
|   |       3x LayerNorm(256), 3x Dropout(0.1)
|   |   tgt reshape -> [1, 52, 256] -> permute -> [1, 256, 52]
|   |
|   |   State: dec_buf [1, 2, 13, 256]
|   +-----------------------------------
|           |
|           v  m = [1, 256, 52]
|   Projection d2e: Conv1d(256, 512, k=1, groups=256) + ReLU  -> [1, 512, 52]
|   Skip connection: m = l_integrated + m
|
+-- End MaskNet -------------------------
        |
        v  m = [1, 512, 52]
+-- Apply mask & decode ----------------
|   x = in_conv_output * m             -> [1, 512, 52]
|   x = cat(out_buf, x)                -> [1, 512, 56]
|   out_buf = x[..., -4:]              (save last 4 frames)
|   output = out_conv(x)               ConvTranspose1d(512, 1, k=80, s=16, pad=64)
|                                       + Tanh
|   State: out_buf [1, 512, 4]
+---------------------------------------
        |
        v
Output audio [1, 1, 832]
```

**Total streaming state: 5 buffers**

| Buffer | Shape | Size | Purpose |
|--------|-------|------|---------|
| `enc_buf` | [1, 512, 510] | 1,044,480 floats | DCC encoder causal context |
| `dec_buf` | [1, 2, 13, 256] | 6,656 floats | Transformer decoder memory |
| `out_buf` | [1, 512, 4] | 2,048 floats | Output conv overlap buffer |
| `convnet_pre_ctx` | [1, 1, 24] | 24 floats | CachedConvNet causal context |
| `prev_chunk_tail` | [32] | 32 floats | Lookahead for next chunk |

---

### ONNX Export Blockers and Solutions

The original `Net` has 6 patterns that prevent `torch.onnx.export()` from tracing the graph. Each is resolved in `onnx_wrapper.py`:

#### Blocker 1: In-Place Buffer Updates

**Problem.** 4 locations use tensor slice assignment (`buf[..., start:end] = new_value`). ONNX tracing cannot represent in-place mutations.

| Location | Code (original) |
|----------|-----------------|
| `cached_convnet.py:171` | `ctx[..., :x.shape[1], buf_start:buf_end] = conv_in[..., -self.buf_lengths[i]:]` |
| `model.py:133` | `ctx_buf[..., buf_start:buf_end] = dcc_in[..., -self.buf_lengths[i]:]` |
| `model.py:255` | `ctx_buf[:, 0, :, :] = mem[:, -self.ctx_len:, :]` |
| `model.py:268` | `ctx_buf[:, i + 1, :, :] = tgt[:, -self.ctx_len:, :]` |

**Solution.** Out-of-place reconstruction via `torch.cat`:

```python
# BEFORE (in-place, not ONNX-compatible):
ctx_buf[..., buf_start:buf_end] = new_values

# AFTER (out-of-place, ONNX-compatible):
new_buf_slice = dcc_in[:, :, -buf_length:]
ctx_buf = torch.cat([
    ctx_buf[:, :, :buf_start],     # everything before the slice
    new_buf_slice,                  # updated slice
    ctx_buf[:, :, buf_end:]        # everything after the slice
], dim=-1)
```

For the decoder buffer (`dec_buf` with 4D shape), slots are collected into a list and reconstructed with `torch.stack(slots, dim=1)` at the end.

**Files:** `onnx_wrapper.py:129-161` (`_convnet_pre_forward`), `onnx_wrapper.py:163-183` (`_encoder_forward`), `onnx_wrapper.py:200-243` (`_decoder_forward`)

#### Blocker 2: `nn.LeakyReLU(inplace=True)`

**Problem.** `cached_convnet.py:65` constructs `nn.LeakyReLU(inplace=True)` inside `CausalConvBlock`. In-place activations create non-deterministic traces.

**Solution.** Walk all submodules at init time and set `inplace=False`:

```python
@staticmethod
def _fix_inplace_relu(module):
    for _, child in module.named_modules():
        if isinstance(child, nn.LeakyReLU) and child.inplace:
            child.inplace = False
```

**File:** `onnx_wrapper.py:114-121`

#### Blocker 3: `nn.Unfold` in `_causal_unfold()`

**Problem.** `model.py:206,231` uses `nn.Unfold(kernel_size=(26, 1), stride=13)` to extract overlapping windows from the sequence. `nn.Unfold` is poorly supported in the ONNX trace.

**Solution.** Replace with explicit static slicing. Since chunk_factor is fixed at export time, the number of windows (4) and their positions are constants:

```python
def _causal_unfold_static(self, x):
    # x: [B, T_with_ctx, C] where T_with_ctx = ctx_len + T_enc_padded
    windows = []
    for i in range(self.num_unfold_windows):  # 4 windows
        start = i * self.dec_chunk_size       # 0, 13, 26, 39
        end = start + self.window_size        # 26, 39, 52, 65
        windows.append(x[:, start:end, :])
    return torch.cat(windows, dim=0)           # [4, 26, 256]
```

**File:** `onnx_wrapper.py:185-198`

#### Blocker 4: Dynamic Loop `for j in range(ceil(...))`

**Problem.** `model.py:275` uses `for j in range(int(math.ceil(tgt.shape[0] / K)))` where `K=1000`. The loop count depends on a runtime tensor shape.

**Solution.** With B=1 and 4 unfold windows, `tgt.shape[0] = 4` which is always < K=1000, so `ceil(4/1000) = 1`. The loop is always exactly 1 iteration. Replace with a single direct call:

```python
tgt_out, _, _ = tf_dec_layer(tgt_ctx, mem_ctx, chunk_size)
```

**File:** `onnx_wrapper.py:235`

#### Blocker 5: Conditional Control Flow

**Problem.** `model.py:440-456` has `if pad:`, `if hasattr(self, 'convnet_pre'):`, `if self.convnet_config['skip_connection'] == 'add':`, etc. ONNX tracing follows only one branch, but dynamic `if/else` on tensor values causes incorrect traces.

**Solution.** Evaluate all conditions once at `__init__` time and store as Python booleans. The wrapper's `forward()` uses only pre-resolved boolean flags:

```python
# At __init__:
self.has_convnet_pre = hasattr(net, 'convnet_pre')
self.has_proj = net.mask_gen.proj
self.has_skip = net.mask_gen.skip_connection
self.convnet_skip_connection = net.convnet_config['skip_connection']  # 'add'
```

Since `pad=False` in streaming mode, all padding-related conditionals are statically removed.

**File:** `onnx_wrapper.py:46-54`, `onnx_wrapper.py:245-302`

#### Blocker 6: `torch.zeros` for Label

**Problem.** `model.py:438` calls `label = torch.zeros(x.shape[0], 1, device=x.device)` every forward pass, then passes it through `label_embedding()`. Dynamic tensor creation is problematic for tracing.

**Solution.** Pre-compute the label embedding at init time and store as a registered buffer:

```python
# At __init__:
with torch.no_grad():
    label_const = net.label_embedding(torch.zeros(1, 1))
self.register_buffer('label_const', label_const)

# At forward time:
l = self.label_const  # [1, 512] -- no computation needed
```

The label is always `zeros(1, 1)` (single-speaker model), so the embedding output is a constant vector.

**File:** `onnx_wrapper.py:59-62`, `onnx_wrapper.py:272`

---

### QAT: Layer-by-Layer Quantization Decisions

49 modules are replaced with FakeQuantize-enabled QAT equivalents. The remaining modules stay float32.

#### Quantized to int8 (49 modules)

| Layer Group | Module Path | Original Type | QAT Replacement | Weight Shape | Rationale |
|-------------|------------|---------------|-----------------|--------------|-----------|
| **Input conv** | `in_conv.0` | Conv1d(1, 512, k=48, s=16) | qat.Conv1d | [512, 1, 48] | Largest single conv, 24K params |
| **Label embed** | `label_embedding.0` | Linear(1, 512) | qat.Linear | [512, 1] | Small but quantizable |
| **Label embed** | `label_embedding.3` | Linear(512, 512) | qat.Linear | [512, 512] | 262K params |
| **DCC encoder** (x8) | `mask_gen.encoder.dcc_layers.dcc_{0-7}.layers.0` | Conv1d(512, 512, k=3, groups=512) | qat.Conv1d | [512, 1, 3] | Depthwise conv, 8 layers |
| **DCC encoder** (x8) | `mask_gen.encoder.dcc_layers.dcc_{0-7}.layers.3` | Conv1d(512, 512, k=1) | qat.Conv1d | [512, 512, 1] | Pointwise conv, 262K params each |
| **Proj e2d_e** | `mask_gen.proj_e2d_e.0` | Conv1d(512, 256, k=1, groups=256) | qat.Conv1d | [256, 2, 1] | Grouped projection |
| **Proj e2d_l** | `mask_gen.proj_e2d_l.0` | Conv1d(512, 256, k=1, groups=256) | qat.Conv1d | [256, 2, 1] | Grouped projection |
| **Proj d2e** | `mask_gen.proj_d2e.0` | Conv1d(256, 512, k=1, groups=256) | qat.Conv1d | [512, 1, 1] | Grouped projection |
| **Decoder FFN** | `mask_gen.decoder.tf_dec_layers.0.linear1` | Linear(256, 512) | qat.Linear | [512, 256] | 131K params |
| **Decoder FFN** | `mask_gen.decoder.tf_dec_layers.0.linear2` | Linear(512, 256) | qat.Linear | [256, 512] | 131K params |
| **Output conv** | `out_conv.0` | ConvTranspose1d(512, 1, k=80, s=16) | FakeQuantizedConvTranspose1d | [512, 1, 80] | Custom wrapper (no built-in qat.ConvTranspose1d) |
| **ConvNet prenet** (x12) | `convnet_pre.down_convs.{0-11}.filter` | Conv1d(1, 1, k=3) | qat.Conv1d | [1, 1, 3] | Gated filter conv |
| **ConvNet prenet** (x12) | `convnet_pre.down_convs.{0-11}.gate` | Conv1d(1, 1, k=3) | qat.Conv1d | [1, 1, 3] | Gated gate conv |

**Total: 1 + 2 + 16 + 3 + 2 + 1 + 24 = 49 modules**

#### Kept at float32 (NOT quantized)

| Module Type | Where | Reason |
|-------------|-------|--------|
| `LayerNorm` / `LayerNormPermuted` | Inside each DSC layer (16 instances), label_embedding (2 instances) | Numerically sensitive -- quantizing normalization layers causes large accuracy degradation |
| `nn.MultiheadAttention` | `self_attn`, `multihead_attn` in decoder layer | Attention score computation (softmax over Q*K^T) is highly sensitive to quantization noise on the dot products |
| `nn.BatchNorm1d` | Inside CachedConvNet CausalConvBlocks (12 instances) | Running mean/variance statistics become inaccurate under int8 |
| `PositionalEncoding` | `mask_gen.decoder.pos_enc` | Fixed sinusoidal encoding (not learnable), no weights to compress |
| `nn.ReLU`, `nn.LeakyReLU`, `nn.Tanh`, `nn.Sigmoid` | Throughout the model | Activation functions have no weights -- nothing to quantize |
| `nn.Dropout` / `nn.Dropout1d` | CachedConvNet, decoder | Training-only -- no effect at inference |
| All discriminators | `hfg_disc.py` | Training-only -- not part of inference graph |

#### FakeQuantize Configuration

```
Weight quantization:
  - Observer:    MovingAveragePerChannelMinMaxObserver
  - Dtype:       qint8 (signed)
  - Range:       [-128, 127]
  - Scheme:      per_channel_symmetric (one scale per output channel)
  - Granularity: Per output channel

Activation quantization:
  - Observer:    MovingAverageMinMaxObserver
  - Dtype:       quint8 (unsigned)
  - Range:       [0, 255]
  - Scheme:      per_tensor_affine (asymmetric, one scale + zero_point per tensor)
  - Granularity: Per tensor
```

**Why per-channel for weights:** Conv1d output channels often have different weight distributions. Per-channel quantization assigns a separate scale factor to each output channel, reducing quantization error vs. per-tensor.

**Why asymmetric for activations:** ReLU outputs are [0, +inf), so symmetric quantization would waste half the int8 range. Asymmetric uses the full [0, 255] range.

#### Custom: `FakeQuantizedConvTranspose1d`

PyTorch's `torch.ao.nn.qat` module provides `qat.Conv1d` and `qat.Linear` but has **no built-in `qat.ConvTranspose1d`**. We implement a custom wrapper:

```python
class FakeQuantizedConvTranspose1d(nn.Module):
    def __init__(self, conv_transpose):
        self.conv_transpose = conv_transpose
        self.weight_fake_quant = FakeQuantize(...)   # per-channel int8
        self.activation_fake_quant = FakeQuantize(...) # per-tensor uint8

    def forward(self, x):
        fq_weight = self.weight_fake_quant(self.conv_transpose.weight)
        out = F.conv_transpose1d(x, fq_weight, ...)
        return self.activation_fake_quant(out)
```

This manually applies FakeQuantize to the weights before calling `conv_transpose1d`, and applies activation FakeQuantize to the output. During QAT training, gradients flow through the straight-through estimator in FakeQuantize.

---

### QAT Training Pipeline

The fine-tuning loop in `qat_train.py` adapts the existing training setup from `train.py` with QAT-specific modifications:

| Aspect | Original `train.py` | QAT `qat_train.py` |
|--------|---------------------|---------------------|
| Parallelism | DDP multi-GPU | Single GPU (simpler for QAT) |
| Mixed precision | AMP (fp16/fp32) | Disabled (conflicts with FakeQuantize) |
| Learning rate | 5e-4 | 5e-5 (10x lower) |
| Training length | Unlimited epochs | 10,000 steps (default) |
| Discriminator | Co-trained, same dtype | Co-trained, stays float32 |
| BN handling | Normal | Freeze stats at 60% of training |
| Observer handling | N/A | Freeze at 80% of training |
| Gradient clipping | Configurable | 1.0 (threshold) |

**Training schedule milestones:**

```
Step 0          Start QAT training, FakeQuantize + observers active
  |
Step 6000       (60%) Freeze BatchNorm running mean/variance
  |               Prevents BN drift from fake-quantized activations
  |
Step 8000       (80%) Freeze observers (scale/zero_point fixed)
  |               FakeQuantize still active for final weight calibration
  |
Step 10000      End. Convert to int8, save quantized model
```

**Loss function (same as original):**
```
L_total = L_adversarial * disc_loss_c + L_feature_matching * feature_loss_c + L_mel * mel_c
```
Where `L_adversarial` = LS-GAN generator loss, `L_feature_matching` = L1 between discriminator feature maps, `L_mel` = multi-resolution STFT mel loss.

---

### ONNX Export Details

#### Float32 Export

```
torch.onnx.export(
    OnnxStreamingWrapper(net),
    (audio, enc_buf, dec_buf, out_buf, convnet_pre_ctx),
    "llvc_fp32.onnx",
    opset_version=17,
    dynamic_axes=None            # all shapes fixed at export time
)
```

**Fixed I/O shapes (no dynamic axes):**

| Name | Direction | Shape | Dtype |
|------|-----------|-------|-------|
| `audio` | input | [1, 1, 864] | float32 |
| `enc_buf` | input | [1, 512, 510] | float32 |
| `dec_buf` | input | [1, 2, 13, 256] | float32 |
| `out_buf` | input | [1, 512, 4] | float32 |
| `convnet_pre_ctx` | input | [1, 1, 24] | float32 |
| `output` | output | [1, 1, 832] | float32 |
| `new_enc_buf` | output | [1, 512, 510] | float32 |
| `new_dec_buf` | output | [1, 2, 13, 256] | float32 |
| `new_out_buf` | output | [1, 512, 4] | float32 |
| `new_convnet_pre_ctx` | output | [1, 1, 24] | float32 |

#### Int8 QDQ Export

After QAT, the FakeQuantize nodes are exported as ONNX `QuantizeLinear` / `DequantizeLinear` (QDQ) operator pairs. ONNX Runtime's graph optimizer recognizes these patterns and fuses them into native int8 kernels at session load time.

---

### ONNX Runtime Inference Engine

`onnx_inference.py` provides `OnnxStreamingInferenceEngine`, a drop-in replacement for `web_app/audio_processor.py:StreamingInferenceEngine`. The key difference: **no PyTorch dependency at inference time** -- only numpy + onnxruntime.

**State management:** All 5 streaming buffers are maintained as numpy arrays and passed as explicit session inputs/outputs on every `process_chunk()` call.

**Session options:**
- `graph_optimization_level = ORT_ENABLE_ALL` (constant folding, operator fusion, etc.)
- `intra_op_num_threads = 4` (parallel execution within operators)
- `inter_op_num_threads = 1` (sequential operator execution -- better for streaming)
- Provider: `CPUExecutionProvider`

---

### Validation Results

Tested on 10 LibriSpeech audio files, chunk-by-chunk streaming comparison:

```
File                                      max_diff    mean_diff   SNR
174-50561-0000.wav                        4.64e-07    1.34e-08    119.8dB
1919-142785-0000.wav                      3.39e-07    1.10e-08    123.6dB
2086-149214-0000.wav                      3.04e-07    1.23e-08    126.9dB
2412-153947-0000.wav                      2.20e-07    7.27e-09    124.5dB
2902-9006-0000.wav                        1.08e-06    2.47e-08    116.0dB
5895-34615-0000.wav                       3.37e-07    9.48e-09    125.6dB
652-129742-0000.wav                       4.16e-07    1.32e-08    121.4dB
777-126732-0000.wav                       2.09e-07    1.07e-08    125.5dB
7850-73752-0000.wav                       2.91e-07    1.27e-08    124.0dB
8842-302196-0000.wav                      6.78e-07    1.71e-08    121.5dB

Overall max diff:  1.08e-06 (threshold: 1e-4)  PASS
Overall mean diff: 1.32e-08
```

**Benchmark (95 chunks, 5 warmup skipped):**

| Metric | ONNX Runtime | PyTorch |
|--------|-------------|---------|
| Mean latency | 2.84 ms | 24.85 ms |
| Median latency | 2.70 ms | 24.77 ms |
| Min latency | 2.53 ms | 23.32 ms |
| Max latency | 3.88 ms | 26.81 ms |
| Audio chunk duration | 52.0 ms | 52.0 ms |
| RTF | 18.29x | 2.09x |
| Real-time capable | YES | YES |

---

### CLI Usage

```bash
# Install quantization dependencies
pip install -r quantization/requirements.txt

# Export float32 ONNX
python quantization/export_onnx.py \
    --checkpoint llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth \
    --config experiments/llvc_hfg/config.json \
    --output quantization/llvc_fp32.onnx

# Validate ONNX vs PyTorch + benchmark
python quantization/validate.py \
    --pytorch_checkpoint llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth \
    --onnx_path quantization/llvc_fp32.onnx \
    --config experiments/llvc_hfg/config.json \
    --test_dir test_wavs

# QAT fine-tuning (requires training dataset)
python quantization/qat_train.py \
    --checkpoint llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth \
    --config experiments/llvc_hfg/config.json \
    --output_dir quantization/qat_output \
    --num_steps 10000 --lr 5e-5 --batch_size 4

# Export int8 ONNX (after QAT)
python quantization/export_onnx.py \
    --checkpoint quantization/qat_output/G_qat_final.pth \
    --config experiments/llvc_hfg/config.json \
    --output quantization/llvc_int8.onnx --quantized
```

---

## Credits

Based on [Koe AI's LLVC](https://github.com/KoeAI/llvc). Additional modules adapted from:
- [rvc-webui](https://github.com/ddPn08/rvc-webui)
- [RVC-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [Voice_Separation_and_Selection](https://github.com/teftef6220/Voice_Separation_and_Selection)

## Citation

```bibtex
@misc{sadov2023lowlatency,
      title={Low-latency Real-time Voice Conversion on CPU},
      author={Konstantine Sadov and Matthew Hutter and Asara Near},
      year={2023},
      eprint={2311.00873},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## License

MIT License -- see [LICENSE](LICENSE) for details.
