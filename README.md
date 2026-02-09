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
