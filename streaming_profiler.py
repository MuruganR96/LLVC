"""
Streaming Inference Profiler for LLVC

Measures frame-level (per-chunk) metrics during streaming inference:
  - Per-chunk inference latency (ms)
  - Per-chunk RSS memory delta (MB)
  - Per-chunk Python heap allocation (tracemalloc)
  - Model weight memory footprint
  - State buffer memory footprint
  - System resource usage (CPU%, RSS, peak RSS)
  - ONNX Runtime comparison (at chunk_factor=4)

Chunk duration options:
  chunk_factor=1 → 208 samples → 13.0ms  (closest to 20ms target, low latency)
  chunk_factor=2 → 416 samples → 26.0ms  (closest to 20ms target, higher throughput)
  chunk_factor=4 → 832 samples → 52.0ms  (default, ONNX-exported size)

Usage:
    python streaming_profiler.py [--chunk_factors 1 2] [--test_dir test_wavs]
                                 [--onnx_path quantization/llvc_fp32.onnx]
                                 [--output_dir profiler_out]
"""

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc

import numpy as np
import psutil
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Net
from utils import model_size, glob_audio_files
from infer import load_audio

# Optional ONNX Runtime
try:
    from quantization.onnx_inference import OnnxStreamingInferenceEngine
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth"
CONFIG_PATH = "experiments/llvc_hfg/config.json"

_process = psutil.Process(os.getpid())


def get_rss_mb():
    return _process.memory_info().rss / (1024 * 1024)


def get_cpu_percent():
    return _process.cpu_percent(interval=None)


def tensor_memory_bytes(t):
    """Memory in bytes consumed by a single tensor."""
    return t.element_size() * t.nelement()


def format_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.2f} KB"
    else:
        return f"{b / (1024 ** 2):.2f} MB"


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_table(headers, rows):
    col_widths = []
    for i, h in enumerate(headers):
        max_data = max((len(str(r[i])) for r in rows), default=0)
        col_widths.append(max(len(str(h)), max_data))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


# ---------------------------------------------------------------------------
# Model Memory Analysis
# ---------------------------------------------------------------------------

def analyze_model_memory(model):
    """Break down model weight memory by component."""
    component_mem = {}
    total_params = 0
    total_bytes = 0

    for name, param in model.named_parameters():
        # Group by top-level component
        parts = name.split(".")
        component = parts[0]
        if len(parts) > 1 and parts[0] in ("mask_gen",):
            component = f"{parts[0]}.{parts[1]}"

        nbytes = param.element_size() * param.nelement()
        nparams = param.nelement()
        if component not in component_mem:
            component_mem[component] = {"params": 0, "bytes": 0}
        component_mem[component]["params"] += nparams
        component_mem[component]["bytes"] += nbytes
        total_params += nparams
        total_bytes += nbytes

    return {
        "components": component_mem,
        "total_params": total_params,
        "total_bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# State Buffer Analysis
# ---------------------------------------------------------------------------

def analyze_state_buffers(model, chunk_factor):
    """Compute state buffer sizes for a given chunk_factor."""
    device = torch.device("cpu")
    enc_buf, dec_buf, out_buf = model.init_buffers(1, device)

    buffers = {
        "enc_buf": {"shape": list(enc_buf.shape), "bytes": tensor_memory_bytes(enc_buf)},
        "dec_buf": {"shape": list(dec_buf.shape), "bytes": tensor_memory_bytes(dec_buf)},
        "out_buf": {"shape": list(out_buf.shape), "bytes": tensor_memory_bytes(out_buf)},
    }

    if hasattr(model, "convnet_pre"):
        ctx = model.convnet_pre.init_ctx_buf(1, device)
        buffers["convnet_pre_ctx"] = {
            "shape": list(ctx.shape), "bytes": tensor_memory_bytes(ctx)
        }

    L = model.L
    prev_tail = torch.zeros(L * 2)
    buffers["prev_chunk_tail"] = {
        "shape": [L * 2], "bytes": tensor_memory_bytes(prev_tail)
    }

    total = sum(b["bytes"] for b in buffers.values())
    buffers["_total_bytes"] = total
    return buffers


# ---------------------------------------------------------------------------
# PyTorch Streaming Profiler
# ---------------------------------------------------------------------------

def profile_pytorch_streaming(model, audio_np, sr, chunk_factor):
    """Profile per-chunk streaming inference with PyTorch.

    Returns:
        dict with per_chunk metrics list, aggregate stats, and buffer info.
    """
    L = model.L
    chunk_len = model.dec_chunk_size * L * chunk_factor

    # Pad audio to multiple of chunk_len
    remainder = len(audio_np) % chunk_len
    if remainder != 0:
        audio_np = np.pad(audio_np, (0, chunk_len - remainder))

    # Init state
    device = torch.device("cpu")
    enc_buf, dec_buf, out_buf = model.init_buffers(1, device)
    if hasattr(model, "convnet_pre"):
        convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, device)
    else:
        convnet_pre_ctx = None
    prev_chunk_tail = torch.zeros(L * 2)

    per_chunk = []
    num_chunks = len(audio_np) // chunk_len

    # Warm-up: run 2 chunks to stabilize
    warmup_chunks = min(2, num_chunks)
    for i in range(warmup_chunks):
        chunk = audio_np[i * chunk_len:(i + 1) * chunk_len]
        chunk_t = torch.from_numpy(chunk).float()
        lookahead = prev_chunk_tail[-L * 2:]
        input_tensor = torch.cat([lookahead, chunk_t])
        prev_chunk_tail = chunk_t[-L * 2:] if len(chunk_t) >= L * 2 else \
            torch.cat([prev_chunk_tail[len(chunk_t):], chunk_t])
        with torch.inference_mode():
            _, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
                input_tensor.unsqueeze(0).unsqueeze(0),
                enc_buf, dec_buf, out_buf, convnet_pre_ctx,
                pad=(not model.lookahead),
            )

    # Reset state for actual measurement
    enc_buf, dec_buf, out_buf = model.init_buffers(1, device)
    if hasattr(model, "convnet_pre"):
        convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, device)
    else:
        convnet_pre_ctx = None
    prev_chunk_tail = torch.zeros(L * 2)

    # Prime CPU percent
    get_cpu_percent()

    # Profile each chunk
    tracemalloc.start()
    baseline_rss = get_rss_mb()

    for i in range(num_chunks):
        chunk = audio_np[i * chunk_len:(i + 1) * chunk_len]
        chunk_t = torch.from_numpy(chunk).float()
        lookahead = prev_chunk_tail[-L * 2:]
        input_tensor = torch.cat([lookahead, chunk_t])
        prev_chunk_tail = chunk_t[-L * 2:] if len(chunk_t) >= L * 2 else \
            torch.cat([prev_chunk_tail[len(chunk_t):], chunk_t])

        # Reset tracemalloc peak for this chunk
        tracemalloc.reset_peak()
        gc.collect()

        rss_before = get_rss_mb()
        t0 = time.perf_counter()

        with torch.inference_mode():
            output, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
                input_tensor.unsqueeze(0).unsqueeze(0),
                enc_buf, dec_buf, out_buf, convnet_pre_ctx,
                pad=(not model.lookahead),
            )

        latency = time.perf_counter() - t0
        rss_after = get_rss_mb()
        current_alloc, peak_alloc = tracemalloc.get_traced_memory()

        audio_duration = chunk_len / sr
        rtf = audio_duration / latency if latency > 0 else 0

        per_chunk.append({
            "chunk_index": i,
            "latency_ms": round(latency * 1000, 3),
            "audio_duration_ms": round(audio_duration * 1000, 2),
            "rtf": round(rtf, 2),
            "rss_mb": round(rss_after, 2),
            "rss_delta_mb": round(rss_after - rss_before, 4),
            "rss_from_baseline_mb": round(rss_after - baseline_rss, 4),
            "tracemalloc_current_mb": round(current_alloc / (1024 ** 2), 4),
            "tracemalloc_peak_chunk_mb": round(peak_alloc / (1024 ** 2), 4),
        })

    _, overall_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Aggregate
    latencies = [c["latency_ms"] for c in per_chunk]
    rss_deltas = [c["rss_delta_mb"] for c in per_chunk]
    peak_allocs = [c["tracemalloc_peak_chunk_mb"] for c in per_chunk]

    aggregate = {
        "num_chunks": num_chunks,
        "chunk_len_samples": chunk_len,
        "chunk_duration_ms": round(chunk_len / sr * 1000, 2),
        "latency_mean_ms": round(np.mean(latencies), 3),
        "latency_p50_ms": round(np.percentile(latencies, 50), 3),
        "latency_p95_ms": round(np.percentile(latencies, 95), 3),
        "latency_p99_ms": round(np.percentile(latencies, 99), 3),
        "latency_min_ms": round(np.min(latencies), 3),
        "latency_max_ms": round(np.max(latencies), 3),
        "rtf_mean": round(np.mean([c["rtf"] for c in per_chunk]), 2),
        "rss_delta_mean_mb": round(np.mean(rss_deltas), 4),
        "rss_delta_max_mb": round(np.max(rss_deltas), 4),
        "tracemalloc_peak_overall_mb": round(overall_peak / (1024 ** 2), 4),
        "tracemalloc_peak_chunk_mean_mb": round(np.mean(peak_allocs), 4),
        "tracemalloc_peak_chunk_max_mb": round(np.max(peak_allocs), 4),
        "real_time_capable": np.mean(latencies) < (chunk_len / sr * 1000),
    }

    return {"per_chunk": per_chunk, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# ONNX Streaming Profiler
# ---------------------------------------------------------------------------

def profile_onnx_streaming(onnx_path, audio_np, sr):
    """Profile per-chunk streaming inference with ONNX Runtime.

    The ONNX model is exported at chunk_factor=4 (832 samples, 52ms).
    """
    if not HAS_ONNX:
        return None

    engine = OnnxStreamingInferenceEngine(num_threads=4)
    info = engine.load_model(onnx_path)
    chunk_len = engine.get_chunk_len()

    # Pad
    remainder = len(audio_np) % chunk_len
    if remainder != 0:
        audio_np = np.pad(audio_np, (0, chunk_len - remainder))

    num_chunks = len(audio_np) // chunk_len

    # Warm-up
    engine.init_streaming_state()
    for i in range(min(2, num_chunks)):
        chunk = audio_np[i * chunk_len:(i + 1) * chunk_len]
        engine.process_chunk(chunk)

    # Reset for measurement
    engine.init_streaming_state()
    get_cpu_percent()

    tracemalloc.start()
    baseline_rss = get_rss_mb()
    per_chunk = []

    for i in range(num_chunks):
        chunk = audio_np[i * chunk_len:(i + 1) * chunk_len]

        tracemalloc.reset_peak()
        rss_before = get_rss_mb()
        t0 = time.perf_counter()

        output, metrics = engine.process_chunk(chunk)

        latency = time.perf_counter() - t0
        rss_after = get_rss_mb()
        current_alloc, peak_alloc = tracemalloc.get_traced_memory()

        audio_duration = chunk_len / sr
        rtf = audio_duration / latency if latency > 0 else 0

        per_chunk.append({
            "chunk_index": i,
            "latency_ms": round(latency * 1000, 3),
            "audio_duration_ms": round(audio_duration * 1000, 2),
            "rtf": round(rtf, 2),
            "rss_mb": round(rss_after, 2),
            "rss_delta_mb": round(rss_after - rss_before, 4),
            "rss_from_baseline_mb": round(rss_after - baseline_rss, 4),
            "tracemalloc_current_mb": round(current_alloc / (1024 ** 2), 4),
            "tracemalloc_peak_chunk_mb": round(peak_alloc / (1024 ** 2), 4),
        })

    _, overall_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies = [c["latency_ms"] for c in per_chunk]
    rss_deltas = [c["rss_delta_mb"] for c in per_chunk]
    peak_allocs = [c["tracemalloc_peak_chunk_mb"] for c in per_chunk]

    aggregate = {
        "num_chunks": num_chunks,
        "chunk_len_samples": chunk_len,
        "chunk_duration_ms": round(chunk_len / sr * 1000, 2),
        "latency_mean_ms": round(np.mean(latencies), 3),
        "latency_p50_ms": round(np.percentile(latencies, 50), 3),
        "latency_p95_ms": round(np.percentile(latencies, 95), 3),
        "latency_p99_ms": round(np.percentile(latencies, 99), 3),
        "latency_min_ms": round(np.min(latencies), 3),
        "latency_max_ms": round(np.max(latencies), 3),
        "rtf_mean": round(np.mean([c["rtf"] for c in per_chunk]), 2),
        "rss_delta_mean_mb": round(np.mean(rss_deltas), 4),
        "rss_delta_max_mb": round(np.max(rss_deltas), 4),
        "tracemalloc_peak_overall_mb": round(overall_peak / (1024 ** 2), 4),
        "tracemalloc_peak_chunk_mean_mb": round(np.mean(peak_allocs), 4),
        "tracemalloc_peak_chunk_max_mb": round(np.max(peak_allocs), 4),
        "real_time_capable": np.mean(latencies) < (chunk_len / sr * 1000),
        "onnx_chunk_factor": 4,
    }

    return {"per_chunk": per_chunk, "aggregate": aggregate, "model_info": info}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLVC Streaming Inference Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Chunk factor reference:
  chunk_factor=1 → 208 samples → 13.0ms
  chunk_factor=2 → 416 samples → 26.0ms
  chunk_factor=4 → 832 samples → 52.0ms (default ONNX export)
        """,
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                        help="Path to .pth checkpoint")
    parser.add_argument("--config", default=CONFIG_PATH,
                        help="Path to config.json")
    parser.add_argument("--chunk_factors", type=int, nargs="+", default=[1, 2],
                        help="Chunk factors to profile (default: 1 2)")
    parser.add_argument("--test_dir", default="test_wavs",
                        help="Directory with test .wav files")
    parser.add_argument("--onnx_path", default="quantization/llvc_fp32.onnx",
                        help="Path to ONNX model (chunk_factor=4)")
    parser.add_argument("--output_dir", default="profiler_out",
                        help="Output directory for JSON report")
    parser.add_argument("--max_files", type=int, default=3,
                        help="Max test files to process (default: 3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {}

    ckpt_path = os.path.join(PROJECT_ROOT, args.checkpoint)
    config_path = os.path.join(PROJECT_ROOT, args.config)

    # ------------------------------------------------------------------
    # 1. Model Loading & Memory
    # ------------------------------------------------------------------
    print_section("1. Model Loading & Weight Memory")

    tracemalloc.start()
    rss_before = get_rss_mb()
    t0 = time.perf_counter()

    with open(config_path) as f:
        config = json.load(f)
    model = Net(**config["model_params"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    sr = config["data"]["sr"]

    load_time = time.perf_counter() - t0
    rss_after = get_rss_mb()
    _, peak_alloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_analysis = analyze_model_memory(model)

    load_report = {
        "load_time_s": round(load_time, 4),
        "model_params_M": model_size(model),
        "model_weight_bytes": mem_analysis["total_bytes"],
        "model_weight_mb": round(mem_analysis["total_bytes"] / (1024 ** 2), 2),
        "rss_before_mb": round(rss_before, 2),
        "rss_after_mb": round(rss_after, 2),
        "rss_delta_mb": round(rss_after - rss_before, 2),
        "tracemalloc_peak_mb": round(peak_alloc / (1024 ** 2), 2),
    }
    report["model_load"] = load_report

    print(f"  Parameters:          {load_report['model_params_M']}M")
    print(f"  Weight memory:       {format_bytes(mem_analysis['total_bytes'])}")
    print(f"  Load time:           {load_report['load_time_s']}s")
    print(f"  RSS delta:           {load_report['rss_delta_mb']} MB")
    print(f"  Tracemalloc peak:    {load_report['tracemalloc_peak_mb']} MB")

    # Component breakdown
    print(f"\n  Weight Memory by Component:")
    comp_rows = []
    for comp, info in sorted(mem_analysis["components"].items(),
                             key=lambda x: -x[1]["bytes"]):
        pct = info["bytes"] / mem_analysis["total_bytes"] * 100
        comp_rows.append([
            comp,
            f"{info['params']:,}",
            format_bytes(info["bytes"]),
            f"{pct:.1f}%",
        ])
    print_table(["Component", "Params", "Memory", "% Total"], comp_rows)

    report["weight_breakdown"] = {
        comp: {"params": info["params"], "bytes": info["bytes"]}
        for comp, info in mem_analysis["components"].items()
    }

    # ------------------------------------------------------------------
    # 2. State Buffer Memory
    # ------------------------------------------------------------------
    print_section("2. Streaming State Buffer Memory")

    for cf in args.chunk_factors:
        buf_info = analyze_state_buffers(model, cf)
        total = buf_info.pop("_total_bytes")
        print(f"\n  chunk_factor={cf} "
              f"({model.dec_chunk_size * model.L * cf} samples, "
              f"{model.dec_chunk_size * model.L * cf / sr * 1000:.1f}ms):")

        buf_rows = []
        for name, info in buf_info.items():
            buf_rows.append([
                name,
                str(info["shape"]),
                format_bytes(info["bytes"]),
            ])
        buf_rows.append(["TOTAL", "", format_bytes(total)])
        print_table(["Buffer", "Shape", "Memory"], buf_rows)

        report[f"state_buffers_cf{cf}"] = {
            name: {"shape": info["shape"], "bytes": info["bytes"]}
            for name, info in buf_info.items()
        }
        report[f"state_buffers_cf{cf}"]["_total_bytes"] = total

    # ------------------------------------------------------------------
    # 3. Load test audio
    # ------------------------------------------------------------------
    fnames = sorted(glob_audio_files(args.test_dir))[:args.max_files]
    if not fnames:
        print(f"\n  No audio files found in {args.test_dir}")
        return

    print(f"\n  Using {len(fnames)} test files")

    # Use longest file for detailed profiling
    audios = []
    for fname in fnames:
        audio = load_audio(fname, sr)
        audios.append((fname, audio))
    # Sort by length descending — use longest for main profile
    audios.sort(key=lambda x: -len(x[1]))
    main_fname, main_audio = audios[0]
    main_duration = len(main_audio) / sr
    print(f"  Primary file: {os.path.basename(main_fname)} "
          f"({main_duration:.2f}s, {len(main_audio)} samples)")

    # ------------------------------------------------------------------
    # 4. PyTorch Streaming Profile (per chunk_factor)
    # ------------------------------------------------------------------
    report["pytorch_streaming"] = {}

    for cf in args.chunk_factors:
        chunk_len = model.dec_chunk_size * model.L * cf
        chunk_ms = chunk_len / sr * 1000
        print_section(
            f"3. PyTorch Streaming — chunk_factor={cf} "
            f"({chunk_len} samples, {chunk_ms:.1f}ms)"
        )

        result = profile_pytorch_streaming(model, main_audio, sr, cf)
        agg = result["aggregate"]
        report["pytorch_streaming"][f"cf{cf}"] = {
            "file": os.path.basename(main_fname),
            "audio_duration_s": round(main_duration, 2),
            "aggregate": agg,
            "per_chunk": result["per_chunk"],
        }

        # Print aggregate
        print(f"  File:                {os.path.basename(main_fname)}")
        print(f"  Audio duration:      {main_duration:.2f}s")
        print(f"  Chunks processed:    {agg['num_chunks']}")
        print(f"  Chunk duration:      {agg['chunk_duration_ms']}ms")
        print()
        print(f"  --- Latency ---")
        print(f"  Mean:                {agg['latency_mean_ms']}ms")
        print(f"  P50:                 {agg['latency_p50_ms']}ms")
        print(f"  P95:                 {agg['latency_p95_ms']}ms")
        print(f"  P99:                 {agg['latency_p99_ms']}ms")
        print(f"  Min / Max:           {agg['latency_min_ms']} / {agg['latency_max_ms']}ms")
        print(f"  RTF (mean):          {agg['rtf_mean']}x")
        print(f"  Real-time capable:   {'YES' if agg['real_time_capable'] else 'NO'}")
        print()
        print(f"  --- Memory (per chunk) ---")
        print(f"  RSS delta mean:      {agg['rss_delta_mean_mb']} MB")
        print(f"  RSS delta max:       {agg['rss_delta_max_mb']} MB")
        print(f"  Heap peak (overall): {agg['tracemalloc_peak_overall_mb']} MB")
        print(f"  Heap peak/chunk mean:{agg['tracemalloc_peak_chunk_mean_mb']} MB")
        print(f"  Heap peak/chunk max: {agg['tracemalloc_peak_chunk_max_mb']} MB")

        # Per-chunk latency table (first 10 + last 5)
        chunks = result["per_chunk"]
        show_chunks = chunks[:10]
        if len(chunks) > 15:
            show_chunks.append({"chunk_index": "...", "latency_ms": "...",
                                "rss_mb": "...", "rss_delta_mb": "...",
                                "tracemalloc_peak_chunk_mb": "..."})
            show_chunks.extend(chunks[-5:])
        elif len(chunks) > 10:
            show_chunks.extend(chunks[10:])

        print(f"\n  --- Per-Chunk Detail ---")
        rows = []
        for c in show_chunks:
            rows.append([
                c["chunk_index"],
                c["latency_ms"],
                c.get("rss_mb", ""),
                c.get("rss_delta_mb", ""),
                c.get("tracemalloc_peak_chunk_mb", ""),
            ])
        print_table(
            ["Chunk#", "Latency(ms)", "RSS(MB)", "RSS_delta(MB)", "Heap_peak(MB)"],
            rows,
        )

    # ------------------------------------------------------------------
    # 5. ONNX Runtime Streaming Profile (chunk_factor=4)
    # ------------------------------------------------------------------
    onnx_path = os.path.join(PROJECT_ROOT, args.onnx_path)
    if HAS_ONNX and os.path.isfile(onnx_path):
        print_section("4. ONNX Runtime Streaming — chunk_factor=4 (832 samples, 52.0ms)")

        result = profile_onnx_streaming(onnx_path, main_audio, sr)
        if result:
            agg = result["aggregate"]
            report["onnx_streaming"] = {
                "file": os.path.basename(main_fname),
                "audio_duration_s": round(main_duration, 2),
                "aggregate": agg,
                "per_chunk": result["per_chunk"],
                "model_info": result.get("model_info"),
            }

            print(f"  File:                {os.path.basename(main_fname)}")
            print(f"  Audio duration:      {main_duration:.2f}s")
            print(f"  Chunks processed:    {agg['num_chunks']}")
            print(f"  Chunk duration:      {agg['chunk_duration_ms']}ms")
            print()
            print(f"  --- Latency ---")
            print(f"  Mean:                {agg['latency_mean_ms']}ms")
            print(f"  P50:                 {agg['latency_p50_ms']}ms")
            print(f"  P95:                 {agg['latency_p95_ms']}ms")
            print(f"  P99:                 {agg['latency_p99_ms']}ms")
            print(f"  Min / Max:           {agg['latency_min_ms']} / {agg['latency_max_ms']}ms")
            print(f"  RTF (mean):          {agg['rtf_mean']}x")
            print(f"  Real-time capable:   {'YES' if agg['real_time_capable'] else 'NO'}")
            print()
            print(f"  --- Memory (per chunk) ---")
            print(f"  RSS delta mean:      {agg['rss_delta_mean_mb']} MB")
            print(f"  RSS delta max:       {agg['rss_delta_max_mb']} MB")
            print(f"  Heap peak (overall): {agg['tracemalloc_peak_overall_mb']} MB")
            print(f"  Heap peak/chunk mean:{agg['tracemalloc_peak_chunk_mean_mb']} MB")
            print(f"  Heap peak/chunk max: {agg['tracemalloc_peak_chunk_max_mb']} MB")

            # Per-chunk detail
            chunks = result["per_chunk"]
            show_chunks = chunks[:10]
            if len(chunks) > 15:
                show_chunks.append({"chunk_index": "...", "latency_ms": "...",
                                    "rss_mb": "...", "rss_delta_mb": "...",
                                    "tracemalloc_peak_chunk_mb": "..."})
                show_chunks.extend(chunks[-5:])
            elif len(chunks) > 10:
                show_chunks.extend(chunks[10:])

            print(f"\n  --- Per-Chunk Detail ---")
            rows = []
            for c in show_chunks:
                rows.append([
                    c["chunk_index"],
                    c["latency_ms"],
                    c.get("rss_mb", ""),
                    c.get("rss_delta_mb", ""),
                    c.get("tracemalloc_peak_chunk_mb", ""),
                ])
            print_table(
                ["Chunk#", "Latency(ms)", "RSS(MB)", "RSS_delta(MB)", "Heap_peak(MB)"],
                rows,
            )
    else:
        if not HAS_ONNX:
            print(f"\n  [skip] ONNX Runtime not installed")
        else:
            print(f"\n  [skip] ONNX model not found at {onnx_path}")

    # ------------------------------------------------------------------
    # 6. System Info
    # ------------------------------------------------------------------
    print_section("5. System Resource Summary")

    vm = psutil.virtual_memory()
    sys_info = {
        "platform": os.uname().sysname,
        "machine": os.uname().machine,
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / (1024 ** 3), 1),
        "available_ram_gb": round(vm.available / (1024 ** 3), 1),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "process_rss_mb": round(get_rss_mb(), 2),
    }
    report["system"] = sys_info

    print(f"  Platform:            {sys_info['platform']} {sys_info['machine']}")
    print(f"  CPU cores:           {sys_info['cpu_count_physical']} physical, "
          f"{sys_info['cpu_count_logical']} logical")
    print(f"  Total RAM:           {sys_info['total_ram_gb']} GB")
    print(f"  Available RAM:       {sys_info['available_ram_gb']} GB")
    print(f"  Python:              {sys_info['python_version']}")
    print(f"  PyTorch:             {sys_info['torch_version']}")
    print(f"  Process RSS (now):   {sys_info['process_rss_mb']} MB")

    # ------------------------------------------------------------------
    # 7. Comparison Summary
    # ------------------------------------------------------------------
    print_section("6. Comparison Summary")

    summary_rows = []
    for cf in args.chunk_factors:
        key = f"cf{cf}"
        if key in report.get("pytorch_streaming", {}):
            a = report["pytorch_streaming"][key]["aggregate"]
            chunk_ms = a["chunk_duration_ms"]
            summary_rows.append([
                f"PyTorch cf={cf}",
                f"{chunk_ms}ms",
                f"{a['latency_mean_ms']}ms",
                f"{a['latency_p95_ms']}ms",
                f"{a['rtf_mean']}x",
                f"{a['tracemalloc_peak_overall_mb']} MB",
                "YES" if a["real_time_capable"] else "NO",
            ])

    if "onnx_streaming" in report:
        a = report["onnx_streaming"]["aggregate"]
        summary_rows.append([
            "ONNX RT cf=4",
            f"{a['chunk_duration_ms']}ms",
            f"{a['latency_mean_ms']}ms",
            f"{a['latency_p95_ms']}ms",
            f"{a['rtf_mean']}x",
            f"{a['tracemalloc_peak_overall_mb']} MB",
            "YES" if a["real_time_capable"] else "NO",
        ])

    if summary_rows:
        print_table(
            ["Engine", "Chunk", "Lat_mean", "Lat_P95", "RTF", "Heap_peak", "Realtime"],
            summary_rows,
        )

    # Total memory budget
    print(f"\n  --- Total Memory Budget (PyTorch, cf=1) ---")
    weight_mb = report["model_load"]["model_weight_mb"]
    buf_key = f"state_buffers_cf{args.chunk_factors[0]}"
    buf_total_mb = report.get(buf_key, {}).get("_total_bytes", 0) / (1024 ** 2)
    pt_key = f"cf{args.chunk_factors[0]}"
    runtime_peak_mb = 0
    if pt_key in report.get("pytorch_streaming", {}):
        runtime_peak_mb = report["pytorch_streaming"][pt_key]["aggregate"][
            "tracemalloc_peak_overall_mb"]

    print(f"  Model weights:       {weight_mb:.2f} MB")
    print(f"  State buffers:       {buf_total_mb:.4f} MB")
    print(f"  Runtime peak heap:   {runtime_peak_mb:.4f} MB")
    print(f"  Process RSS (now):   {sys_info['process_rss_mb']} MB")

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    report_path = os.path.join(args.output_dir, "streaming_profile.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
