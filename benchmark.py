"""
LLVC Benchmark Script
Measures memory usage, latency, and RTF for streaming and non-streaming inference.
"""
import tracemalloc
import psutil
import time
import os
import json
import numpy as np
import torch

from infer import load_model, load_audio, do_infer, infer_stream, infer
from utils import glob_audio_files, model_size


CHECKPOINT_PATH = "llvc_models/models/checkpoints/llvc/G_500000.pth"
CONFIG_PATH = "experiments/llvc/config.json"
TEST_DIR = "test_wavs"
OUT_DIR = "benchmark_out"
CHUNK_FACTOR = 1


def get_rss_mb():
    """Get current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def benchmark_model_load():
    """Benchmark model loading: time and memory."""
    tracemalloc.start()
    rss_before = get_rss_mb()
    t0 = time.perf_counter()
    model, sr = load_model(CHECKPOINT_PATH, CONFIG_PATH)
    load_time = time.perf_counter() - t0
    rss_after = get_rss_mb()
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return model, sr, {
        "load_time_s": round(load_time, 4),
        "model_params_M": model_size(model),
        "rss_before_mb": round(rss_before, 2),
        "rss_after_mb": round(rss_after, 2),
        "rss_delta_mb": round(rss_after - rss_before, 2),
        "tracemalloc_peak_mb": round(peak_tracemalloc / (1024 * 1024), 2),
    }


def benchmark_file_streaming(model, audio, sr, fname):
    """Benchmark streaming inference for a single file."""
    tracemalloc.start()
    rss_before = get_rss_mb()
    t0 = time.perf_counter()
    output, rtf, e2e_latency = do_infer(model, audio, CHUNK_FACTOR, sr, stream=True)
    total_time = time.perf_counter() - t0
    rss_after = get_rss_mb()
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    audio_duration = len(audio) / sr
    return output, {
        "file": os.path.basename(fname),
        "audio_duration_s": round(audio_duration, 3),
        "inference_time_s": round(total_time, 4),
        "rtf": round(rtf, 3) if rtf is not None else None,
        "e2e_latency_ms": round(e2e_latency, 3) if e2e_latency is not None else None,
        "rss_delta_mb": round(rss_after - rss_before, 2),
        "tracemalloc_peak_mb": round(peak_tracemalloc / (1024 * 1024), 2),
    }


def benchmark_file_nonstreaming(model, audio, sr, fname):
    """Benchmark non-streaming inference for a single file."""
    tracemalloc.start()
    rss_before = get_rss_mb()
    t0 = time.perf_counter()
    output, _, _ = do_infer(model, audio, CHUNK_FACTOR, sr, stream=False)
    total_time = time.perf_counter() - t0
    rss_after = get_rss_mb()
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    audio_duration = len(audio) / sr
    return output, {
        "file": os.path.basename(fname),
        "audio_duration_s": round(audio_duration, 3),
        "inference_time_s": round(total_time, 4),
        "rtf": round(audio_duration / total_time, 3) if total_time > 0 else None,
        "rss_delta_mb": round(rss_after - rss_before, 2),
        "tracemalloc_peak_mb": round(peak_tracemalloc / (1024 * 1024), 2),
    }


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_table(headers, rows):
    """Print a simple text table."""
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {}

    # --- Model Loading ---
    print_section("Model Loading")
    model, sr, load_info = benchmark_model_load()
    report["model_load"] = load_info
    print(f"  Model parameters:    {load_info['model_params_M']}M")
    print(f"  Load time:           {load_info['load_time_s']}s")
    print(f"  RSS delta:           {load_info['rss_delta_mb']} MB")
    print(f"  Tracemalloc peak:    {load_info['tracemalloc_peak_mb']} MB")
    print(f"  Sample rate:         {sr} Hz")

    # --- Load audio files ---
    fnames = sorted(glob_audio_files(TEST_DIR))
    print(f"\n  Found {len(fnames)} test files")

    # --- Warm-up run (excluded from metrics) ---
    print_section("Warm-up Run")
    warmup_audio = load_audio(fnames[0], sr)
    with torch.no_grad():
        _ = do_infer(model, warmup_audio, CHUNK_FACTOR, sr, stream=True)
        _ = do_infer(model, warmup_audio, CHUNK_FACTOR, sr, stream=False)
    print("  Warm-up complete (results excluded from report)")

    # --- Streaming Inference ---
    print_section("Streaming Inference (chunk_factor=1)")
    streaming_results = []
    for fname in fnames:
        audio = load_audio(fname, sr)
        output, metrics = benchmark_file_streaming(model, audio, sr, fname)
        streaming_results.append(metrics)
        out_path = os.path.join(OUT_DIR, "stream_" + os.path.basename(fname))
        import torchaudio
        torchaudio.save(out_path, output, sr)

    headers = ["File", "Duration(s)", "Infer(s)", "RTF", "E2E(ms)", "RSS_delta(MB)", "Peak_mem(MB)"]
    rows = []
    for m in streaming_results:
        rows.append([
            m["file"], m["audio_duration_s"], m["inference_time_s"],
            m["rtf"], m["e2e_latency_ms"], m["rss_delta_mb"], m["tracemalloc_peak_mb"]
        ])
    print_table(headers, rows)

    # Aggregate
    rtfs = [m["rtf"] for m in streaming_results if m["rtf"] is not None]
    e2es = [m["e2e_latency_ms"] for m in streaming_results if m["e2e_latency_ms"] is not None]
    infer_times = [m["inference_time_s"] for m in streaming_results]
    peak_mems = [m["tracemalloc_peak_mb"] for m in streaming_results]

    stream_agg = {
        "rtf_mean": round(np.mean(rtfs), 3),
        "rtf_min": round(np.min(rtfs), 3),
        "rtf_max": round(np.max(rtfs), 3),
        "e2e_latency_mean_ms": round(np.mean(e2es), 3),
        "e2e_latency_min_ms": round(np.min(e2es), 3),
        "e2e_latency_max_ms": round(np.max(e2es), 3),
        "inference_time_mean_s": round(np.mean(infer_times), 4),
        "peak_memory_mean_mb": round(np.mean(peak_mems), 2),
        "peak_memory_max_mb": round(np.max(peak_mems), 2),
    }
    report["streaming"] = {"per_file": streaming_results, "aggregate": stream_agg}

    print(f"\n  --- Streaming Aggregate ---")
    print(f"  RTF (mean/min/max):         {stream_agg['rtf_mean']} / {stream_agg['rtf_min']} / {stream_agg['rtf_max']}")
    print(f"  E2E latency (mean/min/max): {stream_agg['e2e_latency_mean_ms']} / {stream_agg['e2e_latency_min_ms']} / {stream_agg['e2e_latency_max_ms']} ms")
    print(f"  Inference time (mean):      {stream_agg['inference_time_mean_s']}s")
    print(f"  Peak memory (mean/max):     {stream_agg['peak_memory_mean_mb']} / {stream_agg['peak_memory_max_mb']} MB")

    # --- Non-Streaming Inference ---
    print_section("Non-Streaming Inference")
    nonstream_results = []
    for fname in fnames:
        audio = load_audio(fname, sr)
        output, metrics = benchmark_file_nonstreaming(model, audio, sr, fname)
        nonstream_results.append(metrics)
        out_path = os.path.join(OUT_DIR, "nonstream_" + os.path.basename(fname))
        torchaudio.save(out_path, output, sr)

    headers = ["File", "Duration(s)", "Infer(s)", "RTF", "RSS_delta(MB)", "Peak_mem(MB)"]
    rows = []
    for m in nonstream_results:
        rows.append([
            m["file"], m["audio_duration_s"], m["inference_time_s"],
            m["rtf"], m["rss_delta_mb"], m["tracemalloc_peak_mb"]
        ])
    print_table(headers, rows)

    # Aggregate
    ns_rtfs = [m["rtf"] for m in nonstream_results if m["rtf"] is not None]
    ns_infer_times = [m["inference_time_s"] for m in nonstream_results]
    ns_peak_mems = [m["tracemalloc_peak_mb"] for m in nonstream_results]

    nonstream_agg = {
        "rtf_mean": round(np.mean(ns_rtfs), 3),
        "rtf_min": round(np.min(ns_rtfs), 3),
        "rtf_max": round(np.max(ns_rtfs), 3),
        "inference_time_mean_s": round(np.mean(ns_infer_times), 4),
        "peak_memory_mean_mb": round(np.mean(ns_peak_mems), 2),
        "peak_memory_max_mb": round(np.max(ns_peak_mems), 2),
    }
    report["non_streaming"] = {"per_file": nonstream_results, "aggregate": nonstream_agg}

    print(f"\n  --- Non-Streaming Aggregate ---")
    print(f"  RTF (mean/min/max):     {nonstream_agg['rtf_mean']} / {nonstream_agg['rtf_min']} / {nonstream_agg['rtf_max']}")
    print(f"  Inference time (mean):  {nonstream_agg['inference_time_mean_s']}s")
    print(f"  Peak memory (mean/max): {nonstream_agg['peak_memory_mean_mb']} / {nonstream_agg['peak_memory_max_mb']} MB")

    # --- Overall System ---
    print_section("System Info")
    rss_final = get_rss_mb()
    sys_info = {
        "platform": os.uname().sysname,
        "machine": os.uname().machine,
        "python_version": f"{__import__('sys').version}",
        "torch_version": torch.__version__,
        "device": "cpu",
        "final_rss_mb": round(rss_final, 2),
    }
    report["system"] = sys_info
    print(f"  Platform:      {sys_info['platform']} {sys_info['machine']}")
    print(f"  Python:        {sys_info['python_version']}")
    print(f"  PyTorch:       {sys_info['torch_version']}")
    print(f"  Device:        {sys_info['device']}")
    print(f"  Final RSS:     {sys_info['final_rss_mb']} MB")

    # --- Save JSON report ---
    report_path = os.path.join(OUT_DIR, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
