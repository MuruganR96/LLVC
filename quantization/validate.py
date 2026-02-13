"""Validation and benchmarking for ONNX export.

Compares PyTorch vs ONNX inference outputs chunk-by-chunk and measures performance.

Usage:
    python quantization/validate.py \
        --pytorch_checkpoint llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth \
        --onnx_path quantization/llvc_fp32.onnx \
        --config experiments/llvc_hfg/config.json \
        --test_dir test_wavs
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torchaudio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Net
from quantization.onnx_inference import OnnxStreamingInferenceEngine


def validate_onnx_vs_pytorch(
    pytorch_checkpoint: str,
    config_path: str,
    onnx_path: str,
    test_dir: str = "test_wavs",
    chunk_factor: int = 4,
):
    """Compare PyTorch and ONNX model outputs chunk by chunk."""
    print("=" * 60)
    print("VALIDATION: PyTorch vs ONNX")
    print("=" * 60)

    # Load PyTorch model
    with open(config_path) as f:
        config = json.load(f)
    sr = config["data"]["sr"]

    net = Net(**config["model_params"])
    state = torch.load(pytorch_checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"])
    net.eval()

    L = net.L
    chunk_len = net.dec_chunk_size * L * chunk_factor

    # Load ONNX model
    onnx_engine = OnnxStreamingInferenceEngine()
    onnx_engine.load_model(onnx_path)

    # Find test files
    test_files = sorted(
        [
            os.path.join(test_dir, f)
            for f in os.listdir(test_dir)
            if f.endswith(".wav")
        ]
    )
    if not test_files:
        print(f"No .wav files found in {test_dir}")
        return

    print(f"Testing {len(test_files)} files, chunk_len={chunk_len}, L={L}")
    print()

    all_max_diffs = []
    all_mean_diffs = []

    for fpath in test_files:
        fname = os.path.basename(fpath)
        audio, file_sr = torchaudio.load(fpath)
        audio = audio.mean(0)  # mono
        if file_sr != sr:
            audio = torchaudio.transforms.Resample(file_sr, sr)(audio)
        audio_np = audio.numpy()

        # Pad to multiple of chunk_len
        remainder = len(audio_np) % chunk_len
        if remainder != 0:
            audio_np = np.pad(audio_np, (0, chunk_len - remainder))

        # PyTorch streaming
        enc_buf, dec_buf, out_buf = net.init_buffers(1, torch.device("cpu"))
        ctx = net.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
        prev_tail = torch.zeros(L * 2)
        pytorch_outputs = []

        for i in range(0, len(audio_np), chunk_len):
            chunk = torch.from_numpy(audio_np[i : i + chunk_len]).float()
            lookahead = prev_tail[-L * 2 :]
            inp = torch.cat([lookahead, chunk]).unsqueeze(0).unsqueeze(0)

            if len(chunk) >= L * 2:
                prev_tail = chunk[-L * 2 :]
            else:
                prev_tail = torch.cat([prev_tail[len(chunk) :], chunk])

            with torch.no_grad():
                out, enc_buf, dec_buf, out_buf, ctx = net(
                    inp, enc_buf, dec_buf, out_buf, ctx, pad=False
                )
            pytorch_outputs.append(out.squeeze().numpy())

        # ONNX streaming
        onnx_engine.init_streaming_state()
        onnx_outputs = []
        for i in range(0, len(audio_np), chunk_len):
            chunk = audio_np[i : i + chunk_len]
            out, _ = onnx_engine.process_chunk(chunk)
            onnx_outputs.append(out)

        # Compare
        pytorch_full = np.concatenate(pytorch_outputs)
        onnx_full = np.concatenate(onnx_outputs)

        max_diff = np.abs(pytorch_full - onnx_full).max()
        mean_diff = np.abs(pytorch_full - onnx_full).mean()
        all_max_diffs.append(max_diff)
        all_mean_diffs.append(mean_diff)

        # SNR
        signal_power = np.mean(pytorch_full**2)
        noise_power = np.mean((pytorch_full - onnx_full) ** 2)
        snr = (
            10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
        )

        print(f"  {fname:40s}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  SNR={snr:.1f}dB")

    print()
    print(f"Overall max diff:  {max(all_max_diffs):.2e}")
    print(f"Overall mean diff: {np.mean(all_mean_diffs):.2e}")
    passed = max(all_max_diffs) < 1e-4
    print(f"PASS: {passed} (threshold: 1e-4)")
    return passed


def benchmark_onnx(onnx_path: str, num_chunks: int = 100, chunk_factor: int = 4):
    """Benchmark ONNX Runtime inference speed."""
    print("=" * 60)
    print("BENCHMARK: ONNX Runtime")
    print("=" * 60)

    engine = OnnxStreamingInferenceEngine()
    info = engine.load_model(onnx_path)
    print(f"Model loaded in {info['load_time_ms']:.1f}ms")
    print(f"Chunk length: {info['chunk_len']} samples ({info['chunk_len']/info['sr']*1000:.1f}ms)")

    chunk_len = engine.get_chunk_len()
    np.random.seed(42)

    times = []
    for i in range(num_chunks):
        audio = np.random.randn(chunk_len).astype(np.float32) * 0.1
        _, metrics = engine.process_chunk(audio)
        times.append(metrics["inference_time_ms"])

    times = np.array(times[5:])  # skip warmup
    audio_duration_ms = chunk_len / engine.sr * 1000

    print(f"\nResults ({len(times)} chunks, excluding 5 warmup):")
    print(f"  Mean latency:  {times.mean():.2f} ms")
    print(f"  Median latency:{np.median(times):.2f} ms")
    print(f"  Min latency:   {times.min():.2f} ms")
    print(f"  Max latency:   {times.max():.2f} ms")
    print(f"  Std:           {times.std():.2f} ms")
    print(f"  Audio chunk:   {audio_duration_ms:.1f} ms")
    print(f"  Mean RTF:      {audio_duration_ms / times.mean():.2f}x")
    print(f"  Real-time:     {'YES' if times.mean() < audio_duration_ms else 'NO'}")


def benchmark_pytorch(
    checkpoint_path: str, config_path: str, num_chunks: int = 100, chunk_factor: int = 4
):
    """Benchmark PyTorch inference speed for comparison."""
    print("=" * 60)
    print("BENCHMARK: PyTorch")
    print("=" * 60)

    with open(config_path) as f:
        config = json.load(f)

    net = Net(**config["model_params"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"])
    net.eval()

    L = net.L
    chunk_len = net.dec_chunk_size * L * chunk_factor
    sr = config["data"]["sr"]

    enc_buf, dec_buf, out_buf = net.init_buffers(1, torch.device("cpu"))
    ctx = net.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
    prev_tail = torch.zeros(L * 2)

    torch.manual_seed(42)
    times = []

    for i in range(num_chunks):
        chunk = torch.randn(chunk_len) * 0.1
        lookahead = prev_tail[-L * 2 :]
        inp = torch.cat([lookahead, chunk]).unsqueeze(0).unsqueeze(0)
        prev_tail = chunk[-L * 2 :]

        t0 = time.perf_counter()
        with torch.no_grad():
            out, enc_buf, dec_buf, out_buf, ctx = net(
                inp, enc_buf, dec_buf, out_buf, ctx, pad=False
            )
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times[5:])
    audio_duration_ms = chunk_len / sr * 1000

    print(f"\nResults ({len(times)} chunks, excluding 5 warmup):")
    print(f"  Mean latency:  {times.mean():.2f} ms")
    print(f"  Median latency:{np.median(times):.2f} ms")
    print(f"  Min latency:   {times.min():.2f} ms")
    print(f"  Max latency:   {times.max():.2f} ms")
    print(f"  Audio chunk:   {audio_duration_ms:.1f} ms")
    print(f"  Mean RTF:      {audio_duration_ms / times.mean():.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ONNX export")
    parser.add_argument("--pytorch_checkpoint", help="PyTorch .pth checkpoint")
    parser.add_argument("--onnx_path", help="ONNX model path")
    parser.add_argument(
        "--config",
        default="experiments/llvc_hfg/config.json",
        help="Config path",
    )
    parser.add_argument("--test_dir", default="test_wavs", help="Test audio dir")
    parser.add_argument(
        "--chunk_factor", type=int, default=4, help="Chunk factor"
    )
    parser.add_argument(
        "--benchmark_chunks", type=int, default=100, help="Chunks for benchmark"
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "benchmark", "all"],
        default="all",
        help="Run mode",
    )
    args = parser.parse_args()

    if args.mode in ("validate", "all"):
        if args.pytorch_checkpoint and args.onnx_path:
            validate_onnx_vs_pytorch(
                args.pytorch_checkpoint,
                args.config,
                args.onnx_path,
                args.test_dir,
                args.chunk_factor,
            )
        else:
            print("Skipping validation (need --pytorch_checkpoint and --onnx_path)")

    if args.mode in ("benchmark", "all"):
        if args.onnx_path:
            benchmark_onnx(args.onnx_path, args.benchmark_chunks, args.chunk_factor)
        if args.pytorch_checkpoint:
            benchmark_pytorch(
                args.pytorch_checkpoint,
                args.config,
                args.benchmark_chunks,
                args.chunk_factor,
            )
