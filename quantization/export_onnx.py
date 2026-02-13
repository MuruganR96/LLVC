"""Export LLVC model to ONNX format.

Supports both float32 and quantized (int8 QDQ) export.

Usage:
    # Float32 export
    python quantization/export_onnx.py \
        --checkpoint llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth \
        --config experiments/llvc_hfg/config.json \
        --output quantization/llvc_fp32.onnx

    # Int8 quantized export (after QAT training)
    python quantization/export_onnx.py \
        --checkpoint quantization/qat_output/G_qat_final.pth \
        --config experiments/llvc_hfg/config.json \
        --output quantization/llvc_int8.onnx --quantized
"""

import argparse
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Net
from quantization.onnx_wrapper import OnnxStreamingWrapper


def export_float32_onnx(
    checkpoint_path: str,
    config_path: str,
    output_path: str = "quantization/llvc_fp32.onnx",
    chunk_factor: int = 4,
    opset_version: int = 17,
):
    """Export float32 model to ONNX."""
    print(f"Loading model from {checkpoint_path}...")
    with open(config_path) as f:
        config = json.load(f)

    net = Net(**config["model_params"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model"])
    net.eval()

    print(f"Creating ONNX wrapper (chunk_factor={chunk_factor})...")
    wrapper = OnnxStreamingWrapper(net, chunk_factor=chunk_factor)
    wrapper.eval()

    L = wrapper.L
    chunk_len = wrapper.chunk_len

    # Create dummy inputs with exact shapes
    audio = torch.randn(1, 1, chunk_len + 2 * L)
    enc_buf, dec_buf, out_buf = net.init_buffers(1, torch.device("cpu"))
    convnet_pre_ctx = (
        net.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
        if hasattr(net, "convnet_pre")
        else torch.zeros(1, 1, 1)
    )

    print(f"Input shapes:")
    print(f"  audio:           {list(audio.shape)}")
    print(f"  enc_buf:         {list(enc_buf.shape)}")
    print(f"  dec_buf:         {list(dec_buf.shape)}")
    print(f"  out_buf:         {list(out_buf.shape)}")
    print(f"  convnet_pre_ctx: {list(convnet_pre_ctx.shape)}")

    # Export
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    t0 = time.perf_counter()

    torch.onnx.export(
        wrapper,
        (audio, enc_buf, dec_buf, out_buf, convnet_pre_ctx),
        output_path,
        opset_version=opset_version,
        input_names=["audio", "enc_buf", "dec_buf", "out_buf", "convnet_pre_ctx"],
        output_names=[
            "output",
            "new_enc_buf",
            "new_dec_buf",
            "new_out_buf",
            "new_convnet_pre_ctx",
        ],
        dynamic_axes=None,  # all shapes fixed
    )
    export_time = time.perf_counter() - t0
    print(f"ONNX export completed in {export_time:.2f}s -> {output_path}")

    # Validate
    import onnx

    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model validation passed")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {file_size_mb:.1f} MB")

    return output_path


def export_quantized_onnx(
    qat_checkpoint_path: str,
    config_path: str,
    output_path: str = "quantization/llvc_int8.onnx",
    chunk_factor: int = 4,
    opset_version: int = 17,
):
    """Export QAT-trained model to ONNX with QDQ nodes.

    The FakeQuantize nodes from QAT are exported as QuantizeLinear/DequantizeLinear
    (QDQ) pairs. ONNX Runtime recognizes these and fuses them to use int8 kernels.
    """
    print(f"Loading QAT model from {qat_checkpoint_path}...")
    with open(config_path) as f:
        config = json.load(f)

    # Load QAT model
    from quantization.qat_model import prepare_qat_model

    qat_model = prepare_qat_model(qat_checkpoint_path, config_path)
    qat_model.eval()

    # The QATNet wraps Net, so extract the inner net for the wrapper
    inner_net = qat_model.net
    wrapper = OnnxStreamingWrapper(inner_net, chunk_factor=chunk_factor)
    wrapper.eval()

    L = wrapper.L
    chunk_len = wrapper.chunk_len

    audio = torch.randn(1, 1, chunk_len + 2 * L)
    enc_buf, dec_buf, out_buf = inner_net.init_buffers(1, torch.device("cpu"))
    convnet_pre_ctx = (
        inner_net.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
        if hasattr(inner_net, "convnet_pre")
        else torch.zeros(1, 1, 1)
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    t0 = time.perf_counter()

    torch.onnx.export(
        wrapper,
        (audio, enc_buf, dec_buf, out_buf, convnet_pre_ctx),
        output_path,
        opset_version=opset_version,
        input_names=["audio", "enc_buf", "dec_buf", "out_buf", "convnet_pre_ctx"],
        output_names=[
            "output",
            "new_enc_buf",
            "new_dec_buf",
            "new_out_buf",
            "new_convnet_pre_ctx",
        ],
        dynamic_axes=None,
    )
    export_time = time.perf_counter() - t0
    print(f"Quantized ONNX export completed in {export_time:.2f}s -> {output_path}")

    import onnx

    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print("ONNX model validation passed")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {file_size_mb:.1f} MB")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LLVC to ONNX")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pth checkpoint"
    )
    parser.add_argument(
        "--config",
        default="experiments/llvc_hfg/config.json",
        help="Path to config.json",
    )
    parser.add_argument(
        "--output",
        default="quantization/llvc_fp32.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--chunk_factor", type=int, default=4, help="Chunk factor (default: 4)"
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Export quantized (QAT) model with QDQ nodes",
    )
    args = parser.parse_args()

    if args.quantized:
        export_quantized_onnx(
            args.checkpoint, args.config, args.output,
            args.chunk_factor, args.opset,
        )
    else:
        export_float32_onnx(
            args.checkpoint, args.config, args.output,
            args.chunk_factor, args.opset,
        )
