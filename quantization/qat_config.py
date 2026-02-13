"""Quantization Aware Training configuration for LLVC.

Defines QConfig (observers, fake quantize), and specifies which modules
to quantize (Conv1d, Linear) and which to skip (LayerNorm, attention,
BatchNorm, activations).
"""

import torch
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torch.ao.quantization.fake_quantize import FakeQuantize


def get_qat_qconfig() -> QConfig:
    """Return the QConfig for LLVC QAT.

    - Activation: asymmetric quint8 [0, 255] with moving average min/max.
    - Weight: symmetric per-channel qint8 [-128, 127] with moving average.
    """
    return QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )


# Module name prefixes to SKIP during QAT (stay float32).
# These are matched as prefixes against the full dotted name.
_SKIP_PREFIXES = [
    # LayerNorm variants (numerically sensitive)
    "mask_gen.encoder.dcc_layers.",  # contains LayerNormPermuted inside DSC
    # We handle DSC layers individually — only quantize their Conv1d children

    # Attention layers (sensitive to quantization noise)
    "mask_gen.decoder.tf_dec_layers.",  # self_attn, multihead_attn, norms inside

    # Positional encoding (fixed, not learnable weights)
    "mask_gen.decoder.pos_enc",

    # Activations and norms that shouldn't be quantized
    "label_embedding.1",   # LayerNorm
    "label_embedding.4",   # LayerNorm
]


def get_modules_to_quantize():
    """Return a dict mapping module name -> description for modules to quantize.

    These are the weight-heavy modules where int8 quantization provides
    meaningful memory savings.
    """
    modules = {}

    # Input conv (Conv1d stride=16, kernel=48)
    modules["in_conv.0"] = "Conv1d(1, 512, k=48, s=16)"

    # Label embedding linear layers
    modules["label_embedding.0"] = "Linear(1, 512)"
    modules["label_embedding.3"] = "Linear(512, 512)"

    # DCC encoder: 8 layers, each with depthwise + pointwise Conv1d
    for i in range(8):
        modules[f"mask_gen.encoder.dcc_layers.dcc_{i}.layers.0"] = (
            f"DCC[{i}] depthwise Conv1d(512, 512, k=3, groups=512)"
        )
        modules[f"mask_gen.encoder.dcc_layers.dcc_{i}.layers.3"] = (
            f"DCC[{i}] pointwise Conv1d(512, 512, k=1)"
        )

    # Projection layers
    modules["mask_gen.proj_e2d_e.0"] = "Conv1d(512, 256, k=1, groups=256)"
    modules["mask_gen.proj_e2d_l.0"] = "Conv1d(512, 256, k=1, groups=256)"
    modules["mask_gen.proj_d2e.0"] = "Conv1d(256, 512, k=1, groups=256)"

    # Decoder FFN (linear1, linear2) inside transformer decoder layers
    for i in range(1):  # num_dec_layers = 1
        modules[f"mask_gen.decoder.tf_dec_layers.{i}.linear1"] = (
            f"Decoder[{i}] FFN Linear(256, 512)"
        )
        modules[f"mask_gen.decoder.tf_dec_layers.{i}.linear2"] = (
            f"Decoder[{i}] FFN Linear(512, 256)"
        )

    # Output conv (ConvTranspose1d)
    modules["out_conv.0"] = "ConvTranspose1d(512, 1, k=80, s=16)"

    # CachedConvNet prenet: 12 layers (ResidualBlock with filter + gate convs)
    for i in range(12):
        modules[f"convnet_pre.down_convs.{i}.filter"] = (
            f"ConvNet[{i}] filter Conv1d(1, 1, k=3)"
        )
        modules[f"convnet_pre.down_convs.{i}.gate"] = (
            f"ConvNet[{i}] gate Conv1d(1, 1, k=3)"
        )

    return modules


def should_skip_module(name: str) -> bool:
    """Check if a module should be skipped (kept float32) during QAT."""
    for prefix in _SKIP_PREFIXES:
        if name.startswith(prefix):
            return True
    return False
