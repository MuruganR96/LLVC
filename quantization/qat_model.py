"""QAT model preparation and conversion for LLVC.

Handles:
- Wrapping Net with QuantStub/DeQuantStub
- Replacing target Conv1d/Linear with QAT equivalents (with FakeQuantize)
- Custom FakeQuantizedConvTranspose1d for the output conv
- Converting QAT model to quantized int8 after training
"""

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Net
from quantization.qat_config import get_qat_qconfig, get_modules_to_quantize


class FakeQuantizedConvTranspose1d(nn.Module):
    """ConvTranspose1d with manual FakeQuantize on weights.

    PyTorch doesn't have a built-in qat.ConvTranspose1d, so we wrap
    the standard module with explicit FakeQuantize on weights during
    forward.
    """

    def __init__(self, conv_transpose: nn.ConvTranspose1d):
        super().__init__()
        self.conv_transpose = conv_transpose
        # Per-channel symmetric weight fake quantization
        self.weight_fake_quant = FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        )()
        # Activation fake quantization on output
        self.activation_fake_quant = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
        )()

    def forward(self, x):
        # Apply fake quant to weights
        fq_weight = self.weight_fake_quant(self.conv_transpose.weight)
        out = torch.nn.functional.conv_transpose1d(
            x,
            fq_weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride,
            self.conv_transpose.padding,
            self.conv_transpose.output_padding,
            self.conv_transpose.groups,
            self.conv_transpose.dilation,
        )
        return self.activation_fake_quant(out)


class QATNet(nn.Module):
    """Wrapper around Net with QuantStub at input and DeQuantStub at output.

    The QuantStub/DeQuantStub pair marks the boundary of the quantized region.
    FakeQuantize nodes inside the network simulate int8 quantization during
    training.
    """

    def __init__(self, net: Net):
        super().__init__()
        self.net = net
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x, init_enc_buf=None, init_dec_buf=None,
                init_out_buf=None, convnet_pre_ctx=None, pad=True):
        x = self.quant(x)
        result = self.net(x, init_enc_buf, init_dec_buf,
                          init_out_buf, convnet_pre_ctx, pad)
        if isinstance(result, tuple):
            out = self.dequant(result[0])
            return (out,) + result[1:]
        return self.dequant(result)


def _replace_module_by_name(model: nn.Module, target_name: str, new_module: nn.Module):
    """Replace a nested module by its dotted name path."""
    parts = target_name.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _get_module_by_name(model: nn.Module, target_name: str) -> nn.Module:
    """Get a nested module by its dotted name path."""
    parts = target_name.split('.')
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def prepare_qat_model(
    checkpoint_path: str,
    config_path: str,
    device: str = "cpu",
) -> QATNet:
    """Load a pretrained Net and prepare it for QAT.

    1. Load pretrained weights from checkpoint
    2. Replace target Conv1d with qat.Conv1d (copies weights)
    3. Replace target Linear with qat.Linear (copies weights)
    4. Replace ConvTranspose1d with FakeQuantizedConvTranspose1d
    5. Wrap in QATNet with QuantStub/DeQuantStub
    6. Set model to train mode for QAT

    Args:
        checkpoint_path: Path to pretrained .pth checkpoint
        config_path: Path to config.json
        device: Device to load model on

    Returns:
        QATNet ready for fine-tuning
    """
    import json

    with open(config_path) as f:
        config = json.load(f)

    net = Net(**config["model_params"])
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(state["model"])
    net.to(device)

    qconfig = get_qat_qconfig()
    modules_to_quantize = get_modules_to_quantize()

    replaced = []
    for name, desc in modules_to_quantize.items():
        try:
            module = _get_module_by_name(net, name)
        except (AttributeError, IndexError):
            print(f"  [skip] {name} not found in model")
            continue

        if isinstance(module, nn.Conv1d):
            # Replace with QAT Conv1d
            qat_conv = torch.ao.nn.qat.Conv1d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
                qconfig=qconfig,
            )
            qat_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                qat_conv.bias.data.copy_(module.bias.data)
            _replace_module_by_name(net, name, qat_conv)
            replaced.append((name, "Conv1d -> qat.Conv1d"))

        elif isinstance(module, nn.Linear):
            qat_linear = torch.ao.nn.qat.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                qconfig=qconfig,
            )
            qat_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                qat_linear.bias.data.copy_(module.bias.data)
            _replace_module_by_name(net, name, qat_linear)
            replaced.append((name, "Linear -> qat.Linear"))

        elif isinstance(module, nn.ConvTranspose1d):
            fq_ct = FakeQuantizedConvTranspose1d(module)
            _replace_module_by_name(net, name, fq_ct)
            replaced.append((name, "ConvTranspose1d -> FakeQuantized"))

        else:
            print(f"  [skip] {name}: unsupported type {type(module).__name__}")

    print(f"QAT preparation: replaced {len(replaced)} modules")
    for name, desc in replaced:
        print(f"  {name}: {desc}")

    qat_model = QATNet(net)
    qat_model.train()

    return qat_model


def convert_to_quantized(qat_model: QATNet) -> nn.Module:
    """Convert a QAT-trained model to quantized int8.

    After QAT training, this freezes the FakeQuantize parameters
    and converts to actual int8 operations.

    Args:
        qat_model: QATNet after QAT fine-tuning

    Returns:
        Quantized model with int8 weights
    """
    qat_model.eval()
    # Disable observers and freeze fake quant parameters
    qat_model.apply(torch.ao.quantization.disable_observer)
    quantized = torch.ao.quantization.convert(qat_model, inplace=False)
    return quantized


def save_qat_checkpoint(
    model: QATNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    lr: float,
    path: str,
):
    """Save QAT training checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "learning_rate": lr,
    }
    torch.save(state, path)
    print(f"Saved QAT checkpoint to {path} (step {step})")


def load_qat_checkpoint(
    path: str,
    model: QATNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> tuple:
    """Load QAT training checkpoint.

    Returns:
        (model, optimizer, lr, epoch, step)
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return model, optimizer, state["learning_rate"], state["epoch"], state["step"]
