"""Stateful streaming inference engine for real-time voice conversion.

Wraps the LLVC model for chunk-by-chunk processing, maintaining state
buffers between chunks. Adapted from infer.py:44-96.
"""

import sys
import os
import time
import json
import numpy as np
import torch

# Add project root to path so we can import model, infer, utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Net
from utils import model_size


# Model registry
MODELS_REGISTRY = {
    "llvc": {
        "name": "LLVC (Standard)",
        "checkpoint": "llvc_models/models/checkpoints/llvc/G_500000.pth",
        "config": "experiments/llvc/config.json",
        "description": "Standard LLVC with ConvNet prenet and RVC discriminator",
    },
    "llvc_hfg": {
        "name": "LLVC-HFG",
        "checkpoint": "llvc_models/models/checkpoints/llvc_hfg/G_500000.pth",
        "config": "experiments/llvc_hfg/config.json",
        "description": "LLVC with HiFi-GAN discriminator",
    },
    "llvc_nc": {
        "name": "LLVC-NC (No Convnet)",
        "checkpoint": "llvc_models/models/checkpoints/llvc_nc/G_500000.pth",
        "config": "experiments/llvc_nc/config.json",
        "description": "LLVC without ConvNet prenet — faster, lower quality",
    },
}


class StreamingInferenceEngine:
    """Chunk-by-chunk streaming voice conversion engine.

    Maintains model state buffers (enc_buf, dec_buf, out_buf, convnet_pre_ctx)
    between calls to process_chunk(). Each chunk is 208 * chunk_factor samples
    at 16kHz.
    """

    def __init__(self):
        self.model = None
        self.sr = None
        self.current_model_key = None
        self.chunk_factor = 4
        self.is_ready = False

        # Streaming state
        self.enc_buf = None
        self.dec_buf = None
        self.out_buf = None
        self.convnet_pre_ctx = None
        self.prev_chunk_tail = None
        self.chunk_index = 0

        # Model params (set on load)
        self.L = None
        self.chunk_len = None

    def load_model(self, model_key):
        """Load a model by registry key."""
        info = MODELS_REGISTRY[model_key]
        checkpoint_path = os.path.join(PROJECT_ROOT, info["checkpoint"])
        config_path = os.path.join(PROJECT_ROOT, info["config"])

        with open(config_path) as f:
            config = json.load(f)

        self.model = Net(**config["model_params"])
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.sr = config["data"]["sr"]
        self.L = self.model.L
        self.chunk_len = self.model.dec_chunk_size * self.L * self.chunk_factor
        self.current_model_key = model_key
        self.init_streaming_state()
        self.is_ready = True

        return {
            "model_key": model_key,
            "name": info["name"],
            "params_M": model_size(self.model),
            "sr": self.sr,
            "L": self.L,
            "chunk_len": self.chunk_len,
        }

    def init_streaming_state(self):
        """Reset all streaming buffers for a new session."""
        if self.model is None:
            return
        self.chunk_len = self.model.dec_chunk_size * self.L * self.chunk_factor
        device = torch.device("cpu")
        self.enc_buf, self.dec_buf, self.out_buf = self.model.init_buffers(1, device)
        if hasattr(self.model, "convnet_pre"):
            self.convnet_pre_ctx = self.model.convnet_pre.init_ctx_buf(1, device)
        else:
            self.convnet_pre_ctx = None
        self.prev_chunk_tail = torch.zeros(self.L * 2)
        self.chunk_index = 0

    def get_chunk_len(self):
        """Return expected chunk length in samples."""
        if self.model is None:
            return 208
        return self.model.dec_chunk_size * self.L * self.chunk_factor

    def process_chunk(self, audio_np):
        """Process a single chunk of audio.

        Args:
            audio_np: numpy float32 array of chunk_len samples (normalized -1..1)

        Returns:
            (output_np, metrics): output audio (float32) and timing metrics dict
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        t0 = time.perf_counter()

        chunk = torch.from_numpy(audio_np).float()

        # Shift chunk down by L (per infer.py line 54)
        # For real-time streaming we apply this per-chunk:
        # prepend lookahead from prev chunk, then the chunk itself
        # The shift is implicit: we take 2*L from prev and chunk_len from current
        lookahead = self.prev_chunk_tail[-self.L * 2:]
        input_tensor = torch.cat([lookahead, chunk])

        # Save tail for next chunk's lookahead
        self.prev_chunk_tail = chunk[-self.L * 2:] if len(chunk) >= self.L * 2 else \
            torch.cat([self.prev_chunk_tail[len(chunk):], chunk])

        with torch.inference_mode():
            output, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = \
                self.model(
                    input_tensor.unsqueeze(0).unsqueeze(0),
                    self.enc_buf,
                    self.dec_buf,
                    self.out_buf,
                    self.convnet_pre_ctx,
                    pad=(not self.model.lookahead),
                )

        output_np = output.squeeze(0).squeeze(0).cpu().numpy()
        inference_time = time.perf_counter() - t0
        audio_duration = len(audio_np) / self.sr if self.sr else 0
        rtf = audio_duration / inference_time if inference_time > 0 else 0

        self.chunk_index += 1

        metrics = {
            "chunk_index": self.chunk_index,
            "inference_time_ms": round(inference_time * 1000, 2),
            "rtf": round(rtf, 3),
            "audio_duration_ms": round(audio_duration * 1000, 2),
        }
        return output_np.astype(np.float32), metrics

    def process_file_audio(self, audio_np):
        """Process an entire audio array in streaming chunks.

        Args:
            audio_np: numpy float32 array, full audio at model sample rate

        Yields:
            (output_chunk_np, metrics) for each chunk
        """
        self.init_streaming_state()
        chunk_len = self.get_chunk_len()

        # Pad to multiple of chunk_len
        remainder = len(audio_np) % chunk_len
        if remainder != 0:
            audio_np = np.pad(audio_np, (0, chunk_len - remainder))

        for i in range(0, len(audio_np), chunk_len):
            chunk = audio_np[i:i + chunk_len]
            output, metrics = self.process_chunk(chunk)
            yield output, metrics
