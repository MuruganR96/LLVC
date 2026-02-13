"""ONNX Runtime streaming inference engine for LLVC.

Drop-in replacement for web_app/audio_processor.py:StreamingInferenceEngine,
using ONNX Runtime instead of PyTorch for inference.

Usage:
    from quantization.onnx_inference import OnnxStreamingInferenceEngine

    engine = OnnxStreamingInferenceEngine()
    engine.load_model("quantization/llvc_fp32.onnx")
    output, metrics = engine.process_chunk(audio_np)
"""

import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")


class OnnxStreamingInferenceEngine:
    """ONNX Runtime streaming inference engine.

    Matches the API of web_app.audio_processor.StreamingInferenceEngine
    for drop-in integration.
    """

    def __init__(self, num_threads: int = 4):
        self.session = None
        self.sr = 16000
        self.L = 16
        self.chunk_factor = 4
        self.chunk_len = None
        self.is_ready = False
        self.current_model_key = None
        self.num_threads = num_threads

        # State buffers (numpy)
        self.enc_buf = None
        self.dec_buf = None
        self.out_buf = None
        self.convnet_pre_ctx = None
        self.prev_chunk_tail = None
        self.chunk_index = 0

        # Buffer shapes (detected from ONNX model inputs)
        self._enc_buf_shape = None
        self._dec_buf_shape = None
        self._out_buf_shape = None
        self._ctx_shape = None

    def load_model(self, onnx_path: str, model_key: str = "onnx"):
        """Load an ONNX model.

        Args:
            onnx_path: Path to .onnx file
            model_key: Identifier string for this model
        Returns:
            dict with model info
        """
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = 1

        providers = ["CPUExecutionProvider"]
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(
            onnx_path, sess_options, providers=providers
        )
        load_time = time.perf_counter() - t0

        # Detect buffer shapes from model inputs
        inputs = {inp.name: inp.shape for inp in self.session.get_inputs()}
        self._enc_buf_shape = tuple(inputs["enc_buf"])
        self._dec_buf_shape = tuple(inputs["dec_buf"])
        self._out_buf_shape = tuple(inputs["out_buf"])
        self._ctx_shape = tuple(inputs["convnet_pre_ctx"])

        # Derive chunk_len from audio input shape
        audio_shape = tuple(inputs["audio"])
        audio_len = audio_shape[-1]
        self.chunk_len = audio_len - 2 * self.L

        self.current_model_key = model_key
        self.init_streaming_state()
        self.is_ready = True

        return {
            "model_key": model_key,
            "name": f"ONNX: {model_key}",
            "sr": self.sr,
            "L": self.L,
            "chunk_len": self.chunk_len,
            "load_time_ms": round(load_time * 1000, 1),
        }

    def init_streaming_state(self):
        """Reset all state buffers to zeros."""
        self.enc_buf = np.zeros(self._enc_buf_shape, dtype=np.float32)
        self.dec_buf = np.zeros(self._dec_buf_shape, dtype=np.float32)
        self.out_buf = np.zeros(self._out_buf_shape, dtype=np.float32)
        self.convnet_pre_ctx = np.zeros(self._ctx_shape, dtype=np.float32)
        self.prev_chunk_tail = np.zeros(self.L * 2, dtype=np.float32)
        self.chunk_index = 0

    def get_chunk_len(self):
        """Return expected chunk length in samples."""
        return self.chunk_len or 832

    def process_chunk(self, audio_np: np.ndarray):
        """Process a single chunk of audio.

        Args:
            audio_np: float32 array of chunk_len samples (normalized -1..1)
        Returns:
            (output_np, metrics): output audio and timing metrics dict
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded")

        t0 = time.perf_counter()

        # Build input with lookahead context
        lookahead = self.prev_chunk_tail[-self.L * 2 :]
        input_audio = np.concatenate([lookahead, audio_np])

        # Save tail for next chunk
        if len(audio_np) >= self.L * 2:
            self.prev_chunk_tail = audio_np[-self.L * 2 :].copy()
        else:
            self.prev_chunk_tail = np.concatenate(
                [self.prev_chunk_tail[len(audio_np) :], audio_np]
            )

        # Reshape to [1, 1, T]
        input_tensor = input_audio.reshape(1, 1, -1).astype(np.float32)

        # Run ONNX inference
        outputs = self.session.run(
            [
                "output",
                "new_enc_buf",
                "new_dec_buf",
                "new_out_buf",
                "new_convnet_pre_ctx",
            ],
            {
                "audio": input_tensor,
                "enc_buf": self.enc_buf,
                "dec_buf": self.dec_buf,
                "out_buf": self.out_buf,
                "convnet_pre_ctx": self.convnet_pre_ctx,
            },
        )

        (
            output_np,
            self.enc_buf,
            self.dec_buf,
            self.out_buf,
            self.convnet_pre_ctx,
        ) = outputs
        output_np = output_np.squeeze()

        inference_time = time.perf_counter() - t0
        audio_duration = len(audio_np) / self.sr
        rtf = audio_duration / inference_time if inference_time > 0 else 0

        self.chunk_index += 1

        metrics = {
            "chunk_index": self.chunk_index,
            "inference_time_ms": round(inference_time * 1000, 2),
            "rtf": round(rtf, 3),
            "audio_duration_ms": round(audio_duration * 1000, 2),
        }
        return output_np.astype(np.float32), metrics

    def process_file_audio(self, audio_np: np.ndarray):
        """Process full audio in streaming chunks.

        Args:
            audio_np: float32 array, full audio at model sample rate
        Yields:
            (output_chunk_np, metrics) for each chunk
        """
        self.init_streaming_state()
        chunk_len = self.get_chunk_len()

        remainder = len(audio_np) % chunk_len
        if remainder != 0:
            audio_np = np.pad(audio_np, (0, chunk_len - remainder))

        for i in range(0, len(audio_np), chunk_len):
            chunk = audio_np[i : i + chunk_len]
            output, metrics = self.process_chunk(chunk)
            yield output, metrics
