"""ONNX-exportable streaming wrapper for the LLVC model.

Resolves all ONNX export blockers:
1. In-place buffer updates -> out-of-place torch.cat reconstruction
2. nn.LeakyReLU(inplace=True) -> inplace=False
3. nn.Unfold in _causal_unfold -> static explicit slices
4. Dynamic loop for j in range(ceil(...)) -> single iteration (fixed B=1)
5. Conditional control flow -> evaluated at __init__ time
6. torch.zeros for label -> pre-computed constant buffer
"""

import sys
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import Net


class OnnxStreamingWrapper(nn.Module):
    """ONNX-exportable streaming inference wrapper for LLVC Net.

    All state buffers are explicit inputs and outputs.
    Fixed chunk_factor at export time (default 4).
    """

    def __init__(self, net: Net, chunk_factor: int = 4):
        super().__init__()
        self.chunk_factor = chunk_factor
        self.L = net.L
        self.enc_dim = net.enc_dim
        self.out_buf_len = net.out_buf_len
        self.dec_chunk_size = net.dec_chunk_size
        self.chunk_len = self.dec_chunk_size * self.L * chunk_factor

        # --- CachedConvNet prenet ---
        self.has_convnet_pre = hasattr(net, 'convnet_pre')
        if self.has_convnet_pre:
            self.convnet_pre = net.convnet_pre
            self.convnet_skip_connection = net.convnet_config['skip_connection']
            self._fix_inplace_relu(self.convnet_pre)
            self.convnet_buf_lengths = list(self.convnet_pre.buf_lengths)
            self.convnet_buf_indices = list(self.convnet_pre.buf_indices)
            self.convnet_num_layers = self.convnet_pre.num_layers
            self.convnet_combine_residuals = self.convnet_pre.combine_residuals

        # --- in_conv ---
        self.in_conv = net.in_conv

        # --- Pre-compute label embedding constant ---
        with torch.no_grad():
            label_const = net.label_embedding(torch.zeros(1, 1))
        self.register_buffer('label_const', label_const)

        # --- MaskNet components ---
        self.encoder = net.mask_gen.encoder
        self.proj_e2d_e = net.mask_gen.proj_e2d_e
        self.proj_e2d_l = net.mask_gen.proj_e2d_l
        self.proj_d2e = net.mask_gen.proj_d2e
        self.has_proj = net.mask_gen.proj
        self.has_skip = net.mask_gen.skip_connection

        # --- Decoder components ---
        self.ctx_len = net.mask_gen.decoder.ctx_len
        self.dec_model_dim = net.mask_gen.decoder.model_dim
        self.num_dec_layers = net.mask_gen.decoder.num_layers
        self.use_pos_enc = net.mask_gen.decoder.use_pos_enc
        self.pos_enc = net.mask_gen.decoder.pos_enc
        self.tf_dec_layers = net.mask_gen.decoder.tf_dec_layers

        # --- Encoder buffer metadata ---
        self.enc_buf_lengths = list(self.encoder.buf_lengths)
        self.enc_buf_indices = list(self.encoder.buf_indices)
        self.enc_num_layers = self.encoder.num_layers

        # --- out_conv ---
        self.out_conv = net.out_conv

        # --- Pre-compute decoder geometry ---
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.chunk_len + 2 * self.L)
            if self.has_convnet_pre:
                ctx = self.convnet_pre.init_ctx_buf(1, torch.device('cpu'))
                conv_out, _ = self.convnet_pre(dummy, ctx)
                if self.convnet_skip_connection == 'add':
                    dummy = dummy + conv_out
                elif self.convnet_skip_connection == 'multiply':
                    dummy = dummy * conv_out
                else:
                    dummy = conv_out
            dummy_enc = self.in_conv(dummy)
            self.T_enc = dummy_enc.shape[-1]

        self.dec_mod = self.T_enc % self.dec_chunk_size
        if self.dec_mod != 0:
            self.dec_pad = self.dec_chunk_size - self.dec_mod
            self.T_enc_padded = self.T_enc + self.dec_pad
        else:
            self.dec_pad = 0
            self.T_enc_padded = self.T_enc

        self.num_unfold_windows = self.T_enc_padded // self.dec_chunk_size
        self.window_size = self.ctx_len + self.dec_chunk_size

    @staticmethod
    def _fix_inplace_relu(module: nn.Module):
        """Replace all inplace LeakyReLU/ReLU with non-inplace versions."""
        for _, child in module.named_modules():
            if isinstance(child, nn.LeakyReLU) and child.inplace:
                child.inplace = False
            elif isinstance(child, nn.ReLU) and child.inplace:
                child.inplace = False

    def _convnet_pre_forward(self, x: Tensor, ctx: Tensor) -> Tuple[Tensor, Tensor]:
        """CachedConvNet forward with out-of-place buffer updates.

        The original CachedConvNet does in-place: ctx[..., start:end] = new_vals.
        We replace with torch.cat reconstruction.
        """
        new_ctx = ctx.clone()

        for i in range(self.convnet_num_layers):
            buf_start = self.convnet_buf_indices[i]
            buf_end = buf_start + self.convnet_buf_lengths[i]

            # Slice context for this layer
            ctx_slice = new_ctx[:, :x.shape[1], buf_start:buf_end]
            conv_in = torch.cat((ctx_slice, x), dim=-1)

            # Out-of-place buffer update
            new_buf_slice = conv_in[:, :, -self.convnet_buf_lengths[i]:]
            # Pad channel dim if needed (ctx may have more channels than x)
            if new_buf_slice.shape[1] < new_ctx.shape[1]:
                pad_size = new_ctx.shape[1] - new_buf_slice.shape[1]
                padded_slice = F.pad(new_buf_slice, (0, 0, 0, pad_size))
            else:
                padded_slice = new_buf_slice
            new_ctx = torch.cat([
                new_ctx[:, :, :buf_start],
                padded_slice,
                new_ctx[:, :, buf_end:]
            ], dim=-1)

            # Apply convolution
            if self.convnet_combine_residuals == 'add':
                x = x + self.convnet_pre.down_convs[i](conv_in)
            elif self.convnet_combine_residuals == 'multiply':
                x = x * self.convnet_pre.down_convs[i](conv_in)
            else:
                x = self.convnet_pre.down_convs[i](conv_in)

        return x, new_ctx

    def _encoder_forward(self, x: Tensor, enc_buf: Tensor) -> Tuple[Tensor, Tensor]:
        """DilatedCausalConvEncoder forward with out-of-place buffer updates."""
        new_enc_buf = enc_buf.clone()

        for i in range(self.enc_num_layers):
            buf_start = self.enc_buf_indices[i]
            buf_end = buf_start + self.enc_buf_lengths[i]

            ctx_slice = new_enc_buf[:, :, buf_start:buf_end]
            dcc_in = torch.cat((ctx_slice, x), dim=-1)

            new_buf_slice = dcc_in[:, :, -self.enc_buf_lengths[i]:]
            new_enc_buf = torch.cat([
                new_enc_buf[:, :, :buf_start],
                new_buf_slice,
                new_enc_buf[:, :, buf_end:]
            ], dim=-1)

            x = x + self.encoder.dcc_layers[i](dcc_in)

        return x, new_enc_buf

    def _causal_unfold_static(self, x: Tensor) -> Tensor:
        """Static causal unfold replacement for nn.Unfold.

        Args:
            x: [B, T_with_ctx, C]
        Returns:
            [B * num_windows, window_size, C]
        """
        windows = []
        for i in range(self.num_unfold_windows):
            start = i * self.dec_chunk_size
            end = start + self.window_size
            windows.append(x[:, start:end, :])
        return torch.cat(windows, dim=0)

    def _decoder_forward(self, tgt: Tensor, mem: Tensor,
                         dec_buf: Tensor) -> Tuple[Tensor, Tensor]:
        """CausalTransformerDecoder forward with out-of-place updates."""
        chunk_size = self.dec_chunk_size

        # mod_pad
        if self.dec_pad > 0:
            mem = F.pad(mem, (0, self.dec_pad))
            tgt = F.pad(tgt, (0, self.dec_pad))

        B, C, T = tgt.shape
        tgt = tgt.permute(0, 2, 1)  # [B, T, C]
        mem = mem.permute(0, 2, 1)

        new_dec_buf_slots = []

        # Prepend mem context
        mem = torch.cat((dec_buf[:, 0, :, :], mem), dim=1)
        new_dec_buf_slots.append(mem[:, -self.ctx_len:, :])

        # Unfold mem
        mem_ctx = self._causal_unfold_static(mem)
        if self.use_pos_enc:
            mem_ctx = mem_ctx + self.pos_enc(mem_ctx)

        # Decoder layers
        for i, tf_dec_layer in enumerate(self.tf_dec_layers):
            tgt = torch.cat((dec_buf[:, i + 1, :, :], tgt), dim=1)
            new_dec_buf_slots.append(tgt[:, -self.ctx_len:, :])

            tgt_ctx = self._causal_unfold_static(tgt)
            if self.use_pos_enc and i == 0:
                tgt_ctx = tgt_ctx + self.pos_enc(tgt_ctx)

            # Single pass (all windows fit in one K-iteration)
            tgt_out, _, _ = tf_dec_layer(tgt_ctx, mem_ctx, chunk_size)
            tgt = tgt_out.reshape(B, T, C)

        tgt = tgt.permute(0, 2, 1)
        if self.dec_mod != 0:
            tgt = tgt[:, :, :-self.dec_pad]

        new_dec_buf = torch.stack(new_dec_buf_slots, dim=1)
        return tgt, new_dec_buf

    def forward(
        self,
        audio: Tensor,
        enc_buf: Tensor,
        dec_buf: Tensor,
        out_buf: Tensor,
        convnet_pre_ctx: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = audio

        # Step 1: CachedConvNet prenet
        if self.has_convnet_pre:
            convnet_out, new_convnet_pre_ctx = self._convnet_pre_forward(
                x, convnet_pre_ctx)
            if self.convnet_skip_connection == 'add':
                x = x + convnet_out
            elif self.convnet_skip_connection == 'multiply':
                x = x * convnet_out
            else:
                x = convnet_out
        else:
            new_convnet_pre_ctx = convnet_pre_ctx

        # Step 2: in_conv
        x = self.in_conv(x)

        # Step 3: Label embedding (pre-computed constant)
        l = self.label_const  # [1, enc_dim]

        # Step 4: Encoder
        e, new_enc_buf = self._encoder_forward(x, enc_buf)

        # Step 5: MaskNet logic
        # In MaskNet.forward: l = l.unsqueeze(2) * e (label * encoder output)
        l_integrated = l.unsqueeze(2) * e

        if self.has_proj:
            e_proj = self.proj_e2d_e(e)
            m_proj = self.proj_e2d_l(l_integrated)
            m, new_dec_buf = self._decoder_forward(m_proj, e_proj, dec_buf)
        else:
            m, new_dec_buf = self._decoder_forward(l_integrated, e, dec_buf)

        if self.has_proj:
            m = self.proj_d2e(m)

        if self.has_skip:
            m = l_integrated + m

        # Step 6: Apply mask to in_conv output
        x = x * m

        # Step 7: Concatenate with out_buf and decode
        x = torch.cat((out_buf, x), dim=-1)
        new_out_buf = x[:, :, -self.out_buf_len:]
        output = self.out_conv(x)

        return output, new_enc_buf, new_dec_buf, new_out_buf, new_convnet_pre_ctx
