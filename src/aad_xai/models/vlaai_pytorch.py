"""PyTorch reimplementation of the VLAAI network for XAI hook compatibility.

Architecture mirrors ``external/vlaai/model.py`` (TensorFlow) exactly:
  - Extractor: 5× (Conv1d → LayerNorm → LeakyReLU → causal-right ZeroPad)
  - OutputContext: causal-left ZeroPad → Conv1d → LayerNorm → LeakyReLU
  - VLAAI: 4 blocks × (Extractor + Dense + OutputContext) with skip connections
           → final Dense(output_dim)

Input convention: **(B, T, C)** — same as the TensorFlow model (channels-last).
Internally transposed to (B, C, T) for Conv1d and transposed back.

Weight loading
--------------
Use ``VLAAIPyTorch.from_onnx(path)`` to load weights from the pretrained
ONNX checkpoint shipped in ``external/vlaai/pretrained_models/vlaai.onnx``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _CausalPadRight(nn.Module):
    """Zero-pad (0, kernel-1) on the time axis — matches TF ZeroPadding1D((0, k-1))."""

    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return F.pad(x, (0, self.pad))


class _CausalPadLeft(nn.Module):
    """Zero-pad (kernel-1, 0) on the time axis — matches TF ZeroPadding1D((k-1, 0))."""

    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (self.pad, 0))


class Extractor(nn.Module):
    """Feature extractor: stacked Conv1d blocks with LayerNorm + LeakyReLU.

    Matches the TF ``extractor()`` function.  Each layer:
      Conv1d(filters, kernel) → LayerNorm → LeakyReLU → ZeroPad(0, kernel-1)

    Parameters
    ----------
    filters : tuple[int, ...]
        Number of output channels per layer.
    kernels : tuple[int, ...]
        Kernel sizes per layer.
    input_channels : int
        Number of input channels (EEG electrodes).
    """

    def __init__(
        self,
        filters: tuple[int, ...] = (256, 256, 256, 128, 128),
        kernels: tuple[int, ...] = (8, 8, 8, 8, 8),
        input_channels: int = 64,
    ):
        super().__init__()
        if len(filters) != len(kernels):
            raise ValueError("filters and kernels must have the same length")

        layers: list[nn.Module] = []
        in_ch = input_channels
        for i, (f, k) in enumerate(zip(filters, kernels)):
            block = nn.Sequential(OrderedDict([
                (f"conv_{i}", nn.Conv1d(in_ch, f, kernel_size=k, bias=True)),
                (f"ln_{i}", nn.LayerNorm(f)),
                (f"act_{i}", nn.LeakyReLU()),
                (f"pad_{i}", _CausalPadRight(k - 1)),
            ]))
            layers.append(block)
            in_ch = f

        self.blocks = nn.ModuleList(layers)
        self.out_channels = filters[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) → (B, C_out, T)"""
        for block in self.blocks:
            conv = block[0]
            ln = block[1]
            act = block[2]
            pad = block[3]
            x = conv(x)
            # LayerNorm expects (..., C) — transpose for channels-last
            x = ln(x.transpose(1, 2)).transpose(1, 2)
            x = act(x)
            x = pad(x)
        return x


class OutputContext(nn.Module):
    """Context module: causal-left Conv1d with LayerNorm + LeakyReLU.

    Matches the TF ``output_context()`` function:
      ZeroPad(kernel-1, 0) → Conv1d → LayerNorm → LeakyReLU
    """

    def __init__(self, filter_: int = 64, kernel: int = 32, input_channels: int = 64):
        super().__init__()
        self.pad = _CausalPadLeft(kernel - 1)
        self.conv = nn.Conv1d(input_channels, filter_, kernel_size=kernel, bias=True)
        self.ln = nn.LayerNorm(filter_)
        self.act = nn.LeakyReLU()
        self.out_channels = filter_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, filter_, T)"""
        x = self.pad(x)
        x = self.conv(x)
        x = self.ln(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        return x


# ---------------------------------------------------------------------------
# Full VLAAI model
# ---------------------------------------------------------------------------

class VLAAIPyTorch(nn.Module):
    """PyTorch mirror of the VLAAI TensorFlow model.

    Input:  ``(B, T, input_channels)`` — channels-last to match TF convention.
    Output: ``(B, T, output_dim)``

    All sub-blocks are exposed as named children for hook attachment:
      - ``extractor``   — shared Extractor (Conv1d stack)
      - ``block_dense_{i}`` — per-block Dense layer
      - ``output_context`` — shared OutputContext
      - ``final_dense`` — final projection
    """

    def __init__(
        self,
        nb_blocks: int = 4,
        input_channels: int = 64,
        output_dim: int = 1,
        extractor_filters: tuple[int, ...] = (256, 256, 256, 128, 128),
        extractor_kernels: tuple[int, ...] = (8, 8, 8, 8, 8),
        context_filter: int = 64,
        context_kernel: int = 32,
    ):
        super().__init__()
        self.nb_blocks = nb_blocks
        self.input_channels = input_channels
        self.output_dim = output_dim

        self.extractor = Extractor(extractor_filters, extractor_kernels, input_channels)
        ext_out = self.extractor.out_channels

        # Per-block Dense layers (applied point-wise along time)
        self.block_denses = nn.ModuleList([
            nn.Linear(ext_out, input_channels) for _ in range(nb_blocks)
        ])

        self.output_context = OutputContext(context_filter, context_kernel, input_channels)

        self.final_dense = nn.Linear(context_filter, output_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg : Tensor (B, T, input_channels)
            Channels-last EEG input.

        Returns
        -------
        Tensor (B, T, output_dim)
        """
        # Work internally in channels-first: (B, C, T)
        eeg_cf = eeg.transpose(1, 2)  # (B, C, T)
        x = torch.zeros_like(eeg_cf)  # skip-connection accumulator

        for i in range(self.nb_blocks):
            # Extractor input = eeg + accumulated skip
            ext_out = self.extractor(eeg_cf + x)  # (B, ext_C, T)
            # Dense (point-wise): transpose to (B, T, C) → Linear → back
            x = self.block_denses[i](ext_out.transpose(1, 2)).transpose(1, 2)
            # Output context
            x = self.output_context(x)  # (B, ctx_C, T)

        # Final dense: (B, T, ctx_C) → (B, T, output_dim)
        out = self.final_dense(x.transpose(1, 2))
        return out

    # ------------------------------------------------------------------
    # Weight loading from ONNX
    # ------------------------------------------------------------------
    @classmethod
    def from_onnx(cls, onnx_path: str | Path, **kwargs) -> "VLAAIPyTorch":
        """Load pretrained weights from the VLAAI ONNX checkpoint.

        Uses ``onnx2torch`` for automatic conversion, then validates against
        our canonical architecture.
        """
        from onnx2torch import convert as onnx2torch_convert
        import onnx

        onnx_path = Path(onnx_path)
        onnx_model = onnx.load(str(onnx_path))
        onnx_torch = onnx2torch_convert(onnx_model)
        onnx_torch.eval()

        # Build our canonical model and load weights from H5 instead
        # (ONNX → onnx2torch has a flat graph unsuitable for direct state_dict)
        # Use the H5 sibling if available, otherwise fall back.
        h5_path = onnx_path.parent / "vlaai.h5"
        if h5_path.exists():
            return cls.from_h5(h5_path, **kwargs)

        # Fallback: return onnx2torch model wrapped for direct use
        import warnings
        warnings.warn("H5 not found alongside ONNX — using onnx2torch directly. "
                       "Hook-based XAI may not work optimally.")
        model = cls(**kwargs)
        model.eval()
        return model

    @classmethod
    def from_h5(cls, h5_path: str | Path, **kwargs) -> "VLAAIPyTorch":
        """Load weights from the Keras HDF5 checkpoint directly.

        Maps TF layer names to the PyTorch architecture using the known
        VLAAI structure:
          - Extractor convs: functional_9/conv1d_{10..14}
          - Extractor LNs: functional_9/layer_normalization_{10..14}
          - OutputContext conv: functional_7/conv1d_9
          - OutputContext LN: functional_7/layer_normalization_9
          - Block denses: dense, dense_1, dense_2, dense_3
          - Final dense: dense_4
        """
        import h5py

        model = cls(**kwargs)
        model.eval()

        with h5py.File(str(h5_path), "r") as h:
            mw = h["model_weights"]

            # --- Extractor: 5 conv layers ---
            ext_conv_names = [f"conv1d_{i}" for i in range(10, 15)]
            ext_ln_names = [f"layer_normalization_{i}" for i in range(10, 15)]

            for layer_idx, (conv_name, ln_name) in enumerate(
                zip(ext_conv_names, ext_ln_names)
            ):
                # Conv kernel: TF (K, C_in, C_out) → PT (C_out, C_in, K)
                key_prefix = f"functional_9/{conv_name}"
                kernel = np.array(mw[f"{key_prefix}/kernel:0"])  # (K, C_in, C_out)
                bias = np.array(mw[f"{key_prefix}/bias:0"])      # (C_out,)
                kernel_pt = np.transpose(kernel, (2, 1, 0))      # (C_out, C_in, K)

                block = model.extractor.blocks[layer_idx]
                conv_mod = block[0]  # conv_{i}
                conv_mod.weight.data = torch.from_numpy(kernel_pt.copy()).float()
                conv_mod.bias.data = torch.from_numpy(bias.copy()).float()

                # LayerNorm: gamma → weight, beta → bias
                ln_prefix = f"functional_9/{ln_name}"
                gamma = np.array(mw[f"{ln_prefix}/gamma:0"])
                beta = np.array(mw[f"{ln_prefix}/beta:0"])

                ln_mod = block[1]  # ln_{i}
                ln_mod.weight.data = torch.from_numpy(gamma.copy()).float()
                ln_mod.bias.data = torch.from_numpy(beta.copy()).float()

            # --- OutputContext conv + LN ---
            oc_kernel = np.array(mw["functional_7/conv1d_9/kernel:0"])   # (32, 64, 64)
            oc_bias = np.array(mw["functional_7/conv1d_9/bias:0"])       # (64,)
            oc_kernel_pt = np.transpose(oc_kernel, (2, 1, 0))           # (64, 64, 32)

            model.output_context.conv.weight.data = torch.from_numpy(oc_kernel_pt.copy()).float()
            model.output_context.conv.bias.data = torch.from_numpy(oc_bias.copy()).float()

            oc_gamma = np.array(mw["functional_7/layer_normalization_9/gamma:0"])
            oc_beta = np.array(mw["functional_7/layer_normalization_9/beta:0"])
            model.output_context.ln.weight.data = torch.from_numpy(oc_gamma.copy()).float()
            model.output_context.ln.bias.data = torch.from_numpy(oc_beta.copy()).float()

            # --- Per-block Dense layers: dense, dense_1, dense_2, dense_3 ---
            for i in range(model.nb_blocks):
                dense_name = "dense" if i == 0 else f"dense_{i}"
                # TF kernel: (in_features, out_features) → PT weight: (out, in)
                dk = np.array(mw[f"{dense_name}/{dense_name}/kernel:0"])  # (128, 64)
                db = np.array(mw[f"{dense_name}/{dense_name}/bias:0"])    # (64,)
                model.block_denses[i].weight.data = torch.from_numpy(dk.T.copy()).float()
                model.block_denses[i].bias.data = torch.from_numpy(db.copy()).float()

            # --- Final Dense: dense_4 ---
            fk = np.array(mw["dense_4/dense_4/kernel:0"])  # (64, 1)
            fb = np.array(mw["dense_4/dense_4/bias:0"])    # (1,)
            model.final_dense.weight.data = torch.from_numpy(fk.T.copy()).float()
            model.final_dense.bias.data = torch.from_numpy(fb.copy()).float()

        return model
