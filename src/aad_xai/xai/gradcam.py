"""GradCAM attribution for PyTorch models (primary path).

Uses Captum's ``LayerGradCam`` for convolutional layers and a manual
gradient-weighted activation map for non-conv layers.

Designed for the VLAAI AADDecisionWrapper but works with any nn.Module
that produces classification logits.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from captum.attr import LayerGradCam


def gradcam_attribution(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    layer: nn.Module,
    relu_attributions: bool = True,
) -> torch.Tensor:
    """Compute GradCAM attribution for a target layer.

    Parameters
    ----------
    model : nn.Module
        Classifier (e.g., AADDecisionEEGOnly wrapping VLAAI).
    x : Tensor (B, ...)
        Input batch.
    target_class : int
        Class index to explain (1 = attended, 0 = unattended).
    layer : nn.Module
        The convolutional layer to target (e.g., ``model.decoder.extractor.blocks[4][0]``).
    relu_attributions : bool
        Apply ReLU to the attribution map (standard GradCAM).

    Returns
    -------
    Tensor — GradCAM heatmap. Shape depends on the target layer's spatial dims.
        For 1D conv layers in VLAAI: ``(B, 1, T)`` or ``(B, T)`` after squeeze.
    """
    model.eval()
    gc = LayerGradCam(model, layer)
    attr = gc.attribute(x, target=target_class, relu_attributions=relu_attributions)
    return attr


def gradcam_all_blocks(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
    relu_attributions: bool = True,
) -> dict[str, torch.Tensor]:
    """Run GradCAM on all Conv1d layers in a VLAAI-based model.

    Automatically discovers Conv1d modules within the model and returns
    a dict mapping ``layer_path → gradcam_heatmap``.

    Parameters
    ----------
    model : nn.Module
        Should be an ``AADDecisionEEGOnly`` wrapping ``VLAAIPyTorch``,
        or any model with Conv1d layers.
    x : Tensor
    target_class : int

    Returns
    -------
    dict[str, Tensor]
    """
    model.eval()
    results: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            try:
                gc = LayerGradCam(model, module)
                attr = gc.attribute(x, target=target_class, relu_attributions=relu_attributions)
                results[name] = attr
            except Exception:
                continue

    return results


def gradcam_temporal_heatmap(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    layer: nn.Module,
) -> np.ndarray:
    """Compute a time-resolved GradCAM heatmap normalized to [0, 1].

    Parameters
    ----------
    model, x, target_class, layer : see ``gradcam_attribution``

    Returns
    -------
    np.ndarray (B, T) — normalized temporal heatmap.
    """
    attr = gradcam_attribution(model, x, target_class, layer, relu_attributions=True)
    # attr may be (B, 1, T) or (B, C, T) — average over channels
    if attr.ndim == 3:
        attr = attr.mean(dim=1)  # (B, T)
    arr = attr.detach().cpu().numpy()
    # Normalize per sample to [0, 1]
    mins = arr.min(axis=-1, keepdims=True)
    maxs = arr.max(axis=-1, keepdims=True)
    denom = maxs - mins + 1e-8
    return (arr - mins) / denom
