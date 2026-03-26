from __future__ import annotations
import torch
from captum.attr import IntegratedGradients

def ig_attribution(model, x: torch.Tensor, target: int, baseline: torch.Tensor | None = None, steps: int = 64):
    """Compute Integrated Gradients attribution for a single batch x.

    x: (B, C, T)
    returns: attributions with same shape
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
    ig = IntegratedGradients(model)
    attr = ig.attribute(x, baselines=baseline, target=target, n_steps=steps)
    return attr
