from __future__ import annotations
import copy
from typing import Callable
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def randomize_parameters(model: nn.Module) -> nn.Module:
    """Return a deep copy of *model* with all parameters re-initialised."""
    m = copy.deepcopy(model)
    for module in m.modules():
        if hasattr(module, "reset_parameters"):
            try:
                module.reset_parameters()  # type: ignore
            except Exception:
                pass
    return m


def cascading_randomization(
    model: nn.Module,
    attr_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Cascading parameter randomisation sanity check (Adebayo et al., 2018).

    Starting from the *top* (last) layer, progressively re-initialise one
    layer at a time and record the attribution map at each step.

    A faithful attribution method should produce maps that *diverge* from the
    original as more layers are randomised.  If the maps stay similar, the
    method may be insensitive to the learned weights.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    attr_fn : callable
        ``attr_fn(model, x) -> attribution_tensor``.  This is the
        attribution method under test (e.g., a wrapper around IG).
    x : torch.Tensor
        A single input batch.

    Returns
    -------
    dict mapping ``layer_name → attribution_ndarray`` after randomising
    that layer (and all layers above it).
    """
    model.eval()
    m = copy.deepcopy(model)

    # Collect named children in *reverse* (top-down) order
    named_children = list(m.named_children())
    named_children.reverse()

    results: dict[str, np.ndarray] = {}
    # First entry: original (trained) model
    with torch.no_grad():
        attr_orig = attr_fn(m, x)
    results["__original__"] = attr_orig.detach().cpu().numpy()

    for name, child in named_children:
        # Re-initialise this layer's parameters
        for submod in child.modules():
            if hasattr(submod, "reset_parameters"):
                try:
                    submod.reset_parameters()
                except Exception:
                    pass
        with torch.no_grad():
            attr = attr_fn(m, x)
        results[name] = attr.detach().cpu().numpy()

    return results

