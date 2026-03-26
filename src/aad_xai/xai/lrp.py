"""LRP stub — Layer-wise Relevance Propagation.

LRP implementations vary by library (zennit, innvestigate, captum LRP).
This module defines the intended interface; plug in your chosen backend.

Example (using zennit)::

    import zennit
    from zennit.composites import EpsilonPlusFlat
    from zennit.attribution import Gradient

    composite = EpsilonPlusFlat()
    with Gradient(model=model, composite=composite) as attributor:
        out, relevance = attributor(input_tensor, torch.eye(n_classes)[[target]])
    return relevance.detach().cpu().numpy()
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_lrp(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    *,
    rule: str = "epsilon",
) -> np.ndarray:
    """Compute Layer-wise Relevance Propagation for input *x*.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    x : torch.Tensor
        Input batch of shape ``(B, C, T)``.
    target : int
        Target class index for relevance computation.
    rule : str
        LRP rule variant (``"epsilon"``, ``"alpha_beta"``, …).
        Only used once a real backend is wired in.

    Returns
    -------
    relevance : np.ndarray
        Same shape as *x* — per-input-feature relevance scores.

    Raises
    ------
    NotImplementedError
        Always — this is a stub awaiting a concrete LRP backend.
    """
    raise NotImplementedError(
        "LRP is not implemented yet. "
        "Plug in zennit (recommended) or innvestigate. "
        "See the docstring of this module for a zennit example."
    )
