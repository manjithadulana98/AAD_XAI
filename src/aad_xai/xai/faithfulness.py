from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def deletion_curve(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    importance: torch.Tensor,
    steps: int = 20,
) -> list[float]:
    """Deletion test: progressively mask top-importance features → track confidence.

    At each step, the top-``frac`` most important features (by absolute value)
    are set to zero and the model's softmax confidence on the true class is
    recorded.  A faithful explanation should cause a *steep* drop.

    Parameters
    ----------
    model : nn.Module
    x : Tensor (B, C, T)
    y : Tensor (B,) — true labels
    importance : Tensor — same shape as x
    steps : int

    Returns
    -------
    confs : list[float]
        Confidence values from 0 % masked → 100 % masked.
    """
    model.eval()
    B, C, T = x.shape
    imp = importance.abs().detach().cpu().numpy().reshape(B, -1)
    x0 = x.detach().clone()

    order = np.argsort(-imp, axis=1)

    confs: list[float] = []
    for k in range(steps + 1):
        frac = k / steps
        n_mask = int(round(frac * C * T))
        xk = x0.clone()
        for i in range(B):
            flat = xk[i].view(-1)
            idx = order[i, :n_mask]
            flat[idx] = 0.0
        with torch.no_grad():
            logits = model(xk)
        probs = F.softmax(logits, dim=-1)
        conf = probs.gather(1, y.view(-1, 1)).mean().item()
        confs.append(conf)
    return confs


def insertion_curve(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    importance: torch.Tensor,
    steps: int = 20,
) -> list[float]:
    """Insertion test: progressively reveal top-importance features → track confidence.

    Starts from a zero baseline and inserts the most important features first.
    A faithful explanation should cause a *steep rise*.

    Parameters
    ----------
    model : nn.Module
    x : Tensor (B, C, T)
    y : Tensor (B,) — true labels
    importance : Tensor — same shape as x
    steps : int

    Returns
    -------
    confs : list[float]
        Confidence values from 0 % revealed → 100 % revealed.
    """
    model.eval()
    B, C, T = x.shape
    imp = importance.abs().detach().cpu().numpy().reshape(B, -1)
    x0 = x.detach().clone()

    order = np.argsort(-imp, axis=1)  # most important first

    confs: list[float] = []
    for k in range(steps + 1):
        frac = k / steps
        n_reveal = int(round(frac * C * T))
        xk = torch.zeros_like(x0)
        for i in range(B):
            flat_src = x0[i].view(-1)
            flat_dst = xk[i].view(-1)
            idx = order[i, :n_reveal]
            flat_dst[idx] = flat_src[idx]
        with torch.no_grad():
            logits = model(xk)
        probs = F.softmax(logits, dim=-1)
        conf = probs.gather(1, y.view(-1, 1)).mean().item()
        confs.append(conf)
    return confs

