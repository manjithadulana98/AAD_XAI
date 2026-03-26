from __future__ import annotations
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def linear_probe_accuracy(
    activations: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    test_size: float = 0.2,
) -> float:
    """Train a linear probe on frozen activations and report *held-out* accuracy.

    Parameters
    ----------
    activations : (N, D)
    labels : (N,)
    seed : int
    test_size : float
        Fraction of data reserved for evaluation (avoids resubstitution bias).

    Returns
    -------
    float — probe accuracy on the held-out portion.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        activations, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(X_te)))


def probe_all_layers(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device | None = None,
    max_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Hook every named child of *model* and report linear-probe accuracy.

    Parameters
    ----------
    model : nn.Module
        Trained model (will be set to eval mode).
    dataset : Dataset
        Yields ``(x, label)`` pairs.
    device : torch.device | None
    max_samples : int
        Cap the number of samples to avoid memory issues.
    seed : int

    Returns
    -------
    dict mapping ``layer_name → probe_accuracy``.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # --- Collect a batch of inputs and labels ---
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=min(max_samples, len(dataset)), shuffle=False)
    xb, yb = next(iter(dl))
    xb = xb.to(device)
    labels = np.asarray(yb)

    # --- Register hooks on all named children ---
    activations: dict[str, np.ndarray] = {}
    hooks = []

    def _make_hook(name: str):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                act = output.detach().cpu().numpy()
                # Flatten spatial / temporal dims → (B, D)
                if act.ndim > 2:
                    act = act.reshape(act.shape[0], -1)
                activations[name] = act
        return hook_fn

    for name, child in model.named_children():
        hooks.append(child.register_forward_hook(_make_hook(name)))

    # Forward pass to populate activations
    with torch.no_grad():
        model(xb)

    for h in hooks:
        h.remove()

    # --- Run probes ---
    results: dict[str, float] = {}
    for name, act in activations.items():
        try:
            results[name] = linear_probe_accuracy(act, labels, seed=seed)
        except ValueError:
            # Can happen if a layer produces a constant output
            results[name] = float("nan")

    return results

