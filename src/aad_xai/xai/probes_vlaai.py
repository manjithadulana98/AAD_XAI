"""Comprehensive probing for VLAAI: attention decoding, auditory, linguistic.

Extends the base ``linear_probe_accuracy`` from ``probes.py`` with:
  - VLAAI-specific layer extraction (PyTorch hooks + TF sub-models)
  - Attention decoding probes (attended vs unattended)
  - Auditory feature probes (envelope amplitude, onset, spectral centroid)
  - Linguistic feature probes (stub — requires aligned annotations)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .probes import linear_probe_accuracy


# ---------------------------------------------------------------------------
# PyTorch: recursive layer activation extraction
# ---------------------------------------------------------------------------

def extract_all_activations_pt(
    model: nn.Module,
    x: torch.Tensor,
    recursive: bool = True,
) -> dict[str, np.ndarray]:
    """Extract activations from all (or immediate) children via hooks.

    Parameters
    ----------
    model : nn.Module
    x : Tensor — input batch
    recursive : bool
        If True, hook into every named *module* (recursive). 
        If False, only immediate ``named_children()``.

    Returns
    -------
    dict mapping ``module_path → activation (N, D)``.
    """
    model.eval()
    activations: dict[str, np.ndarray] = {}
    hooks = []

    def _make_hook(name: str):
        def hook_fn(module, inp, output):
            if isinstance(output, torch.Tensor):
                act = output.detach().cpu().numpy()
                if act.ndim > 2:
                    act = act.reshape(act.shape[0], -1)
                activations[name] = act
        return hook_fn

    iterator = model.named_modules() if recursive else model.named_children()
    for name, child in iterator:
        if name == "":
            continue
        hooks.append(child.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return activations


# ---------------------------------------------------------------------------
# TF: layer activation extraction
# ---------------------------------------------------------------------------

def extract_all_activations_tf(
    model,
    eeg: np.ndarray,
    layer_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract intermediate activations from a TF Keras VLAAI model.

    Parameters
    ----------
    model : tf.keras.Model
    eeg : (B, T, 64)
    layer_names : specific layers to extract. None = all.

    Returns
    -------
    dict mapping ``layer_name → activation (N, D)``.
    """
    import tensorflow as tf

    if layer_names is None:
        layer_names = [
            l.name for l in model.layers
            if hasattr(l, "output") and l.output is not None and l.name != model.layers[0].name
        ]

    activations: dict[str, np.ndarray] = {}
    for name in layer_names:
        try:
            layer = model.get_layer(name)
            sub = tf.keras.Model(inputs=model.input, outputs=layer.output)
            act = sub.predict(eeg, verbose=0)
            if act.ndim > 2:
                act = act.reshape(act.shape[0], -1)
            activations[name] = act
        except Exception:
            continue

    return activations


# ---------------------------------------------------------------------------
# Attention decoding probes
# ---------------------------------------------------------------------------

def attention_decoding_probes(
    activations: dict[str, np.ndarray],
    labels: np.ndarray,
    seed: int = 0,
    test_size: float = 0.2,
) -> dict[str, float]:
    """Train linear probes on each layer to decode attended speaker.

    Parameters
    ----------
    activations : dict from ``extract_all_activations_pt`` or ``_tf``
    labels : (N,) binary labels (0 = unattended, 1 = attended)
    seed, test_size : passed to ``linear_probe_accuracy``

    Returns
    -------
    dict mapping ``layer_name → probe_accuracy``.
    """
    results: dict[str, float] = {}
    for name, act in activations.items():
        if act.shape[0] != len(labels):
            continue
        try:
            results[name] = linear_probe_accuracy(act, labels, seed=seed, test_size=test_size)
        except ValueError:
            results[name] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Auditory feature probes
# ---------------------------------------------------------------------------

def _discretize_envelope(envelope: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Discretize mean envelope amplitude into bins."""
    means = envelope.mean(axis=1).ravel()
    bins = np.quantile(means, np.linspace(0, 1, n_bins + 1))
    return np.digitize(means, bins[1:-1])


def _detect_onsets(envelope: np.ndarray, threshold_factor: float = 1.5) -> np.ndarray:
    """Binary onset label: 1 if window contains a sharp envelope rise."""
    diff = np.diff(envelope.mean(axis=-1), axis=1)
    max_diff = diff.max(axis=1)
    threshold = np.median(max_diff) * threshold_factor
    return (max_diff > threshold).astype(int)


def auditory_feature_probes(
    activations: dict[str, np.ndarray],
    envelopes: np.ndarray,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Probe each layer for auditory features: amplitude bins, onset detection.

    Parameters
    ----------
    activations : dict from ``extract_all_activations_*``
    envelopes : (N, T, 1) — speech envelopes matching the activation samples
    seed : int

    Returns
    -------
    dict mapping ``layer_name → {'amplitude': acc, 'onset': acc}``.
    """
    n = envelopes.shape[0]
    amp_labels = _discretize_envelope(envelopes)
    onset_labels = _detect_onsets(envelopes)

    results: dict[str, dict[str, float]] = {}
    for name, act in activations.items():
        if act.shape[0] != n:
            continue
        res: dict[str, float] = {}
        try:
            res["amplitude"] = linear_probe_accuracy(act, amp_labels, seed=seed)
        except ValueError:
            res["amplitude"] = float("nan")
        try:
            res["onset"] = linear_probe_accuracy(act, onset_labels, seed=seed)
        except ValueError:
            res["onset"] = float("nan")
        results[name] = res
    return results


# ---------------------------------------------------------------------------
# Linguistic feature probes (stub)
# ---------------------------------------------------------------------------

def linguistic_feature_probes(
    activations: dict[str, np.ndarray],
    labels: np.ndarray,
    label_name: str = "phoneme",
    seed: int = 0,
) -> dict[str, float]:
    """Probe each layer for a linguistic feature.

    Parameters
    ----------
    activations : dict from ``extract_all_activations_*``
    labels : (N,) — linguistic labels aligned to each window
        (e.g., phoneme identity, word boundary indicator)
    label_name : str — descriptive name for the label type
    seed : int

    Returns
    -------
    dict mapping ``layer_name → probe_accuracy``.

    Notes
    -----
    This is currently a stub. The DTU evaluation dataset does not include
    aligned linguistic annotations. When such annotations become available,
    pass them as ``labels`` and this function will train probes identically
    to the attention decoding probes.
    """
    results: dict[str, float] = {}
    for name, act in activations.items():
        if act.shape[0] != len(labels):
            continue
        try:
            results[name] = linear_probe_accuracy(act, labels, seed=seed)
        except ValueError:
            results[name] = float("nan")
    return results


# ---------------------------------------------------------------------------
# Unified probe runner (convenience)
# ---------------------------------------------------------------------------

def run_all_probes_pt(
    model: nn.Module,
    eeg: torch.Tensor,
    labels_attention: np.ndarray,
    envelopes: np.ndarray,
    labels_linguistic: np.ndarray | None = None,
    recursive: bool = True,
    seed: int = 0,
) -> dict[str, dict]:
    """Run all probe types on a PyTorch model in one call.

    Parameters
    ----------
    model : nn.Module (e.g., VLAAIPyTorch or AADDecisionEEGOnly)
    eeg : (B, T, C) — tensor
    labels_attention : (B,) — binary attention labels
    envelopes : (B, T, 1) — attended envelopes
    labels_linguistic : (B,) or None — linguistic labels (optional)
    recursive : bool — recurse into nested modules
    seed : int

    Returns
    -------
    dict with keys 'attention', 'auditory', 'linguistic' (if available),
    each mapping layer_name → probe accuracy.
    """
    activations = extract_all_activations_pt(model, eeg, recursive=recursive)

    results: dict[str, dict] = {}
    results["attention"] = attention_decoding_probes(activations, labels_attention, seed=seed)
    results["auditory"] = auditory_feature_probes(activations, envelopes, seed=seed)

    if labels_linguistic is not None:
        results["linguistic"] = linguistic_feature_probes(
            activations, labels_linguistic, seed=seed,
        )

    return results
