"""Probe result visualization utilities for VLAAI XAI.

Generates publication-quality plots showing probe accuracy across layers
for attention, auditory, and linguistic features.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_probe_accuracy_by_layer(
    probe_results: dict[str, float],
    title: str = "Linear Probe Accuracy by Layer",
    ylabel: str = "Probe Accuracy",
    chance_level: float = 0.5,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Bar chart of probe accuracy across layers.

    Parameters
    ----------
    probe_results : dict mapping layer_name → accuracy
    title : str
    ylabel : str
    chance_level : float — horizontal line for chance performance
    save_path : path to save PNG (or None to just return figure)
    figsize : figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(probe_results.keys())
    accs = [probe_results[n] for n in names]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(names))
    ax.bar(x, accs, color="steelblue", alpha=0.8)
    ax.axhline(y=chance_level, color="red", linestyle="--", label=f"Chance ({chance_level:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_probe_comparison(
    results: dict[str, dict[str, float]],
    title: str = "Probe Comparison Across Feature Types",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Overlay line plots comparing probe types (attention, auditory, linguistic).

    Parameters
    ----------
    results : dict mapping ``feature_type → {layer_name: accuracy}``.
        E.g., ``{'attention': {...}, 'auditory_amplitude': {...}}``.
    title : str
    save_path : path or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    all_layers = set()
    for feat_results in results.values():
        all_layers.update(feat_results.keys())
    layer_order = sorted(all_layers)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.cm.tab10.colors

    for i, (feat_name, feat_results) in enumerate(results.items()):
        accs = [feat_results.get(l, float("nan")) for l in layer_order]
        ax.plot(range(len(layer_order)), accs, marker="o", markersize=4,
                label=feat_name, color=colors[i % len(colors)], linewidth=1.5)

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels(layer_order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Probe Accuracy")
    ax.set_xlabel("Layer (depth →)")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def plot_auditory_probes(
    auditory_results: dict[str, dict[str, float]],
    title: str = "Auditory Feature Probes by Layer",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Grouped bar chart for auditory sub-probes (amplitude, onset).

    Parameters
    ----------
    auditory_results : dict mapping layer_name → {'amplitude': acc, 'onset': acc}
    title, save_path, figsize : see above

    Returns
    -------
    matplotlib.figure.Figure
    """
    layers = list(auditory_results.keys())
    amp_accs = [auditory_results[l].get("amplitude", float("nan")) for l in layers]
    onset_accs = [auditory_results[l].get("onset", float("nan")) for l in layers]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w / 2, amp_accs, w, label="Amplitude bins", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, onset_accs, w, label="Onset detection", color="coral", alpha=0.8)
    ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.4, label="Chance (4-class)")
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.4, label="Chance (binary)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Probe Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


def save_probe_results(
    results: dict,
    save_path: str | Path,
):
    """Save probe results as JSON."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python natives for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(save_path, "w") as f:
        json.dump(_convert(results), f, indent=2)
