"""EEG-adapted LIME explainer.

Superpixels are defined as EEG channel groups × coarse time bins.
Perturbation = zeroing out superpixels.  This is fully model-agnostic:
works with both PyTorch and TF predict functions.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# Default 10 EEG ROIs (same as SHAP for consistency)
DEFAULT_CHANNEL_GROUPS = {
    "frontal_left": list(range(0, 6)),
    "frontal_right": list(range(6, 12)),
    "frontal_central": list(range(12, 18)),
    "central_left": list(range(18, 24)),
    "central_right": list(range(24, 30)),
    "temporal_left": list(range(30, 36)),
    "temporal_right": list(range(36, 42)),
    "parietal_left": list(range(42, 48)),
    "parietal_right": list(range(48, 54)),
    "occipital": list(range(54, 64)),
}


def lime_attribution(
    predict_fn: Callable,
    x: np.ndarray,
    n_time_bins: int = 10,
    channel_groups: dict[str, list[int]] | None = None,
    n_samples: int = 1000,
    target_class: int = 1,
) -> dict[str, object]:
    """Compute LIME attributions on EEG data.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(eeg_batch) → (B, n_classes)`` — class probabilities.
        Must handle ``(B, T, C)`` numpy input.
    x : (T, C) — single EEG sample to explain
    n_time_bins : int — number of temporal segments
    channel_groups : dict mapping group_name → channel indices
    n_samples : int — number of perturbed samples
    target_class : int — class to explain

    Returns
    -------
    dict with:
        'importance': (n_superpixels,) — LIME weights
        'feature_names': list[str]
        'heatmap_channels': (C,) — importance aggregated per channel
        'heatmap_time': (T,) — importance aggregated per time step
        'intercept': float
        'score': float — local model R²
    """
    from lime.lime_tabular import LimeTabularExplainer

    if channel_groups is None:
        channel_groups = DEFAULT_CHANNEL_GROUPS

    T, C = x.shape
    n_groups = len(channel_groups)
    time_bin_size = T // n_time_bins
    n_superpixels = n_groups * n_time_bins

    group_names = list(channel_groups.keys())
    feature_names = [
        f"{g}_t{t}" for g in group_names for t in range(n_time_bins)
    ]

    # Build superpixel-level representation of x (binary: all 1s = original)
    x_sp = np.ones(n_superpixels, dtype=np.float32)

    # Map from superpixel index → (channel_indices, time_slice)
    sp_map: list[tuple[list[int], slice]] = []
    for gi, (gname, channels) in enumerate(channel_groups.items()):
        for ti in range(n_time_bins):
            t_start = ti * time_bin_size
            t_end = min(t_start + time_bin_size, T)
            sp_map.append((channels, slice(t_start, t_end)))

    def _perturbed_predict(sp_activations: np.ndarray) -> np.ndarray:
        """Map superpixel activations → full EEG → predict."""
        B_ = sp_activations.shape[0]
        batch = np.tile(x[np.newaxis], (B_, 1, 1)).copy()

        for b in range(B_):
            for sp_idx in range(n_superpixels):
                if sp_activations[b, sp_idx] == 0:
                    channels, t_slice = sp_map[sp_idx]
                    batch[b, t_slice, channels] = 0.0

        return predict_fn(batch)

    # Create a binary training dataset for LIME to sample from
    rng = np.random.RandomState(0)
    training_data = rng.randint(0, 2, size=(100, n_superpixels)).astype(np.float32)

    explainer = LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        mode="classification",
        discretize_continuous=False,
        categorical_features=list(range(n_superpixels)),
    )

    explanation = explainer.explain_instance(
        x_sp,
        _perturbed_predict,
        num_features=n_superpixels,
        num_samples=n_samples,
        labels=(target_class,),
    )

    # Extract weights for the target class
    weights_map = dict(explanation.as_list(label=target_class))
    importance = np.array([
        weights_map.get(fname, 0.0) for fname in feature_names
    ])

    # Aggregate into channel-level and time-level heatmaps
    heatmap_channels = np.zeros(C, dtype=np.float32)
    heatmap_time = np.zeros(T, dtype=np.float32)

    for sp_idx in range(n_superpixels):
        channels, t_slice = sp_map[sp_idx]
        w = importance[sp_idx]
        for ch in channels:
            heatmap_channels[ch] += w
        t_indices = range(*t_slice.indices(T))
        for ti in t_indices:
            heatmap_time[ti] += w

    return {
        "importance": importance,
        "feature_names": feature_names,
        "heatmap_channels": heatmap_channels,
        "heatmap_time": heatmap_time,
        "intercept": explanation.intercept.get(target_class, 0.0),
        "score": explanation.score,
    }


def lime_batch_attribution(
    predict_fn: Callable,
    x_batch: np.ndarray,
    n_time_bins: int = 10,
    channel_groups: dict[str, list[int]] | None = None,
    n_samples: int = 500,
    target_class: int = 1,
) -> list[dict]:
    """Run LIME on each sample in a batch.

    Parameters
    ----------
    predict_fn : callable — ``f(batch) → (B, n_classes)``
    x_batch : (B, T, C)
    n_time_bins, channel_groups, n_samples, target_class : see ``lime_attribution``

    Returns
    -------
    list of dicts (one per sample), same structure as ``lime_attribution`` output.
    """
    B = x_batch.shape[0]
    results = []
    for i in range(B):
        r = lime_attribution(
            predict_fn, x_batch[i], n_time_bins, channel_groups,
            n_samples, target_class,
        )
        results.append(r)
    return results
