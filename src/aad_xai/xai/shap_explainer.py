"""SHAP explainer for VLAAI-based AAD models.

Provides DeepSHAP (PyTorch, fast) and KernelSHAP (model-agnostic, TF cross-check).

DeepSHAP works with the full ``(B, T, C)`` input dimensionality.
KernelSHAP uses channel-group summaries for scalability.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DeepSHAP (PyTorch)
# ---------------------------------------------------------------------------

def shap_deep_attribution(
    model: nn.Module,
    x: torch.Tensor,
    background: torch.Tensor,
    target_class: int = 1,
) -> np.ndarray:
    """Compute DeepSHAP values for each input feature.

    Parameters
    ----------
    model : nn.Module
        Classifier producing ``(B, n_classes)`` logits (e.g., AADDecisionEEGOnly).
    x : Tensor (B, T, C) — samples to explain
    background : Tensor (N_bg, T, C) — reference distribution (50–100 samples)
    target_class : int — class to explain (1 = attended)

    Returns
    -------
    np.ndarray (B, T, C) — Shapley values per input feature.
    """
    import shap

    model.eval()

    # shap.DeepExplainer expects numpy or torch tensors
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(x)

    # shap_values may be a list (one per class) or a single array
    if isinstance(shap_values, list):
        return shap_values[target_class]
    return shap_values


# ---------------------------------------------------------------------------
# KernelSHAP (model-agnostic)
# ---------------------------------------------------------------------------

# Default 10 EEG channel groups (ROIs) for KernelSHAP scalability
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


def shap_kernel_attribution(
    predict_fn: Callable,
    x: np.ndarray,
    background: np.ndarray,
    n_time_bins: int = 10,
    channel_groups: dict[str, list[int]] | None = None,
    n_samples: int = 500,
) -> dict[str, np.ndarray]:
    """Compute KernelSHAP with grouped EEG features for scalability.

    Superpixels = channel_groups × time_bins.  This reduces the feature
    space from ``64 × 320 = 20480`` to ``10 × 10 = 100``, making KernelSHAP
    tractable.

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(eeg_batch) → (B, n_classes)`` probabilities or logits.
        Works with both PyTorch and TF models.
    x : (B, T, C) — samples to explain
    background : (N_bg, T, C) — reference data
    n_time_bins : int — number of temporal bins
    channel_groups : dict mapping group_name → channel indices.
        Defaults to 10 EEG ROIs.
    n_samples : int — number of KernelSHAP samples

    Returns
    -------
    dict with:
        'shap_values': (B, n_groups × n_time_bins) — grouped Shapley values
        'feature_names': list[str] — names for each grouped feature
        'group_map': dict — channel group definitions
    """
    import shap

    if channel_groups is None:
        channel_groups = DEFAULT_CHANNEL_GROUPS

    B, T, C = x.shape
    n_groups = len(channel_groups)
    time_bin_size = T // n_time_bins

    group_names = list(channel_groups.keys())
    feature_names = [
        f"{g}_t{t}" for g in group_names for t in range(n_time_bins)
    ]
    n_features = n_groups * n_time_bins

    def _summarize(data: np.ndarray) -> np.ndarray:
        """Reduce (B, T, C) → (B, n_features) by averaging within groups."""
        B_ = data.shape[0]
        out = np.zeros((B_, n_features), dtype=np.float32)
        for gi, (gname, channels) in enumerate(channel_groups.items()):
            for ti in range(n_time_bins):
                t_start = ti * time_bin_size
                t_end = min(t_start + time_bin_size, data.shape[1])
                feat_idx = gi * n_time_bins + ti
                out[:, feat_idx] = data[:, t_start:t_end, channels].mean(axis=(1, 2))
        return out

    def _reconstruct_and_predict(summary: np.ndarray) -> np.ndarray:
        """Map grouped features back to full EEG and predict."""
        B_ = summary.shape[0]
        # Start from the background mean as baseline
        bg_mean = background.mean(axis=0, keepdims=True)
        full = np.tile(bg_mean, (B_, 1, 1)).astype(np.float32)

        for gi, (gname, channels) in enumerate(channel_groups.items()):
            for ti in range(n_time_bins):
                t_start = ti * time_bin_size
                t_end = min(t_start + time_bin_size, full.shape[1])
                feat_idx = gi * n_time_bins + ti
                # Scale the region by the summary value ratio
                scale = summary[:, feat_idx:feat_idx+1, np.newaxis, np.newaxis]
                # Replace with original data scaled
                pass

        # Simpler approach: use masking
        # For each sample, use the summary to determine which groups are "on"
        return predict_fn(full)

    # Use summarized features directly
    x_summary = _summarize(x)
    bg_summary = _summarize(background)

    def _mask_predict(summary_samples: np.ndarray) -> np.ndarray:
        """Predict from masked full-resolution EEG."""
        B_ = summary_samples.shape[0]
        bg_mean = background.mean(axis=0)  # (T, C)

        # Vectorized: start from tiled background mean
        batch = np.tile(bg_mean[np.newaxis], (B_, 1, 1)).astype(np.float32)

        # Build a boolean mask of which (sample, superpixel) should use original
        bg_mean_summary = bg_summary.mean(axis=0)
        # Replace region with original x[0] when summary differs from background
        for sp_idx in range(n_features):
            gi = sp_idx // n_time_bins
            ti = sp_idx % n_time_bins
            gname = group_names[gi]
            channels = channel_groups[gname]
            t_start = ti * time_bin_size
            t_end = min(t_start + time_bin_size, batch.shape[1])
            ref_val = bg_mean_summary[sp_idx]

            # Find which samples in the batch differ from background
            mask = np.abs(summary_samples[:, sp_idx] - ref_val) > 1e-8
            if mask.any():
                # Use original x[0] for those samples in this region
                for ch in channels:
                    batch[mask, t_start:t_end, ch] = x[0, t_start:t_end, ch]

        return predict_fn(batch)

    explainer = shap.KernelExplainer(_mask_predict, bg_summary[:50])
    sv = explainer.shap_values(x_summary, nsamples=n_samples)

    # sv is list of arrays (one per class) or single array
    if isinstance(sv, list):
        sv = np.array(sv)  # (n_classes, B, n_features)

    return {
        "shap_values": sv,
        "feature_names": feature_names,
        "group_map": channel_groups,
    }
