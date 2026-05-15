"""GradCAM via TensorFlow GradientTape — cross-check for PyTorch GradCAM.

Works directly with the TF Keras VLAAI model (no PyTorch conversion needed).
Used to validate that the PyTorch GradCAM produces consistent results.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def gradcam_tf(
    model,
    eeg: np.ndarray,
    target_layer_name: str,
    output_index: int | None = None,
) -> np.ndarray:
    """Compute GradCAM using TensorFlow GradientTape.

    Parameters
    ----------
    model : tf.keras.Model
        The VLAAI Keras model (or any Keras model).
    eeg : (B, T, C) — input EEG
    target_layer_name : str
        Name of the convolutional layer to target.
    output_index : int, optional
        If the model output has multiple features, index to explain.
        For VLAAI (output_dim=1), leave as None → uses full output mean.

    Returns
    -------
    np.ndarray (B, T) — GradCAM temporal heatmap normalized to [0, 1].
    """
    import tensorflow as tf

    # Build sub-model that outputs both the target layer and final output
    layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[layer.output, model.output],
    )

    eeg_tensor = tf.constant(eeg, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(eeg_tensor)
        conv_output, predictions = grad_model(eeg_tensor, training=False)

        if output_index is not None:
            target_output = predictions[:, :, output_index]
        else:
            target_output = predictions

        loss = tf.reduce_mean(target_output)

    # Gradients of the output w.r.t. the conv layer output
    grads = tape.gradient(loss, conv_output)

    if grads is None:
        return np.zeros((eeg.shape[0], eeg.shape[1]))

    # Global average pooling of gradients over the time axis
    # grads shape: (B, T, filters)
    weights = tf.reduce_mean(grads, axis=1, keepdims=True)  # (B, 1, filters)

    # Weighted combination of conv outputs
    cam = tf.reduce_sum(weights * conv_output, axis=-1)  # (B, T)

    # ReLU
    cam = tf.nn.relu(cam)

    # Normalize to [0, 1] per sample
    cam_np = cam.numpy()
    mins = cam_np.min(axis=-1, keepdims=True)
    maxs = cam_np.max(axis=-1, keepdims=True)
    denom = maxs - mins + 1e-8
    return (cam_np - mins) / denom


def gradcam_tf_all_convs(
    model,
    eeg: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run GradCAM on all Conv1D layers in a TF Keras model.

    Parameters
    ----------
    model : tf.keras.Model
    eeg : (B, T, C)

    Returns
    -------
    dict[str, np.ndarray] — mapping layer_name → (B, T) heatmap.
    """
    import tensorflow as tf

    results: dict[str, np.ndarray] = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            try:
                heatmap = gradcam_tf(model, eeg, layer.name)
                results[layer.name] = heatmap
            except Exception:
                continue
    return results


def compare_gradcam_pytorch_tf(
    heatmap_pt: np.ndarray,
    heatmap_tf: np.ndarray,
) -> dict[str, float]:
    """Compare PyTorch and TF GradCAM outputs.

    Parameters
    ----------
    heatmap_pt, heatmap_tf : (B, T) arrays

    Returns
    -------
    dict with 'cosine_sim', 'rank_corr', 'mse' averaged over the batch.
    """
    from scipy.stats import spearmanr

    B = heatmap_pt.shape[0]
    cosines, ranks, mses = [], [], []

    for i in range(B):
        a = heatmap_pt[i].ravel()
        b = heatmap_tf[i].ravel()

        # Cosine similarity
        dot = np.dot(a, b)
        norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        cosines.append(dot / norm)

        # Spearman rank correlation
        r, _ = spearmanr(a, b)
        ranks.append(r if not np.isnan(r) else 0.0)

        # MSE
        mses.append(float(np.mean((a - b) ** 2)))

    return {
        "cosine_sim": float(np.mean(cosines)),
        "rank_corr": float(np.mean(ranks)),
        "mse": float(np.mean(mses)),
    }
