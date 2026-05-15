"""Thin wrapper around the TF Keras VLAAI model for native TensorFlow XAI.

Provides:
  - Easy loading from HDF5 / SavedModel with custom loss/metric objects
  - Block-level intermediate-activation extraction
  - ``predict_and_correlate()`` for attended-vs-unattended Pearson-r framing
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np


def _import_vlaai_tf():
    """Dynamically import the upstream TF VLAAI module."""
    root = Path(__file__).resolve().parents[3]
    vlaai_dir = root / "external" / "vlaai"
    if str(vlaai_dir) not in sys.path:
        sys.path.insert(0, str(vlaai_dir))
    from model import vlaai, pearson_loss, pearson_metric, pearson_tf
    return vlaai, pearson_loss, pearson_metric, pearson_tf


class VLAAITFWrapper:
    """Wrapper exposing the TF VLAAI model for XAI workflows.

    Parameters
    ----------
    model : tf.keras.Model
        A compiled / loaded VLAAI Keras model.
    """

    def __init__(self, model):
        self.model = model

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_h5(cls, h5_path: str | Path) -> "VLAAITFWrapper":
        """Load from HDF5 weights file."""
        import tensorflow as tf
        _, pearson_loss, pearson_metric, _ = _import_vlaai_tf()
        vlaai_fn, *_ = _import_vlaai_tf()

        model = vlaai_fn()
        model.load_weights(str(h5_path))
        return cls(model)

    @classmethod
    def from_saved_model(cls, saved_model_dir: str | Path) -> "VLAAITFWrapper":
        """Load from TF SavedModel directory."""
        import tensorflow as tf
        _, pearson_loss, pearson_metric, _ = _import_vlaai_tf()
        model = tf.keras.models.load_model(
            str(saved_model_dir),
            custom_objects={
                "pearson_loss": pearson_loss,
                "pearson_metric": pearson_metric,
            },
        )
        return cls(model)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, eeg: np.ndarray) -> np.ndarray:
        """Predict speech envelope from EEG.

        Parameters
        ----------
        eeg : (B, T, 64) or (T, 64)

        Returns
        -------
        np.ndarray — predicted envelope, same batch/time shape.
        """
        import tensorflow as tf
        if eeg.ndim == 2:
            eeg = eeg[np.newaxis]
        return self.model.predict(eeg, verbose=0)

    def predict_and_correlate(
        self,
        eeg: np.ndarray,
        env_attended: np.ndarray,
        env_unattended: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict envelope and compute Pearson-r with attended & unattended.

        Parameters
        ----------
        eeg : (B, T, 64)
        env_attended : (B, T, 1)
        env_unattended : (B, T, 1)

        Returns
        -------
        r_attended : (B,) Pearson-r per sample
        r_unattended : (B,) Pearson-r per sample
        """
        from scipy.stats import pearsonr
        pred = self.predict(eeg)  # (B, T, 1)

        B = pred.shape[0]
        r_att = np.zeros(B)
        r_unatt = np.zeros(B)
        for i in range(B):
            p = pred[i].ravel()
            r_att[i] = pearsonr(p, env_attended[i].ravel())[0]
            r_unatt[i] = pearsonr(p, env_unattended[i].ravel())[0]
        return r_att, r_unatt

    # ------------------------------------------------------------------
    # Intermediate activations (for TF-native probing)
    # ------------------------------------------------------------------

    def get_layer_outputs(
        self,
        eeg: np.ndarray,
        layer_names: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract intermediate activations for specified layers.

        Parameters
        ----------
        eeg : (B, T, 64)
        layer_names : list of layer names (from ``model.layers``).
            If None, extracts from all layers.

        Returns
        -------
        dict mapping ``layer_name → activation_ndarray``.
        """
        import tensorflow as tf

        if eeg.ndim == 2:
            eeg = eeg[np.newaxis]

        if layer_names is None:
            layer_names = [l.name for l in self.model.layers if l.output.shape is not None]

        outputs = {}
        for name in layer_names:
            try:
                layer = self.model.get_layer(name)
                sub_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=layer.output,
                )
                act = sub_model.predict(eeg, verbose=0)
                outputs[name] = act
            except Exception:
                continue
        return outputs

    # ------------------------------------------------------------------
    # GradientTape-based gradient computation
    # ------------------------------------------------------------------

    def compute_gradients(
        self,
        eeg: np.ndarray,
        target_layer_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute gradients of the output w.r.t. input (or a target layer).

        Parameters
        ----------
        eeg : (B, T, 64)
        target_layer_name : str, optional
            If provided, returns gradients w.r.t. that layer's output AND
            the layer's activation.

        Returns
        -------
        grads_input : (B, T, 64) — gradients w.r.t. input EEG
        grads_layer : (B, ...) or None — gradients w.r.t. target layer
        """
        import tensorflow as tf

        eeg_t = tf.constant(eeg, dtype=tf.float32)

        if target_layer_name is None:
            with tf.GradientTape() as tape:
                tape.watch(eeg_t)
                pred = self.model(eeg_t, training=False)
                loss = tf.reduce_mean(pred)
            grads = tape.gradient(loss, eeg_t)
            return grads.numpy(), None

        layer = self.model.get_layer(target_layer_name)
        sub_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[layer.output, self.model.output],
        )

        with tf.GradientTape() as tape:
            tape.watch(eeg_t)
            layer_out, pred = sub_model(eeg_t, training=False)
            loss = tf.reduce_mean(pred)

        grads_input, grads_layer = tape.gradient(loss, [eeg_t, layer_out])
        return (
            grads_input.numpy() if grads_input is not None else np.zeros_like(eeg),
            grads_layer.numpy() if grads_layer is not None else None,
        )

    @property
    def layer_names(self) -> list[str]:
        """List all layer names in the TF model."""
        return [l.name for l in self.model.layers]
