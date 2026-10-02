"""StimulusGAT decision wrapper: adapts (eeg, env) -> (B,2) to (eeg) -> (B,2).

Mirrors vlaai_decision.py::AADDecisionEEGOnly / trf_decision.py's wrapper
pattern (stash a second fixed input as a registered buffer so generic XAI
tools that call ``model(eeg)`` with exactly one positional tensor still
work), but simpler: StimulusGAT is already a native 2-class classifier, not
an envelope regressor, so no Pearson-correlation reduction is needed here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .sgat import StimulusGAT


class StimulusGATDecisionWrapper(nn.Module):
    """Wrap StimulusGAT's ``(eeg, env) -> (B,2)`` interface into
    ``(eeg) -> (B,2)``, for xai/ tools that call ``model(x)`` with exactly
    one positional tensor (faithfulness.py, probes.py, integrated_gradients.py,
    gradcam.py, shap_explainer.py, lime_explainer.py, sanity_checks.py).

    Usage::

        wrapper = StimulusGATDecisionWrapper(sgat_model)
        wrapper.set_env(env)          # env: (B, 2, T)
        logits = wrapper(eeg)         # eeg: (B, C, T) -> (B, 2)
    """

    def __init__(self, model: StimulusGAT):
        super().__init__()
        self.model = model
        self.register_buffer("_env", torch.zeros(1, 2, 1))

    def set_env(self, env: torch.Tensor) -> None:
        """Set the reference candidate envelopes for the current batch.

        Parameters
        ----------
        env : Tensor, shape (B, 2, T) -- env[:, 0] = candidate A, env[:, 1] = candidate B.
        """
        self._env = env

    def _matched_env(self, batch_size: int) -> torch.Tensor:
        """Return ``self._env`` expanded/trimmed to exactly *batch_size* rows.

        Handles the common case where an XAI tool internally multiplies the
        batch (e.g. Captum's IntegratedGradients, which expands the batch by
        ``n_steps``) by repeating the stored env cyclically -- same logic as
        AADDecisionEEGOnly.forward. Also handles the (currently unused by any
        known caller, but cheap to guard) shrink case, where the stashed env
        batch is larger than the requested one.
        """
        env = self._env
        if env.shape[0] < batch_size:
            reps = (batch_size + env.shape[0] - 1) // env.shape[0]
            env = env.repeat(reps, 1, 1)[:batch_size]
        elif env.shape[0] > batch_size:
            env = env[:batch_size]
        return env

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg : Tensor, shape (B, C, T) -- channels-first EEG window.

        Returns
        -------
        logits : Tensor, shape (B, 2)
        """
        env = self._matched_env(eeg.shape[0])
        return self.model(eeg, env, return_attention_weights=False)

    def forward_with_attention(self, eeg: torch.Tensor):
        """Non-generic escape hatch: only ``xai/sgat_explain.py`` calls this
        directly (no generic xai/ tool asks for attention weights).

        Returns
        -------
        logits : Tensor, shape (B, 2)
        attn : dict[str, Tensor] -- {"attn_a": (B,heads,N,N), "attn_b": (B,heads,N,N)}
        """
        env = self._matched_env(eeg.shape[0])
        return self.model(eeg, env, return_attention_weights=True)
