"""AAD decision wrapper: turns VLAAI regression into a 2-class AAD problem.

VLAAI predicts a speech envelope from EEG. This wrapper computes the
differentiable Pearson correlation between the predicted envelope and both
attended / unattended reference envelopes, returning ``[r_unatt, r_att]``
as 2-class logits.

This enables standard classification-based XAI methods (GradCAM, SHAP, LIME)
to target class 1 (= attended) or class 0 (= unattended).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _pearson_torch(x: torch.Tensor, y: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Differentiable Pearson correlation along *dim*.

    Parameters
    ----------
    x, y : Tensor with matching shape
    dim : int — axis along which to compute correlation

    Returns
    -------
    Tensor — correlation values (reduced along *dim*).
    """
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    xc = x - x_mean
    yc = y - y_mean
    num = (xc * yc).sum(dim=dim, keepdim=True)
    denom = torch.sqrt((xc ** 2).sum(dim=dim, keepdim=True) * (yc ** 2).sum(dim=dim, keepdim=True))
    return (num / (denom + 1e-8)).squeeze(dim)


class AADDecisionWrapper(nn.Module):
    """Wrap a VLAAI envelope decoder into a 2-class attended/unattended classifier.

    Forward signature::

        logits = wrapper(eeg, env_attended, env_unattended)
        # logits shape: (B, 2) — [r_unattended, r_attended]

    The wrapper is fully differentiable end-to-end, so gradient-based XAI
    methods (GradCAM, IG, DeepSHAP) can back-propagate through it.

    Parameters
    ----------
    decoder : nn.Module
        A PyTorch VLAAI model that maps ``(B, T, C) → (B, T, 1)``.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        eeg: torch.Tensor,
        env_attended: torch.Tensor,
        env_unattended: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg : (B, T, C) — input EEG
        env_attended : (B, T, 1) — attended speech envelope
        env_unattended : (B, T, 1) — unattended speech envelope

        Returns
        -------
        logits : (B, 2) — ``[:, 0]`` = unattended corr, ``[:, 1]`` = attended corr
        """
        pred_env = self.decoder(eeg)  # (B, T, 1)

        r_att = _pearson_torch(pred_env.squeeze(-1), env_attended.squeeze(-1), dim=1)
        r_unatt = _pearson_torch(pred_env.squeeze(-1), env_unattended.squeeze(-1), dim=1)

        # Stack as 2-class logits: [unattended, attended]
        logits = torch.stack([r_unatt, r_att], dim=-1)  # (B, 2)
        return logits


class AADDecisionEEGOnly(nn.Module):
    """Variant that takes only EEG input (envelopes are frozen references).

    This is useful for standard XAI methods that expect a single-input model
    ``f(eeg) → logits``.  The reference envelopes are registered as buffers
    (not learnable) and can be set per-batch before calling the XAI method.

    Example::

        wrapper = AADDecisionEEGOnly(decoder)
        wrapper.set_envelopes(env_att, env_unatt)
        # Now GradCAM / IG / SHAP can call  wrapper(eeg) → (B, 2)
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("_env_att", torch.zeros(1))
        self.register_buffer("_env_unatt", torch.zeros(1))

    def set_envelopes(
        self,
        env_attended: torch.Tensor,
        env_unattended: torch.Tensor,
    ):
        """Set the reference envelopes for the current batch."""
        self._env_att = env_attended
        self._env_unatt = env_unattended

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg : (B, T, C) — channels-last EEG

        Returns
        -------
        logits : (B, 2)
        """
        pred_env = self.decoder(eeg)  # (B, T, 1)
        B = eeg.shape[0]

        # Handle batch expansion by XAI tools (e.g., Captum IG multiplies
        # batch by n_steps).  Repeat stored envelopes cyclically.
        env_att = self._env_att
        env_unatt = self._env_unatt
        if env_att.shape[0] < B:
            reps = (B + env_att.shape[0] - 1) // env_att.shape[0]
            env_att = env_att.repeat(reps, 1, 1)[:B]
            env_unatt = env_unatt.repeat(reps, 1, 1)[:B]

        r_att = _pearson_torch(pred_env.squeeze(-1), env_att.squeeze(-1), dim=1)
        r_unatt = _pearson_torch(pred_env.squeeze(-1), env_unatt.squeeze(-1), dim=1)
        return torch.stack([r_unatt, r_att], dim=-1)
