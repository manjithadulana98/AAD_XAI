"""TRF decision wrapper: turns TRFDecoder regression into a 2-class AAD classifier.

Mirrors the interface of AADDecisionEEGOnly so that model-agnostic XAI
sections (C, D, H, I, J) can call ``wrapper(eeg) -> (B, 2)`` identically.

Unlike the VLAAI wrapper, TRFDecoder is a numpy/sklearn model, so this
wrapper handles numpy<->torch conversion internally.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .trf_baseline import TRFDecoder


def _pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation returning 0.0 for constant signals."""
    a, b = a.ravel(), b.ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


class TRFDecisionWrapper(nn.Module):
    """Wrap a fitted TRFDecoder into a 2-class attended/unattended classifier.

    Interface matches AADDecisionEEGOnly:
        wrapper.set_envelopes(env_att, env_unatt)
        logits = wrapper(eeg)  # (B, 2)

    Parameters
    ----------
    trf : TRFDecoder
        A *fitted* TRFDecoder instance.
    """

    def __init__(self, trf: TRFDecoder):
        super().__init__()
        self.trf = trf
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
        eeg : (B, T, C) — channels-last EEG (matches VLAAIDTUDataset output)

        Returns
        -------
        logits : (B, 2) — ``[:, 0]`` = unattended corr, ``[:, 1]`` = attended corr
        """
        B = eeg.shape[0]
        eeg_np = eeg.detach().cpu().numpy()  # (B, T, C)

        env_att = self._env_att
        env_unatt = self._env_unatt

        # Handle batch expansion (same as AADDecisionEEGOnly)
        if env_att.shape[0] < B:
            reps = (B + env_att.shape[0] - 1) // env_att.shape[0]
            env_att = env_att.repeat(reps, 1, 1)[:B]
            env_unatt = env_unatt.repeat(reps, 1, 1)[:B]

        att_np = env_att.detach().cpu().numpy()    # (B, T, 1)
        unatt_np = env_unatt.detach().cpu().numpy()  # (B, T, 1)

        logits = np.zeros((B, 2), dtype=np.float32)
        for i in range(B):
            # TRFDecoder.predict expects (n_channels, n_times)
            eeg_i = eeg_np[i].T  # (C, T)
            pred_env = self.trf.predict(eeg_i)  # (T,)

            att_i = att_np[i, :, 0]    # (T,)
            unatt_i = unatt_np[i, :, 0]  # (T,)

            # Trim to common length (lag matrix may shorten slightly)
            n = min(len(pred_env), len(att_i))
            r_att = _pearson_np(pred_env[:n], att_i[:n])
            r_unatt = _pearson_np(pred_env[:n], unatt_i[:n])

            logits[i, 0] = r_unatt
            logits[i, 1] = r_att

        return torch.tensor(logits, dtype=torch.float32, device=eeg.device)
