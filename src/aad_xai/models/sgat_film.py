from __future__ import annotations
import torch
import torch.nn as nn


class TimeResolvedFiLM(nn.Module):
    """Condition per-electrode temporal EEG features on a candidate's speech
    sequence at every time step, then average over time.

    ``out[b,c,f] = mean_t( h[b,c,f,t] * (1 + gamma[b,f,t]) + beta[b,f,t] )``

    Conditioning must happen *before* time-pooling: the term
    ``mean_t(h * gamma)`` contains the EEG-speech temporal covariance, which
    is the signal auditory attention decoding relies on. Pooling each stream
    first keeps only their marginal statistics, and the synthetic double
    dissociation (scripts/diag_sgat_synthetic_dissociation.py) showed the
    model then cannot learn envelope tracking at all.

    Init: zero weight, gamma bias = FILM_GAMMA_INIT, beta bias = 0, so both
    candidates' outputs are identical at init while ``proj.weight`` still
    receives gradient from the first step.

    Pooling of the EEG-speech term:

    - ``"cov"`` (default): ``mean_t(h * gamma)`` as above.
    - ``"corr"``: the Pearson correlation over time of ``h[b,c,f,:]`` with
      ``gamma[b,f,:]``, as in AADNet's correlation layer. It is invariant to
      the scale and offset of either signal within a window, so the
      candidate-dependent part of the output no longer depends on EEG
      amplitude. The other two terms are unchanged:
      ``out = mean_t(h) + corr_t(h, gamma) + mean_t(beta)``.

    Parameters
    ----------
    speech_dim : int
        Channels of the speech-encoder output sequence.
    n_features : int
        Temporal features per electrode (F of ``h``).
    pooling : {"cov", "corr"}
    """

    FILM_GAMMA_INIT = 0.1
    CORR_EPS = 1e-5

    def __init__(self, speech_dim: int, n_features: int, pooling: str = "cov"):
        super().__init__()
        if pooling not in ("cov", "corr"):
            raise ValueError(f"pooling must be 'cov' or 'corr', got {pooling!r}")
        self.n_features = n_features
        self.pooling = pooling
        self.proj = nn.Conv1d(speech_dim, 2 * n_features, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias[:n_features].fill_(self.FILM_GAMMA_INIT)
            self.proj.bias[n_features:].zero_()

    def forward(self, h: torch.Tensor, speech_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : Tensor, shape (B, C, F, T) -- per-electrode temporal EEG features
        speech_seq : Tensor, shape (B, D_s, T) -- one candidate's speech sequence

        Returns
        -------
        Tensor, shape (B, C, F)
        """
        gamma, beta = self.proj(speech_seq).chunk(2, dim=1)          # (B, F, T) each
        T = h.shape[-1]
        if self.pooling == "corr":
            # z-score over time; eps keeps it finite when gamma is constant
            # (at init), where the term is exactly 0 for every candidate.
            hz = _zscore_time(h, self.CORR_EPS)                         # (B, C, F, T)
            gz = _zscore_time(gamma, self.CORR_EPS)                     # (B, F, T)
            term = torch.einsum("bcft,bft->bcf", hz, gz) / T            # (B, C, F), in [-1, 1]
        else:
            # Equivalent to (h * (1 + gamma[:, None]) + beta[:, None]).mean(-1)
            # without materializing the (B, C, F, T) product.
            term = torch.einsum("bcft,bft->bcf", h, gamma) / T          # (B, C, F)
        return h.mean(dim=-1) + term + beta.mean(dim=-1).unsqueeze(1)   # (B, C, F)


def _zscore_time(x: torch.Tensor, eps: float) -> torch.Tensor:
    """(..., T) -> (..., T), zero mean and (up to eps) unit variance over T."""
    x = x - x.mean(dim=-1, keepdim=True)
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
