from __future__ import annotations
import torch
import torch.nn as nn


class SharedSpeechEncoder(nn.Module):
    """Minimal speech-envelope encoder, shared across both candidates.

    A single instance of this module is invoked twice by ``StimulusGAT``
    (once per candidate) -- weight sharing comes from Python-level reuse of
    the same instance, not from anything in this class. Deliberately a small
    2-layer 1D-CNN, not a Transformer/Conformer (per the project's
    development rule against adding extra attention blocks automatically).

    Parameters
    ----------
    embed_dim : int
        Channels of the output sequence.
    hidden : int
        Number of channels after the first conv layer.
    kernel_sizes : tuple[int, int]
    dropout : float
    in_channels : int
        Envelope channels per candidate: 1 for a single broadband envelope
        (input (B, T)), K for K sub-band envelopes (input (B, K, T)).
    """

    def __init__(
        self,
        embed_dim: int = 16,
        hidden: int = 8,
        kernel_sizes: tuple[int, int] = (9, 9),
        dropout: float = 0.3,
        in_channels: int = 1,
    ):
        super().__init__()
        k1, k2 = kernel_sizes
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, embed_dim, kernel_size=k2, padding=k2 // 2),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
        )

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        env : Tensor, shape (B, T) if in_channels == 1, else (B, in_channels, T)

        Returns
        -------
        Tensor, shape (B, embed_dim, T) -- a sequence, not a pooled vector, so
        it can modulate EEG features time step by time step.
        """
        if env.dim() == 2:
            env = env.unsqueeze(1)                     # (B, 1, T)
        if env.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} envelope channels, got shape {tuple(env.shape)}")
        return self.net(env)                           # (B, in_channels, T) -> (B, embed_dim, T)
