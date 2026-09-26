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
        Output embedding dimension.
    hidden : int
        Number of channels after the first conv layer.
    kernel_sizes : tuple[int, int]
    dropout : float
    """

    def __init__(
        self,
        embed_dim: int = 16,
        hidden: int = 8,
        kernel_sizes: tuple[int, int] = (9, 9),
        dropout: float = 0.3,
    ):
        super().__init__()
        k1, k2 = kernel_sizes
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, embed_dim, kernel_size=k2, padding=k2 // 2),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        env : Tensor, shape (B, T)

        Returns
        -------
        Tensor, shape (B, embed_dim)
        """
        h = env.unsqueeze(1)          # (B, 1, T)
        return self.net(h).squeeze(-1)  # (B, embed_dim)
