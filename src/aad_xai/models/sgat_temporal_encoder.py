from __future__ import annotations
import torch
import torch.nn as nn


class PerChannelTemporalEncoder(nn.Module):
    """Per-electrode temporal feature extractor.

    Uses depthwise (grouped, ``groups=n_channels``) Conv1d layers throughout so
    every electrode's own trace determines its own output features, with zero
    mixing across electrodes at this stage. This avoids ST-GCN's temporal
    stage, which pools all channels through one shared Conv1d and then
    artificially re-projects a single pooled vector back into per-node
    features via a Linear layer, losing genuine per-electrode identity.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (electrodes).
    out_features : int
        Number of temporal features produced per electrode.
    hidden_features : int
        Number of features per electrode after the first conv layer.
    kernel_sizes : tuple[int, int]
        Kernel sizes for the two depthwise conv layers.
    dropout : float
    """

    def __init__(
        self,
        n_channels: int,
        out_features: int = 16,
        hidden_features: int = 16,
        kernel_sizes: tuple[int, int] = (9, 9),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.out_features = out_features
        k1, k2 = kernel_sizes

        # groups=n_channels: each electrode's 1 input channel is convolved
        # independently into `hidden_features` (then `out_features`) output
        # channels, with the grouped-conv output layout automatically staying
        # electrode-major: [ch0_f0..ch0_f{F-1}, ch1_f0..ch1_f{F-1}, ...].
        self.block1 = nn.Sequential(
            nn.Conv1d(
                n_channels, n_channels * hidden_features,
                kernel_size=k1, padding=k1 // 2, groups=n_channels,
            ),
            nn.BatchNorm1d(n_channels * hidden_features),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(
                n_channels * hidden_features, n_channels * out_features,
                kernel_size=k2, padding=k2 // 2, groups=n_channels,
            ),
            nn.BatchNorm1d(n_channels * out_features),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, C, T)

        Returns
        -------
        Tensor, shape (B, C, out_features)
        """
        B = x.size(0)
        h = self.block1(x)              # (B, C*hidden_features, T)
        h = self.block2(h)               # (B, C*out_features, T)
        h = self.pool(h).squeeze(-1)      # (B, C*out_features)
        return h.view(B, self.n_channels, self.out_features)  # (B, C, out_features)
