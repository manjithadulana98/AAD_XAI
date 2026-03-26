from __future__ import annotations
import torch
import torch.nn as nn


class AADNet(nn.Module):
    """End-to-end CNN for Auditory Attention Decoding (AADNet-style).

    Architecture::

        Conv1d → BN → ELU  (spatial mixing)
        Conv1d → BN → ELU  (temporal filtering 1)
        Conv1d → BN → ELU  (temporal filtering 2)
        AdaptiveAvgPool1d(1) → Flatten
        Linear → head (2-class logits)

    The first three conv blocks are grouped into ``feature_extractor`` so
    that interpretability tools (probes, IG) can easily access intermediate
    representations via forward hooks.

    Parameters
    ----------
    n_channels : int
        Number of input EEG channels (spatial dimension).
    n_classes : int
        Number of output classes (default 2: attended left/right).
    F1, F2, F3 : int
        Number of filters in the three conv layers.
    K1, K2, K3 : int
        Kernel sizes (temporal) for the three conv layers.
    dropout : float
        Dropout probability applied after each ELU.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 2,
        F1: int = 32,
        F2: int = 64,
        F3: int = 64,
        K1: int = 9,
        K2: int = 9,
        K3: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Block 1 — spatial mixing
            nn.Conv1d(n_channels, F1, kernel_size=K1, padding=K1 // 2),
            nn.BatchNorm1d(F1),
            nn.ELU(),
            nn.Dropout(dropout),
            # Block 2 — temporal filter 1
            nn.Conv1d(F1, F2, kernel_size=K2, padding=K2 // 2),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(dropout),
            # Block 3 — temporal filter 2
            nn.Conv1d(F2, F3, kernel_size=K3, padding=K3 // 2),
            nn.BatchNorm1d(F3),
            nn.ELU(),
            nn.Dropout(dropout),
            # Pooling
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(F3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, n_channels, T)``.

        Returns
        -------
        logits : Tensor
            Shape ``(B, n_classes)``.
        """
        h = self.feature_extractor(x).squeeze(-1)  # (B, F3)
        return self.head(h)


# Keep a lightweight alias for backwards-compatibility with old configs
AADNetLite = AADNet

