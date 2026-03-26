from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


def _import_upstream_aadnet():
    root = Path(__file__).resolve().parents[3]
    submodule_root = root / "external" / "AADNet"
    if str(submodule_root) not in sys.path:
        sys.path.insert(0, str(submodule_root))
    from aadnet.EnvelopeAAD import AADNet as UpstreamAADNet
    return UpstreamAADNet


class ExternalAADNet(nn.Module):
    """Adapter for the upstream AADNet submodule model.

    Expects EEG and audio envelopes:
      eeg: (B, C, T)
      env: (B, 2, T)

    Returns 2-class logits: (B, 2)
    """

    def __init__(self, n_channels: int, window_samples: int, sfreq: int = 64):
        super().__init__()
        UpstreamAADNet = _import_upstream_aadnet()

        config = {
            "in_channels": n_channels,
            "chns_1": [32, [16, 8], [8, 8], [4, 8], [2, 8], 8],
            "kernels_1": [1, 19, 25, 33, 39, 3],
            "chns_1_aud": [1, [1, 4], [1, 4], 0],
            "kernels_1_aud": [1, 65, 81, 3],
            "act_1": "relu",
            "pool_stride_1": 2,
            "hidden_size": 0,
            "dropout": 0.4,
            "feature_freeze": False,
        }
        channels = list(range(n_channels))
        self.model = UpstreamAADNet(config=config, L=window_samples, n_streams=2, sr=sfreq, channels=channels)

    def forward(self, eeg: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        return self.model(eeg, env)
