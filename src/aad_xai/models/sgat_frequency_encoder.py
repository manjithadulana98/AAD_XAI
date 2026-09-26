from __future__ import annotations
import torch
import torch.nn as nn

# Same band edges (Hz) as the FAConformer SI port's FFT-band decomposition
# (notebooks/kaggle_faconformer_si_port_pilot.py).
DEFAULT_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (1.0, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 10.0),   # alpha1
    (10.0, 13.0),  # alpha2
    (13.0, 16.0),  # beta1
    (16.0, 20.0),  # beta2
    (20.0, 26.0),  # beta3
    (26.0, 32.0),  # gamma1
)


class PerChannelFrequencyEncoder(nn.Module):
    """Per-electrode frequency-domain feature extractor.

    Computes per-channel log band-power via ``torch.fft.rfft`` on the raw
    window, then projects the fixed-band feature vector for each electrode
    into ``out_features`` with a per-electrode Linear (implemented as a
    grouped 1x1 Conv1d, ``groups=n_channels``) -- no cross-channel mixing.

    Computed fresh per forward call (per-batch), not precomputed for the
    whole dataset, matching the FAConformer port's own OOM-avoidance
    rationale for the same FFT-band-decomposition idea. Uses no
    cross-subject-pooled statistic anywhere (pure per-window FFT), avoiding
    the cross-subject covariance-pooling failure that port hit with CSP.

    Parameters
    ----------
    n_channels : int
    sfreq : float
        EEG sampling rate in Hz, used to map band edges to FFT bins.
    out_features : int
        Number of frequency features produced per electrode.
    bands_hz : tuple[tuple[float, float], ...]
        Frequency band edges in Hz.
    """

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        out_features: int = 8,
        bands_hz: tuple[tuple[float, float], ...] = DEFAULT_BANDS_HZ,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.sfreq = float(sfreq)
        self.out_features = out_features
        self.bands_hz = bands_hz
        self.eps = eps
        n_bands = len(bands_hz)

        # Per-electrode Linear(n_bands -> out_features): groups=n_channels
        # means group i only ever reads electrode i's own n_bands values.
        self.proj = nn.Conv1d(
            n_channels * n_bands, n_channels * out_features,
            kernel_size=1, groups=n_channels,
        )

    def _band_power(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, n_bands) log band power (no learned params)."""
        n_times = x.size(-1)
        spectrum = torch.fft.rfft(x, dim=-1)                       # (B, C, T//2+1) complex
        power = spectrum.real ** 2 + spectrum.imag ** 2            # (B, C, T//2+1)
        freqs = torch.fft.rfftfreq(n_times, d=1.0 / self.sfreq).to(x.device)  # (T//2+1,)

        band_feats = []
        for lo, hi in self.bands_hz:
            mask = (freqs >= lo) & (freqs < hi)
            if not bool(mask.any()):
                # Window too short / sfreq too low to resolve this band at all.
                band_feats.append(torch.zeros(x.size(0), x.size(1), device=x.device, dtype=power.dtype))
                continue
            band_feats.append(power[..., mask].mean(dim=-1))        # (B, C)
        feat = torch.stack(band_feats, dim=-1)                       # (B, C, n_bands)
        return torch.log1p(feat + self.eps)

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
        feat = self._band_power(x)                     # (B, C, n_bands)
        feat = feat.reshape(B, -1, 1)                    # (B, C*n_bands, 1), electrode-major
        out = self.proj(feat).squeeze(-1)                 # (B, C*out_features)
        return out.view(B, self.n_channels, self.out_features)  # (B, C, out_features)
