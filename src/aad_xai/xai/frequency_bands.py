"""Reusable frequency-band ablation: isolate and subtractively remove one
frequency band from EEG, to measure its contribution to a model's decision.

Consolidates the numerically-sound SOS+reflect-pad bandpass approach that
previously existed only as two divergent inline copies in
scripts/run_focused_xai.py (not as a reusable library function) --
NOT scripts/run_focused_xai.py's accompanying but weaker single-tap variant
used elsewhere, and NOT notebooks/kaggle_run_xai_aadnet.py's
``bandpass_channel_block`` (plain ``butter``+``filtfilt``, no SOS, no
padding), which is not numerically equivalent to this version.

BANDS below matches scripts/run_focused_xai.py's own ``BANDS`` exactly, for
cross-pipeline comparability. This is deliberately coarser than
``sgat_frequency_encoder.py``'s internal 8-band split -- that is the model's
*learned* feature decomposition; this module is a separate, coarser,
human-interpretable *post-hoc ablation probe*. The two are not meant to be
reconciled.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.signal import butter, sosfiltfilt

BANDS: "OrderedDict[str, tuple[float, float]]" = OrderedDict([
    ("delta", (0.5, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 13.0)),
    ("beta", (13.0, 30.0)),
])


def band_filtered_component(
    eeg: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
    *,
    time_axis: int = -1,
    pad_samples: int = 64,
) -> np.ndarray:
    """Isolate one frequency band's content via a 4th-order Butterworth SOS
    bandpass with zero-phase, reflect-padded filtering (``sosfiltfilt``).

    Parameters
    ----------
    eeg : np.ndarray
        Any shape with time along ``time_axis``, e.g. (B, C, T)
        [StimulusGAT's convention, time_axis=-1 is fine] or (B, T, C)
        [VLAAI/AADNet's convention, pass time_axis=1].
    sfreq : float
        Sampling rate, Hz.
    low_hz, high_hz : float
        Passband edges, Hz.
    time_axis : int
        Which axis of eeg is time.
    pad_samples : int
        Reflect-pad length (matches the 64-sample pad already established
        in scripts/run_focused_xai.py).

    Returns
    -------
    np.ndarray, same shape as eeg -- the isolated band-limited component.
    """
    nyq = sfreq / 2.0
    lo_n = max(low_hz / nyq, 1e-4)
    hi_n = min(high_hz / nyq, 1.0 - 1e-4)
    sos = butter(4, [lo_n, hi_n], btype="bandpass", output="sos")

    def _filt_1d(sig: np.ndarray) -> np.ndarray:
        padded = np.pad(sig, pad_samples, mode="reflect")
        filtered = sosfiltfilt(sos, padded)
        return filtered[pad_samples:pad_samples + sig.shape[-1]]

    return np.apply_along_axis(_filt_1d, time_axis, eeg)


def ablate_band(
    eeg: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    *,
    channel_idx: list[int] | None = None,
    time_axis: int = -1,
    channel_axis: int = -2,
    pad_samples: int = 64,
) -> np.ndarray:
    """Subtractively remove one frequency band from (optionally, a subset
    of) channels: ``eeg_ablated = eeg - band_filtered_component(eeg, band)``,
    restricted to ``channel_idx`` if given. Matches
    scripts/run_focused_xai.py's ablation semantics exactly (subtract the
    isolated band, don't replace the signal).

    Parameters
    ----------
    eeg : np.ndarray
        e.g. (B, C, T) [StimulusGAT] or (B, T, C) [VLAAI/AADNet].
    sfreq : float
    band : tuple[float, float]
        (low_hz, high_hz), typically one of ``BANDS.values()``.
    channel_idx : list[int] | None
        If given, ablate only these channels (all others untouched); if
        None, ablate the band from every channel.
    time_axis, channel_axis : int
        Axis roles in eeg. Defaults assume (..., C, T); pass time_axis=1,
        channel_axis=2 for (B, T, C).
    pad_samples : int

    Returns
    -------
    np.ndarray, same shape as eeg.
    """
    low_hz, high_hz = band
    band_content = band_filtered_component(
        eeg, sfreq, low_hz, high_hz, time_axis=time_axis, pad_samples=pad_samples
    )
    x = eeg.copy()
    if channel_idx is None:
        x -= band_content
    else:
        idx = [slice(None)] * x.ndim
        idx[channel_axis] = channel_idx
        idx = tuple(idx)
        x[idx] -= band_content[idx]
    return x
