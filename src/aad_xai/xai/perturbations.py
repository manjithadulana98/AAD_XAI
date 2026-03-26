from __future__ import annotations
import numpy as np

def band_limited_attenuation(eeg: np.ndarray, sfreq: float, low_hz: float, high_hz: float, factor: float = 0.0) -> np.ndarray:
    """Attenuate a frequency band by scaling its bandpassed component (simple, linear)."""
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [low_hz/(sfreq/2), high_hz/(sfreq/2)], btype="band")
    band = filtfilt(b, a, eeg, axis=-1)
    return eeg - (1.0 - factor) * band

def suppress_lag_range(eeg: np.ndarray, sfreq: float, tmin_s: float, tmax_s: float) -> np.ndarray:
    """Zero out a latency range inside a window (useful only if your window is aligned to stimulus onset)."""
    x = eeg.copy()
    i0 = int(round(tmin_s * sfreq))
    i1 = int(round(tmax_s * sfreq))
    x[:, max(0, i0):max(0, i1)] = 0.0
    return x

def remove_channel_group(eeg: np.ndarray, ch_idx: list[int]) -> np.ndarray:
    x = eeg.copy()
    x[ch_idx, :] = 0.0
    return x
