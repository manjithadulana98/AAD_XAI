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
    """Zero out a group of channels. Supports (T, C) and (B, T, C) inputs."""
    x = eeg.copy()
    if x.ndim == 2:
        # (T, C)
        x[:, ch_idx] = 0.0
    elif x.ndim == 3:
        # (B, T, C)
        x[:, :, ch_idx] = 0.0
    else:
        # fallback: try last axis
        x[..., ch_idx] = 0.0
    return x


def permute_channel_group(
    eeg: np.ndarray, ch_idx: list[int], *, channel_axis: int = -1, seed: int = 42
) -> np.ndarray:
    """Shuffle a group of channels' values across the batch axis (axis 0).

    Unlike remove_channel_group's zero-occlusion (which destroys the signal
    entirely), this preserves each channel's own marginal distribution while
    breaking its window-specific pairing with the label -- a window at batch
    index i takes channel ch's values from a different, randomly chosen batch
    index. Requires a batched input (ndim >= 2 with a leading batch axis);
    permutation across batch is meaningless for a single unbatched window.

    Parameters
    ----------
    eeg : np.ndarray
        Batched input, e.g. (B, T, C) -- channel_axis=-1 (default), matching
        remove_channel_group's convention -- or (B, C, T), pass channel_axis=1.
    ch_idx : list[int]
        Channels (indexed along channel_axis) to permute.
    channel_axis : int
        Which axis of eeg indexes channels.
    seed : int
        RNG seed for the batch permutation (deterministic).

    Returns
    -------
    np.ndarray, same shape as eeg, with ch_idx channels' values shuffled
    across the batch axis.
    """
    if eeg.ndim < 2:
        raise ValueError("permute_channel_group requires a batched input (ndim >= 2).")
    x = np.asarray(eeg)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(x.shape[0])
    x_permuted_batch = x[perm]  # single fancy index along axis 0 only -- safe, no axis reordering

    # Boolean mask, True only at ch_idx along channel_axis, broadcastable against x's full shape.
    # (Combining the batch-permutation index and the channel-selection index in one `x[...,]`
    # expression would trigger NumPy's advanced-indexing axis-reordering gotcha when the two
    # fancy indices aren't adjacent -- np.where sidesteps that entirely.)
    mask = np.zeros(x.shape[channel_axis], dtype=bool)
    mask[ch_idx] = True
    mask_shape = [1] * x.ndim
    mask_shape[channel_axis] = x.shape[channel_axis]
    mask = mask.reshape(mask_shape)

    return np.where(mask, x_permuted_batch, x)
