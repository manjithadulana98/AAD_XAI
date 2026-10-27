from __future__ import annotations
import numpy as np
from scipy.signal import hilbert, butter, filtfilt, resample, gammatone, lfilter
from scipy.stats import zscore


def powerlaw_subbands_envelope(
    audio: np.ndarray,
    sr: int,
    *,
    target_sr: int,
    bandpass_hz: tuple[float, float] = (1.0, 9.0),
    power: float = 0.6,
) -> np.ndarray:
    """Powerlaw-subband envelope extraction (Biesmans-style).

    This mirrors the envelope extraction used in the external AADNet codebase
    (gammatone filterbank -> per-band envelope -> powerlaw compression -> sum).

    Steps
    -----
    1) Low-pass the audio to 4 kHz (via intermediate sr=8 kHz).
    2) Resample to 8 kHz.
    3) Apply a fixed 15-channel gammatone FIR filterbank.
    4) Per-channel magnitude and power-law compression (default exponent 0.6).
    5) Sum subbands to a broadband envelope.
    6) Resample to 128 Hz, bandpass filter, then resample to target_sr.
    7) Z-score the resulting envelope.

    Parameters
    ----------
    audio:
        Mono waveform.
    sr:
        Sampling rate of the waveform.
    target_sr:
        Target sampling rate for the envelope (should match EEG rate).
    bandpass_hz:
        Bandpass applied to the envelope at 128 Hz, matching the linear-model
        preprocessing band (e.g., 1-9 Hz in Vandecappelle et al., 2021).
    """

    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32)

    env = np.sum(_gammatone_powerlaw_bands(audio, sr, power), axis=0)
    env = _band_limit_and_resample(env, bandpass_hz, target_sr)
    return zscore(env).astype(np.float32, copy=False)


# Gammatone stage of the power-law envelope: fixed rates and center
# frequencies used by the AADNet preprocessing.
_SR_GAMMATONE = 8000
_SR_ENVELOPE_FILTER = 128
GAMMATONE_CENTER_FREQS_HZ = np.array(
    [178.7, 250.3, 334.5, 433.5, 549.9, 686.8, 847.7, 1036.9,
     1259.3, 1520.9, 1828.4, 2190.0, 2615.1, 3114.9, 3702.6],
    dtype=np.float64,
)


def _gammatone_powerlaw_bands(audio: np.ndarray, sr: int, power: float) -> np.ndarray:
    """(n_samples,) waveform -> (15, n_samples_8k) power-law compressed sub-band magnitudes at 8 kHz."""
    from mne.filter import filter_data  # imported here to keep module import light

    # Low-pass to 4 kHz before resampling to 8 kHz.
    audio_lp = filter_data(audio, sr, None, _SR_GAMMATONE / 2.0, verbose="CRITICAL")
    if int(sr) != int(_SR_GAMMATONE):
        audio_lp = resample(audio_lp, int(round(len(audio_lp) * _SR_GAMMATONE / sr)))

    envs = []
    for f in GAMMATONE_CENTER_FREQS_HZ:
        b, a = gammatone(freq=float(f), ftype="fir", order=4, fs=_SR_GAMMATONE)
        sub = np.real(lfilter(b, a, audio_lp))
        envs.append(np.abs(sub) ** float(power))
    return np.asarray(envs)


def _band_limit_and_resample(env: np.ndarray, bandpass_hz: tuple[float, float], target_sr: int) -> np.ndarray:
    """(..., n_samples_8k) -> (..., n_samples_target): resample to 128 Hz, band-pass, resample to target."""
    from mne.filter import filter_data

    if int(_SR_ENVELOPE_FILTER) != int(_SR_GAMMATONE):
        env = resample(env, int(round(env.shape[-1] * _SR_ENVELOPE_FILTER / _SR_GAMMATONE)), axis=-1)

    env = filter_data(env, _SR_ENVELOPE_FILTER, float(bandpass_hz[0]), float(bandpass_hz[1]), verbose="CRITICAL")

    if int(target_sr) != int(_SR_ENVELOPE_FILTER):
        env = resample(env, int(round(env.shape[-1] * target_sr / _SR_ENVELOPE_FILTER)), axis=-1)
    return env


def powerlaw_subband_envelopes(
    audio: np.ndarray,
    sr: int,
    *,
    target_sr: int,
    bandpass_hz: tuple[float, float] = (1.0, 9.0),
    power: float = 0.6,
    zscore_bands: bool = True,
) -> np.ndarray:
    """Per-band power-law envelopes: the 15 sub-bands of
    :func:`powerlaw_subbands_envelope` kept separate instead of summed.

    Returns
    -------
    np.ndarray, shape (15, n_samples_target), float32. Each band is z-scored
    over time if ``zscore_bands``. With ``zscore_bands=False``,
    ``zscore(bands.sum(0))`` equals :func:`powerlaw_subbands_envelope` up to
    float rounding (every step after the gammatone stage is linear).
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    n_bands = len(GAMMATONE_CENTER_FREQS_HZ)
    if audio.size == 0:
        return np.zeros((n_bands, 0), dtype=np.float32)

    bands = _band_limit_and_resample(_gammatone_powerlaw_bands(audio, sr, power), bandpass_hz, target_sr)
    if zscore_bands:
        bands = zscore(bands, axis=-1)
    return bands.astype(np.float32, copy=False)

def speech_envelope(audio: np.ndarray, sr: int, low_hz: float = 1.0, high_hz: float = 8.0) -> np.ndarray:
    """Simple amplitude envelope + bandpass to match common AAD pipelines."""
    audio = np.asarray(audio).astype(float)
    analytic = hilbert(audio)
    env = np.abs(analytic)

    b, a = butter(4, [low_hz/(sr/2), high_hz/(sr/2)], btype="band")
    env = filtfilt(b, a, env)
    return env
