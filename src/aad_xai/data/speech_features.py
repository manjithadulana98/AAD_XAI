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

    # Imports here to keep module import light.
    from mne.filter import filter_data

    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32)

    sr_int1 = 8000
    sr_int2 = 128

    # Fixed center frequencies used by the AADNet preprocessing.
    freqs = np.array(
        [
            178.7,
            250.3,
            334.5,
            433.5,
            549.9,
            686.8,
            847.7,
            1036.9,
            1259.3,
            1520.9,
            1828.4,
            2190.0,
            2615.1,
            3114.9,
            3702.6,
        ],
        dtype=np.float64,
    )

    # Low-pass to 4 kHz before resampling to 8 kHz.
    audio_lp = filter_data(audio, sr, None, sr_int1 / 2.0, verbose="CRITICAL")
    if int(sr) != int(sr_int1):
        audio_lp = resample(audio_lp, int(round(len(audio_lp) * sr_int1 / sr)))

    envs = []
    for f in freqs:
        b, a = gammatone(freq=float(f), ftype="fir", order=4, fs=sr_int1)
        sub = np.real(lfilter(b, a, audio_lp))
        envs.append(np.abs(sub) ** float(power))

    env = np.sum(np.asarray(envs), axis=0)

    # Resample for filtering.
    if int(sr_int2) != int(sr_int1):
        env = resample(env, int(round(len(env) * sr_int2 / sr_int1)))

    env = filter_data(env, sr_int2, float(bandpass_hz[0]), float(bandpass_hz[1]), verbose="CRITICAL")

    if int(target_sr) != int(sr_int2):
        env = resample(env, int(round(len(env) * target_sr / sr_int2)))

    return zscore(env).astype(np.float32, copy=False)

def speech_envelope(audio: np.ndarray, sr: int, low_hz: float = 1.0, high_hz: float = 8.0) -> np.ndarray:
    """Simple amplitude envelope + bandpass to match common AAD pipelines."""
    audio = np.asarray(audio).astype(float)
    analytic = hilbert(audio)
    env = np.abs(analytic)

    b, a = butter(4, [low_hz/(sr/2), high_hz/(sr/2)], btype="band")
    env = filtfilt(b, a, env)
    return env
