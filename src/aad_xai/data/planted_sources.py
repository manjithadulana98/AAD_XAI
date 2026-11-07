"""Synthetic two-talker AAD data with a planted, known explanation ground truth.

Level 3 of the interpretability ladder: real EEG has no ground truth for explanations,
so this generator plants one, and explanation methods are scored on whether they
recover it.

Every simulated subject hears the same two-talker segments (as in DTU), and trials are
named like DTU segments, so ``subject_group_story_out`` applies unchanged. A subject's
EEG on one trial is

    background  1/f activity, volume-conducted over the real electrode positions, plus
                posterior alpha; average-referenced
  + planted     attended envelope -> TRF with a subject-specific peak lag -> band-pass
                to the planted band, on a chosen electrode set at a set SNR
  + decoy       (env_A + env_B) / sqrt(2) -> the same TRF and band, on one electrode at
                a high SNR: stimulus-driven, but blind to which talker is attended

The planted and decoy terms are added after re-referencing, so every other electrode
carries no envelope information at all. SNR is the ratio of planted (or decoy) power to
the background power of that electrode on that trial, over the whole 1-31 Hz band.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, fftconvolve, sosfiltfilt

from .base import Trial

MONTAGE_PATH = Path(__file__).resolve().parents[3] / "config" / "aadnet_dtu_channel_montage.csv"
# DTU male-story names; content groups 1 (story1, story2), 2 (story4, story6), 3 and 4
# (story3, story5) map onto the three story folds of subject_group_story_out.
STORIES = ("aske_story1", "aske_story2", "aske_story4", "aske_story6", "aske_story3", "aske_story5")
EEG_BAND_HZ = (1.0, 31.0)
ENVELOPE_BAND_HZ = (1.0, 8.0)


@dataclass(frozen=True)
class PlantedSpec:
    """What is planted, where, and how strongly."""
    center: str = "T7"                       # planted set = this electrode + its nearest neighbours
    n_planted: int = 6
    band_hz: tuple[float, float] = (1.0, 4.0)
    lag_s: tuple[float, float] = (0.10, 0.25)    # range of the per-subject TRF peak lag
    snr_db: Optional[float] = -15.0          # None: no planted response (null data)
    decoy: Optional[str] = "Cz"              # None: no decoy
    decoy_snr_db: float = 0.0


@dataclass(frozen=True)
class PlantedTruth:
    """Ground truth of one generated dataset."""
    names: list[str]
    planted_idx: np.ndarray                  # (n_planted,) electrode indices, sorted
    decoy_idx: Optional[int]
    band_hz: tuple[float, float]
    lags_s: dict[str, float]                 # subject -> TRF peak lag (s)

    def planted_mask(self) -> np.ndarray:
        m = np.zeros(len(self.names), dtype=bool)
        m[self.planted_idx] = True
        return m

    def planted_edges(self, edges) -> np.ndarray:
        """Boolean mask over directed edges (i, j): True iff both ends are planted."""
        s = set(int(i) for i in self.planted_idx)
        return np.array([int(i) in s and int(j) in s for i, j in edges], dtype=bool)


def load_montage(path: Path | None = None) -> tuple[list[str], np.ndarray]:
    """Electrode names (channel order) and positions (C, 3) in metres."""
    df = pd.read_csv(path or MONTAGE_PATH).sort_values("channel_index")
    return df["electrode_name"].tolist(), df[["x", "y", "z"]].to_numpy(dtype=float)


def nearest_electrodes(pos: np.ndarray, center_idx: int, n: int) -> np.ndarray:
    """Sorted indices of the n electrodes closest to center_idx (itself included)."""
    d = np.linalg.norm(pos - pos[center_idx], axis=1)
    return np.sort(np.argsort(d, kind="stable")[:n])


def bandpass(x: np.ndarray, sfreq: float, band: tuple[float, float], order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def colored_noise(rng: np.random.Generator, shape: tuple[int, ...], sfreq: float,
                  exponent: float = 1.0, band: tuple[float, float] = EEG_BAND_HZ) -> np.ndarray:
    """Gaussian noise with power ~ 1 / f**exponent inside band and none outside;
    unit variance along the last axis. shape (..., T)."""
    n = shape[-1]
    f = np.fft.rfftfreq(n, 1.0 / sfreq)
    amp = np.zeros_like(f)
    inside = (f >= band[0]) & (f <= band[1])
    amp[inside] = f[inside] ** (-exponent / 2.0)
    spec = rng.standard_normal(shape[:-1] + (len(f),)) + 1j * rng.standard_normal(shape[:-1] + (len(f),))
    x = np.fft.irfft(spec * amp, n=n, axis=-1)
    return x / x.std(axis=-1, keepdims=True)


def background_eeg(rng: np.random.Generator, pos: np.ndarray, n_times: int, sfreq: float, *,
                   length_scale_m: float = 0.04, alpha_rel: float = 0.5, alpha_hz: float = 10.0,
                   alpha_center_idx: Optional[int] = None, gains: Optional[np.ndarray] = None) -> np.ndarray:
    """(C, T) background: 1/f sources mixed by an exponential distance kernel (volume
    conduction), plus a posterior alpha generator, per-electrode gains, average reference."""
    C = len(pos)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)                     # (C, C)
    L = np.linalg.cholesky(np.exp(-d / length_scale_m) + 1e-6 * np.eye(C))
    x = L @ colored_noise(rng, (C, n_times), sfreq, 1.0)                     # (C, T)
    if alpha_center_idx is not None and alpha_rel > 0:
        a = colored_noise(rng, (n_times,), sfreq, 0.0, (alpha_hz - 1.5, alpha_hz + 1.5))
        w = np.exp(-d[alpha_center_idx] ** 2 / (2 * 0.05 ** 2))               # (C,) posterior topography
        x = x + np.sqrt(alpha_rel) * np.outer(w, a)
    if gains is not None:
        x = gains[:, None] * x
    return x - x.mean(axis=0, keepdims=True)


def speech_like_envelope(rng: np.random.Generator, n_times: int, sfreq: float) -> np.ndarray:
    """(T,) positive, compressed envelope with syllable- and phrase-rate fluctuations,
    band-passed like the pipeline's envelope (1-8 Hz) and z-scored."""
    m = colored_noise(rng, (n_times,), sfreq, 0.0, (2.0, 8.0)) \
        + 0.7 * colored_noise(rng, (n_times,), sfreq, 0.0, (0.3, 2.0))
    env = np.log1p(np.exp(1.5 * m)) ** 0.6                                    # softplus, power-law compression
    env = bandpass(env, sfreq, ENVELOPE_BAND_HZ)
    return (env - env.mean()) / env.std()


def trf_kernel(peak_lag_s: float, sfreq: float) -> np.ndarray:
    """Causal gamma-like kernel h(t) = (t / tau) exp(1 - t / tau), peak at t = tau,
    length 4 tau, unit norm."""
    t = np.arange(int(round(4 * peak_lag_s * sfreq)) + 1) / sfreq
    h = (t / peak_lag_s) * np.exp(1.0 - t / peak_lag_s)
    return h / np.linalg.norm(h)


def neural_response(env: np.ndarray, kernel: np.ndarray, sfreq: float,
                    band: tuple[float, float]) -> np.ndarray:
    """(T,) envelope convolved with the TRF (causal), band-passed, z-scored."""
    r = fftconvolve(env, kernel)[: len(env)]
    r = bandpass(r, sfreq, band)
    return (r - r.mean()) / r.std()


def make_planted_trials(spec: PlantedSpec, *, n_subjects: int = 18, trials_per_story: int = 4,
                        trial_s: float = 50.0, sfreq: float = 64.0, seed: int = 0,
                        montage_path: Path | None = None) -> tuple[list[Trial], PlantedTruth]:
    """Generate the dataset. Labels are balanced within each subject and drawn
    independently of story, so no talker preference can be learned from the folds.
    Random streams are separate for envelopes, backgrounds and subject parameters, so the
    same seed with a different spec (e.g. no plant) gives identical backgrounds."""
    names, pos = load_montage(montage_path)
    C, n = len(names), int(round(trial_s * sfreq))
    planted = nearest_electrodes(pos, names.index(spec.center), spec.n_planted)
    decoy = names.index(spec.decoy) if spec.decoy else None
    if decoy is not None and decoy in planted:
        raise ValueError(f"decoy {spec.decoy} is inside the planted set")
    alpha_center = names.index("POz") if "POz" in names else None

    rng_env = np.random.default_rng([seed, 1])
    segments = [(story, k) for story in STORIES for k in range(trials_per_story)]
    envs = {seg: (speech_like_envelope(rng_env, n, sfreq), speech_like_envelope(rng_env, n, sfreq))
            for seg in segments}

    trials, lags = [], {}
    for s in range(n_subjects):
        sid = f"S{s + 1}"
        rng_sub = np.random.default_rng([seed, 2, s])
        rng_bg = np.random.default_rng([seed, 3, s])
        lag = float(rng_sub.uniform(*spec.lag_s))
        lags[sid] = lag
        kernel = trf_kernel(lag, sfreq)
        bg_kw = dict(length_scale_m=rng_sub.uniform(0.03, 0.05), alpha_rel=rng_sub.uniform(0.3, 1.0),
                     alpha_hz=rng_sub.uniform(9.0, 11.0), alpha_center_idx=alpha_center,
                     gains=rng_sub.uniform(0.8, 1.2, C))
        labels = rng_sub.permutation(np.arange(len(segments)) % 2)
        for ti, ((story, k), y) in enumerate(zip(segments, labels)):
            env_a, env_b = envs[(story, k)]
            eeg = background_eeg(rng_bg, pos, n, sfreq, **bg_kw)              # (C, T)
            p_bg = eeg.var(axis=1)                                            # (C,)
            if spec.snr_db is not None:
                r = neural_response(env_a if y == 0 else env_b, kernel, sfreq, spec.band_hz)
                eeg[planted] += np.sqrt(10 ** (spec.snr_db / 10) * p_bg[planted])[:, None] * r
            if decoy is not None:
                dcy = neural_response((env_a + env_b) / np.sqrt(2), kernel, sfreq, spec.band_hz)
                eeg[decoy] += np.sqrt(10 ** (spec.decoy_snr_db / 10) * p_bg[decoy]) * dcy
            trials.append(Trial(eeg=eeg.astype(np.float32), sfreq=sfreq, label=int(y), subject_id=sid,
                                trial_id=f"{sid}_T{ti:03d}", group_id=f"{sid}_{story}_trial_{k + 1}.wav",
                                audio_a=env_a.astype(np.float32), audio_b=env_b.astype(np.float32),
                                audio_sr=int(sfreq)))
    truth = PlantedTruth(names=names, planted_idx=planted, decoy_idx=decoy, band_hz=tuple(spec.band_hz),
                         lags_s=lags)
    return trials, truth
