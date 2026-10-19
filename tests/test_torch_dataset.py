"""Tests for WindowedEEGAudioDataset's standardize_eeg option."""
from __future__ import annotations

import numpy as np
import torch

from aad_xai.data.base import Trial
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset


def _make_trial(seed: int, scale: float, n_channels: int = 4, n_times: int = 320, sfreq: float = 64.0) -> Trial:
    rng = np.random.default_rng(seed)
    eeg = (rng.standard_normal((n_channels, n_times)) * scale).astype(np.float32)
    audio_a = rng.standard_normal(n_times).astype(np.float32)
    audio_b = rng.standard_normal(n_times).astype(np.float32)
    return Trial(
        eeg=eeg, sfreq=sfreq, label=0, subject_id="S1", trial_id="S1_T0", group_id="G1",
        audio_a=audio_a, audio_b=audio_b, audio_sr=int(sfreq),
    )


class TestStandardizeEeg:
    def test_default_is_off_and_preserves_raw_scale(self):
        trial = _make_trial(seed=0, scale=50.0)
        ds = WindowedEEGAudioDataset([trial], window_s=1.0, overlap_s=0.0)
        eeg, _, _ = ds[0]
        # Raw amplitude preserved -- std should be close to the injected scale, not ~1.
        assert eeg.std().item() > 10.0

    def test_enabled_produces_zero_mean_unit_std_per_channel(self):
        trial = _make_trial(seed=0, scale=50.0)
        ds = WindowedEEGAudioDataset([trial], window_s=1.0, overlap_s=0.0, standardize_eeg=True)
        eeg, _, _ = ds[0]
        # Per-channel (dim 0), over the window's own time axis (dim 1).
        means = eeg.mean(dim=1)
        stds = eeg.std(dim=1, unbiased=False)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)
        assert torch.allclose(stds, torch.ones_like(stds), atol=1e-2)

    def test_standardization_removes_cross_subject_scale_difference(self):
        """The whole point of the fix: two trials with very different raw
        amplitude scales should look statistically identical after
        standardization, whereas without it their scales remain distinct."""
        trial_small = _make_trial(seed=1, scale=1.0)
        trial_large = _make_trial(seed=1, scale=100.0)  # same seed -> same underlying pattern, different scale

        # window_s=5.0 with n_times=320 @ 64Hz -> exactly 1 window per trial,
        # so ds[0]/ds[1] map cleanly to trial_small/trial_large respectively
        # (otherwise each trial yields multiple windows and both indices
        # would land inside the same first trial).
        ds_raw = WindowedEEGAudioDataset([trial_small, trial_large], window_s=5.0, overlap_s=0.0)
        eeg_small_raw, _, _ = ds_raw[0]
        eeg_large_raw, _, _ = ds_raw[1]
        assert not torch.allclose(eeg_small_raw.std(), eeg_large_raw.std(), atol=0.5)

        ds_std = WindowedEEGAudioDataset([trial_small, trial_large], window_s=5.0, overlap_s=0.0, standardize_eeg=True)
        eeg_small_std, _, _ = ds_std[0]
        eeg_large_std, _, _ = ds_std[1]
        assert torch.allclose(eeg_small_std, eeg_large_std, atol=1e-3)

    def test_env_untouched_by_standardize_eeg(self):
        """standardize_eeg must only affect the eeg tensor, not env."""
        trial = _make_trial(seed=0, scale=50.0)
        ds_raw = WindowedEEGAudioDataset([trial], window_s=1.0, overlap_s=0.0)
        ds_std = WindowedEEGAudioDataset([trial], window_s=1.0, overlap_s=0.0, standardize_eeg=True)
        _, env_raw, _ = ds_raw[0]
        _, env_std, _ = ds_std[0]
        assert torch.equal(env_raw, env_std)
