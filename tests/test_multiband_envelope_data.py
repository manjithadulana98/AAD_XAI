"""Multi-band (K, T) speech envelopes through the DTU length matching and the
windowed EEG+audio dataset; 1-D envelopes must behave exactly as before."""
from __future__ import annotations

import numpy as np
import pytest

from aad_xai.data.base import Trial
from aad_xai.data.dtu_dataset import _match_length
from aad_xai.data.torch_dataset import WindowedEEGAudioDataset


class TestMatchLength:
    def test_1d_unchanged(self):
        env = np.arange(10, dtype=np.float32)
        np.testing.assert_array_equal(_match_length(env, 6), env[:6])
        np.testing.assert_array_equal(_match_length(env, 12), np.r_[env, 0, 0])

    def test_2d_matches_time_axis(self):
        env = np.arange(30, dtype=np.float32).reshape(3, 10)
        assert _match_length(env, 6).shape == (3, 6)
        padded = _match_length(env, 12)
        assert padded.shape == (3, 12)
        np.testing.assert_array_equal(padded[:, :10], env)
        assert not padded[:, 10:].any()


def _trial(audio_shape, n_eeg=320, sfreq=64.0, seed=0):
    rng = np.random.default_rng(seed)
    return Trial(eeg=rng.standard_normal((4, n_eeg)).astype(np.float32), sfreq=sfreq, label=1,
                 subject_id="S1", trial_id="S1_T000", group_id="S1_x.wav",
                 audio_a=rng.standard_normal(audio_shape).astype(np.float32),
                 audio_b=rng.standard_normal(audio_shape).astype(np.float32), audio_sr=int(sfreq))


class TestWindowedMultiBand:
    def test_single_band_shape_unchanged(self):
        ds = WindowedEEGAudioDataset([_trial((320,))], window_s=1.0)
        eeg, env, label = ds[0]
        assert eeg.shape == (4, 64) and env.shape == (2, 64) and label == 1

    @pytest.mark.parametrize("k", [1, 15])
    def test_multi_band_windows(self, k):
        t = _trial((k, 300))  # envelope shorter than EEG: windows follow the shorter one
        ds = WindowedEEGAudioDataset([t], window_s=1.0)
        assert len(ds) == 300 // 64
        eeg, env, _ = ds[2]
        assert env.shape == (2, k, 64)
        np.testing.assert_array_equal(env[0].numpy(), t.audio_a[:, 128:192])
        np.testing.assert_array_equal(env[1].numpy(), t.audio_b[:, 128:192])
