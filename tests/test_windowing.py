"""Tests for windowing and cross-split overlap detection."""
import pytest
import numpy as np

from aad_xai.data.windowing import make_windows, assert_no_cross_split_overlap, WindowIndex


class TestMakeWindows:
    def test_basic_no_overlap(self):
        """Non-overlapping windows tile the trial correctly."""
        wins = make_windows(
            n_times=256, sfreq=64.0, window_s=1.0, overlap_s=0.0,
            label=1, subject_id="S0", trial_id="T0",
        )
        assert len(wins) == 4  # 256 / 64 = 4 windows of length 1s
        for w in wins:
            assert w.stop - w.start == 64

    def test_overlap_within_trial(self):
        """Overlapping windows produce more windows than non-overlapping."""
        wins_no = make_windows(
            n_times=256, sfreq=64.0, window_s=1.0, overlap_s=0.0,
            label=0, subject_id="S0", trial_id="T0",
        )
        wins_ov = make_windows(
            n_times=256, sfreq=64.0, window_s=1.0, overlap_s=0.5,
            label=0, subject_id="S0", trial_id="T0",
        )
        assert len(wins_ov) > len(wins_no)

    def test_window_longer_than_trial(self):
        """If the window is longer than the trial, no windows are produced."""
        wins = make_windows(
            n_times=32, sfreq=64.0, window_s=1.0, overlap_s=0.0,
            label=0, subject_id="S0", trial_id="T0",
        )
        assert len(wins) == 0

    def test_window_exact_fit(self):
        """Trial length exactly equal to one window → 1 window."""
        wins = make_windows(
            n_times=64, sfreq=64.0, window_s=1.0, overlap_s=0.0,
            label=1, subject_id="S0", trial_id="T0",
        )
        assert len(wins) == 1
        assert wins[0].start == 0
        assert wins[0].stop == 64

    def test_no_window_exceeds_trial(self):
        """No window should ever go past n_times."""
        wins = make_windows(
            n_times=100, sfreq=64.0, window_s=1.0, overlap_s=0.0,
            label=0, subject_id="S0", trial_id="T0",
        )
        for w in wins:
            assert w.stop <= 100


class TestCrossSplitOverlap:
    def test_no_overlap_detected(self):
        """Non-overlapping window sets should pass."""
        train_w = [WindowIndex(0, 64, 1, "S0", "T0")]
        val_w = [WindowIndex(0, 64, 0, "S1", "T1")]
        test_w = [WindowIndex(0, 64, 1, "S2", "T2")]
        assert_no_cross_split_overlap(train_w, val_w, test_w)  # should not raise

    def test_overlap_detected(self):
        """Same (subject, trial, start, stop) in two splits must raise."""
        shared = WindowIndex(0, 64, 1, "S0", "T0")
        train_w = [shared]
        val_w = [shared]
        test_w = []
        with pytest.raises(AssertionError, match="Cross-split window overlap"):
            assert_no_cross_split_overlap(train_w, val_w, test_w)
