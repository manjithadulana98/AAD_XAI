"""Tests for the SyntheticDataset and full-pipeline smoke-test."""
import pytest
import numpy as np

from aad_xai.data.synthetic_dataset import SyntheticDataset
from aad_xai.data.splits import Record, subject_independent_split, assert_no_leakage
from aad_xai.data.windowing import assert_no_cross_split_overlap
from aad_xai.data.torch_dataset import WindowedEEGDataset


class TestSyntheticDataset:
    def test_yields_correct_number_of_trials(self):
        ds = SyntheticDataset(n_subjects=4, trials_per_subject=5, seed=0)
        trials = list(ds.trials())
        assert len(trials) == 20

    def test_unique_subject_ids(self):
        ds = SyntheticDataset(n_subjects=6, trials_per_subject=3, seed=0)
        trials = list(ds.trials())
        sids = {t.subject_id for t in trials}
        assert len(sids) == 6

    def test_unique_group_ids(self):
        ds = SyntheticDataset(n_subjects=4, trials_per_subject=5, seed=0)
        trials = list(ds.trials())
        gids = [t.group_id for t in trials]
        assert len(set(gids)) == len(gids), "group_ids must be unique per trial"

    def test_trial_shapes(self):
        ds = SyntheticDataset(n_channels=32, duration_s=5.0, sfreq=64.0, seed=0)
        trial = next(ds.trials())
        assert trial.eeg.shape == (32, 320)
        assert trial.sfreq == 64.0

    def test_has_audio(self):
        ds = SyntheticDataset(seed=0)
        trial = next(ds.trials())
        assert trial.audio_a is not None
        assert trial.audio_b is not None


class TestFullPipeline:
    """Smoke-test: SyntheticDataset → split → windowing → leakage checks."""

    def test_pipeline_no_leakage(self):
        ds = SyntheticDataset(n_subjects=8, trials_per_subject=10, seed=42)
        trials = list(ds.trials())
        subject_ids = sorted({t.subject_id for t in trials})

        split = subject_independent_split(subject_ids, train_frac=0.6, val_frac=0.2, seed=42)
        records = [Record(subject_id=t.subject_id, trial_id=t.trial_id, group_id=t.group_id) for t in trials]
        assert_no_leakage(records, split)

        s_tr, s_v, s_te = set(split.train), set(split.val), set(split.test)
        ds_tr = WindowedEEGDataset([t for t in trials if t.subject_id in s_tr], window_s=1.0)
        ds_v = WindowedEEGDataset([t for t in trials if t.subject_id in s_v], window_s=1.0)
        ds_te = WindowedEEGDataset([t for t in trials if t.subject_id in s_te], window_s=1.0)

        assert len(ds_tr) > 0
        assert len(ds_v) > 0
        assert len(ds_te) > 0

        assert_no_cross_split_overlap(
            ds_tr.window_indices, ds_v.window_indices, ds_te.window_indices,
        )

    def test_windowed_dataset_getitem(self):
        ds = SyntheticDataset(n_subjects=4, trials_per_subject=5, sfreq=64.0, duration_s=5.0, seed=0)
        trials = list(ds.trials())
        wds = WindowedEEGDataset(trials, window_s=1.0)
        x, label = wds[0]
        assert x.shape == (64, 64)  # (n_channels, win_samples)
        assert label in (0, 1)


class TestReproducibility:
    def test_same_seed_same_data(self):
        ds1 = SyntheticDataset(n_subjects=4, trials_per_subject=3, seed=0)
        ds2 = SyntheticDataset(n_subjects=4, trials_per_subject=3, seed=0)
        t1 = list(ds1.trials())
        t2 = list(ds2.trials())
        for a, b in zip(t1, t2):
            np.testing.assert_array_equal(a.eeg, b.eeg)
            assert a.label == b.label
