"""Tests for the DTU subject-group x story-group-out CV split."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from aad_xai.data.base import Trial
from aad_xai.data.cv_splits import (
    CV_STRATEGIES,
    CVFold,
    assert_story_disjoint_fold_integrity,
    dtu_content_group,
    subject_group_story_out,
)

# (male story, number of segments) as in DTU; content group in comments.
_STORIES = [
    ("aske_story1", 3),  # 1
    ("aske_story2", 2),  # 1
    ("aske_story4", 3),  # 2
    ("aske_story6", 2),  # 2
    ("aske_story3", 3),  # 3
    ("aske_story5", 2),  # 4
]


def _make_trials(n_subjects: int = 18, n_single_talker: int = 2) -> list[Trial]:
    """DTU-shaped trials: every subject hears every segment once, plus a few
    single-talker ``dss.wav`` trials (no audio, as in the real loader)."""
    rng = np.random.default_rng(0)
    trials = []
    for si in range(n_subjects):
        sid = f"S{si + 1}"
        wavs = [f"{story}_trial_{k + 1}.wav" for story, n in _STORIES for k in range(n)]
        wavs += ["dss.wav"] * n_single_talker
        for ti, wav in enumerate(wavs):
            two_talker = wav != "dss.wav"
            trials.append(Trial(
                eeg=np.zeros((2, 8), dtype=np.float32), sfreq=4.0, label=int(rng.integers(2)),
                subject_id=sid, trial_id=f"{sid}_T{ti:03d}", group_id=f"{sid}_{wav}",
                audio_a=np.zeros(8, dtype=np.float32) if two_talker else None,
                audio_b=np.zeros(8, dtype=np.float32) if two_talker else None,
                audio_sr=4,
            ))
    return trials


class TestDtuContentGroup:
    @pytest.mark.parametrize("wav,group", [
        ("aske_story1_trial_3.wav", 1), ("aske_story2_trial_1.wav", 1),
        ("aske_story4_trial_10.wav", 2), ("aske_story6_trial_2.wav", 2),
        ("aske_story3_trial_13.wav", 3), ("aske_story5_trial_6.wav", 4),
        ("dss.wav", None),
    ])
    def test_mapping(self, wav, group):
        t = Trial(eeg=np.zeros((1, 1)), sfreq=1.0, label=0, subject_id="S12",
                  trial_id="S12_T000", group_id=f"S12_{wav}")
        assert dtu_content_group(t) == group


class TestSubjectGroupStoryOut:
    def test_registered(self):
        assert CV_STRATEGIES["dtu_subject_group_story_out"] is subject_group_story_out

    def test_fold_count_and_integrity(self):
        trials = _make_trials()
        folds = list(subject_group_story_out(trials, seed=42))
        assert len(folds) == 6 * 3
        for f in folds:
            assert f.train_idx and f.val_idx and f.test_idx
            assert len(f.meta["test_subjects"]) == 3
            assert len(f.meta["val_subjects"]) == 2
            assert_story_disjoint_fold_integrity(trials, f)  # should not raise

    def test_every_two_talker_trial_tested_exactly_once(self):
        trials = _make_trials()
        counts = Counter(i for f in subject_group_story_out(trials, seed=42) for i in f.test_idx)
        two_talker = {i for i, t in enumerate(trials) if dtu_content_group(t) is not None}
        assert set(counts) == two_talker
        assert set(counts.values()) == {1}

    def test_single_talker_trials_never_used(self):
        trials = _make_trials()
        single = {i for i, t in enumerate(trials) if dtu_content_group(t) is None}
        for f in subject_group_story_out(trials, seed=42):
            assert single.isdisjoint(f.train_idx + f.val_idx + f.test_idx)

    def test_train_excludes_held_out_stories_and_keeps_the_rest(self):
        trials = _make_trials()
        for f in subject_group_story_out(trials, seed=42):
            held = set(f.meta["held_out_content_groups"])
            train_groups = {dtu_content_group(trials[i]) for i in f.train_idx}
            assert train_groups == {1, 2, 3, 4} - held
            assert {dtu_content_group(trials[i]) for i in f.test_idx} == held
            assert {dtu_content_group(trials[i]) for i in f.val_idx} == held

    def test_deterministic_with_seed(self):
        trials = _make_trials()
        a = [(f.train_idx, f.val_idx, f.test_idx) for f in subject_group_story_out(trials, seed=3)]
        b = [(f.train_idx, f.val_idx, f.test_idx) for f in subject_group_story_out(trials, seed=3)]
        assert a == b

    def test_integrity_check_catches_story_leak(self):
        trials = _make_trials()
        f = next(iter(subject_group_story_out(trials, seed=42)))
        held = set(f.meta["held_out_content_groups"])
        excluded = set(f.meta["test_subjects"]) | set(f.meta["val_subjects"])
        leak = next(i for i, t in enumerate(trials)
                    if t.subject_id not in excluded and dtu_content_group(t) in held)
        corrupted = CVFold(f.fold_id, f.train_idx + [leak], f.val_idx, f.test_idx, f.meta)
        with pytest.raises(AssertionError, match="content-group overlap"):
            assert_story_disjoint_fold_integrity(trials, corrupted)

    def test_integrity_check_catches_subject_leak(self):
        trials = _make_trials()
        f = next(iter(subject_group_story_out(trials, seed=42)))
        test_subj = f.meta["test_subjects"][0]
        held = set(f.meta["held_out_content_groups"])
        leak = next(i for i, t in enumerate(trials)
                    if t.subject_id == test_subj and dtu_content_group(t) not in (held | {None}))
        corrupted = CVFold(f.fold_id, f.train_idx + [leak], f.val_idx, f.test_idx, f.meta)
        with pytest.raises(AssertionError, match="subject overlap"):
            assert_story_disjoint_fold_integrity(trials, corrupted)
