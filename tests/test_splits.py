"""Tests for leakage-safe subject-independent splits."""
import pytest
import numpy as np

from aad_xai.data.splits import subject_independent_split, Record, assert_no_leakage


def test_subject_split_no_overlap():
    """Subjects must not appear in more than one partition."""
    subs = [f"S{i}" for i in range(10)]
    sp = subject_independent_split(subs, train_frac=0.7, val_frac=0.15, seed=0)
    assert set(sp.train).isdisjoint(sp.val)
    assert set(sp.train).isdisjoint(sp.test)
    assert set(sp.val).isdisjoint(sp.test)


def test_no_group_overlap():
    """Unique group_ids per subject should pass the leakage check."""
    subs = [f"S{i}" for i in range(6)]
    sp = subject_independent_split(subs, train_frac=0.5, val_frac=0.17, seed=0)

    # unique groups per subject -> should pass
    recs = [Record(subject_id=s, trial_id="T0", group_id=f"G_{s}") for s in subs]
    assert_no_leakage(recs, sp)


def test_shared_group_detected():
    """Shared stimulus group across splits MUST be caught."""
    subs = [f"S{i}" for i in range(6)]
    sp = subject_independent_split(subs, train_frac=0.5, val_frac=0.17, seed=0)

    # Give every subject the *same* group — should trigger assertion
    recs = [Record(subject_id=s, trial_id="T0", group_id="SHARED_STORY") for s in subs]
    with pytest.raises(AssertionError, match="Shared stimulus groups"):
        assert_no_leakage(recs, sp)


def test_all_subjects_assigned():
    """Every subject must appear in exactly one partition."""
    subs = [f"S{i}" for i in range(12)]
    sp = subject_independent_split(subs, train_frac=0.7, val_frac=0.15, seed=99)
    assigned = set(sp.train) | set(sp.val) | set(sp.test)
    assert assigned == set(subs)


def test_deterministic_with_same_seed():
    """Same seed must produce the same split."""
    subs = [f"S{i}" for i in range(10)]
    sp1 = subject_independent_split(subs, train_frac=0.7, val_frac=0.15, seed=42)
    sp2 = subject_independent_split(subs, train_frac=0.7, val_frac=0.15, seed=42)
    assert sp1.train == sp2.train
    assert sp1.val == sp2.val
    assert sp1.test == sp2.test


def test_different_seed_gives_different_split():
    """Different seeds should (almost certainly) give different assignments."""
    subs = [f"S{i}" for i in range(20)]
    sp1 = subject_independent_split(subs, train_frac=0.7, val_frac=0.15, seed=0)
    sp2 = subject_independent_split(subs, train_frac=0.7, val_frac=0.15, seed=999)
    assert sp1.train != sp2.train  # extremely unlikely to match

