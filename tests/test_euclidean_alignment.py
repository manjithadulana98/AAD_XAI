"""Per-subject Euclidean alignment: whitening, subject independence, rank-deficient
(average-referenced) EEG, that nothing but the EEG changes, and alignment from separate
calibration EEG."""
from __future__ import annotations

import numpy as np
import pytest

from aad_xai.data.alignment import (calibration_eeg, euclidean_align, euclidean_align_to, euclidean_reference,
                                    inverse_sqrtm)
from aad_xai.data.base import Trial


def _subject(sid, mixing, n_trials=6, n_times=2000, seed=0, avg_ref=False):
    rng = np.random.default_rng(seed)
    trials = []
    for k in range(n_trials):
        x = mixing @ rng.standard_normal((mixing.shape[1], n_times))            # (C, T)
        if avg_ref:
            x = x - x.mean(axis=0, keepdims=True)
        trials.append(Trial(eeg=x.astype(np.float32), sfreq=64.0, label=k % 2, subject_id=sid,
                            trial_id=f"{sid}_T{k:03d}", group_id=f"story{k}",
                            audio_a=rng.standard_normal(n_times).astype(np.float32),
                            audio_b=rng.standard_normal(n_times).astype(np.float32), audio_sr=64))
    return trials


def _mean_cov(trials):
    return np.mean([np.cov(t.eeg.astype(np.float64), bias=True) for t in trials], axis=0)


def test_inverse_sqrtm():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    R = A @ A.T + 0.5 * np.eye(5)
    W = inverse_sqrtm(R)
    np.testing.assert_allclose(W, W.T, atol=1e-12)
    np.testing.assert_allclose(W @ R @ W, np.eye(5), atol=1e-9)
    with pytest.raises(ValueError):
        inverse_sqrtm(np.diag([1.0, 0.0, 2.0]))


def test_mean_covariance_becomes_identity_for_each_subject():
    rng = np.random.default_rng(2)
    trials = _subject("S1", rng.standard_normal((8, 8)) * 3, seed=3) + \
        _subject("S2", np.diag(np.linspace(0.1, 50, 8)), seed=4)
    aligned = euclidean_align(trials, shrinkage=0.0)
    for sid in ("S1", "S2"):
        cov = _mean_cov([t for t in aligned if t.subject_id == sid])
        np.testing.assert_allclose(cov, np.eye(8), atol=1e-4)


def test_invariant_to_subject_mixing():
    """Same latent sources, different subject-specific mixing -> same aligned covariance."""
    rng = np.random.default_rng(5)
    a = euclidean_align(_subject("S1", rng.standard_normal((6, 6)), seed=7), shrinkage=0.0)
    b = euclidean_align(_subject("S1", 10 * rng.standard_normal((6, 6)), seed=7), shrinkage=0.0)
    np.testing.assert_allclose(_mean_cov(a), _mean_cov(b), atol=1e-4)


def test_subject_reference_ignores_other_subjects():
    rng = np.random.default_rng(6)
    s1 = _subject("S1", rng.standard_normal((5, 5)), seed=8)
    s2 = _subject("S2", rng.standard_normal((5, 5)), seed=9)
    alone = euclidean_align(s1)
    together = euclidean_align(s2 + s1)[len(s2):]
    for x, y in zip(alone, together):
        np.testing.assert_array_equal(x.eeg, y.eeg)


def test_average_referenced_eeg_stays_average_referenced():
    rng = np.random.default_rng(10)
    C = 8
    trials = _subject("S1", rng.standard_normal((C, C)), seed=11, avg_ref=True)
    assert np.linalg.matrix_rank(_mean_cov(trials), tol=1e-6) == C - 1
    with pytest.raises(ValueError):
        inverse_sqrtm(euclidean_reference([t.eeg for t in trials], shrinkage=0.0) - 1e-9 * np.eye(C))
    aligned = euclidean_align(trials, shrinkage=1e-3)
    for t in aligned:
        assert np.isfinite(t.eeg).all()
        assert np.abs(t.eeg.mean(axis=0)).max() < 1e-3                 # common mode still ~0
    # Whitened on the C-1 dimensional complement of the common mode.
    P = np.eye(C) - np.ones((C, C)) / C
    np.testing.assert_allclose(_mean_cov(aligned), P, atol=5e-3)


def test_only_eeg_changes_and_order_is_kept():
    rng = np.random.default_rng(12)
    trials = _subject("S2", rng.standard_normal((4, 4)), seed=13) + _subject("S1", rng.standard_normal((4, 4)), seed=14)
    aligned = euclidean_align(trials)
    assert [t.trial_id for t in aligned] == [t.trial_id for t in trials]
    for t, u in zip(trials, aligned):
        assert u.eeg.shape == t.eeg.shape and u.eeg.dtype == t.eeg.dtype
        assert (u.label, u.subject_id, u.group_id, u.sfreq, u.audio_sr) == \
            (t.label, t.subject_id, t.group_id, t.sfreq, t.audio_sr)
        assert u.audio_a is t.audio_a and u.audio_b is t.audio_b
        assert not np.array_equal(u.eeg, t.eeg)


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        euclidean_reference([])


def test_euclidean_align_unchanged_by_refactor():
    """Bit-identical to the original per-subject loop."""
    rng = np.random.default_rng(20)
    trials = _subject("S2", rng.standard_normal((6, 6)), seed=21) + _subject("S1", rng.standard_normal((6, 6)), seed=22)
    for t, u in zip(trials, euclidean_align(trials)):
        W = inverse_sqrtm(euclidean_reference([x.eeg for x in trials if x.subject_id == t.subject_id], 1e-3))
        np.testing.assert_array_equal(u.eeg, (W @ t.eeg.astype(np.float64)).astype(t.eeg.dtype))


def test_calibration_eeg_takes_the_first_seconds_in_order():
    rng = np.random.default_rng(23)
    trials = _subject("S1", rng.standard_normal((4, 4)), n_trials=3, n_times=640, seed=24)   # 10 s each at 64 Hz
    trials += _subject("S2", rng.standard_normal((4, 4)), n_trials=2, n_times=640, seed=25)
    cal = calibration_eeg(trials, seconds=15)
    assert set(cal) == {"S1", "S2"} and cal["S1"].shape == (4, 960) and cal["S2"].shape == (4, 960)
    centred = [t.eeg.astype(np.float64) - t.eeg.astype(np.float64).mean(axis=1, keepdims=True) for t in trials[:2]]
    np.testing.assert_array_equal(cal["S1"][:, :640], centred[0])
    np.testing.assert_array_equal(cal["S1"][:, 640:], centred[1][:, :320])          # second trial cut
    assert calibration_eeg(trials)["S1"].shape == (4, 1920)                        # None = everything
    assert calibration_eeg(trials, seconds=5)["S1"].shape == (4, 320)
    with pytest.raises(ValueError):
        calibration_eeg(trials, seconds=25)                                        # S2 has only 20 s


def test_align_to_whitens_the_calibration_not_the_trials():
    rng = np.random.default_rng(26)
    M = rng.standard_normal((6, 6))
    calib = _subject("S1", M, n_trials=2, seed=27)
    scored = _subject("S1", M @ np.diag(np.linspace(1, 4, 6)), seed=28)             # covariance differs
    cal = calibration_eeg(calib)
    aligned = euclidean_align_to(scored, cal, shrinkage=0.0)
    W = inverse_sqrtm(euclidean_reference([cal["S1"]], 0.0))
    np.testing.assert_allclose(W @ (cal["S1"] @ cal["S1"].T / cal["S1"].shape[1]) @ W, np.eye(6), atol=1e-8)
    for t, u in zip(scored, aligned):
        np.testing.assert_allclose(u.eeg, (W @ t.eeg.astype(np.float64)).astype(np.float32), rtol=1e-6)
    assert np.abs(_mean_cov(aligned) - np.eye(6)).max() > 0.5                     # scored data not whitened


def test_align_to_with_the_trials_as_calibration_equals_euclidean_align():
    """Equal-length trials: the time-weighted calibration reference equals the trial-mean one."""
    rng = np.random.default_rng(29)
    trials = _subject("S1", rng.standard_normal((5, 5)), seed=30) + _subject("S2", rng.standard_normal((5, 5)), seed=31)
    a = euclidean_align(trials)
    b = euclidean_align_to(trials, calibration_eeg(trials))
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.eeg, y.eeg, rtol=1e-4, atol=1e-5)
        assert (x.trial_id, x.label, x.audio_a is y.audio_a) == (y.trial_id, y.label, True)


def test_align_to_needs_every_subject():
    rng = np.random.default_rng(32)
    trials = _subject("S1", rng.standard_normal((4, 4)), seed=33) + _subject("S2", rng.standard_normal((4, 4)), seed=34)
    with pytest.raises(KeyError):
        euclidean_align_to(trials, calibration_eeg(trials[:6]))
