"""DTU ocular derivations and EOG regression (aad_xai.data.ocular)."""
from __future__ import annotations

import numpy as np

from aad_xai.config import PreprocessConfig
from aad_xai.data.base import Trial
from aad_xai.data.ocular import (DTU_EXG, N_SCALP, dtu_derivations, fit_eog_regression, regress_eog_trials,
                                 regress_out)
from aad_xai.data.preprocessing import preprocess_eeg


def test_derivations_follow_the_zenodo_channel_map():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((72, 500))
    d = dtu_derivations(x)
    assert d["eeg"].shape == (64, 500) and d["mastoid"].shape == (2, 500) and d["eog"].shape == (3, 500)
    np.testing.assert_allclose(d["eog"][0], x[68] - x[66])        # 1-based 69 above R - 67 below R
    np.testing.assert_allclose(d["eog"][1], x[71] - x[69])        # 72 above L - 70 below L
    np.testing.assert_allclose(d["eog"][2], x[67] - x[70])        # 68 right - 71 left
    car = x[:N_SCALP].mean(0)
    np.testing.assert_allclose(d["mastoid"], x[[64, 65]] - car)
    np.testing.assert_allclose(d["eeg"].mean(0), 0, atol=1e-12)
    assert set(DTU_EXG.values()) == set(range(64, 72))


def test_derivations_are_reference_free():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((72, 300))
    common = rng.standard_normal(300)
    a, b = dtu_derivations(x), dtu_derivations(x + common)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-12)


def test_rereferencing_after_preprocessing_matches_the_standard_pipeline():
    """preprocess (no reref) then average reference == preprocess with reref='average'
    on the 64 scalp channels (filter and resample are linear, per channel)."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((72, 512 * 6)).astype(np.float64) * 1e-5
    std, _ = preprocess_eeg(x[:64], 512.0, PreprocessConfig(bandpass_hz=(1.0, 8.0), reref="average"))
    raw, _ = preprocess_eeg(x, 512.0, PreprocessConfig(bandpass_hz=(1.0, 8.0), reref=None))
    np.testing.assert_allclose(dtu_derivations(raw)["eeg"], std, rtol=1e-6, atol=1e-12 * np.abs(std).max() + 1e-18)


def test_preprocess_does_not_modify_its_input():
    rng = np.random.default_rng(4)
    for dtype in (np.float64, np.float32):
        x = (rng.standard_normal((8, 512 * 4)) * 1e-5).astype(dtype)
        x0 = x.copy()
        preprocess_eeg(x, 512.0, PreprocessConfig(bandpass_hz=(1.0, 8.0)))
        np.testing.assert_array_equal(x, x0)


def test_regression_removes_the_ocular_part_and_keeps_the_rest():
    rng = np.random.default_rng(3)
    T = 20000
    eog = rng.standard_normal((3, T)) * np.array([[5.0], [4.0], [2.0]])
    neural = rng.standard_normal((10, T))
    mix = rng.standard_normal((3, 10))
    eeg = neural + mix.T @ eog + 0.3
    B = fit_eog_regression(eeg, eog)
    np.testing.assert_allclose(B, mix, atol=0.05)
    clean = regress_out(eeg, eog, B)
    assert clean.shape == eeg.shape
    assert np.abs(np.corrcoef(np.vstack([clean, eog]))[:10, 10:]).max() < 0.05
    assert np.corrcoef(clean[0], neural[0])[0, 1] > 0.99


def _ocular_trials(rng, subjects=("S1", "S2"), n_trials=3, T=4000, C=8):
    """Trials whose EEG = neural + subject-specific mix of the EOG; returns trials, eog, mixes, neural."""
    trials, eog, mixes, neural = [], {}, {}, {}
    for s in subjects:
        mixes[s] = rng.standard_normal((3, C)) * 3
        for k in range(n_trials):
            tid = f"{s}_t{k}"
            e = rng.standard_normal((3, T)) * 4
            n = rng.standard_normal((C, T))
            eog[tid], neural[tid] = e, n
            trials.append(Trial(eeg=(n + mixes[s].T @ e).astype(np.float32), sfreq=64.0, label=k % 2,
                                subject_id=s, trial_id=tid, group_id=f"g{k}"))
    return trials, eog, mixes, neural


def test_trial_regression_uses_each_subjects_own_weights():
    rng = np.random.default_rng(5)
    trials, eog, mixes, neural = _ocular_trials(rng)
    out = regress_eog_trials(trials, eog)
    assert [t.trial_id for t in out] == [t.trial_id for t in trials]
    for t, u in zip(trials, out):
        assert u.eeg.shape == t.eeg.shape and u.eeg.dtype == t.eeg.dtype
        assert (u.label, u.subject_id, u.group_id) == (t.label, t.subject_id, t.group_id)
        r = np.corrcoef(np.vstack([u.eeg, eog[t.trial_id]]))[:-3, -3:]                 # (C, 3)
        assert np.abs(r).max() < 0.06
        assert min(np.corrcoef(a, b)[0, 1] for a, b in zip(u.eeg, neural[t.trial_id])) > 0.99


def test_trial_regression_is_fitted_only_on_the_trials_it_is_given():
    """A subject's result does not depend on other subjects' trials, and equals a fit on
    exactly that subject's trials in the list (split isolation)."""
    rng = np.random.default_rng(6)
    trials, eog, _, _ = _ocular_trials(rng)
    s1 = [t for t in trials if t.subject_id == "S1"]
    alone = regress_eog_trials(s1, eog)
    mixed = [u for u in regress_eog_trials(trials, eog) if u.subject_id == "S1"]
    for a, b in zip(alone, mixed):
        np.testing.assert_allclose(a.eeg, b.eeg, rtol=1e-6, atol=1e-6)
    # Fitting on a subset of S1's trials gives different weights (it really uses only those).
    sub = regress_eog_trials(s1[:1], eog)[0]
    B_all = fit_eog_regression(np.concatenate([t.eeg for t in s1], 1), np.concatenate([eog[t.trial_id] for t in s1], 1))
    np.testing.assert_allclose(sub.eeg, regress_out(s1[0].eeg, eog[s1[0].trial_id],
                                                    fit_eog_regression(s1[0].eeg, eog[s1[0].trial_id])), atol=1e-5)
    assert not np.allclose(sub.eeg, regress_out(s1[0].eeg, eog[s1[0].trial_id], B_all), atol=1e-7)


def test_trial_regression_rejects_missing_or_misaligned_eog():
    rng = np.random.default_rng(7)
    trials, eog, _, _ = _ocular_trials(rng, subjects=("S1",), n_trials=1)
    import pytest
    with pytest.raises(KeyError):
        regress_eog_trials(trials, {})
    with pytest.raises(ValueError):
        regress_eog_trials(trials, {trials[0].trial_id: eog[trials[0].trial_id][:, :-1]})
