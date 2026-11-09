"""DTU ocular derivations and EOG regression (aad_xai.data.ocular)."""
from __future__ import annotations

import numpy as np

from aad_xai.config import PreprocessConfig
from aad_xai.data.ocular import DTU_EXG, N_SCALP, dtu_derivations, fit_eog_regression, regress_out
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
