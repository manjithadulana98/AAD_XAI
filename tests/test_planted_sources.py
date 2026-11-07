"""Planted-source synthetic data for the explanation ground-truth benchmark (XAI level 3)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aad_xai.data import cv_splits
from aad_xai.data import planted_sources as ps

SMALL = dict(n_subjects=6, trials_per_story=2, trial_s=20.0, sfreq=64.0)


@pytest.fixture(scope="module")
def pair():
    """The same seed with and without the plant and decoy: identical backgrounds."""
    spec = ps.PlantedSpec(center="T7", band_hz=(1.0, 4.0), snr_db=-10.0, decoy="Cz", decoy_snr_db=0.0)
    with_plant, truth = ps.make_planted_trials(spec, seed=3, **SMALL)
    null, _ = ps.make_planted_trials(dataclasses.replace(spec, snr_db=None, decoy=None), seed=3, **SMALL)
    return spec, truth, with_plant, null


def test_trials_fit_the_story_disjoint_protocol(pair):
    _, truth, trials, _ = pair
    folds = list(cv_splits.subject_group_story_out(trials, n_subject_groups=3, n_val_subjects=1, seed=0))
    assert len(folds) == 9
    for f in folds:
        cv_splits.assert_story_disjoint_fold_integrity(trials, f)
        assert f.train_idx and f.val_idx and f.test_idx
    by_seg = {}
    for t in trials:                                  # every subject hears the same segments
        seg = t.group_id.split("_", 1)[1]
        if seg in by_seg:
            np.testing.assert_array_equal(by_seg[seg][0], t.audio_a)
            np.testing.assert_array_equal(by_seg[seg][1], t.audio_b)
        else:
            by_seg[seg] = (t.audio_a, t.audio_b)
    for s in {t.subject_id for t in trials}:          # labels balanced within each subject
        assert np.mean([t.label for t in trials if t.subject_id == s]) == 0.5
    assert truth.names[truth.decoy_idx] == "Cz" and "T7" in [truth.names[i] for i in truth.planted_idx]


def test_plant_touches_only_the_planted_set_and_decoy_at_the_set_snr(pair):
    spec, truth, trials, null = pair
    others = np.ones(len(truth.names), dtype=bool)
    others[truth.planted_idx] = False
    others[truth.decoy_idx] = False
    for t, t0 in zip(trials, null):
        diff = t.eeg.astype(float) - t0.eeg
        assert np.abs(diff[others]).max() < 1e-5
        ratio = diff[truth.planted_idx].var(axis=1) / t0.eeg[truth.planted_idx].astype(float).var(axis=1)
        np.testing.assert_allclose(10 * np.log10(ratio), spec.snr_db, atol=0.05)
        dec = diff[truth.decoy_idx].var() / t0.eeg[truth.decoy_idx].astype(float).var()
        assert 10 * np.log10(dec) == pytest.approx(spec.decoy_snr_db, abs=0.05)


def test_planted_response_is_in_band_and_tracks_only_the_attended_talker(pair):
    _, truth, trials, null = pair
    sf = 64.0
    for t, t0 in list(zip(trials, null))[:8]:
        r = (t.eeg.astype(float) - t0.eeg)[truth.planted_idx[0]]
        p = np.abs(np.fft.rfft(r)) ** 2
        f = np.fft.rfftfreq(len(r), 1 / sf)
        assert p[(f >= 0.8) & (f <= 4.5)].sum() / p.sum() > 0.9            # delta band (filter skirts)
        kern = ps.trf_kernel(truth.lags_s[t.subject_id], sf)
        att, unatt = (t.audio_a, t.audio_b) if t.label == 0 else (t.audio_b, t.audio_a)
        r_att = np.corrcoef(r, ps.neural_response(att, kern, sf, truth.band_hz))[0, 1]
        r_un = np.corrcoef(r, ps.neural_response(unatt, kern, sf, truth.band_hz))[0, 1]
        assert r_att > 0.999 and abs(r_un) < 0.3


def test_decoy_is_stimulus_driven_but_attention_blind(pair):
    _, truth, trials, null = pair
    for t, t0 in list(zip(trials, null))[:8]:
        d = (t.eeg.astype(float) - t0.eeg)[truth.decoy_idx]
        kern = ps.trf_kernel(truth.lags_s[t.subject_id], 64.0)
        ca = np.corrcoef(d, ps.neural_response(t.audio_a, kern, 64.0, truth.band_hz))[0, 1]
        cb = np.corrcoef(d, ps.neural_response(t.audio_b, kern, 64.0, truth.band_hz))[0, 1]
        assert min(ca, cb) > 0.4 and abs(ca - cb) < 0.3


def test_background_is_volume_conducted_1_over_f_and_average_referenced(pair):
    _, truth, _, null = pair
    names, pos = ps.load_montage()
    x = np.concatenate([t.eeg for t in null[:6]], axis=1).astype(float)     # (C, T)
    assert np.abs(x.sum(axis=0)).max() < 1e-3                               # average reference
    c = np.corrcoef(x)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    near = c[(d > 0) & (d < 0.03)].mean()
    far = c[d > 0.12].mean()
    assert near > far + 0.2
    p = np.mean([np.abs(np.fft.rfft(t.eeg.astype(float), axis=1)) ** 2 for t in null[:6]], axis=0)  # per trial
    f = np.fft.rfftfreq(null[0].eeg.shape[1], 1 / 64.0)
    assert p[:, (f >= 2) & (f <= 4)].mean() > 3 * p[:, (f >= 20) & (f <= 30)].mean()
    assert p[:, (f > 31.5)].mean() < 1e-6 * p.mean()                        # band-limited to 1-31 Hz


def test_generation_is_deterministic_and_validates_the_decoy():
    spec = ps.PlantedSpec(center="P8", band_hz=(4.0, 8.0), snr_db=-15.0, decoy="Cz")
    a, ta = ps.make_planted_trials(spec, seed=1, n_subjects=2, trials_per_story=1, trial_s=10.0)
    b, tb = ps.make_planted_trials(spec, seed=1, n_subjects=2, trials_per_story=1, trial_s=10.0)
    assert all(np.array_equal(x.eeg, y.eeg) for x, y in zip(a, b)) and ta.lags_s == tb.lags_s
    assert all(0.10 <= v <= 0.25 for v in ta.lags_s.values())
    with pytest.raises(ValueError):
        ps.make_planted_trials(dataclasses.replace(spec, decoy="P8"), n_subjects=1, trials_per_story=1,
                               trial_s=5.0)


def test_planted_edges_and_kernel_peak():
    truth = ps.PlantedTruth(names=list("abcd"), planted_idx=np.array([1, 2]), decoy_idx=3,
                            band_hz=(1.0, 4.0), lags_s={})
    assert truth.planted_edges([(1, 2), (2, 1), (0, 1), (2, 3)]).tolist() == [True, True, False, False]
    assert truth.planted_mask().tolist() == [False, True, True, False]
    k = ps.trf_kernel(0.15, 64.0)
    assert np.argmax(k) == round(0.15 * 64) and np.linalg.norm(k) == pytest.approx(1.0)
