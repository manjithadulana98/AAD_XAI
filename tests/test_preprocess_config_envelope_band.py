"""Tests for PreprocessConfig.envelope_bandpass_hz: widening the EEG band must
not widen the speech-envelope band unless asked to."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from aad_xai.config import PreprocessConfig
from aad_xai.data.dtu_dataset import DTUDataset
from tests.test_dtu_dataset_preprocess import _setup_dataset


class TestEnvelopeBandProperty:
    def test_default_follows_eeg_band(self):
        assert PreprocessConfig().envelope_band == (1.0, 8.0)
        assert PreprocessConfig(bandpass_hz=(1.0, 31.0)).envelope_band == (1.0, 31.0)

    def test_override(self):
        cfg = PreprocessConfig(bandpass_hz=(1.0, 31.0), envelope_bandpass_hz=(1.0, 8.0))
        assert cfg.bandpass_hz == (1.0, 31.0)
        assert cfg.envelope_band == (1.0, 8.0)


class TestDtuDatasetUsesEnvelopeBand:
    @pytest.fixture
    def spies(self, monkeypatch):
        """Record the band passed to the EEG filter and to the envelope loader."""
        import aad_xai.data.dtu_dataset as dtu_mod

        # Stand-in for aad_xai.data.preprocessing (imports mne, which these
        # tests do not need); DTUDataset imports it lazily inside the loop.
        pp_mod = types.ModuleType("aad_xai.data.preprocessing")
        monkeypatch.setitem(sys.modules, "aad_xai.data.preprocessing", pp_mod)
        calls = {"eeg": [], "env": []}

        def fake_preprocess_eeg(eeg, sfreq_in, cfg):
            calls["eeg"].append(cfg.bandpass_hz)
            n_out = round(eeg.shape[1] * cfg.sfreq_out / sfreq_in)
            return np.zeros((eeg.shape[0], n_out), dtype=np.float32), float(cfg.sfreq_out)

        def fake_load_env(wav_path, target_sfreq=64.0, low_hz=1.0, high_hz=8.0, method="hilbert"):
            calls["env"].append((low_hz, high_hz))
            calls.setdefault("method", []).append(method)
            return np.arange(int(10 * target_sfreq), dtype=np.float32)

        pp_mod.preprocess_eeg = fake_preprocess_eeg
        monkeypatch.setattr(dtu_mod, "_load_wav_envelope", fake_load_env)
        return calls

    def _load(self, tmp_path, monkeypatch, cfg, n_subjects=1, **kw):
        eeg_dir, _ = _setup_dataset(tmp_path, monkeypatch, sfreq=512.0, n_samples=4096, onset=100, offset=1124)
        for s in range(2, n_subjects + 1):
            (eeg_dir / f"S{s}.mat").write_bytes(b"")  # loadmat is faked; same trial per subject
        return list(DTUDataset(root=tmp_path, preprocess=cfg, **kw).trials())

    def test_envelope_method_is_forwarded(self, tmp_path, monkeypatch, spies):
        self._load(tmp_path, monkeypatch, PreprocessConfig(reref=None), envelope_method="powerlaw_subbands")
        assert spies["method"] == ["powerlaw_subbands", "powerlaw_subbands"]

    def test_envelopes_cached_across_subjects(self, tmp_path, monkeypatch, spies):
        trials = self._load(tmp_path, monkeypatch, PreprocessConfig(reref=None), n_subjects=3)
        assert [t.subject_id for t in trials] == ["S1", "S2", "S3"]
        assert len(spies["env"]) == 2  # one call per WAV, not per subject
        np.testing.assert_array_equal(trials[0].audio_a, trials[2].audio_a)
        trials[0].audio_a[:] = -1.0  # mutating one trial must not leak into others
        assert trials[1].audio_a[0] != -1.0

    def test_default_uses_eeg_band_for_envelope(self, tmp_path, monkeypatch, spies):
        self._load(tmp_path, monkeypatch, PreprocessConfig(reref=None))
        assert spies["eeg"] == [(1.0, 8.0)]
        assert spies["env"] == [(1.0, 8.0), (1.0, 8.0)]

    def test_wide_eeg_band_keeps_envelope_band(self, tmp_path, monkeypatch, spies):
        cfg = PreprocessConfig(bandpass_hz=(1.0, 31.0), envelope_bandpass_hz=(1.0, 8.0), reref=None)
        trials = self._load(tmp_path, monkeypatch, cfg)
        assert spies["eeg"] == [(1.0, 31.0)]
        assert spies["env"] == [(1.0, 8.0), (1.0, 8.0)]
        t = trials[0]
        assert t.audio_a.shape[0] == t.eeg.shape[1]
