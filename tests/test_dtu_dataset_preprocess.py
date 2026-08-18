"""Tests for DTUDataset's preprocessing/sample-rate-alignment fix.

Builds a fake `scipy.io.loadmat` return value directly in memory (matching
the exact nested-struct access pattern `_parse_subject_file` expects) rather
than round-tripping through a real `.mat` file: `scipy.io.savemat`'s
dict-to-struct conversion does not reproduce the double-object-cell wrapping
real MATLAB-authored structs have, and reproducing that wrapping exactly (to
get a genuine byte-for-byte `.mat` fixture) collides with an unrelated,
pre-existing numpy>=2.0 incompatibility in the untouched `float(...)` calls
in `_parse_subject_file`'s .mat-parsing code -- orthogonal to what this test
is actually verifying (the NEW preprocessing/envelope-alignment logic that
runs *after* those values are already extracted).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io.wavfile

from aad_xai.config import PreprocessConfig
from aad_xai.data.dtu_dataset import DTUDataset


def _cell(value):
    """A (1,1) object-dtype array wrapping `value` -- unwraps via one [0,0]."""
    c = np.empty((1, 1), dtype=object)
    c[0, 0] = value
    return c


def _struct(fields: dict) -> np.ndarray:
    """A (1,1) structured array (MATLAB scalar struct) from pre-built field values."""
    dtype = np.dtype([(k, "O") for k in fields])
    arr = np.zeros((1, 1), dtype=dtype)
    for k, v in fields.items():
        arr[0, 0][k] = v
    return arr


def _fake_loadmat_dict(
    *, sfreq: float, n_samples: int, n_channels: int, onset: int, offset: int,
    attend_mf: float, wavfile_male: str, wavfile_female: str, seed: int = 0,
) -> dict:
    """Build the exact dict `scipy.io.loadmat(squeeze_me=False)` would need
    to return for `_parse_subject_file` to extract one trial (onset:offset)
    with the given metadata. One EEG channel's data (channel 0) is set to a
    fixed sinusoid so downstream length assertions have something concrete
    to check against.
    """
    rng = np.random.default_rng(seed)
    eeg_cont = rng.standard_normal((n_samples, n_channels))

    event_dtype = np.dtype([("sample", "O"), ("value", "O")])
    event_arr = np.zeros((2, 1), dtype=event_dtype)
    event_arr[0, 0]["sample"] = np.array(float(onset))
    event_arr[0, 0]["value"] = np.array(1.0)
    event_arr[1, 0]["sample"] = np.array(float(offset))
    event_arr[1, 0]["value"] = np.array(2.0)

    data = _struct({
        "eeg": _cell(eeg_cont),
        "fsample": _cell(_struct({"eeg": np.array(sfreq)})),
        "event": _cell(_struct({"eeg": event_arr})),
    })

    expinfo_dtype = np.dtype([("attend_mf", "O"), ("wavfile_male", "O"), ("wavfile_female", "O")])
    expinfo_arr = np.zeros((1, 1), dtype=expinfo_dtype)
    expinfo_arr[0, 0]["attend_mf"] = _cell(np.array(attend_mf))
    expinfo_arr[0, 0]["wavfile_male"] = np.array([wavfile_male])
    expinfo_arr[0, 0]["wavfile_female"] = np.array([wavfile_female])

    return {"data": data, "expinfo": expinfo_arr}


def _write_wav(path: Path, duration_s: float, sr: int, freq_hz: float = 5.0) -> None:
    t = np.arange(int(duration_s * sr)) / sr
    tone = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    scipy.io.wavfile.write(str(path), sr, tone)


def _setup_dataset(tmp_path: Path, monkeypatch, *, sfreq: float, n_samples: int, onset: int, offset: int):
    eeg_dir = tmp_path / "eeg_new"
    eeg_dir.mkdir(parents=True)
    mat_path = eeg_dir / "S1.mat"
    mat_path.write_bytes(b"")  # placeholder; loadmat is monkeypatched below

    audio_dir = tmp_path / "Audio"
    audio_dir.mkdir(parents=True)
    _write_wav(audio_dir / "male1.wav", duration_s=10.0, sr=44100, freq_hz=4.0)
    _write_wav(audio_dir / "female1.wav", duration_s=10.0, sr=44100, freq_hz=6.0)

    fake_mat = _fake_loadmat_dict(
        sfreq=sfreq, n_samples=n_samples, n_channels=73, onset=onset, offset=offset,
        attend_mf=1.0, wavfile_male="male1.wav", wavfile_female="female1.wav",
    )

    import aad_xai.data.dtu_dataset as dtu_mod
    monkeypatch.setattr(dtu_mod.scipy.io, "loadmat", lambda path, **kw: fake_mat)

    return eeg_dir, audio_dir


class TestDtuDatasetPreprocessAlignment:
    def test_preprocess_none_reproduces_historical_behavior(self, tmp_path, monkeypatch):
        """preprocess=None: EEG stays at native rate; envelope hardcoded to
        64 Hz/1-8 Hz, un-truncated -- exactly today's (mismatched) behavior,
        for backward compatibility with the two existing callers."""
        sfreq = 512.0
        _setup_dataset(tmp_path, monkeypatch, sfreq=sfreq, n_samples=4096, onset=100, offset=1124)

        ds = DTUDataset(root=tmp_path, preprocess=None)
        trials = list(ds.trials())
        assert len(trials) == 1
        t = trials[0]

        assert t.sfreq == sfreq
        assert t.eeg.shape[1] == 1124 - 100  # native-rate sample count, untouched
        assert t.audio_sr == 64
        # Historical behavior: envelope length is NOT truncated/padded to match
        # the (much longer, native-rate) EEG length.
        assert t.audio_a.shape[0] != t.eeg.shape[1]

    def test_preprocess_set_aligns_eeg_and_envelope(self, tmp_path, monkeypatch):
        """preprocess set: EEG is resampled/band-limited to the config's
        target rate, and the envelope is extracted at that SAME rate/band,
        then truncated/padded to exactly match -- fixing the alignment bug."""
        sfreq = 512.0
        cfg = PreprocessConfig(sfreq_out=64, bandpass_hz=(1.0, 8.0), reref=None)
        _setup_dataset(tmp_path, monkeypatch, sfreq=sfreq, n_samples=4096, onset=100, offset=1124)

        ds = DTUDataset(root=tmp_path, preprocess=cfg)
        trials = list(ds.trials())
        assert len(trials) == 1
        t = trials[0]

        assert t.sfreq == cfg.sfreq_out
        assert t.audio_sr == cfg.sfreq_out
        # The whole point of the fix: EEG and both envelopes now share one
        # length, so index-based slicing (as TRF fold training does) never
        # misaligns or runs out of bounds.
        assert t.audio_a.shape[0] == t.eeg.shape[1]
        assert t.audio_b.shape[0] == t.eeg.shape[1]

    def test_preprocess_set_changes_eeg_sample_count_vs_native(self, tmp_path, monkeypatch):
        """Sanity check that preprocessing actually resamples (native 512 Hz
        -> configured 64 Hz), not a no-op."""
        sfreq = 512.0
        n_native = 1124 - 100
        cfg = PreprocessConfig(sfreq_out=64, bandpass_hz=(1.0, 8.0), reref=None)
        _setup_dataset(tmp_path, monkeypatch, sfreq=sfreq, n_samples=4096, onset=100, offset=1124)

        ds = DTUDataset(root=tmp_path, preprocess=cfg)
        t = next(iter(ds.trials()))

        expected_n = round(n_native * cfg.sfreq_out / sfreq)
        assert abs(t.eeg.shape[1] - expected_n) <= 2  # resampling can be off by a sample or two
