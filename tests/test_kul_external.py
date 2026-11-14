"""KUL external-test pieces: loader (dry envelopes, trial info), content groups and
story folds, and the content-disjoint trial-swap control."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.io.wavfile

from aad_xai.data.base import Trial
from aad_xai.data.cv_splits import (
    KUL_STORY_FOLDS,
    _content_group,
    assert_story_disjoint_fold_integrity,
    kul_story_part,
    subject_group_story_out,
)
from aad_xai.data.kul_dataset import KULeuvenDataset, _load_wav_envelope, _stimulus_fields
from aad_xai.data.negatives import partner_trial_permutation, same_stratum_other_content_partners

FS_EEG = 128
N_EEG = 3 * FS_EEG          # 3 s of EEG per trial
AUDIO_SR = 16000            # above 8 kHz: the gammatone bank reaches 4 kHz
AUDIO_S = 4.0               # audio longer than the EEG, as in KUL


# --------------------------------------------------------------------------- #
#  Fake KUL release: 1 subject, experiments 1-3 with the real design
# --------------------------------------------------------------------------- #

def _kul_trial(k, exp, part, rep, cond, ear, track, left, right, rng):
    return {"RawData": {"Channels": np.arange(1, 65, dtype=np.uint8)[:, None],
                        "EegData": rng.standard_normal((N_EEG, 64))},
            "FileHeader": {"SampleRate": float(FS_EEG)},
            "TrialID": float(k + 1), "attended_ear": ear,
            "stimuli": np.array([[left], [right]], dtype=object),
            "condition": cond, "experiment": float(exp), "part": float(part),
            "attended_track": float(track), "repetition": float(rep), "subject": "S1"}


@pytest.fixture(scope="module")
def kul_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("kul")
    stim = root / "stimuli"
    stim.mkdir()
    rng = np.random.default_rng(0)
    t = np.arange(int(AUDIO_SR * AUDIO_S)) / AUDIO_SR
    for rep in ("", "rep_"):
        for p in range(1, 5):
            for tr in (1, 2):
                for c in ("dry", "hrtf"):
                    # different amplitude modulation per file, so envelopes differ
                    am = 1 + 0.8 * np.sin(2 * np.pi * rng.uniform(1.5, 6.0) * t)
                    x = (am * rng.standard_normal(t.size) * 3000).astype(np.int16)
                    scipy.io.wavfile.write(stim / f"{rep}part{p}_track{tr}_{c}.wav", AUDIO_SR, x)
    trials, k = [], 0
    for exp in (1, 2):                       # each part twice: attend track 1, then track 2
        for p in range(1, 5):
            cond = "hrtf" if p <= 2 else "dry"
            ear = "R" if p % 2 else "L"
            att = exp
            left, right = (3 - att, att) if ear == "R" else (att, 3 - att)
            trials.append(_kul_trial(k, exp, p, 0, cond, ear, att, f"part{p}_track{left}_{cond}.wav",
                                     f"part{p}_track{right}_{cond}.wav", rng)); k += 1
    for i in range(12):                      # presentation index 1-12 cycles rep_part1..4
        p = i % 4 + 1
        cond = "hrtf" if p <= 2 else "dry"
        ear = "R" if p % 2 else "L"
        left, right = (2, 1) if ear == "R" else (1, 2)
        trials.append(_kul_trial(k, 3, i + 1, 1, cond, ear, 1, f"rep_part{p}_track{left}_{cond}.wav",
                                 f"rep_part{p}_track{right}_{cond}.wav", rng)); k += 1
    cell = np.empty((1, len(trials)), dtype=object)
    for j, d in enumerate(trials):
        cell[0, j] = d
    scipy.io.savemat(root / "S1.mat", {"trials": cell})
    return root


def _expected_env(root, name, n=N_EEG):
    return _load_wav_envelope(root / "stimuli" / name, target_sfreq=FS_EEG, low_hz=1.0, high_hz=8.0)[:n]


# --------------------------------------------------------------------------- #
#  Loader
# --------------------------------------------------------------------------- #

def test_stimulus_fields_parse_and_reject():
    assert _stimulus_fields("part3_track2_hrtf.wav", "part3_track1_hrtf.wav") == \
        {"story_part": 3, "left_track": 2, "right_track": 1}
    assert _stimulus_fields("rep_part1_track1_dry.wav", "rep_part1_track2_dry.wav")["story_part"] == 1
    with pytest.raises(ValueError):
        _stimulus_fields("part1_track1_dry.wav", "part2_track2_dry.wav")      # different parts
    with pytest.raises(ValueError):
        _stimulus_fields("part1_track1_dry.wav", "part1_track1_dry.wav")      # same track
    with pytest.raises(ValueError):
        _stimulus_fields("story.wav", "part1_track2_dry.wav")


def test_dry_envelopes_for_hrtf_trials(kul_root):
    ds = KULeuvenDataset(kul_root, envelope_source="dry", include_experiments=(1, 2))
    trials = list(ds.trials())
    hrtf = [t for t in trials if ds.trial_info[t.trial_id]["condition"] == "hrtf"]
    assert hrtf, "fixture has HRTF trials"
    for t in hrtf:
        info = ds.trial_info[t.trial_id]
        assert info["left_wav"].endswith("_hrtf.wav")
        assert info["left_envelope_wav"] == info["left_wav"].replace("_hrtf.wav", "_dry.wav")
        np.testing.assert_allclose(t.audio_a, _expected_env(kul_root, info["left_envelope_wav"]), rtol=1e-6)
        np.testing.assert_allclose(t.audio_b, _expected_env(kul_root, info["right_envelope_wav"]), rtol=1e-6)
        # and it is not the HRTF file's envelope
        assert not np.allclose(t.audio_a, _expected_env(kul_root, info["left_wav"]))


def test_presented_source_keeps_historical_behaviour(kul_root):
    ds = KULeuvenDataset(kul_root, include_experiments=(1,))
    for t in ds.trials():
        info = ds.trial_info[t.trial_id]
        assert info["left_envelope_wav"] == info["left_wav"]
        np.testing.assert_allclose(t.audio_a, _expected_env(kul_root, info["left_wav"]), rtol=1e-6)


def test_invalid_envelope_source(kul_root):
    with pytest.raises(ValueError):
        KULeuvenDataset(kul_root, envelope_source="hrtf")


def test_trial_info_and_experiment_filter(kul_root):
    ds = KULeuvenDataset(kul_root, load_audio=False, include_experiments=(1, 2))
    trials = list(ds.trials())
    assert len(trials) == 8
    assert set(ds.trial_info) == {t.trial_id for t in trials}
    for t in trials:
        info = ds.trial_info[t.trial_id]
        assert info["experiment"] in (1, 2) and info["part"] == info["story_part"]
        att_track = info["right_track"] if info["attended_ear"] == "R" else info["left_track"]
        assert att_track == info["attended_track"] == info["experiment"]
        assert t.label == (1 if info["attended_ear"] == "R" else 0)
        assert t.eeg.shape == (64, N_EEG)
    # each part twice, attending each track once
    pairs = Counter((ds.trial_info[t.trial_id]["part"], ds.trial_info[t.trial_id]["attended_track"]) for t in trials)
    assert pairs == Counter({(p, a): 1 for p in range(1, 5) for a in (1, 2)})


def test_multiband_envelope_truncated_on_time_axis(kul_root):
    ds = KULeuvenDataset(kul_root, envelope_method="powerlaw_subbands_multi", envelope_source="dry",
                         include_experiments=(1,))
    t = next(iter(ds.trials()))
    assert t.audio_a.ndim == 2 and t.audio_a.shape[1] == N_EEG     # (K, T)
    assert t.audio_a.shape[0] > 1


# --------------------------------------------------------------------------- #
#  Content groups and story folds
# --------------------------------------------------------------------------- #

def test_content_group_experiment3_follows_stimulus_files(kul_root):
    ds = KULeuvenDataset(kul_root, load_audio=False)
    trials = list(ds.trials())
    assert len(trials) == 20
    for t in trials:
        assert _content_group(t) == ds.trial_info[t.trial_id]["story_part"], t.group_id


def test_kul_story_part(kul_root):
    ds = KULeuvenDataset(kul_root, load_audio=False)
    for t in ds.trials():
        info = ds.trial_info[t.trial_id]
        expected = info["story_part"] if info["experiment"] in (1, 2) else None
        assert kul_story_part(t) == expected


def _kul_like_trials(n_subjects=16):
    """Experiment-1/2 KUL trials, 8 per subject, plus experiment-3 trials that must be dropped."""
    out = []
    for s in range(1, n_subjects + 1):
        k = 0
        for exp in (1, 2):
            for p in range(1, 5):
                out.append(Trial(eeg=np.zeros((2, 10), np.float32), sfreq=64.0, label=(p + s) % 2,
                                 subject_id=f"S{s}", trial_id=f"S{s}_T{k:03d}",
                                 group_id=f"S{s}_exp{exp}_p{p}_rep0_dry")); k += 1
        for i in range(12):
            out.append(Trial(eeg=np.zeros((2, 10), np.float32), sfreq=64.0, label=0, subject_id=f"S{s}",
                             trial_id=f"S{s}_T{k:03d}", group_id=f"S{s}_exp3_p{i + 1}_rep1_hrtf")); k += 1
    return out


def test_kul_subject_group_story_out():
    trials = _kul_like_trials()
    folds = list(subject_group_story_out(trials, content_group=kul_story_part, story_folds=KUL_STORY_FOLDS,
                                         n_subject_groups=8, n_val_subjects=2, seed=42))
    assert len(folds) == 16
    tested = Counter(i for f in folds for i in f.test_idx)
    exp12 = [i for i, t in enumerate(trials) if kul_story_part(t) is not None]
    assert set(tested) == set(exp12) and set(tested.values()) == {1}      # every exp-1/2 trial once
    for f in folds:
        assert_story_disjoint_fold_integrity(trials, f, content_group=kul_story_part)
        held = set(f.meta["held_out_content_groups"])
        assert held in ({1, 2}, {3, 4})
        assert len(f.meta["test_subjects"]) == 2 and len(f.meta["val_subjects"]) == 2
        assert {kul_story_part(trials[i]) for i in f.test_idx} == held
        assert {kul_story_part(trials[i]) for i in f.val_idx} == held
        assert {kul_story_part(trials[i]) for i in f.train_idx} == {1, 2, 3, 4} - held
        assert len({trials[i].subject_id for i in f.train_idx}) == 12
        assert len(f.test_idx) == 2 * 4 and len(f.train_idx) == 12 * 4


# --------------------------------------------------------------------------- #
#  Content-disjoint trial-swap control
# --------------------------------------------------------------------------- #

def test_partners_same_subject_and_experiment_other_part():
    keys = {f"S{s}_e{e}_p{p}": (f"S{s}", e, p) for s in (1, 2) for e in (1, 2) for p in (1, 2)}
    partners = same_stratum_other_content_partners(keys)
    for tid, pid in partners.items():
        (s, e, p), (s2, e2, p2) = keys[tid], keys[pid]
        assert (s, e) == (s2, e2) and p != p2
        assert partners[pid] == tid


def test_partners_reject_ambiguous_or_missing():
    with pytest.raises(ValueError):          # two candidates
        same_stratum_other_content_partners({"a": ("S1", 1, 1), "b": ("S1", 1, 2), "c": ("S1", 1, 3)})
    with pytest.raises(ValueError):          # none
        same_stratum_other_content_partners({"a": ("S1", 1, 1), "b": ("S1", 2, 2)})


def test_partner_trial_permutation_positions_and_wrap():
    # trial A: 5 windows, trial B: 3 windows (shorter partner wraps), shuffled order
    trial = np.array(["A"] * 5 + ["B"] * 3)
    start = np.array([0, 320, 640, 960, 1280, 0, 320, 640])
    order = np.random.default_rng(1).permutation(len(trial))
    trial, start = trial[order], start[order]
    perm = partner_trial_permutation(trial, start, {"A": "B", "B": "A"})
    for i, j in enumerate(perm):
        assert trial[j] != trial[i]
        rank_i = int(start[i] // 320)
        n_partner = 3 if trial[i] == "A" else 5
        assert start[j] // 320 == rank_i % n_partner


def test_partner_trial_permutation_rejects_self_or_missing():
    with pytest.raises(ValueError):
        partner_trial_permutation(["A", "B"], [0, 0], {"A": "A", "B": "A"})
    with pytest.raises(ValueError):
        partner_trial_permutation(["A", "B"], [0, 0], {"A": "B"})
