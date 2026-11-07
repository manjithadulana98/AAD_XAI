"""AADNet / CA-TCN subject-independent folds (cv_splits.loso_attended_track_out)."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold, train_test_split

from aad_xai.data.base import Trial
from aad_xai.data.cv_splits import assert_loso_attended_track_integrity, loso_attended_track_out

N_SUBJ, N_SEG = 10, 20


def _fake():
    """DTU-like: every subject hears the same N_SEG two-talker segments (in its own
    order) plus one single-talker trial; trial names include trial_1 and trial_11 so
    a substring match would differ from an exact match."""
    rng = np.random.default_rng(0)
    pairs = [(f"aske_story1_trial_{k + 1}.wav", f"marianne_story4_trial_{k + 1}.wav") for k in range(N_SEG)]
    trials, track, data = [], {}, {}
    for s in range(N_SUBJ):
        sid = f"S{s + 1}"
        order = rng.permutation(N_SEG)
        rows = []
        for pos, seg in enumerate(list(order[:5]) + [-1] + list(order[5:])):   # single-talker trial inserted
            tid = f"{sid}_T{pos:03d}"
            if seg < 0:
                trials.append(Trial(eeg=np.zeros((2, 4)), sfreq=64.0, label=0, subject_id=sid, trial_id=tid,
                                    group_id=f"{sid}_dss.wav"))
                continue
            label = int(rng.integers(2))
            trials.append(Trial(eeg=np.zeros((2, 4)), sfreq=64.0, label=label, subject_id=sid, trial_id=tid,
                                group_id=f"{sid}_{pairs[seg][0]}"))
            track[tid] = pairs[seg][label]
            rows.append({"tid": tid, "label": label, "stimuli": list(pairs[seg])})
        data[s] = rows
    return trials, track, data


def _aadnet_reference(data, n_fold=8):
    """Literal transcription of AADNet's DTUDataset.createSICrossValidation plus the
    train/validation split in cross_validate_loso.trainLOSO (trial ids instead of arrays)."""
    out = []
    all_sbjs = np.arange(N_SUBJ)
    for subject in all_sbjs:
        tr_sbjs = np.delete(all_sbjs, np.where(all_sbjs == subject)[0])
        test_data = data[subject]
        kf = KFold(n_splits=n_fold, random_state=int(subject), shuffle=True)
        for train, test in kf.split(range(len(test_data))):
            test_attended = [test_data[t]["stimuli"][test_data[t]["label"]] for t in test]
            te = [test_data[t]["tid"] for t in test]
            tr = []
            for tr_s in tr_sbjs:
                for row in data[tr_s]:
                    attd = row["stimuli"][row["label"]]
                    if any(i in attd for i in test_attended):
                        continue
                    tr.append(row["tid"])
            tr2, va = train_test_split(tr, test_size=0.2, random_state=int(subject))
            out.append((te, tr2, va))
    return out


def test_folds_equal_aadnet_reference_implementation():
    trials, track, data = _fake()
    ours = list(loso_attended_track_out(trials, track))
    ref = _aadnet_reference(data)
    assert len(ours) == len(ref) == N_SUBJ * 8
    for f, (te, tr, va) in zip(ours, ref):
        ids = lambda idx: [trials[i].trial_id for i in idx]
        assert ids(f.test_idx) == te
        assert ids(f.train_idx) == tr
        assert ids(f.val_idx) == va
        assert_loso_attended_track_integrity(trials, f, track)


def test_subject_order_is_natural_and_test_folds_partition_each_subject():
    trials, track, _ = _fake()
    folds = list(loso_attended_track_out(trials, track))
    order = [f.meta["test_subjects"][0] for f in folds[::8]]
    assert order == [f"S{i + 1}" for i in range(N_SUBJ)]                     # S10 after S9
    for s in order:
        tested = [trials[i].trial_id for f in folds if f.meta["test_subjects"] == [s] for i in f.test_idx]
        usable = [t.trial_id for t in trials if t.subject_id == s and t.trial_id in track]
        assert sorted(tested) == sorted(usable) and len(tested) == len(set(tested))
    assert not any(t.group_id.endswith("dss.wav") for f in folds for i in f.train_idx + f.test_idx
                   for t in [trials[i]])


def test_exclusion_is_by_attended_file_only():
    trials, track, _ = _fake()
    f = next(iter(loso_attended_track_out(trials, track)))
    test_tracks = {track[trials[i].trial_id] for i in f.test_idx}
    test_segments = {trials[i].group_id.split("_", 1)[1] for i in f.test_idx}
    pool = [trials[i] for i in f.train_idx + f.val_idx]
    assert not test_tracks & {track[t.trial_id] for t in pool}
    # same segment with the other talker attended is kept
    assert any(t.group_id.split("_", 1)[1] in test_segments for t in pool)
    assert f.meta["n_excluded"] == sum(1 for t in trials if t.subject_id != "S1" and t.trial_id in track
                                       and track[t.trial_id] in test_tracks)
    assert len(f.val_idx) == int(np.ceil(0.2 * (len(f.val_idx) + len(f.train_idx))))


def test_integrity_check_catches_leaks():
    trials, track, _ = _fake()
    f = next(iter(loso_attended_track_out(trials, track)))
    leak = next(i for i, t in enumerate(trials) if t.subject_id != "S1" and t.trial_id in track
                and track[t.trial_id] == track[trials[f.test_idx[0]].trial_id])
    bad = type(f)(f.fold_id, f.train_idx + [leak], f.val_idx, f.test_idx, f.meta)
    with pytest.raises(AssertionError):
        assert_loso_attended_track_integrity(trials, bad, track)
    same_subject = next(i for i, t in enumerate(trials) if t.subject_id == "S1" and i not in f.test_idx
                        and t.trial_id in track)
    bad = type(f)(f.fold_id, f.train_idx + [same_subject], f.val_idx, f.test_idx, f.meta)
    with pytest.raises(AssertionError):
        assert_loso_attended_track_integrity(trials, bad, track)
