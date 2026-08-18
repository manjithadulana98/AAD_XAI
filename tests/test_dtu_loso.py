"""Tests for the DTU LOSO TRF baseline pipeline (Phase 1, WP1-3).

Uses a hand-built, DTU-shaped `Trial` fixture rather than
`aad_xai.data.synthetic_dataset.SyntheticDataset`: that dataset sets
`audio_a` as *always* the attended envelope regardless of `label`, which
does not match the real `Trial` contract (fixed per-speaker streams;
`label` selects which one is attended, as in `DTUDataset`). Using it here
would silently feed the wrong training target on roughly half the trials.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pytest

from aad_xai.data.base import Trial
from aad_xai.data.cv_splits import CV_STRATEGIES, CVFold, assert_loso_fold_integrity
from aad_xai.data.windowing import assert_no_cross_split_overlap, make_windows
from aad_xai.evaluation.baseline_table import write_baseline_performance_csv
from aad_xai.evaluation.loso_runner import run_dtu_loso_trf_on_trials
from aad_xai.evaluation.schema import (
    REQUIRED_COLUMNS,
    TRF_EXTRA_COLUMNS,
    SubjectRecord,
    aggregate_subject_records,
    build_prediction_records,
    write_prediction_csv,
)
from aad_xai.xai.composite_stability import bootstrap_ci


def _make_dtu_shaped_trials(
    n_subjects: int = 3,
    n_trials_per_subject: int = 2,
    n_times: int = 256,
    sfreq: float = 32.0,
    n_channels: int = 4,
    seed: int = 0,
) -> list[Trial]:
    """DTU-shaped fixture: audio_a/audio_b are fixed per-speaker streams
    (not label-dependent); `label` selects which one is attended, matching
    DTUDataset's real convention (0 = audio_a attended, 1 = audio_b attended).
    """
    rng = np.random.default_rng(seed)
    trials: list[Trial] = []
    for si in range(n_subjects):
        subject_id = f"S{si + 1}"
        for ti in range(n_trials_per_subject):
            trial_id = f"{subject_id}_T{ti:03d}"
            group_id = f"{subject_id}_story{ti}.wav"
            label = ti % 2
            eeg = rng.standard_normal((n_channels, n_times)).astype(np.float32)
            audio_a = (rng.standard_normal(n_times) * 0.1).astype(np.float32)
            audio_b = (rng.standard_normal(n_times) * 0.1).astype(np.float32)
            if label == 0:
                audio_a = audio_a + 0.3 * eeg[0]
            else:
                audio_b = audio_b + 0.3 * eeg[0]
            trials.append(
                Trial(
                    eeg=eeg, sfreq=sfreq, label=label, subject_id=subject_id,
                    trial_id=trial_id, group_id=group_id,
                    audio_a=audio_a, audio_b=audio_b, audio_sr=int(sfreq),
                )
            )
    return trials


class TestLosoFoldCorrectness:
    def test_loso_folds_cover_every_subject_exactly_once(self):
        trials = _make_dtu_shaped_trials(n_subjects=4)
        folds = list(CV_STRATEGIES["loso"](trials, seed=42))
        test_subjects = [f.meta["test_subject"] for f in folds]
        assert sorted(test_subjects) == ["S1", "S2", "S3", "S4"]
        assert len(folds) == 4

    def test_loso_val_never_equals_test_subject(self):
        trials = _make_dtu_shaped_trials(n_subjects=5)
        folds = list(CV_STRATEGIES["loso"](trials, seed=7))
        for f in folds:
            assert f.meta["val_subject"] != f.meta["test_subject"]

    def test_assert_loso_fold_integrity_passes_on_real_folds(self):
        trials = _make_dtu_shaped_trials(n_subjects=4)
        for fold in CV_STRATEGIES["loso"](trials, seed=42):
            assert_loso_fold_integrity(trials, fold)  # should not raise

    def test_assert_loso_fold_integrity_catches_corruption(self):
        trials = _make_dtu_shaped_trials(n_subjects=4)
        fold = next(iter(CV_STRATEGIES["loso"](trials, seed=42)))
        corrupted = CVFold(
            fold_id=fold.fold_id,
            train_idx=fold.train_idx + [fold.test_idx[0]],
            val_idx=fold.val_idx,
            test_idx=fold.test_idx,
            meta=fold.meta,
        )
        with pytest.raises(AssertionError):
            assert_loso_fold_integrity(trials, corrupted)


class TestNoCrossSplitWindowOverlap:
    def test_no_cross_split_window_overlap_on_dtu_shaped_trials(self):
        trials = _make_dtu_shaped_trials(n_subjects=3, n_trials_per_subject=2)
        fold = next(iter(CV_STRATEGIES["loso"](trials, seed=42)))

        def _windows_for(idx: list[int]):
            out = []
            for i in idx:
                t = trials[i]
                out.extend(
                    make_windows(
                        n_times=t.eeg.shape[-1], sfreq=t.sfreq, window_s=1.0, overlap_s=0.0,
                        label=t.label, subject_id=t.subject_id, trial_id=t.trial_id,
                    )
                )
            return out

        train_w = _windows_for(fold.train_idx)
        val_w = _windows_for(fold.val_idx)
        test_w = _windows_for(fold.test_idx)
        assert_no_cross_split_overlap(train_w, val_w, test_w)  # should not raise

        corrupted_train = train_w + [val_w[0]]
        with pytest.raises(AssertionError, match="Cross-split window overlap"):
            assert_no_cross_split_overlap(corrupted_train, val_w, test_w)


class TestBuildPredictionRecords:
    def test_correlation_swap_margin_and_tie_correctness(self):
        rows = [
            {"subject_id": "S1", "trial_id": "T0", "start": 64, "y_true": 0, "y_pred": 0, "corr_a": 0.5, "corr_b": 0.1},
            {"subject_id": "S1", "trial_id": "T1", "start": 0, "y_true": 1, "y_pred": 1, "corr_a": 0.2, "corr_b": 0.6},
            {"subject_id": "S1", "trial_id": "T0", "start": 0, "y_true": 0, "y_pred": 0, "corr_a": 0.4, "corr_b": 0.1},
            {"subject_id": "S1", "trial_id": "T1", "start": 64, "y_true": 1, "y_pred": 0, "corr_a": 0.5, "corr_b": 0.3},
            # exact tie: decision rule assigns prediction=1; correct because target==1 despite margin==0
            {"subject_id": "S1", "trial_id": "T0", "start": 128, "y_true": 1, "y_pred": 1, "corr_a": 0.3, "corr_b": 0.3},
        ]
        records = build_prediction_records(rows, dataset="dtu", model="trf", window_seconds=1.0, seed=42)
        by_key = {(r.trial_id, r.window_id): r for r in records}

        r0 = by_key[("T0", 0)]  # start=0, target=0
        assert r0.target == 0
        assert r0.attended_correlation == pytest.approx(0.4)
        assert r0.unattended_correlation == pytest.approx(0.1)
        assert r0.correlation_margin == pytest.approx(0.3)
        assert r0.correct == 1
        assert 0.0 <= r0.probability <= 1.0

        r1 = by_key[("T0", 1)]  # start=64, target=0
        assert r1.attended_correlation == pytest.approx(0.5)
        assert r1.unattended_correlation == pytest.approx(0.1)
        assert r1.correct == 1

        r2 = by_key[("T0", 2)]  # start=128, exact tie, target=1
        assert r2.target == 1 and r2.prediction == 1
        assert r2.attended_correlation == pytest.approx(0.3)
        assert r2.unattended_correlation == pytest.approx(0.3)
        assert r2.correlation_margin == pytest.approx(0.0)
        assert r2.correct == 1

        t1_0 = by_key[("T1", 0)]  # start=0, target=1
        assert t1_0.attended_correlation == pytest.approx(0.6)
        assert t1_0.unattended_correlation == pytest.approx(0.2)
        assert t1_0.correct == 1

        t1_1 = by_key[("T1", 1)]  # start=64, target=1, prediction=0 -> incorrect
        assert t1_1.target == 1 and t1_1.prediction == 0
        assert t1_1.correct == 0
        assert t1_1.attended_correlation == pytest.approx(0.3)
        assert t1_1.unattended_correlation == pytest.approx(0.5)
        assert t1_1.correlation_margin == pytest.approx(-0.2)

    def test_write_prediction_csv_columns_match_spec_exactly(self, tmp_path):
        rows = [{"subject_id": "S1", "trial_id": "T0", "start": 0, "y_true": 0, "y_pred": 0, "corr_a": 0.4, "corr_b": 0.1}]
        records = build_prediction_records(rows, dataset="dtu", model="trf", window_seconds=1.0, seed=42)
        path = tmp_path / "window_predictions.csv"
        write_prediction_csv(path, records)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == REQUIRED_COLUMNS + TRF_EXTRA_COLUMNS


class TestAggregateSubjectRecords:
    def test_aggregate_subject_records_accuracy(self):
        rows = [
            {"subject_id": "S1", "trial_id": "T0", "start": 0, "y_true": 0, "y_pred": 0, "corr_a": 0.5, "corr_b": 0.1},
            {"subject_id": "S1", "trial_id": "T0", "start": 64, "y_true": 1, "y_pred": 0, "corr_a": 0.5, "corr_b": 0.3},
            {"subject_id": "S1", "trial_id": "T1", "start": 0, "y_true": 1, "y_pred": 1, "corr_a": 0.2, "corr_b": 0.6},
        ]
        records = build_prediction_records(rows, dataset="dtu", model="trf", window_seconds=1.0, seed=42)
        subj = aggregate_subject_records(records)
        assert len(subj) == 1
        s = subj[0]
        assert s.subject_id == "S1"
        assert s.n_windows == 3
        assert s.n_correct == 2
        assert s.accuracy == pytest.approx(2 / 3)


class TestBaselineTable:
    def test_baseline_performance_csv_mean_std_ci(self, tmp_path):
        accuracies = [0.6, 0.7, 0.8, 0.9]
        subj = [
            SubjectRecord(
                dataset="dtu", model="trf", subject_id=f"S{i}", window_seconds=1.0, seed=42,
                n_windows=10, n_correct=int(round(acc * 10)), accuracy=acc,
            )
            for i, acc in enumerate(accuracies)
        ]
        out_path = tmp_path / "baseline_performance.csv"
        summary = write_baseline_performance_csv(out_path, subj, seed=42)

        expected_mean, expected_lo, expected_hi = bootstrap_ci(accuracies, seed=42)
        assert summary["mean_accuracy"] == pytest.approx(expected_mean)
        assert summary["ci_low"] == pytest.approx(expected_lo)
        assert summary["ci_high"] == pytest.approx(expected_hi)
        assert summary["std_accuracy"] == pytest.approx(float(np.std(accuracies)))
        assert out_path.exists()
        assert out_path.with_name("baseline_performance_summary.json").exists()


class TestDtuLosoTrfSmoke:
    def test_dtu_loso_trf_smoke_end_to_end(self, tmp_path):
        # >=3 subjects: LOSO needs a non-empty train set (1 test + 1 val + >=1 train).
        trials = _make_dtu_shaped_trials(
            n_subjects=3, n_trials_per_subject=2, n_times=256, sfreq=32.0, n_channels=4, seed=1,
        )
        t0 = time.time()
        summary = run_dtu_loso_trf_on_trials(
            trials, output_dir=tmp_path, window_s=1.0, seed=42, trf_max_train_seconds=5.0,
        )
        elapsed = time.time() - t0
        assert elapsed < 30.0  # generous guard, not a strict timing test

        run_dir = Path(summary["run_dir"])
        assert (run_dir / "window_predictions.csv").exists()
        assert (run_dir / "subject_predictions.csv").exists()
        assert (run_dir / "baseline_performance.csv").exists()
        assert summary["n_folds"] == 3  # one fold per subject
        assert summary["n_windows"] > 0

        with (run_dir / "window_predictions.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert {r["subject_id"] for r in rows} <= {"S1", "S2", "S3"}
