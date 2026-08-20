"""Tests for DTU LOSO TRF explainability (aad_xai.xai.trf_explain).

Fixture style matches test_dtu_loso.py's `_make_dtu_shaped_trials`: audio_a/
audio_b are fixed per-speaker streams, `label` selects which one is
attended -- NOT aad_xai.data.synthetic_dataset.SyntheticDataset's
always-audio_a-is-attended convention, which would silently feed the wrong
training target on roughly half the trials.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aad_xai.data.base import Trial
from aad_xai.evaluation.loso_runner import run_dtu_loso_trf_on_trials
from aad_xai.models.trf_baseline import TRFDecoder
from aad_xai.xai.trf_explain import (
    _haufe_from_eeg,
    load_trf_decoder,
    sanity_check_lag_cascade,
    subject_level_stats,
    window_level_combined_table,
    window_level_importance,
)


def _make_dtu_shaped_trials(
    n_subjects: int = 3,
    n_trials_per_subject: int = 2,
    n_times: int = 256,
    sfreq: float = 32.0,
    n_channels: int = 4,
    seed: int = 0,
) -> list[Trial]:
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


class TestLoadTrfDecoder:
    def test_reloaded_predictions_correlate_perfectly_with_original(self, tmp_path):
        # load_trf_decoder deliberately skips intercept_/_y_mean/_y_std --
        # this test is the decisive check that the claim in its docstring
        # (every downstream consumer is shift/positive-scale invariant, so
        # skipping those 3 fields cannot change any decision) actually holds.
        rng = np.random.default_rng(0)
        n_ch, n_t = 4, 2000
        eeg = rng.standard_normal((n_ch, n_t)).astype(np.float32)
        env = (0.5 * eeg[0] + 0.1 * rng.standard_normal(n_t)).astype(np.float32)

        original = TRFDecoder(tmin_s=0.0, tmax_s=0.2, alpha=10.0)
        original.fit(eeg, env, sfreq=32.0)

        npz_path = tmp_path / "S1.npz"
        np.savez(
            npz_path, coef=original.model.coef_, lags=original.lags_,
            x_mean=original._X_mean, x_std=original._X_std,
        )
        reloaded = load_trf_decoder(npz_path, sfreq=32.0)

        eeg_test = rng.standard_normal((n_ch, 500)).astype(np.float32)
        pred_orig = original.predict(eeg_test)
        pred_reloaded = reloaded.predict(eeg_test)
        assert np.corrcoef(pred_orig, pred_reloaded)[0, 1] == pytest.approx(1.0, abs=1e-5)


class TestHaufePattern:
    def test_haufe_pattern_identifies_the_informative_channel(self):
        # Decisive test of the lag-major/channel-minor reshape convention:
        # if pattern_full.reshape(n_lags, n_channels) were transposed, the
        # informative channel's magnitude would land at the wrong index.
        rng = np.random.default_rng(1)
        n_ch, n_t = 6, 5000
        informative_ch = 3
        eeg = (rng.standard_normal((n_ch, n_t)) * 0.05).astype(np.float32)
        env = (2.0 * eeg[informative_ch] + 0.01 * rng.standard_normal(n_t)).astype(np.float32)

        decoder = TRFDecoder(tmin_s=0.0, tmax_s=0.1, alpha=1.0)
        decoder.fit(eeg, env, sfreq=32.0)

        channel_magnitude, pattern_matrix = _haufe_from_eeg(decoder, eeg)
        assert pattern_matrix.shape == (len(decoder.lags_), n_ch)
        assert int(np.argmax(channel_magnitude)) == informative_ch


class TestWindowLevelImportance:
    def test_runs_and_produces_finite_shapes(self):
        trials = _make_dtu_shaped_trials(
            n_subjects=1, n_trials_per_subject=4, n_times=640, sfreq=32.0, n_channels=4, seed=2,
        )
        eeg_cat = np.concatenate([t.eeg for t in trials], axis=1)
        env_cat = np.concatenate(
            [(t.audio_a if t.label == 0 else t.audio_b) for t in trials]
        )
        decoder = TRFDecoder(tmin_s=0.0, tmax_s=0.1, alpha=1.0)
        decoder.fit(eeg_cat, env_cat, sfreq=32.0)

        imp = window_level_importance(decoder, trials, window_s=1.0, n_boot=200, seed=42)
        assert imp["occ_pw"].shape[1] == 4
        assert imp["perm_pw"].shape == imp["occ_pw"].shape
        assert imp["n_windows"] > 0
        assert np.all(np.isfinite(imp["occ_pw"]))
        assert np.all(np.isfinite(imp["perm_pw"]))
        assert len(imp["occ_results"]) == 4
        assert all("fdr_sig" in r for r in imp["occ_results"])

    def test_max_windows_caps_and_subsamples_deterministically(self):
        # DTU subjects have ~600 test windows each and each forward pass is
        # Python-loop-bound (~9ms/window locally) -- max_windows keeps
        # window_level_importance/faithfulness_curves tractable across 18
        # folds. This must actually cap the window count, and do so
        # deterministically for a fixed seed (reproducible reporting).
        trials = _make_dtu_shaped_trials(
            n_subjects=1, n_trials_per_subject=6, n_times=640, sfreq=32.0, n_channels=4, seed=6,
        )
        eeg_cat = np.concatenate([t.eeg for t in trials], axis=1)
        env_cat = np.concatenate(
            [(t.audio_a if t.label == 0 else t.audio_b) for t in trials]
        )
        decoder = TRFDecoder(tmin_s=0.0, tmax_s=0.1, alpha=1.0)
        decoder.fit(eeg_cat, env_cat, sfreq=32.0)

        imp_full = window_level_importance(decoder, trials, window_s=1.0, n_boot=50, seed=42, max_windows=None)
        n_available = imp_full["n_windows"]
        assert n_available > 5  # sanity: fixture actually has multiple windows

        cap = max(2, n_available - 2)
        imp_capped = window_level_importance(decoder, trials, window_s=1.0, n_boot=50, seed=42, max_windows=cap)
        assert imp_capped["n_windows"] == cap

        imp_capped_again = window_level_importance(decoder, trials, window_s=1.0, n_boot=50, seed=42, max_windows=cap)
        np.testing.assert_array_equal(imp_capped["occ_pw"], imp_capped_again["occ_pw"])


class TestSubjectLevelStats:
    def test_promotes_strong_consistent_channel_to_tier1(self):
        rng = np.random.default_rng(3)
        n_subj, n_ch = 18, 6
        strong_ch = 2

        occ = rng.normal(0, 0.0005, size=(n_subj, n_ch))
        perm = rng.normal(0, 0.0005, size=(n_subj, n_ch))
        haufe = rng.normal(0, 0.0005, size=(n_subj, n_ch))
        occ[:, strong_ch] = rng.normal(0.05, 0.002, size=n_subj)
        perm[:, strong_ch] = rng.normal(0.05, 0.002, size=n_subj)
        haufe[:, strong_ch] = rng.normal(0.5, 0.01, size=n_subj)

        window_table = window_level_combined_table(
            occ_pw_pooled=np.repeat(occ, 5, axis=0),
            perm_pw_pooled=np.repeat(perm, 5, axis=0),
            occ_subject_profile=occ,
            perm_subject_profile=perm,
        )
        stats = subject_level_stats(occ, perm, haufe, window_table)

        assert stats["n_subjects"] == 18
        assert stats["tier1_stability_frac"] == pytest.approx(12 / 18)
        assert stats["tier2_stability_frac"] == pytest.approx(10 / 18)
        tier1_channels = {r["channel"] for r in stats["tier1_channels"]}
        assert strong_ch in tier1_channels

    def test_custom_tier_fractions_are_honored(self):
        rng = np.random.default_rng(5)
        n_subj, n_ch = 10, 3
        occ = rng.normal(0, 0.001, size=(n_subj, n_ch))
        perm = rng.normal(0, 0.001, size=(n_subj, n_ch))
        haufe = rng.normal(0, 0.001, size=(n_subj, n_ch))
        window_table = window_level_combined_table(
            occ_pw_pooled=occ, perm_pw_pooled=perm, occ_subject_profile=occ, perm_subject_profile=perm,
        )
        stats = subject_level_stats(
            occ, perm, haufe, window_table,
            tier1_stability_frac=0.9, tier2_stability_frac=0.5,
        )
        assert stats["tier1_stability_frac"] == 0.9
        assert stats["tier2_stability_frac"] == 0.5


class TestSanityCheckLagCascade:
    def test_covers_every_lag_and_diverges_from_original(self):
        rng = np.random.default_rng(4)
        n_ch, n_t = 4, 3000
        eeg = rng.standard_normal((n_ch, n_t)).astype(np.float32)
        env = (0.5 * eeg[0] + 0.1 * rng.standard_normal(n_t)).astype(np.float32)
        decoder = TRFDecoder(tmin_s=0.0, tmax_s=0.1, alpha=1.0)
        decoder.fit(eeg, env, sfreq=32.0)

        trial = Trial(
            eeg=eeg, sfreq=32.0, label=0, subject_id="S1", trial_id="T0",
            group_id="g0", audio_a=env, audio_b=(env * 0.0).astype(np.float32), audio_sr=32,
        )
        result = sanity_check_lag_cascade(decoder, [trial], seg_len_s=1.0, max_total_s=10.0, seed=0)

        assert len(result["cascade_steps"]) == len(decoder.lags_)
        for step in result["cascade_steps"]:
            assert -1.0 <= step["rho_vs_original"] <= 1.0
        # Fully randomized (last step) should not coincidentally reproduce
        # the exact original pattern.
        assert result["cascade_steps"][-1]["rho_vs_original"] != pytest.approx(1.0, abs=1e-6)


class TestLosoRunnerWeightSaving:
    def test_saves_one_npz_per_held_out_test_subject(self, tmp_path):
        trials = _make_dtu_shaped_trials(
            n_subjects=3, n_trials_per_subject=2, n_times=256, sfreq=32.0, n_channels=4, seed=1,
        )
        summary = run_dtu_loso_trf_on_trials(
            trials, output_dir=tmp_path, window_s=1.0, seed=42, trf_max_train_seconds=5.0,
        )
        run_dir = Path(summary["run_dir"])
        weights_dir = run_dir / "trf_weights"
        assert weights_dir.exists()

        for subj in ("S1", "S2", "S3"):
            npz_path = weights_dir / f"{subj}.npz"
            assert npz_path.exists(), f"missing weights for held-out subject {subj}"
            npz = np.load(npz_path)
            assert set(npz.files) == {"coef", "lags", "x_mean", "x_std"}
            assert npz["coef"].ndim == 1
            assert npz["x_mean"].shape == npz["coef"].shape
            assert npz["x_std"].shape == npz["coef"].shape
