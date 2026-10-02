"""Tests for the StimulusGAT interpretability pipeline."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from tests.test_stimulus_gat import _small_adjacency, _make_stimulus_gat
from aad_xai.models.sgat_decision import StimulusGATDecisionWrapper
from aad_xai.xai.perturbations import permute_channel_group
from aad_xai.xai.frequency_bands import BANDS, band_filtered_component, ablate_band
from aad_xai.xai.composite_stability import sign_flip_p_value
from aad_xai.xai import sgat_explain


# ============================================================================
#  Shared fixtures
# ============================================================================

@pytest.fixture
def sgat_model():
    model = _make_stimulus_gat(n_channels=6, sfreq=64.0)
    model.eval()
    return model


@pytest.fixture
def eeg_batch():
    """Synthetic EEG: (B=4, C=6, T=128)"""
    torch.manual_seed(0)
    return torch.randn(4, 6, 128)


@pytest.fixture
def env_batch(eeg_batch):
    """(B, 2, T)"""
    torch.manual_seed(1)
    return torch.randn(eeg_batch.shape[0], 2, eeg_batch.shape[-1])


@pytest.fixture
def labels(eeg_batch):
    """(B,) int in {0, 1}"""
    torch.manual_seed(2)
    return torch.randint(0, 2, (eeg_batch.shape[0],))


# ============================================================================
#  1. StimulusGATDecisionWrapper
# ============================================================================

class TestStimulusGATDecisionWrapper:
    def test_forward_shape(self, sgat_model, eeg_batch, env_batch):
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        wrapper.set_env(env_batch)
        logits = wrapper(eeg_batch)
        assert logits.shape == (eeg_batch.shape[0], 2)

    def test_batch_expansion_repeat(self, sgat_model, eeg_batch, env_batch):
        """A bigger eeg batch than the stashed env (e.g. Captum IG expanding
        the batch by n_steps) must not crash, and should cyclically repeat."""
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        wrapper.set_env(env_batch)  # env batch = 4
        eeg_expanded = eeg_batch.repeat(3, 1, 1)  # eeg batch = 12
        logits = wrapper(eeg_expanded)
        assert logits.shape == (12, 2)

    def test_batch_shrink(self, sgat_model, eeg_batch, env_batch):
        """A smaller eeg batch than the stashed env must not crash either."""
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        wrapper.set_env(env_batch)  # env batch = 4
        logits = wrapper(eeg_batch[:2])
        assert logits.shape == (2, 2)

    def test_differentiable(self, sgat_model, eeg_batch, env_batch):
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        wrapper.set_env(env_batch)
        eeg = eeg_batch.clone().requires_grad_(True)
        logits = wrapper(eeg)
        logits.sum().backward()
        assert eeg.grad is not None
        assert eeg.grad.abs().sum() > 0

    def test_forward_with_attention_shape(self, sgat_model, eeg_batch, env_batch):
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        wrapper.set_env(env_batch)
        logits, attn = wrapper.forward_with_attention(eeg_batch)
        assert logits.shape == (eeg_batch.shape[0], 2)
        n = sgat_model.gat.n_channels
        heads = sgat_model.gat.n_heads
        assert attn["attn_a"].shape == (eeg_batch.shape[0], heads, n, n)
        assert attn["attn_b"].shape == (eeg_batch.shape[0], heads, n, n)


# ============================================================================
#  2. permute_channel_group
# ============================================================================

class TestPermuteChannelGroup:
    def test_shape_preserved(self):
        rng = np.random.default_rng(0)
        eeg = rng.standard_normal((5, 10, 3)).astype(np.float32)  # (B, T, C)
        out = permute_channel_group(eeg, [0], channel_axis=-1, seed=1)
        assert out.shape == eeg.shape

    def test_permuted_channel_is_value_permutation(self):
        rng = np.random.default_rng(0)
        eeg = rng.standard_normal((5, 10, 3)).astype(np.float32)
        out = permute_channel_group(eeg, [0], channel_axis=-1, seed=1)
        # Same multiset of per-window (T,) blocks for channel 0, just reassigned across batch.
        orig_sorted = np.sort(eeg[:, :, 0], axis=0)
        out_sorted = np.sort(out[:, :, 0], axis=0)
        np.testing.assert_allclose(orig_sorted, out_sorted)

    def test_other_channels_untouched(self):
        rng = np.random.default_rng(0)
        eeg = rng.standard_normal((5, 10, 3)).astype(np.float32)
        out = permute_channel_group(eeg, [0], channel_axis=-1, seed=1)
        np.testing.assert_array_equal(out[:, :, 1:], eeg[:, :, 1:])

    def test_channel_axis_equivalence(self):
        rng = np.random.default_rng(0)
        eeg_btc = rng.standard_normal((5, 10, 3)).astype(np.float32)  # (B, T, C)
        eeg_bct = eeg_btc.transpose(0, 2, 1)  # (B, C, T)

        out_btc = permute_channel_group(eeg_btc, [0], channel_axis=-1, seed=7)
        out_bct = permute_channel_group(eeg_bct, [0], channel_axis=1, seed=7)

        np.testing.assert_allclose(out_btc, out_bct.transpose(0, 2, 1))

    def test_requires_batched_input(self):
        with pytest.raises(ValueError):
            permute_channel_group(np.zeros(10), [0])  # ndim=1, no batch axis to permute


# ============================================================================
#  3. frequency_bands
# ============================================================================

class TestFrequencyBands:
    def test_band_filtered_component_shape(self):
        rng = np.random.default_rng(0)
        eeg = rng.standard_normal((2, 4, 256)).astype(np.float64)  # (B, C, T)
        out = band_filtered_component(eeg, sfreq=64.0, low_hz=1.0, high_hz=8.0, time_axis=-1)
        assert out.shape == eeg.shape

    def test_pure_tone_dominant_band(self):
        """A pure 10 Hz sinusoid should have much higher alpha-band power
        than delta-band power after isolation."""
        sfreq = 64.0
        t = np.arange(512) / sfreq
        sine = np.sin(2 * math.pi * 10.0 * t).reshape(1, 1, -1)

        alpha = band_filtered_component(sine, sfreq, *BANDS["alpha"])
        delta = band_filtered_component(sine, sfreq, *BANDS["delta"])

        assert np.var(alpha) > 10 * np.var(delta)

    def test_ablate_band_is_subtractive(self):
        rng = np.random.default_rng(0)
        eeg = rng.standard_normal((2, 4, 256)).astype(np.float64)
        band = BANDS["theta"]
        ablated = ablate_band(eeg, sfreq=64.0, band=band, time_axis=-1, channel_axis=-2)
        component = band_filtered_component(eeg, sfreq=64.0, low_hz=band[0], high_hz=band[1], time_axis=-1)
        np.testing.assert_allclose(eeg - ablated, component)

    def test_channel_idx_restricts_ablation(self):
        rng = np.random.default_rng(0)
        eeg = rng.standard_normal((2, 4, 256)).astype(np.float64)
        band = BANDS["beta"]
        ablated = ablate_band(eeg, sfreq=64.0, band=band, channel_idx=[0], time_axis=-1, channel_axis=-2)
        np.testing.assert_array_equal(ablated[:, 1:, :], eeg[:, 1:, :])
        assert not np.allclose(ablated[:, 0, :], eeg[:, 0, :])


# ============================================================================
#  4. sign_flip_p_value
# ============================================================================

class TestSignFlipPValue:
    def test_zero_mean_gives_high_p(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal(200) * 0.0  # exactly zero
        p = sign_flip_p_value(values, n_perm=500, seed=0)
        assert p > 0.5

    def test_strong_signal_gives_low_p(self):
        values = np.full(200, 5.0)  # every value identically +5 -- no sign flip can match it
        p = sign_flip_p_value(values, n_perm=500, seed=0)
        assert p < 0.01

    def test_deterministic_with_seed(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal(50)
        p1 = sign_flip_p_value(values, n_perm=500, seed=123)
        p2 = sign_flip_p_value(values, n_perm=500, seed=123)
        assert p1 == p2


# ============================================================================
#  5. sgat_explain -- channel importance
# ============================================================================

class TestChannelImportance:
    def test_occlusion_shape_and_keys(self, sgat_model, eeg_batch, env_batch, labels):
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        result = sgat_explain.channel_occlusion_importance(wrapper, eeg_batch, env_batch, labels, n_boot=50)
        assert len(result) == eeg_batch.shape[1]
        for row in result:
            for key in ("channel", "mean_dp", "ci_lo", "ci_hi", "p_value", "fdr_p", "fdr_sig"):
                assert key in row

    def test_permutation_shape_and_keys(self, sgat_model, eeg_batch, env_batch, labels):
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        result = sgat_explain.channel_permutation_importance(wrapper, eeg_batch, env_batch, labels, n_boot=50)
        assert len(result) == eeg_batch.shape[1]
        for row in result:
            for key in ("channel", "mean_dp", "ci_lo", "ci_hi", "p_value", "fdr_p", "fdr_sig"):
                assert key in row

    def test_occlusion_vs_permutation_distinct(self, sgat_model, eeg_batch, env_batch, labels):
        """Zero-occlusion and value-shuffling are different perturbations --
        they should not coincidentally produce identical delta-P everywhere."""
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        occ = sgat_explain.channel_occlusion_importance(wrapper, eeg_batch, env_batch, labels, n_boot=50)
        perm = sgat_explain.channel_permutation_importance(wrapper, eeg_batch, env_batch, labels, n_boot=50)
        occ_vals = [r["mean_dp"] for r in occ]
        perm_vals = [r["mean_dp"] for r in perm]
        assert not np.allclose(occ_vals, perm_vals)


# ============================================================================
#  6. sgat_explain -- attention summary
# ============================================================================

class TestAttentionSummary:
    def test_extract_attention_shapes(self, sgat_model, eeg_batch, env_batch):
        attn = sgat_explain.extract_attention(sgat_model, eeg_batch, env_batch)
        n = sgat_model.gat.n_channels
        heads = sgat_model.gat.n_heads
        assert attn["attn_a"].shape == (eeg_batch.shape[0], heads, n, n)
        assert attn["attn_b"].shape == (eeg_batch.shape[0], heads, n, n)

    def test_summarize_attention_respects_mask(self, sgat_model, eeg_batch, env_batch):
        attn = sgat_explain.extract_attention(sgat_model, eeg_batch, env_batch)
        summary = sgat_explain.summarize_attention(attn["attn_a"], sgat_model.gat.adj_mask)
        adj_mask_np = sgat_model.gat.adj_mask.cpu().numpy()
        assert np.all(summary["edge_importance"][~adj_mask_np] == 0)

    def test_summary_bounded_and_has_caveat(self, sgat_model, eeg_batch, env_batch):
        attn = sgat_explain.extract_attention(sgat_model, eeg_batch, env_batch)
        summary = sgat_explain.summarize_attention(attn["attn_a"], sgat_model.gat.adj_mask)
        assert summary["edge_importance"].min() >= 0.0
        assert summary["edge_importance"].max() <= 1.0 + 1e-6
        assert "_caveat" in summary


# ============================================================================
#  7. sgat_explain -- frequency-band contribution
# ============================================================================

class TestChannelBandContribution:
    def test_shape_and_keys(self, sgat_model, eeg_batch, env_batch, labels):
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        result = sgat_explain.channel_band_contribution(
            wrapper, eeg_batch, env_batch, labels, sfreq=64.0, channel_idx=[0, 1], n_boot=50,
        )
        assert set(result.keys()) == {0, 1}
        for ch_stats in result.values():
            assert set(ch_stats.keys()) == set(sgat_explain.BANDS.keys())
            for band_stats in ch_stats.values():
                for key in ("mean_dp", "ci_lo", "ci_hi", "p_value", "fdr_p", "fdr_sig"):
                    assert key in band_stats


# ============================================================================
#  8. sgat_explain -- faithfulness comparison
# ============================================================================

class TestFaithfulnessComparison:
    def test_runs_both_rankings_and_random_control(self, sgat_model, eeg_batch, env_batch, labels):
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        n = eeg_batch.shape[1]
        attention_ranking = list(range(n))
        occlusion_ranking = list(reversed(range(n)))
        result = sgat_explain.faithfulness_comparison(
            wrapper, eeg_batch, env_batch, labels, attention_ranking, occlusion_ranking, steps=5,
        )
        assert set(result.keys()) == {"attention", "occlusion", "random"}
        for curves in result.values():
            assert len(curves["deletion"]) == 6  # steps=5 -> 6 points (0..steps inclusive)
            assert len(curves["insertion"]) == 6

    def test_deletion_curve_moves(self, sgat_model, eeg_batch, env_batch, labels):
        wrapper = StimulusGATDecisionWrapper(sgat_model)
        n = eeg_batch.shape[1]
        ranking = list(range(n))
        curves = sgat_explain.faithfulness_from_channel_ranking(
            wrapper, eeg_batch, env_batch, labels, ranking, steps=5,
        )
        # Loose sign-of-effect check (untrained/tiny fixture model, not a strict guarantee):
        # full deletion should not leave confidence higher than no deletion by more than noise.
        assert curves["deletion"][-1] <= curves["deletion"][0] + 0.2


# ============================================================================
#  9. sgat_explain -- sanity check
# ============================================================================

class TestSanityCheckCascadingRandomization:
    def test_occlusion_mode_returns_one_entry_per_named_child(self, sgat_model, eeg_batch, env_batch, labels):
        result = sgat_explain.sanity_check_cascading_randomization(
            sgat_model, eeg_batch, env_batch, importance_fn="occlusion", labels=labels,
        )
        expected_children = {name for name, _ in sgat_model.named_children()}
        assert set(result["rho_by_layer"].keys()) == expected_children
        for stats in result["rho_by_layer"].values():
            assert "rho" in stats and "p_value" in stats

    def test_attention_mode_runs(self, sgat_model, eeg_batch, env_batch):
        result = sgat_explain.sanity_check_cascading_randomization(
            sgat_model, eeg_batch, env_batch, importance_fn="attention",
        )
        expected_children = {name for name, _ in sgat_model.named_children()}
        assert set(result["rho_by_layer"].keys()) == expected_children

    def test_occlusion_requires_labels(self, sgat_model, eeg_batch, env_batch):
        with pytest.raises(ValueError):
            sgat_explain.sanity_check_cascading_randomization(
                sgat_model, eeg_batch, env_batch, importance_fn="occlusion", labels=None,
            )

    def test_invalid_importance_fn_raises(self, sgat_model, eeg_batch, env_batch, labels):
        with pytest.raises(ValueError):
            sgat_explain.sanity_check_cascading_randomization(
                sgat_model, eeg_batch, env_batch, importance_fn="bogus", labels=labels,
            )


# ============================================================================
#  10. sgat_explain -- top-level orchestrator
# ============================================================================

class TestRunSgatExplainOrchestrator:
    def test_output_is_json_serializable(self, sgat_model, eeg_batch, env_batch, labels):
        result = sgat_explain.run_sgat_explain(
            sgat_model, eeg_batch, env_batch, labels, sfreq=64.0, n_boot=20, top_k_freq_channels=2,
            faithfulness_steps=3,
        )
        json.dumps(result)  # must not raise

    def test_all_top_level_keys_present(self, sgat_model, eeg_batch, env_batch, labels):
        result = sgat_explain.run_sgat_explain(
            sgat_model, eeg_batch, env_batch, labels, sfreq=64.0, n_boot=20, top_k_freq_channels=2,
            faithfulness_steps=3,
        )
        for key in (
            "channel_occlusion_importance", "channel_permutation_importance",
            "attention_summary_a", "attention_summary_b",
            "frequency_band_contribution", "faithfulness_comparison", "sanity_check",
        ):
            assert key in result
