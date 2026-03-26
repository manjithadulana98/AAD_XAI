"""Tests for XAI modules: IG, probes, faithfulness, sanity checks, perturbations."""
import pytest
import torch
import numpy as np

from aad_xai.models.aadnet import AADNet
from aad_xai.xai.integrated_gradients import ig_attribution
from aad_xai.xai.faithfulness import deletion_curve, insertion_curve
from aad_xai.xai.sanity_checks import randomize_parameters, cascading_randomization
from aad_xai.xai.perturbations import band_limited_attenuation, suppress_lag_range, remove_channel_group
from aad_xai.xai.lrp import compute_lrp


class TestIG:
    def test_ig_shape(self):
        model = AADNet(n_channels=16, n_classes=2)
        x = torch.randn(2, 16, 64)
        attr = ig_attribution(model, x, target=1, steps=8)
        assert attr.shape == x.shape


class TestFaithfulness:
    def test_deletion_curve_length(self):
        model = AADNet(n_channels=16, n_classes=2)
        x = torch.randn(2, 16, 32)
        y = torch.tensor([0, 1])
        imp = torch.randn_like(x)
        curve = deletion_curve(model, x, y, imp, steps=5)
        assert len(curve) == 6  # steps + 1

    def test_insertion_curve_length(self):
        model = AADNet(n_channels=16, n_classes=2)
        x = torch.randn(2, 16, 32)
        y = torch.tensor([0, 1])
        imp = torch.randn_like(x)
        curve = insertion_curve(model, x, y, imp, steps=5)
        assert len(curve) == 6

    def test_deletion_starts_high_ends_low(self):
        """Deletion of all features should reduce or maintain confidence."""
        model = AADNet(n_channels=8, n_classes=2)
        model.eval()
        x = torch.randn(4, 8, 32)
        y = torch.zeros(4, dtype=torch.long)
        imp = torch.randn_like(x)
        curve = deletion_curve(model, x, y, imp, steps=10)
        # Confidence at 0% deletion should be >= confidence at 100%
        assert curve[0] >= curve[-1] - 0.01  # small tolerance


class TestSanityChecks:
    def test_randomize_changes_params(self):
        model = AADNet(n_channels=16)
        rand_model = randomize_parameters(model)
        # At least one param should differ
        for p1, p2 in zip(model.parameters(), rand_model.parameters()):
            if not torch.equal(p1, p2):
                return
        pytest.fail("Randomized model has identical parameters")

    def test_cascading_returns_all_layers(self):
        model = AADNet(n_channels=16)
        x = torch.randn(2, 16, 32)

        def attr_fn(m, inp):
            return ig_attribution(m, inp, target=0, steps=4)

        results = cascading_randomization(model, attr_fn, x)
        assert "__original__" in results
        # Should have entries for named children
        assert len(results) >= 2


class TestPerturbations:
    def test_band_limited_attenuation(self):
        eeg = np.random.randn(8, 256)
        out = band_limited_attenuation(eeg, sfreq=64.0, low_hz=1.0, high_hz=4.0)
        assert out.shape == eeg.shape
        assert not np.allclose(out, eeg)

    def test_suppress_lag_range(self):
        eeg = np.random.randn(8, 256)
        out = suppress_lag_range(eeg, sfreq=64.0, tmin_s=0.1, tmax_s=0.3)
        assert out.shape == eeg.shape
        # Zeroed region
        i0, i1 = int(0.1 * 64), int(0.3 * 64)
        np.testing.assert_array_equal(out[:, i0:i1], 0.0)

    def test_remove_channel_group(self):
        eeg = np.random.randn(16, 128)
        out = remove_channel_group(eeg, ch_idx=[0, 3, 7])
        np.testing.assert_array_equal(out[0], 0.0)
        np.testing.assert_array_equal(out[3], 0.0)
        np.testing.assert_array_equal(out[7], 0.0)
        assert not np.allclose(out[1], 0.0)


class TestLRP:
    def test_lrp_stub_raises(self):
        model = AADNet(n_channels=8)
        x = torch.randn(1, 8, 32)
        with pytest.raises(NotImplementedError, match="LRP is not implemented"):
            compute_lrp(model, x, target=0)
