"""Tests for VLAAI XAI modules.

Tests cover:
  - VLAAIPyTorch forward pass shape
  - AADDecisionWrapper + AADDecisionEEGOnly output shape
  - GradCAM attribution shape
  - SHAP (DeepSHAP) attribution shape
  - LIME attribution structure
  - Probing utilities (recursive activation extraction)
  - Probe visualization (no-crash)
  - DTU dataset adapter loading
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vlaai_model():
    from aad_xai.models.vlaai_pytorch import VLAAIPyTorch
    model = VLAAIPyTorch(nb_blocks=2, input_channels=64, output_dim=1)
    model.eval()
    return model


@pytest.fixture
def eeg_batch():
    """Synthetic EEG: (B=4, T=320, C=64)"""
    torch.manual_seed(42)
    return torch.randn(4, 320, 64)


@pytest.fixture
def env_att(eeg_batch):
    """Synthetic attended envelope: (B, T, 1)"""
    return torch.randn(eeg_batch.shape[0], eeg_batch.shape[1], 1)


@pytest.fixture
def env_unatt(eeg_batch):
    """Synthetic unattended envelope: (B, T, 1)"""
    return torch.randn(eeg_batch.shape[0], eeg_batch.shape[1], 1)


# ---------------------------------------------------------------------------
# VLAAIPyTorch
# ---------------------------------------------------------------------------

class TestVLAAIPyTorch:
    def test_forward_shape(self, vlaai_model, eeg_batch):
        out = vlaai_model(eeg_batch)
        assert out.shape == (4, 320, 1), f"Expected (4, 320, 1), got {out.shape}"

    def test_gradient_flow(self, vlaai_model, eeg_batch):
        eeg_batch.requires_grad_(True)
        out = vlaai_model(eeg_batch)
        loss = out.mean()
        loss.backward()
        assert eeg_batch.grad is not None
        assert eeg_batch.grad.abs().sum() > 0

    def test_named_modules_exist(self, vlaai_model):
        names = [n for n, _ in vlaai_model.named_modules() if n]
        assert "extractor" in names
        assert "output_context" in names
        assert "final_dense" in names


# ---------------------------------------------------------------------------
# Decision Wrappers
# ---------------------------------------------------------------------------

class TestAADDecisionWrapper:
    def test_forward_shape(self, vlaai_model, eeg_batch, env_att, env_unatt):
        from aad_xai.models.vlaai_decision import AADDecisionWrapper
        wrapper = AADDecisionWrapper(vlaai_model)
        logits = wrapper(eeg_batch, env_att, env_unatt)
        assert logits.shape == (4, 2)

    def test_eeg_only_shape(self, vlaai_model, eeg_batch, env_att, env_unatt):
        from aad_xai.models.vlaai_decision import AADDecisionEEGOnly
        wrapper = AADDecisionEEGOnly(vlaai_model)
        wrapper.set_envelopes(env_att, env_unatt)
        logits = wrapper(eeg_batch)
        assert logits.shape == (4, 2)

    def test_differentiable(self, vlaai_model, eeg_batch, env_att, env_unatt):
        from aad_xai.models.vlaai_decision import AADDecisionEEGOnly
        wrapper = AADDecisionEEGOnly(vlaai_model)
        wrapper.set_envelopes(env_att, env_unatt)
        eeg_batch.requires_grad_(True)
        logits = wrapper(eeg_batch)
        logits[:, 1].sum().backward()
        assert eeg_batch.grad is not None


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

class TestGradCAM:
    def test_gradcam_attribution_shape(self, vlaai_model, eeg_batch, env_att, env_unatt):
        from aad_xai.models.vlaai_decision import AADDecisionEEGOnly
        from aad_xai.xai.gradcam import gradcam_attribution

        wrapper = AADDecisionEEGOnly(vlaai_model)
        wrapper.set_envelopes(env_att[:2], env_unatt[:2])

        # Find a conv layer
        target_conv = None
        for name, mod in wrapper.named_modules():
            if isinstance(mod, nn.Conv1d):
                target_conv = mod
                break
        assert target_conv is not None

        attr = gradcam_attribution(wrapper, eeg_batch[:2], target_class=1, layer=target_conv)
        assert attr.shape[0] == 2
        assert attr.ndim >= 2

    def test_temporal_heatmap(self, vlaai_model, eeg_batch, env_att, env_unatt):
        from aad_xai.models.vlaai_decision import AADDecisionEEGOnly
        from aad_xai.xai.gradcam import gradcam_temporal_heatmap

        wrapper = AADDecisionEEGOnly(vlaai_model)
        wrapper.set_envelopes(env_att[:2], env_unatt[:2])

        target_conv = None
        for mod in wrapper.modules():
            if isinstance(mod, nn.Conv1d):
                target_conv = mod
        assert target_conv is not None

        heatmap = gradcam_temporal_heatmap(wrapper, eeg_batch[:2], target_class=1, layer=target_conv)
        assert heatmap.shape[0] == 2
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

class TestProbing:
    def test_extract_activations_recursive(self, vlaai_model, eeg_batch):
        from aad_xai.xai.probes_vlaai import extract_all_activations_pt

        acts = extract_all_activations_pt(vlaai_model, eeg_batch, recursive=True)
        assert len(acts) > 3, f"Expected >3 layers, got {len(acts)}"
        # Each activation should have batch dim = 4
        for name, act in acts.items():
            assert act.shape[0] == 4, f"{name}: bad batch dim {act.shape[0]}"

    def test_extract_activations_shallow(self, vlaai_model, eeg_batch):
        from aad_xai.xai.probes_vlaai import extract_all_activations_pt

        acts = extract_all_activations_pt(vlaai_model, eeg_batch, recursive=False)
        assert len(acts) >= 3  # extractor, output_context, final_dense, block_denses

    def test_attention_probes(self, vlaai_model, eeg_batch):
        from aad_xai.xai.probes_vlaai import extract_all_activations_pt, attention_decoding_probes

        acts = extract_all_activations_pt(vlaai_model, eeg_batch, recursive=False)
        # Create synthetic binary labels
        labels = np.array([0, 1, 0, 1])
        results = attention_decoding_probes(acts, labels, seed=42)
        assert len(results) > 0
        for name, acc in results.items():
            assert 0.0 <= acc <= 1.0 or np.isnan(acc)

    def test_auditory_probes(self, vlaai_model, eeg_batch, env_att):
        from aad_xai.xai.probes_vlaai import extract_all_activations_pt, auditory_feature_probes

        acts = extract_all_activations_pt(vlaai_model, eeg_batch, recursive=False)
        envs = env_att.cpu().numpy()
        results = auditory_feature_probes(acts, envs, seed=42)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Probe Visualization (no-crash tests)
# ---------------------------------------------------------------------------

class TestProbeViz:
    def test_plot_probe_accuracy(self, tmp_path):
        from aad_xai.xai.probe_viz import plot_probe_accuracy_by_layer
        data = {"layer_0": 0.65, "layer_1": 0.72, "layer_2": 0.58}
        fig = plot_probe_accuracy_by_layer(data, save_path=tmp_path / "test.png")
        assert (tmp_path / "test.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save_probe_results(self, tmp_path):
        from aad_xai.xai.probe_viz import save_probe_results
        import json
        data = {"attention": {"l0": 0.5, "l1": 0.7}}
        save_probe_results(data, tmp_path / "results.json")
        loaded = json.loads((tmp_path / "results.json").read_text())
        assert loaded["attention"]["l0"] == 0.5


# ---------------------------------------------------------------------------
# LIME (quick structural test)
# ---------------------------------------------------------------------------

class TestLIME:
    def test_lime_output_structure(self, eeg_batch):
        from aad_xai.xai.lime_explainer import lime_attribution

        def _dummy_predict(batch):
            B = batch.shape[0]
            return np.column_stack([np.full(B, 0.3), np.full(B, 0.7)])

        result = lime_attribution(
            _dummy_predict,
            eeg_batch[0].cpu().numpy(),
            n_time_bins=5,
            n_samples=50,
            target_class=1,
        )
        assert "importance" in result
        assert "heatmap_channels" in result
        assert "heatmap_time" in result
        assert len(result["heatmap_channels"]) == 64
        assert len(result["heatmap_time"]) == 320


# ---------------------------------------------------------------------------
# DTU Dataset (structural — actual data may not be present in CI)
# ---------------------------------------------------------------------------

class TestVLAAIDTUDataset:
    def test_window_data(self):
        from aad_xai.data.vlaai_dataset import window_data

        data = np.random.randn(1000, 64).astype(np.float32)
        windowed = window_data(data, 320, 64)
        expected_n = (1000 - 320) // 64
        assert windowed.shape == (expected_n, 320, 64)

    def test_dataset_with_synthetic_data(self, tmp_path):
        """Create synthetic .npz files and test dataset loading."""
        from aad_xai.data.vlaai_dataset import VLAAIDTUDataset

        # Create 3 synthetic trials for subject S1
        for i in range(3):
            eeg = np.random.randn(1000, 64).astype(np.float32)
            env = np.random.randn(1000, 1).astype(np.float32)
            np.savez(tmp_path / f"DTU_S1_{i:03d}.npz", eeg=eeg, envelope=env)

        ds = VLAAIDTUDataset(str(tmp_path), window_length=320, hop=64)
        assert len(ds) > 0
        eeg, att, unatt, label = ds[0]
        assert eeg.shape == (320, 64)
        assert att.shape == (320, 1)
        assert unatt.shape == (320, 1)
        assert label.item() == 1
