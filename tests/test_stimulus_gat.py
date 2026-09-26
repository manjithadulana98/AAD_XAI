"""Tests for the stimulus-conditioned GAT model components (StimulusGAT)."""
from __future__ import annotations

import torch

import math

from aad_xai.models.sgat_temporal_encoder import PerChannelTemporalEncoder
from aad_xai.models.sgat_frequency_encoder import PerChannelFrequencyEncoder
from aad_xai.models.sgat_speech_encoder import SharedSpeechEncoder
from aad_xai.models.sgat_gat_layer import CandidateConditionedGAT
from aad_xai.models.sgat import NodeFeatureFusion, StimulusGAT


class TestPerChannelTemporalEncoder:
    def test_forward_shape(self):
        model = PerChannelTemporalEncoder(n_channels=8, out_features=4)
        x = torch.randn(3, 8, 64)
        out = model(x)
        assert out.shape == (3, 8, 4)

    def test_gradient_flows(self):
        model = PerChannelTemporalEncoder(n_channels=8, out_features=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_no_cross_channel_leakage(self):
        """Perturbing one electrode's raw input must not change any other
        electrode's output features -- the regression test for the specific
        cross-channel-mixing flaw this encoder is designed to avoid (unlike
        ST-GCN's shared-Conv1d + re-projection temporal stage)."""
        model = PerChannelTemporalEncoder(n_channels=8, out_features=4)
        model.eval()  # avoid BatchNorm batch-statistics/dropout noise
        x = torch.randn(2, 8, 64)
        out_ref = model(x)

        x_perturbed = x.clone()
        x_perturbed[:, 0, :] += 5.0  # perturb only electrode 0
        out_perturbed = model(x_perturbed)

        # Electrode 0's own features may change; all others must not.
        assert not torch.allclose(out_ref[:, 0, :], out_perturbed[:, 0, :])
        assert torch.allclose(out_ref[:, 1:, :], out_perturbed[:, 1:, :], atol=1e-6)


class TestPerChannelFrequencyEncoder:
    def test_forward_shape(self):
        model = PerChannelFrequencyEncoder(n_channels=8, sfreq=64.0, out_features=4)
        x = torch.randn(3, 8, 128)
        out = model(x)
        assert out.shape == (3, 8, 4)

    def test_gradient_flows(self):
        model = PerChannelFrequencyEncoder(n_channels=8, sfreq=64.0, out_features=4)
        x = torch.randn(2, 8, 128, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_band_power_sanity(self):
        """A pure 9 Hz sinusoid (alpha1 band, 8-10 Hz) should dominate the
        alpha1 log band-power relative to every other band."""
        sfreq = 64.0
        n_times = 256
        t = torch.arange(n_times) / sfreq
        sine = torch.sin(2 * math.pi * 9.0 * t)
        x = sine.view(1, 1, -1).repeat(1, 3, 1)  # (B=1, C=3, T)

        model = PerChannelFrequencyEncoder(n_channels=3, sfreq=sfreq)
        band_power = model._band_power(x)  # (1, C, n_bands), no learned params involved
        alpha1_idx = model.bands_hz.index((8.0, 10.0))

        for ch in range(3):
            powers = band_power[0, ch]
            assert torch.argmax(powers).item() == alpha1_idx


class TestSharedSpeechEncoder:
    def test_forward_shape(self):
        model = SharedSpeechEncoder(embed_dim=6)
        env = torch.randn(4, 64)
        out = model(env)
        assert out.shape == (4, 6)

    def test_gradient_flows(self):
        model = SharedSpeechEncoder(embed_dim=6)
        env = torch.randn(2, 64, requires_grad=True)
        out = model(env)
        loss = out.sum()
        loss.backward()
        assert env.grad is not None
        assert env.grad.abs().sum() > 0


class TestNodeFeatureFusion:
    def test_forward_shape(self):
        model = NodeFeatureFusion(n_channels=5, in_features=6, out_features=3)
        temporal = torch.randn(2, 5, 4)
        freq = torch.randn(2, 5, 2)
        out = model(temporal, freq)
        assert out.shape == (2, 5, 3)

    def test_gradient_flows(self):
        model = NodeFeatureFusion(n_channels=5, in_features=6, out_features=3)
        temporal = torch.randn(2, 5, 4, requires_grad=True)
        freq = torch.randn(2, 5, 2, requires_grad=True)
        out = model(temporal, freq)
        loss = out.sum()
        loss.backward()
        assert temporal.grad is not None and temporal.grad.abs().sum() > 0
        assert freq.grad is not None and freq.grad.abs().sum() > 0


def _small_adjacency(n: int) -> torch.Tensor:
    """Deterministic small adjacency: a ring (each node connects to its two
    immediate neighbors), used across GAT/StimulusGAT tests to avoid touching
    real montage CSVs."""
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[i, (i - 1) % n] = 1.0
    return A


class TestCandidateConditionedGAT:
    def test_forward_shape(self):
        n = 5
        adj = _small_adjacency(n)
        model = CandidateConditionedGAT(
            n_channels=n, in_features=6, speech_dim=4, out_features=8, adjacency=adj, n_heads=2,
        )
        node_feats = torch.randn(3, n, 6)
        speech = torch.randn(3, 4)
        out = model(node_feats, speech)
        assert out.shape == (3, n, 8)

    def test_gradient_flows(self):
        n = 5
        adj = _small_adjacency(n)
        model = CandidateConditionedGAT(
            n_channels=n, in_features=6, speech_dim=4, out_features=8, adjacency=adj, n_heads=2,
        )
        node_feats = torch.randn(2, n, 6, requires_grad=True)
        speech = torch.randn(2, 4, requires_grad=True)
        out = model(node_feats, speech)
        loss = out.sum()
        loss.backward()
        assert node_feats.grad is not None and node_feats.grad.abs().sum() > 0
        assert speech.grad is not None and speech.grad.abs().sum() > 0

    def test_attention_respects_adjacency(self):
        n = 5
        adj = _small_adjacency(n)
        model = CandidateConditionedGAT(
            n_channels=n, in_features=6, speech_dim=4, out_features=8, adjacency=adj, n_heads=2, dropout=0.0,
        )
        model.eval()
        node_feats = torch.randn(2, n, 6)
        speech = torch.randn(2, 4)
        out, attn = model(node_feats, speech, return_attention_weights=True)
        assert attn.shape == (2, 2, n, n)

        adj_mask = model.adj_mask  # (n, n) bool, includes self-loops
        # Masked (non-adjacent) entries carry exactly zero attention mass.
        assert torch.all(attn[:, :, ~adj_mask] == 0)
        # Each row's attention mass over its valid neighbors sums to 1.
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_film_conditioning_changes_output(self):
        n = 5
        adj = _small_adjacency(n)
        model = CandidateConditionedGAT(
            n_channels=n, in_features=6, speech_dim=4, out_features=8, adjacency=adj, n_heads=2, dropout=0.0,
        )
        model.eval()
        node_feats = torch.randn(2, n, 6)
        speech_1 = torch.randn(2, 4)
        speech_2 = torch.randn(2, 4)
        out_1 = model(node_feats, speech_1)
        out_2 = model(node_feats, speech_2)
        assert not torch.allclose(out_1, out_2)


def _make_stimulus_gat(n_channels: int = 6, sfreq: float = 64.0, **kwargs) -> StimulusGAT:
    adj = _small_adjacency(n_channels)
    defaults = dict(
        temporal_features=4, freq_features=4, node_features=8,
        speech_embed_dim=6, gat_hidden=8, gat_heads=2, dropout=0.0,
    )
    defaults.update(kwargs)
    return StimulusGAT(n_channels=n_channels, sfreq=sfreq, adjacency=adj, **defaults)


class TestStimulusGAT:
    def test_forward_shape(self):
        model = _make_stimulus_gat()
        eeg = torch.randn(3, 6, 128)
        env = torch.randn(3, 2, 128)
        out = model(eeg, env)
        assert out.shape == (3, 2)

    def test_gradient_flows(self):
        model = _make_stimulus_gat()
        eeg = torch.randn(2, 6, 128, requires_grad=True)
        env = torch.randn(2, 2, 128)
        out = model(eeg, env)
        loss = out.sum()
        loss.backward()
        assert eeg.grad is not None
        assert eeg.grad.abs().sum() > 0

    def test_custom_adjacency(self):
        n = 4
        adj = torch.eye(n) + torch.randn(n, n).abs() * 0.1
        model = StimulusGAT(n_channels=n, sfreq=64.0, adjacency=adj)
        eeg = torch.randn(2, n, 128)
        env = torch.randn(2, 2, 128)
        out = model(eeg, env)
        assert out.shape == (2, 2)

    def test_attention_weights_introspection(self):
        model = _make_stimulus_gat()
        model.eval()
        eeg = torch.randn(3, 6, 128)
        env = torch.randn(3, 2, 128)
        logits, attn = model(eeg, env, return_attention_weights=True)
        assert logits.shape == (3, 2)
        assert attn["attn_a"].shape == (3, 2, 6, 6)  # (B, heads, N, N)
        assert attn["attn_b"].shape == (3, 2, 6, 6)

    def test_shared_weight_symmetry(self):
        """The same shared speech_encoder/gat must produce both scores, with
        no learned per-candidate asymmetry: swapping which candidate occupies
        env[:,0] vs env[:,1] must swap the scores correspondingly."""
        model = _make_stimulus_gat()
        model.eval()
        eeg = torch.randn(3, 6, 128)
        env = torch.randn(3, 2, 128)

        logits = model(eeg, env)
        logits_swapped = model(eeg, env[:, [1, 0], :])

        assert torch.allclose(logits_swapped[:, 0], logits[:, 1], atol=1e-5)
        assert torch.allclose(logits_swapped[:, 1], logits[:, 0], atol=1e-5)

    def test_genuine_weight_sharing(self):
        """speech_encoder and gat must each be a single submodule invoked
        twice per forward call (not two separately-parameterized copies)."""
        model = _make_stimulus_gat()
        call_counts = {"speech_encoder": 0, "gat": 0}

        def _count(name):
            def hook(module, inputs, output):
                call_counts[name] += 1
            return hook

        model.speech_encoder.register_forward_hook(_count("speech_encoder"))
        model.gat.register_forward_hook(_count("gat"))

        eeg = torch.randn(2, 6, 128)
        env = torch.randn(2, 2, 128)
        model(eeg, env)

        assert call_counts["speech_encoder"] == 2
        assert call_counts["gat"] == 2

        # No separate per-candidate submodules exist anywhere in the model.
        names = [n for n, _ in model.named_modules()]
        assert not any("_a" in n or "_b" in n for n in names)

        # A loss on score_a alone still back-props through the shared modules.
        model.zero_grad()
        eeg2 = torch.randn(2, 6, 128)
        env2 = torch.randn(2, 2, 128)
        logits = model(eeg2, env2)
        score_a = logits[:, 0]
        score_a.sum().backward()

        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.speech_encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gat.parameters())
