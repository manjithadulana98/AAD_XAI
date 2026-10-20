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

    def test_output_is_normalized(self):
        """Regression test for the LayerNorm fix (Oct 2026): per-node output
        must have ~zero mean / ~unit std over the out_features axis,
        regardless of input scale -- this is what keeps node_fusion's output
        bounded as its proj weights evolve during training (the observed
        failure mode: unbounded output growth saturating the GAT's attention
        softmax into a degenerate near-one-hot pattern)."""
        model = NodeFeatureFusion(n_channels=5, in_features=6, out_features=8)
        # Large input scale -- without LayerNorm this would produce a
        # correspondingly large-magnitude output.
        temporal = torch.randn(3, 5, 4) * 100.0
        freq = torch.randn(3, 5, 2) * 100.0
        out = model(temporal, freq)
        means = out.mean(dim=-1)
        stds = out.std(dim=-1, unbiased=False)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)
        assert torch.allclose(stds, torch.ones_like(stds), atol=1e-1)

    def test_output_scale_bounded_across_simulated_weight_growth(self):
        """Simulate several large gradient-style updates to proj's weights
        (the mechanism that produced unbounded growth pre-fix) and confirm
        output std stays bounded post-fix, rather than scaling with the
        weight magnitude."""
        model = NodeFeatureFusion(n_channels=4, in_features=3, out_features=6)
        temporal = torch.randn(2, 4, 2)
        freq = torch.randn(2, 4, 1)

        stds = []
        with torch.no_grad():
            for _ in range(5):
                model.proj.weight.mul_(3.0)  # simulate runaway weight growth
                out = model(temporal, freq)
                stds.append(out.std().item())

        # Despite proj's weights growing by 3^5 ~= 243x, LayerNorm keeps the
        # final output's overall scale close to the first iteration's, not
        # growing proportionally with the weight magnitude.
        assert stds[-1] < stds[0] * 3  # loose bound -- NOT ~243x like pre-fix


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
        # Note: speech.grad is EXACTLY ZERO at init with the FiLM-init fix,
        # because film.weight is initialized to zero -- so d(out)/d(speech) =
        # film.weight * ... = 0. This is the intended behavior: the gradient
        # flows into film.weight (which updates film's response to speech) on
        # the first backward pass, giving the FiLM mechanism a non-degenerate
        # gradient regime to escape from, rather than relying on each
        # candidate's noise-level FiLM output to break symmetry. Check
        # film.weight.grad instead -- THAT is what must receive signal.
        assert model.film.weight.grad is not None and model.film.weight.grad.abs().sum() > 0

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

    def test_film_init_breaks_candidate_symmetry_at_epoch_1(self):
        """Regression test for the FiLM-init fix (Oct 18). film.weight must be
        exactly zero and film.bias[:in_features] must be exactly 0.1 at init,
        so FiLM output at epoch 1 is non-zero (gamma=0.1, beta=0) regardless
        of speech input. This gives film.weight real gradient signal from the
        first backward pass; the prior default-Kaiming init produced gamma,
        beta ~= 0, which combined with the model's two identical candidate
        passes at init to leave FiLM in a near-degenerate gradient regime
        that the full-dataset LOSO training couldn't escape. If this test
        fails, re-check sgat_gat_layer.py's CandidateConditionedGAT.__init__
        -- someone probably ran an auto-reformat or revert that stripped the
        explicit init."""
        n, in_features = 5, 6
        adj = _small_adjacency(n)
        model = CandidateConditionedGAT(
            n_channels=n, in_features=in_features, speech_dim=4, out_features=8,
            adjacency=adj, n_heads=2, dropout=0.0,
        )
        assert torch.all(model.film.weight == 0.0)
        assert torch.all(model.film.bias[:in_features] == 0.1)
        assert torch.all(model.film.bias[in_features:] == 0.0)

        # Direct behavioral check: FiLM output on any input is (gamma=0.1, beta=0).
        model.eval()
        speech = torch.randn(3, 4)
        gamma, beta = model.film(speech).chunk(2, dim=-1)
        assert torch.allclose(gamma, torch.full_like(gamma, 0.1))
        assert torch.allclose(beta, torch.zeros_like(beta))

    def test_film_conditioning_changes_output_after_training(self):
        """FiLM is INTENTIONALLY input-independent at init (film.weight = 0,
        so FiLM(speech) is a constant gamma=0.1, beta=0 regardless of speech).
        That's what the FiLM-init fix guarantees. The property that FiLM
        becomes genuinely input-sensitive after training is checked by
        simulating one gradient step that nudges film.weight away from zero
        -- which is all the full training needs to escape the symmetric
        starting point."""
        n = 5
        adj = _small_adjacency(n)
        model = CandidateConditionedGAT(
            n_channels=n, in_features=6, speech_dim=4, out_features=8, adjacency=adj, n_heads=2, dropout=0.0,
        )
        # Manually perturb film.weight to simulate "after at least one update step"
        with torch.no_grad():
            model.film.weight.add_(torch.randn_like(model.film.weight) * 0.1)
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

    def test_readout_input_is_normalized_regardless_of_gat_output_scale(self):
        """Regression test for the readout-LayerNorm fix (Oct 2026): scaling
        the GAT's pooled output by a large common factor must not change the
        readout's score proportionally -- LayerNorm bounds the SHARED
        (candidate-independent) magnitude that was previously free to grow
        without affecting loss (2-class softmax cross-entropy is exactly
        shift-invariant to equal growth in both candidates' scores)."""
        model = _make_stimulus_gat()
        model.eval()

        captured = {}

        def _capture(module, inputs, output):
            captured["readout_input"] = inputs[0]

        model.readout.register_forward_hook(_capture)

        eeg = torch.randn(2, 6, 128)
        env = torch.randn(2, 2, 128)
        model(eeg, env)
        readout_input = captured["readout_input"]

        # LayerNorm output (readout's actual input) must be ~zero-mean/unit-std
        # per sample, regardless of whatever scale gat_out.mean(dim=1) had.
        means = readout_input.mean(dim=-1)
        stds = readout_input.std(dim=-1, unbiased=False)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)
        assert torch.allclose(stds, torch.ones_like(stds), atol=1e-1)

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
        # Nudge film.weight off zero first -- at init it's exactly zero
        # (FiLM-init fix), so d(output)/d(speech_encoder params) = film.weight
        # * ... = 0 through FiLM on epoch 1. This test checks the structural
        # weight-sharing property that applies AFTER the first update step
        # has moved film off its symmetric starting point.
        with torch.no_grad():
            model.gat.film.weight.add_(torch.randn_like(model.gat.film.weight) * 0.1)

        model.zero_grad()
        eeg2 = torch.randn(2, 6, 128)
        env2 = torch.randn(2, 2, 128)
        logits = model(eeg2, env2)
        score_a = logits[:, 0]
        score_a.sum().backward()

        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.speech_encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gat.parameters())
