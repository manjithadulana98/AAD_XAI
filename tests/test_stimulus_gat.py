"""Tests for the stimulus-conditioned GAT model components (StimulusGAT)."""
from __future__ import annotations

import pytest
import torch

import math

from aad_xai.models.sgat_temporal_encoder import PerChannelTemporalEncoder
from aad_xai.models.sgat_frequency_encoder import PerChannelFrequencyEncoder
from aad_xai.models.sgat_speech_encoder import SharedSpeechEncoder
from aad_xai.models.sgat_film import TimeResolvedFiLM
from aad_xai.models.sgat_gat_layer import CandidateConditionedGAT
from aad_xai.models.sgat import NodeFeatureFusion, StimulusGAT


class TestPerChannelTemporalEncoder:
    def test_forward_shape(self):
        model = PerChannelTemporalEncoder(n_channels=8, out_features=4)
        x = torch.randn(3, 8, 64)
        out = model(x)
        assert out.shape == (3, 8, 4, 64)  # time axis kept for TimeResolvedFiLM

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
        assert out.shape == (4, 6, 64)  # a sequence, not a pooled vector

    def test_gradient_flows(self):
        model = SharedSpeechEncoder(embed_dim=6)
        env = torch.randn(2, 64, requires_grad=True)
        out = model(env)
        loss = out.sum()
        loss.backward()
        assert env.grad is not None
        assert env.grad.abs().sum() > 0


class TestTimeResolvedFiLM:
    def _perturbed(self, speech_dim=3, n_features=4):
        film = TimeResolvedFiLM(speech_dim=speech_dim, n_features=n_features)
        with torch.no_grad():
            film.proj.weight.add_(torch.randn_like(film.proj.weight))
        return film

    def test_forward_shape(self):
        film = TimeResolvedFiLM(speech_dim=3, n_features=4)
        out = film(torch.randn(2, 5, 4, 32), torch.randn(2, 3, 32))
        assert out.shape == (2, 5, 4)

    def test_init_is_speech_independent(self):
        """Zero weight, gamma bias 0.1, beta bias 0: output is 1.1 * mean_t(h)
        for any speech input, so both candidates start identical."""
        film = TimeResolvedFiLM(speech_dim=3, n_features=4)
        assert torch.all(film.proj.weight == 0.0)
        h = torch.randn(2, 5, 4, 32)
        out = film(h, torch.randn(2, 3, 32))
        assert torch.allclose(out, 1.1 * h.mean(dim=-1), atol=1e-6)

    def test_matches_explicit_modulate_then_pool(self):
        film = self._perturbed()
        h, s = torch.randn(2, 5, 4, 32), torch.randn(2, 3, 32)
        gamma, beta = film.proj(s).chunk(2, dim=1)                       # (B, F, T)
        expected = (h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)).mean(dim=-1)
        assert torch.allclose(film(h, s), expected, atol=1e-5)

    def test_sensitive_to_temporal_alignment(self):
        """The property the pooled-then-FiLM design lacked: circularly shifting
        speech in time leaves every time-averaged statistic of both streams
        unchanged but breaks their alignment, so the output must change."""
        film = self._perturbed()
        h, s = torch.randn(2, 5, 4, 64), torch.randn(2, 3, 64)
        s_shifted = torch.roll(s, shifts=17, dims=-1)
        assert torch.allclose(s.mean(-1), s_shifted.mean(-1), atol=1e-6)
        assert not torch.allclose(film(h, s), film(h, s_shifted), atol=1e-3)

    def test_gradient_reaches_proj_weight_at_init(self):
        film = TimeResolvedFiLM(speech_dim=3, n_features=4)
        film(torch.randn(2, 5, 4, 32), torch.randn(2, 3, 32)).pow(2).sum().backward()
        assert film.proj.weight.grad.abs().sum() > 0


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
            n_channels=n, in_features=6, out_features=8, adjacency=adj, n_heads=2,
        )
        out = model(torch.randn(3, n, 6))
        assert out.shape == (3, n, 8)

    def test_gradient_flows(self):
        n = 5
        adj = _small_adjacency(n)
        model = CandidateConditionedGAT(
            n_channels=n, in_features=6, out_features=8, adjacency=adj, n_heads=2,
        )
        node_feats = torch.randn(2, n, 6, requires_grad=True)
        model(node_feats).sum().backward()
        assert node_feats.grad is not None and node_feats.grad.abs().sum() > 0

    def test_attention_respects_adjacency(self):
        n = 5
        adj = _small_adjacency(n)
        model = CandidateConditionedGAT(
            n_channels=n, in_features=6, out_features=8, adjacency=adj, n_heads=2, dropout=0.0,
        )
        model.eval()
        node_feats = torch.randn(2, n, 6)
        out, attn = model(node_feats, return_attention_weights=True)
        assert attn.shape == (2, 2, n, n)

        adj_mask = model.adj_mask  # (n, n) bool, includes self-loops
        # Masked (non-adjacent) entries carry exactly zero attention mass.
        assert torch.all(attn[:, :, ~adj_mask] == 0)
        # Each row's attention mass over its valid neighbors sums to 1.
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


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
        """Both candidates go through the same speech_encoder and gat instance
        in one batched call of size 2B (not two parameterized copies)."""
        model = _make_stimulus_gat()
        batch_sizes = {"speech_encoder": [], "gat": []}

        def _record(name):
            def hook(module, inputs, output):
                batch_sizes[name].append(inputs[0].shape[0])
            return hook

        model.speech_encoder.register_forward_hook(_record("speech_encoder"))
        model.gat.register_forward_hook(_record("gat"))

        eeg = torch.randn(2, 6, 128)
        env = torch.randn(2, 2, 128)
        model(eeg, env)

        assert batch_sizes["speech_encoder"] == [4]
        assert batch_sizes["gat"] == [4]

        # No separate per-candidate submodules exist anywhere in the model.
        names = [n for n, _ in model.named_modules()]
        assert not any("_a" in n or "_b" in n for n in names)

        # A loss on score_a alone still back-props through the shared modules.
        # FiLM's weight is zero at init, which blocks gradient to the speech
        # encoder, so nudge it to simulate "after the first update step".
        with torch.no_grad():
            model.film.proj.weight.add_(torch.randn_like(model.film.proj.weight) * 0.1)

        model.zero_grad()
        eeg2 = torch.randn(2, 6, 128)
        env2 = torch.randn(2, 2, 128)
        logits = model(eeg2, env2)
        score_a = logits[:, 0]
        score_a.sum().backward()

        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.speech_encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gat.parameters())

    def test_readout_has_no_bias(self):
        """A readout bias shifts both candidates' scores equally, so it can't
        change any prediction; it was observed drifting under Adam anyway."""
        assert _make_stimulus_gat().readout.bias is None

    def test_candidates_identical_at_init(self):
        model = _make_stimulus_gat()
        model.eval()
        logits = model(torch.randn(3, 6, 128), torch.randn(3, 2, 128))
        assert torch.allclose(logits[:, 0], logits[:, 1], atol=1e-6)

    def test_scores_depend_on_eeg_speech_alignment(self):
        """End-to-end version of TimeResolvedFiLM's alignment test: shifting a
        candidate's envelope in time (same marginal statistics, broken
        alignment with the EEG) must change that candidate's score."""
        model = _make_stimulus_gat()
        with torch.no_grad():
            model.film.proj.weight.add_(torch.randn_like(model.film.proj.weight))
        model.eval()
        eeg, env = torch.randn(3, 6, 128), torch.randn(3, 2, 128)
        env_shifted = env.clone()
        env_shifted[:, 0] = torch.roll(env[:, 0], shifts=40, dims=-1)
        logits, logits_shifted = model(eeg, env), model(eeg, env_shifted)
        assert not torch.allclose(logits[:, 0], logits_shifted[:, 0], atol=1e-4)
        assert torch.allclose(logits[:, 1], logits_shifted[:, 1], atol=1e-5)


class TestStimulusGATMultiBandSpeech:
    """speech_bands=K: each candidate's speech is K sub-band envelopes (B, 2, K, T)."""

    K = 5

    def test_default_is_single_band(self):
        model = _make_stimulus_gat()
        assert model.speech_bands == 1
        assert model.speech_encoder.net[0].weight.shape[1] == 1  # Conv1d(1, hidden, k)

    def test_encoder_reads_all_bands(self):
        enc = SharedSpeechEncoder(embed_dim=6, hidden=4, dropout=0.0, in_channels=self.K)
        assert enc.net[0].weight.shape[1] == self.K
        assert enc(torch.randn(3, self.K, 64)).shape == (3, 6, 64)
        with pytest.raises(ValueError):
            enc(torch.randn(3, 64))  # single-band input to a K-band encoder

    def test_forward_shape(self):
        model = _make_stimulus_gat(speech_bands=self.K)
        assert model(torch.randn(3, 6, 128), torch.randn(3, 2, self.K, 128)).shape == (3, 2)

    def test_wrong_env_shape_raises(self):
        with pytest.raises(ValueError):
            _make_stimulus_gat(speech_bands=self.K)(torch.randn(2, 6, 128), torch.randn(2, 2, 128))
        with pytest.raises(ValueError):
            _make_stimulus_gat()(torch.randn(2, 6, 128), torch.randn(2, 2, self.K, 128))

    def test_shared_weight_symmetry(self):
        model = _make_stimulus_gat(speech_bands=self.K)
        model.eval()
        eeg, env = torch.randn(3, 6, 128), torch.randn(3, 2, self.K, 128)
        logits, swapped = model(eeg, env), model(eeg, env[:, [1, 0]])
        assert torch.allclose(swapped[:, 0], logits[:, 1], atol=1e-5)
        assert torch.allclose(swapped[:, 1], logits[:, 0], atol=1e-5)

    def test_one_band_of_one_candidate_moves_only_its_score(self):
        model = _make_stimulus_gat(speech_bands=self.K)
        with torch.no_grad():
            model.film.proj.weight.add_(torch.randn_like(model.film.proj.weight))
        model.eval()
        eeg, env = torch.randn(3, 6, 128), torch.randn(3, 2, self.K, 128)
        env2 = env.clone()
        env2[:, 0, 2] = torch.roll(env[:, 0, 2], shifts=40, dims=-1)
        logits, logits2 = model(eeg, env), model(eeg, env2)
        assert not torch.allclose(logits[:, 0], logits2[:, 0], atol=1e-4)
        assert torch.allclose(logits[:, 1], logits2[:, 1], atol=1e-5)
