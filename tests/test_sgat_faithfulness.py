"""Faithfulness harness for StimulusGAT explanations (src/aad_xai/xai/sgat_faithfulness.py)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from aad_xai.xai import sgat_faithfulness as sf
from tests.test_stimulus_gat import _make_stimulus_gat


class OnlyChannelZero(nn.Module):
    """Toy decoder that, by construction, uses electrode 0 only: the score of
    candidate k is the mean of eeg[:, 0] * env[:, k]."""

    def forward(self, eeg, env):
        return torch.stack([(eeg[:, 0] * env[:, k]).mean(-1) for k in range(2)], dim=1)   # (B, 2)


def _toy_data(n=24, C=6, T=128, seed=0):
    """Electrode 0 carries the attended envelope; the rest is noise."""
    g = torch.Generator().manual_seed(seed)
    env = torch.randn(n, 2, T, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    x = torch.randn(n, C, T, generator=g)
    x[:, 0] = 2.0 * env[torch.arange(n), y] + 0.5 * torch.randn(n, T, generator=g)
    return x, env, y


# --- electrode-space scoring ----------------------------------------------------------

def test_scorer_matches_model_on_aligned_zscored_input():
    torch.manual_seed(0)
    model = _make_stimulus_gat().eval()
    x, env = torch.randn(4, 6, 128), torch.randn(4, 2, 128)
    W = torch.eye(6) + 0.1 * torch.randn(6, 6)
    xa = torch.einsum("ij,bjt->bit", W, x)
    xa = (xa - xa.mean(-1, keepdim=True)) / (xa.std(-1, unbiased=False, keepdim=True) + 1e-8)
    with torch.no_grad():
        torch.testing.assert_close(sf.ElectrodeSpaceScorer(model, W)(x, env), model(xa, env))


def test_alignment_locality():
    assert sf.alignment_locality(np.eye(5), remove_common_mode=False) == 1.0
    assert sf.alignment_locality(np.ones((4, 4)), remove_common_mode=False) == pytest.approx(0.25)
    # a large common-mode term in W does not count as mixing for average-referenced data
    W = np.eye(5) + 30.0 * np.ones((5, 5)) / 5
    assert sf.alignment_locality(W, remove_common_mode=False) < 0.3     # looks heavily mixed
    assert sf.alignment_locality(W) == pytest.approx(sf.alignment_locality(np.eye(5)))


def test_cross_fit_halves_split_whole_trials():
    trials = np.array(["a", "a", "b", "c", "c", "d", "b"])
    h1, h2 = sf.cross_fit_halves(trials)
    assert not set(trials[h1]) & set(trials[h2]) and (h1 ^ h2).all()


# --- deletion and attributions on a model with a known answer ---------------------------

def test_delete_electrodes_modes():
    x = torch.arange(3 * 4 * 5, dtype=torch.float32).view(3, 4, 5)
    z = sf.delete_electrodes(x, [1], "zero")
    assert (z[:, 1] == 0).all() and torch.equal(z[:, [0, 2, 3]], x[:, [0, 2, 3]])
    p = sf.delete_electrodes(x, [1], "permute")
    assert not any(torch.equal(p[b, 1], x[b, 1]) for b in range(3))
    assert torch.equal(p[:, [0, 2, 3]], x[:, [0, 2, 3]])


def test_all_methods_find_the_only_informative_electrode():
    x, env, y = _toy_data()
    scorer = sf.ElectrodeSpaceScorer(OnlyChannelZero())
    occ = sf.occlusion_scores(scorer, x, env, y)
    ig = sf.ig_channel_scores(scorer, x, env, y, steps=16)
    assert np.argmax(occ) == 0 and np.argmax(ig) == 0
    assert np.abs(occ[1:]).max() < 1e-6                     # other electrodes are never read


def test_deletion_curve_drops_at_once_for_the_true_ranking():
    x, env, y = _toy_data()
    scorer = sf.ElectrodeSpaceScorer(OnlyChannelZero())
    true_first = sf.deletion_curve(scorer, x, env, y, [0, 1, 2, 3, 4, 5], ks=[0, 1])
    true_last = sf.deletion_curve(scorer, x, env, y, [1, 2, 3, 4, 5, 0], ks=[0, 1])
    assert true_first[0][1] == 1.0 and true_first[1][1] < 0.8   # accuracy collapses after 1 deletion
    assert true_last[1][1] == 1.0                                # deleting a noise electrode changes nothing
    kept = sf.deletion_curve(scorer, x, env, y, [0, 1, 2, 3, 4, 5], ks=[1], keep=True)
    assert kept[0][1] == 1.0                                     # keeping only electrode 0 is enough


def test_phase_randomized_keeps_spectrum_and_covariance():
    x, _, _ = _toy_data(n=3, T=256)
    s = sf.phase_randomized(x)
    torch.testing.assert_close(torch.fft.rfft(s).abs(), torch.fft.rfft(x).abs(), atol=1e-3, rtol=1e-4)
    cov = lambda v: torch.einsum("bct,bdt->bcd", v - v.mean(-1, keepdim=True), v - v.mean(-1, keepdim=True))
    torch.testing.assert_close(cov(s), cov(x), atol=1e-2, rtol=1e-3)
    assert not torch.allclose(s, x)


def test_integrated_gradients_completeness():
    torch.manual_seed(0)
    model = _make_stimulus_gat(film_pooling="corr").eval()
    scorer = sf.ElectrodeSpaceScorer(model)
    x, env = torch.randn(3, 6, 128), torch.randn(3, 2, 128)
    y = torch.tensor([0, 1, 0])
    attr = sf.integrated_gradients(scorer, x, env, y, steps=128, seed=5, batch=8)
    base = sf.phase_randomized(x, 5)
    with torch.no_grad():
        gap = sf.margin(scorer(x, env), y) - sf.margin(scorer(base, env), y)
    torch.testing.assert_close(attr.sum(dim=(1, 2)), gap, atol=2e-2, rtol=5e-2)


# --- attention and edges ------------------------------------------------------------------

def test_attended_attention_selects_each_windows_attended_candidate():
    torch.manual_seed(0)
    model = _make_stimulus_gat(film_pooling="corr").eval()
    xp, env, y = torch.randn(4, 6, 128), torch.randn(4, 2, 128), torch.tensor([0, 1, 1, 0])
    attn, _ = sf._attended_attention(model, xp, env, y)
    with torch.no_grad():
        _, ref = model(xp, env, return_attention_weights=True)
    for b in range(4):
        torch.testing.assert_close(attn[b], ref["attn_b" if y[b] else "attn_a"][b])


def test_edges_removed_restores_the_mask_and_k0_is_unchanged():
    torch.manual_seed(0)
    model = _make_stimulus_gat(film_pooling="corr").eval()
    scorer = sf.ElectrodeSpaceScorer(model)
    x, env, y = torch.randn(8, 6, 128), torch.randn(8, 2, 128), torch.randint(0, 2, (8,))
    edges = sf.directed_edges(model)
    before = model.gat.adj_mask.clone()
    with sf.edges_removed(model, edges):                      # every edge gone: only self-loops left
        assert model.gat.adj_mask.sum() == model.gat.adj_mask.shape[0]
        logits = scorer(x, env)
        assert torch.isfinite(logits).all()
    assert torch.equal(model.gat.adj_mask, before)
    curve = sf.edge_deletion_curve(scorer, x, env, y, edges, list(range(len(edges))), ks=[0])
    assert curve[0][1:] == sf.evaluate(scorer, x, env, y)


def test_edge_scores_shapes_and_attention_rows_sum_to_one():
    torch.manual_seed(0)
    model = _make_stimulus_gat(film_pooling="corr").eval()
    scorer = sf.ElectrodeSpaceScorer(model)
    x, env, y = torch.randn(6, 6, 128), torch.randn(6, 2, 128), torch.randint(0, 2, (6,))
    edges = sf.directed_edges(model)
    att = sf.edge_scores(scorer, x, env, y, edges, method="attention")
    axg = sf.edge_scores(scorer, x, env, y, edges, method="attn_x_grad")
    assert att.shape == axg.shape == (len(edges),) and np.isfinite(axg).all()
    attn, _ = sf._attended_attention(model, scorer.prepare(x), env, y)
    torch.testing.assert_close(attn.sum(-1), torch.ones_like(attn.sum(-1)))
    node = sf.attention_node_scores(scorer, x, env, y)
    assert node.shape == (6,) and node.sum() == pytest.approx(6.0, rel=1e-4)


# --- frequency bands ----------------------------------------------------------------------

def test_band_stop_removes_its_band_only():
    t = torch.arange(320) / 64.0
    x = (torch.sin(2 * np.pi * 9 * t) + torch.sin(2 * np.pi * 3 * t)).repeat(2, 3, 1)   # (2, 3, 320)
    y = sf.band_stop(x, 64.0, (8.0, 10.0))
    P = lambda v, f: torch.fft.rfft(v)[..., int(f * 320 / 64)].abs().mean()
    assert P(y, 9) < 0.1 * P(x, 9) and P(y, 3) > 0.9 * P(x, 3)


def test_band_feature_attribution_completeness():
    torch.manual_seed(0)
    model = _make_stimulus_gat(film_pooling="corr", freq_length_normalize=True).eval()
    scorer = sf.ElectrodeSpaceScorer(model)
    x, env, y = torch.randn(5, 6, 128), torch.randn(5, 2, 128), torch.randint(0, 2, (5,))
    attr = sf.band_feature_attribution(scorer, x, env, y, steps=128, batch=8)
    assert attr.shape == (8,)
    # completeness: summed attribution = mean margin change from batch-mean to actual band features
    enc, xp = model.frequency_encoder, scorer.prepare(x)
    feat = enc._band_power(xp)
    base = feat.mean(dim=0, keepdim=True).expand_as(feat)
    def margin_with(f):
        h = enc.proj.register_forward_pre_hook(lambda m, inp: (f.reshape(f.shape[0], -1, 1),))
        try:
            with torch.no_grad():
                return sf.margin(model(xp, env), y)
        finally:
            h.remove()
    gap = (margin_with(feat) - margin_with(base)).mean()
    assert attr.sum() == pytest.approx(float(gap), abs=2e-2, rel=5e-2)


# --- sanity -------------------------------------------------------------------------------

def test_cascade_sanity_randomises_every_module_and_decorrelates():
    torch.manual_seed(0)
    model = _make_stimulus_gat(film_pooling="corr").eval()
    scorer = sf.ElectrodeSpaceScorer(model)
    x, env, y = torch.randn(16, 6, 128), torch.randn(16, 2, 128), torch.randint(0, 2, (16,))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rho = sf.cascade_sanity(model, lambda m: sf.occlusion_scores(sf.ElectrodeSpaceScorer(m), x, env, y))
    assert list(rho)[0] == "readout" and list(rho)[-1] == "temporal_encoder"
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())   # original untouched
    assert all(-1.0 <= r <= 1.0 for r in rho.values())
